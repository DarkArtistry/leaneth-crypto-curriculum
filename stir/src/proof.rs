//! The on-the-wire STIR proof object.
//!
//! ## What this module does
//!
//! [`StirProof`] is the **data the prover sends to the (non-interactive,
//! Fiat-Shamir-compiled) verifier**. Everything else in the crate either
//! produces a `StirProof` (the prover) or consumes one (the verifier).
//! The struct is a pure data container — no behaviour beyond what
//! `Clone` and `Debug` give you.
//!
//! Concretely, the proof is the round-by-round transcript of (writing
//! `F = F_p` for Goldilocks, `L_i` for the round-`i` evaluation domain,
//! `f_i: L_i → F` for the round-`i` committed function, `d_i = d_0 / k^i`
//! for the round-`i` degree bound, `s` for the OOD sample count, and
//! `k` for the folding factor):
//!
//! 1. A **Merkle commitment** to the round's committed function `f_i: L_i → F`
//!    (the Merkle root over its evaluation table on `L_i`, plus the table
//!    size).
//! 2. **OOD replies** — the prover's claimed evaluations, at the verifier-
//!    chosen out-of-domain points `z_1, ..., z_s`, of the unique degree-`< d_i`
//!    polynomial the committed function `f_i` is being tested for proximity to.
//! 3. **Shift-query answers** — the prover's claimed values of `f_i` at the
//!    verifier-chosen shift positions in `L_i`, plus the `k` leaves along the
//!    fold-axis at each query.
//! 4. **Merkle opening paths** — authentication paths against the round's
//!    Merkle root for each opened leaf.
//! 5. (Optionally) **PoW nonce** — a proof-of-work bump used to lower the
//!    verifier's per-round repetition count without sacrificing soundness.
//!
//! Plus, at the end of the protocol:
//!
//! 6. The **final polynomial** in coefficient form — degree-`< stopping_degree`
//!    and sent in the clear, so the verifier can run cheap degree and
//!    consistency checks instead of another Merkle-commit round.
//!
//! ## Sizing example
//!
//! With `log_initial_domain_size = 26` (so `|L_0| = 2^26 ≈ 6.7 × 10^7`),
//! `folding_factor = 16`, `num_rounds = 7`, `security_bits = 128`,
//! `ood_samples = 2`:
//!
//! - **Commitments.** 7 roots × 32 bytes ≈ **224 bytes**.
//! - **OOD replies.** 7 rounds × 2 samples × 8 bytes (Fp) = **112 bytes**.
//! - **Shift answers.** ≈ 3000 field elements total across all rounds (the
//!   harmonic-decreasing repetition schedule front-loads queries on cheap
//!   early rounds) → ≈ **24 KB**.
//! - **Merkle paths.** ≈ 3000 paths × `log_2(2^26) = 26` siblings × 32 bytes
//!   per sibling ≈ **2.5 MB**. *Dominant cost.*
//! - **Final polynomial.** `stopping_degree` coefficients × 8 bytes ≈
//!   **64-256 bytes**.
//!
//! Total ≈ a few hundred KB to a couple of MB depending on the schedule. The
//! whole point of STIR over FRI is that this is **1.25-2.46× smaller than the
//! FRI proof for the same security level** — fewer queries × `O(log|L_0|)`
//! Merkle hashes per query is the dominant saving.
//!
//! ## Worked example, n_rounds = 2
//!
//! Take `log_initial_domain_size = 6`, `folding_factor = 4`, `num_rounds = 2`,
//! `repetition_schedule = [4, 2, 1]`, `ood_samples = 1`, `stopping_degree = 4`.
//!
//! The proof looks like:
//!
//! ```text
//! round_commitments    = [c_0, c_1]                       // 2 roots
//! ood_replies          = [[ood_0], [ood_1]]               // 1 reply per round
//! shift_answers        = [[a_0_0, a_0_1, a_0_2, a_0_3],   // 4 shift queries in round 0
//!                         [a_1_0, a_1_1]]                  // 2 in round 1
//! merkle_paths         = [[p_0_0, p_0_1, p_0_2, p_0_3],   // one path per shift query
//!                         [p_1_0, p_1_1]]
//! merkle_opened_leaves = [[l_0_0, l_0_1, l_0_2, l_0_3],   // the leaves the paths open to
//!                         [l_1_0, l_1_1]]
//! final_polynomial     = UnivariatePoly(degree < 4)        // sent in the clear
//! pow_nonces           = [nonce_0, nonce_1]                // 1 per round
//! ```
//!
//! The outer indexing is **round**; the inner indexing is **query within
//! round** (or **OOD sample within round** for `ood_replies`). The verifier
//! iterates rounds in order, absorbing each commitment into the Fiat-Shamir
//! transcript before deriving the next round's challenges.
//!
//! `// CAUTION:` the verifier MUST absorb each `round_commitments[i]` into the
//! transcript BEFORE deriving the next round's OOD points / shift queries.
//! Order matters for Fiat-Shamir security: if the prover could see the
//! verifier's next-round challenges before committing, the prover could pick a
//! commitment that aligns with those challenges, breaking soundness. The
//! contract is "commit, then challenge, then respond" — the same order the
//! interactive protocol uses, faithfully recompiled.
//!
//! **Question for the reader.** Why doesn't the proof carry the full
//! evaluation tables of `f_i: L_i → F`? Wouldn't that let the verifier check
//! everything directly?
//!
//! Yes, it would — and that would be a linear-size proof, which is exactly
//! what STIR (and FRI before it) exists to avoid. The whole point of an IOP of
//! proximity is that the verifier never reads the full evaluation table: it
//! sees only a Merkle commitment plus `O(log² d)` openings, and the soundness
//! argument turns "few random openings agree with the committed root" into
//! "the committed function is `δ`-close (in Hamming distance) to a Reed-
//! Solomon codeword". Sending the full evaluation table would also defeat
//! the *non-interactive* compilation — the proof would grow with the domain
//! size instead of being polylogarithmic in it.
//!
//! ## Indexing convention
//!
//! Every per-round field is a `Vec<Vec<_>>` indexed as `[round][inner_index]`.
//! There are exactly `num_rounds` entries in each outer vec for the per-round
//! fields, except `final_polynomial` (one) and `pow_nonces` (length matches
//! the number of rounds that actually used PoW; may be empty for runs with
//! `pow_bits = 0`).

use reed_solomon::{Fp, UnivariatePoly};

use crate::commitment::StirCommitment;
use crate::merkle::MerklePath;

/// The on-the-wire STIR proof.
///
/// Constructed by [`crate::prover::StirProver::prove`], consumed by
/// [`crate::verifier::StirVerifier::verify`]. Every field is `pub` because
/// the verifier reads them all directly; the prover serialises this struct
/// (or its proof-bytes equivalent) over whatever channel carries the
/// transcript.
///
/// All `Vec<Vec<_>>` fields use `[round][inner_index]` indexing. The outer
/// length equals `num_rounds` for `round_commitments`, `ood_replies`,
/// `shift_answers`, `merkle_paths`, `merkle_opened_leaves`. `pow_nonces` has
/// one entry per round only if PoW is enabled; otherwise it may be empty.
#[derive(Clone, Debug)]
pub struct StirProof {
    /// `round_commitments[i]` is the Merkle commitment to the round-`i`
    /// function `f_i: L_i → F` (the Merkle root over its evaluation table on
    /// `L_i`). Length `= num_rounds`. The verifier absorbs
    /// `round_commitments[i].root` into the transcript at the start of round
    /// `i`; everything else in round `i` is derived (challenge-side) from the
    /// post-absorption transcript state.
    pub round_commitments: Vec<StirCommitment>,
    /// `ood_replies[i][s]` is the prover's claimed value of `f_i(z_{i,s})`
    /// where `z_{i,s}` is the `s`-th out-of-domain point sampled in round
    /// `i`. Outer length `= num_rounds`; inner length `= params.ood_samples`.
    pub ood_replies: Vec<Vec<Fp>>,
    /// `shift_answers[i][q]` is the prover's claimed value of `f_i(α_{i,q})`
    /// at the `q`-th shift query in round `i`. Outer length `= num_rounds`;
    /// inner length `= params.repetition_schedule[i]` (the number of shift
    /// queries in round `i`).
    ///
    /// In the chunked variant of STIR (not implemented here at first pass)
    /// each entry would be a `k`-vector — one leaf for each fold-axis offset.
    /// For the basic variant we treat each query as one field-element answer.
    pub shift_answers: Vec<Vec<Fp>>,
    /// `merkle_paths[i][q]` is the authentication path that opens the leaf
    /// at the shift-query position in `round_commitments[i]`. The verifier
    /// checks each path against `round_commitments[i].root` using
    /// [`crate::merkle::MerkleTree::verify`]. Outer length `= num_rounds`;
    /// inner length `= params.repetition_schedule[i]`.
    pub merkle_paths: Vec<Vec<MerklePath>>,
    /// `merkle_opened_leaves[i][q]` is the field element that
    /// `merkle_paths[i][q]` claims to open. Carrying the leaves explicitly
    /// (rather than reading them off `shift_answers`) keeps the Merkle-
    /// verification call site self-contained and matches the structure the
    /// verifier already needs: it cross-checks `merkle_opened_leaves[i][q]`
    /// against `shift_answers[i][q]` separately from the Merkle-path check.
    pub merkle_opened_leaves: Vec<Vec<Fp>>,
    /// The final-round polynomial sent in the clear. Its degree must satisfy
    /// `final_polynomial.degree() < params.stopping_degree`, which the
    /// verifier explicitly checks (a wrong-degree final polynomial is one of
    /// the protocol's named rejection modes — see
    /// [`crate::verifier::StirVerifier::verify`]).
    pub final_polynomial: UnivariatePoly,
    /// Optional per-round PoW nonces. Empty when `params.pow_bits == 0` for
    /// every round; otherwise length equals `num_rounds` (with zero-nonce
    /// entries for any rounds where the schedule disabled PoW).
    ///
    /// The verifier checks each nonce by re-hashing the transcript state ||
    /// nonce and confirming the result has the required number of leading
    /// zero bits. PoW lowers the per-round repetition count by trading
    /// prover work (the brute-force search) for proof-size and verifier
    /// query budget.
    pub pow_nonces: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-round outer vectors all have length `num_rounds`.
    #[test]
    fn proof_round_count_matches_params() {
        // TODO:
        //   1. Build a StirParams with num_rounds = 2.
        //   2. Construct (or `run_stir` to produce) a StirProof.
        //   3. Assert all of round_commitments, ood_replies, shift_answers,
        //      merkle_paths, merkle_opened_leaves have len() == 2.
        // WHY: outer-length consistency is the most likely place for the
        // prover to drift out of sync with the params; the verifier's first
        // structural check pins this down before any cryptography runs.
        todo!()
    }

    /// The proof carries exactly one final polynomial.
    #[test]
    fn proof_has_one_final_polynomial() {
        // TODO:
        //   1. Build a StirProof (via the prover).
        //   2. Assert `proof.final_polynomial.degree() < stopping_degree`
        //      (or is the zero polynomial, which has no degree).
        // WHY: the final polynomial is the protocol's terminator — sent in
        // the clear, degree-bounded, single instance. This test is a
        // contract anchor for the prover's `final_polynomial` step.
        todo!()
    }
}
