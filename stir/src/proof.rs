//! The on-the-wire STIR proof object.
//!
//! ## What this module does
//!
//! [`StirProof`] is the **data the prover sends to the (non-interactive,
//! Fiat-Shamir-compiled) verifier**. Everything else in the crate either
//! produces a `StirProof` (the prover) or consumes one (the verifier).
//! The struct is a pure data container — no behaviour beyond what
//! `Clone` and `Debug` give you, plus a single inspection helper
//! ([`StirProof::num_rounds`]).
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
//! 5. (Optionally, but **disabled for this educational iteration**) PoW
//!    nonces — see the [`StirProof::pow_nonces`] field doc.
//!
//! Plus, at the end of the protocol:
//!
//! 6. The **final polynomial** in coefficient form — degree-`< stopping_degree`
//!    and sent in the clear, so the verifier can run cheap degree and
//!    consistency checks instead of another Merkle-commit round.
//!
//! ## Anchor: what the prover sends, what the verifier reads
//!
//! [`StirProof`] is the **complete, self-contained payload** of a STIR
//! run. It is what the prover [`crate::prover::StirProver::prove`]
//! returns, and the **only** input the verifier
//! [`crate::verifier::StirVerifier::verify`] gets besides the public
//! [`crate::params::StirParams`] and the original commitment. Reading
//! this struct is therefore the fastest way to understand the protocol:
//!
//! - **The prover's job** is, for each round `i ∈ 0..num_rounds`, to
//!   produce `(commitment_i, ood_replies_i, shift_answers_i, paths_i,
//!   opened_leaves_i)` consistent with a single underlying low-degree
//!   polynomial, and at the end to send `final_polynomial`.
//! - **The verifier's job** is, before running any cryptography, to
//!   check the **shape** of the proof (the per-round invariant theorem
//!   below); then for each round `i`, to absorb `round_commitments[i]`
//!   into the Fiat-Shamir transcript, *derive* the OOD points / shift
//!   indices / folding randomness from the post-absorption transcript
//!   state, and check that the prover's responses are consistent with
//!   the derived challenges. Finally it degree-bounds `final_polynomial`
//!   and runs the terminal consistency check.
//!
//! `// CAUTION:` the verifier MUST absorb each `round_commitments[i]`
//! into the transcript BEFORE deriving round-`i`'s OOD points / shift
//! queries. Order matters for Fiat-Shamir security: if the prover could
//! see the verifier's next-round challenges before committing, the
//! prover could pick a commitment that aligns with those challenges,
//! breaking soundness. The contract is "commit, then challenge, then
//! respond" — the same order the interactive protocol uses, faithfully
//! recompiled.
//!
//! ## Named theorem: Proof-Shape Invariant
//!
//! > **Proof-Shape Invariant theorem.** Let `proof: StirProof` be the
//! > output of an honest run of STIR on parameters `p: StirParams`,
//! > with `M = p.num_rounds`, `t_i = p.repetition_schedule[i]`,
//! > `s = p.ood_samples`, and `k = p.folding_factor`. Then every one of
//! > the following holds:
//! >
//! > ```text
//! > proof.round_commitments.len()    == M
//! > proof.ood_replies.len()          == M
//! > proof.shift_answers.len()        == M
//! > proof.merkle_paths.len()         == M
//! > proof.merkle_opened_leaves.len() == M
//! >
//! > for every i ∈ 0..M:
//! >     proof.ood_replies[i].len()          == s
//! >     proof.shift_answers[i].len()        == t_i
//! >     proof.merkle_paths[i].len()         == t_i
//! >     proof.merkle_opened_leaves[i].len() == t_i
//! >     proof.round_commitments[i].tree_size == 2^p.round_log_domain_size(i)
//! >     for every q ∈ 0..t_i:
//! >         proof.shift_answers[i][q].len()        == k
//! >         proof.merkle_paths[i][q].len()         == k
//! >         proof.merkle_opened_leaves[i][q].len() == k
//! >
//! > proof.final_polynomial.degree() < p.stopping_degree   (or is the zero polynomial)
//! > proof.pow_nonces.is_empty()                            (PoW descoped, see field doc)
//! > ```
//!
//! **Proof sketch.** Each invariant is a direct restatement of the
//! prover's round-`i` loop body: the prover commits exactly once per
//! round (one Merkle root), answers exactly `s` OOD samples (one per
//! verifier-derived OOD point), and opens exactly `t_i` shift queries.
//! For each shift query, the prover opens **exactly `k` Merkle paths**
//! — one per preimage of the query under the fold map — and reports
//! the `k` corresponding leaves both in `merkle_opened_leaves[i][q]`
//! and (redundantly, as a defense-in-depth cross-check) in
//! `shift_answers[i][q]`. The triple-nested shape `[round][query][preimage]`
//! is therefore intrinsic to the per-paper Construction 5.2 fold step:
//! verifying `g_i = Fold(f_i, α_i)` at a single query point requires
//! all `k` fibre siblings, not just one. The inner `[j ∈ 0..k]`
//! dimension is the `k` preimages opened per shift query; see §5 of
//! `stir-full-spec.md` for the index formula
//! (`preimage_indices[j] = q_idx + j · (n_i / k)`) and `fold.rs` for
//! the natural fibre layout that pins the sibling stride. The final-
//! round polynomial is sent once and only once. The
//! `pow_nonces.is_empty()` clause is a property of *this* implementation
//! (PoW is descoped); a production STIR run would have
//! `pow_nonces.len() == M` instead. ∎
//!
//! Why this matters: the verifier's **first structural check** (before
//! any cryptography runs) is to reject proofs whose shape mismatches
//! the invariant. A length-mismatched proof is malformed at the wire
//! level; rejecting fast saves CPU on adversarial inputs. The check is
//! also useful as a *prover sanity assertion*: it pins down off-by-one
//! drifts in the round loop at the boundary of the proof object,
//! instead of letting them propagate into cryptic Merkle-path-length
//! failures deep in the verifier.
//!
//! See [`StirProof::num_rounds`] for the convenience accessor the
//! verifier uses to cross-check `M`.
//!
//! ## Worked example — demo parameters
//!
//! Take the curriculum's demo parameters:
//!
//! ```text
//! log_initial_domain_size = 6     →  |L_0| = 64
//! folding_factor          = 4     →  k     = 4
//! num_rounds              = 2     →  M     = 2
//! ood_samples             = 2     →  s     = 2
//! repetition_schedule     = [8, 4, 2]   (length M + 1 = 3)
//! ```
//!
//! So `t_0 = 8`, `t_1 = 4`, `t_final = 2`. The per-round outer fields
//! of [`StirProof`] therefore have length `M = 2` (the final-round
//! `t_final = 2` governs the verifier's terminal index queries against
//! `round_commitments[M-1]`, not a fresh outer-vec entry).
//!
//! The proof shape (writing each per-query inner vec as `[·; k]` for
//! brevity — each is exactly `k = 4` preimages opened against the
//! round-`i` Merkle tree):
//!
//! ```text
//! round_commitments    = [c_0, c_1]                                  // 2 roots
//! ood_replies          = [[ood_0_0, ood_0_1],                        // s = 2 per round
//!                         [ood_1_0, ood_1_1]]
//! shift_answers        = [[ [·; 4]; 8 ],                             // t_0 = 8 queries × k = 4 preimages
//!                         [ [·; 4]; 4 ]]                             // t_1 = 4 queries × k = 4 preimages
//! merkle_paths         = [[ [·; 4]; 8 ],                             // t_0 × k paths
//!                         [ [·; 4]; 4 ]]                             // t_1 × k paths
//! merkle_opened_leaves = [[ [·; 4]; 8 ],                             // t_0 × k leaves
//!                         [ [·; 4]; 4 ]]                             // t_1 × k leaves
//! final_polynomial     = UnivariatePoly(degree < stopping_degree)
//! pow_nonces           = []                                           // descoped
//! ```
//!
//! Total per-round openings: `k · (t_0 + t_1) = 4 · 12 = 48` Merkle
//! paths + 48 leaves + 48 shift-answer claims, plus `2 · s = 4` OOD
//! field elements, plus 2 Merkle roots, plus one short final polynomial.
//!
//! ## Sizing example (production-scale)
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
//! `ood_replies` is a `Vec<Vec<Fp>>` indexed as `[round][ood_sample]`.
//! `shift_answers`, `merkle_paths`, and `merkle_opened_leaves` are each
//! `Vec<Vec<Vec<_>>>` indexed as `[round][query][preimage]` — the inner
//! dimension is the `k` fold-axis preimages opened per shift query
//! (see §5 of `stir-full-spec.md` for the index formula and `fold.rs`
//! for the fibre layout). There are exactly `num_rounds` entries in
//! each outer vec for the per-round fields, except `final_polynomial`
//! (one) and `pow_nonces` (always empty in this implementation — PoW
//! is descoped).

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
/// `ood_replies` uses `[round][ood_sample]` indexing. `shift_answers`,
/// `merkle_paths`, and `merkle_opened_leaves` use `[round][query][preimage]`
/// indexing — the inner dimension is the `k = params.folding_factor`
/// fold-axis preimages opened per shift query. The outer length equals
/// `num_rounds` for all five per-round fields. `pow_nonces` is always
/// empty in this implementation (PoW is descoped — see field doc).
///
/// The shape invariants are stated and proved in the module-level
/// **Proof-Shape Invariant theorem**. Use [`StirProof::num_rounds`] to
/// read off `M = num_rounds` directly from the proof for cross-checks.
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
    /// `shift_answers[i][q][j]` is the prover's claimed value of `f_i(x_{i,q,j})`
    /// where `x_{i,q,j}` is the `j`-th of the `k` preimages (under the fold
    /// map) corresponding to the `q`-th shift query in round `i`. Outer length
    /// `= num_rounds`; middle length `= params.repetition_schedule[i]` (the
    /// number of shift queries in round `i`); inner length
    /// `= params.folding_factor = k`.
    ///
    /// The inner `[j ∈ 0..k]` dimension is the `k` preimages opened per shift
    /// query — one Merkle-authenticated value per fold-axis fibre sibling.
    /// See §5 of `stir-full-spec.md` for the preimage index formula
    /// (`stride = n_i / k`) and `fold.rs` for the natural fibre layout that
    /// pins those sibling positions.
    pub shift_answers: Vec<Vec<Vec<Fp>>>,
    /// `merkle_paths[i][q][j]` is the authentication path that opens the
    /// `j`-th preimage leaf at the `q`-th shift query in `round_commitments[i]`.
    /// The verifier checks each path against `round_commitments[i].root` using
    /// [`crate::merkle::MerkleTree::verify`]. Outer length `= num_rounds`;
    /// middle length `= params.repetition_schedule[i]`; inner length `= k`.
    pub merkle_paths: Vec<Vec<Vec<MerklePath>>>,
    /// `merkle_opened_leaves[i][q][j]` is the field element that
    /// `merkle_paths[i][q][j]` claims to open. Carrying the leaves explicitly
    /// (rather than reading them off `shift_answers`) keeps the Merkle-
    /// verification call site self-contained and matches the structure the
    /// verifier already needs: it cross-checks `merkle_opened_leaves[i][q][j]`
    /// against `shift_answers[i][q][j]` separately from the Merkle-path check.
    /// Outer length `= num_rounds`; middle `= params.repetition_schedule[i]`;
    /// inner `= k`.
    pub merkle_opened_leaves: Vec<Vec<Vec<Fp>>>,
    /// The final-round polynomial sent in the clear. Its degree must satisfy
    /// `final_polynomial.degree() < params.stopping_degree`, which the
    /// verifier explicitly checks (a wrong-degree final polynomial is one of
    /// the protocol's named rejection modes — see
    /// [`crate::verifier::StirVerifier::verify`]).
    pub final_polynomial: UnivariatePoly,
    /// **Always empty in this educational implementation.** Proof-of-work
    /// is descoped from the curriculum — the prover produces `Vec::new()`
    /// and the verifier asserts emptiness. A production STIR would set
    /// `pow_nonces.len() == num_rounds`, one nonce per round, where each
    /// nonce is a `u64` such that re-hashing the round's transcript state
    /// concatenated with the nonce produces a digest with the required
    /// number of leading zero bits. PoW lowers the per-round repetition
    /// count by trading prover work (the brute-force search) for proof-size
    /// and verifier query budget; we omit it because the security analysis
    /// of the unboosted protocol is already complete and clearer to study.
    pub pow_nonces: Vec<u64>,
}

impl StirProof {
    /// The number of folding rounds this proof spans.
    ///
    /// By the **Proof-Shape Invariant theorem** (see module docs), this
    /// equals `params.num_rounds` for any honest proof — and that
    /// equality is the verifier's first structural check. We read it off
    /// `round_commitments.len()` because that field is the protocol's
    /// canonical per-round outer vector: every other per-round field is
    /// required to have the same outer length.
    ///
    /// # Cross-reference
    ///
    /// The verifier in [`crate::verifier::StirVerifier::verify`] should
    /// assert `proof.num_rounds() == params.num_rounds as usize` before
    /// any cryptography runs; see the **Proof-Shape Invariant theorem**
    /// for the full list of per-round length checks that follow.
    pub fn num_rounds(&self) -> usize {
        self.round_commitments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::StirCommitment;
    use crate::merkle::{MerklePath, MerkleRoot};

    /// Build a default-shape `StirProof` matching the demo parameters
    /// (`log|L_0| = 6, k = 4, M = 2, s = 2, repetition_schedule = [8, 4, 2]`).
    ///
    /// The contents are placeholders — the only thing this helper pins
    /// down is the **shape**, exactly as required by the Proof-Shape
    /// Invariant theorem in the module docs.
    fn demo_proof() -> StirProof {
        // Demo parameters: M = 2 rounds, s = 2 OOD samples, t_0 = 8,
        // t_1 = 4. Final-round t_final = 2 is the verifier's terminal
        // query count against `round_commitments[M-1]` and does NOT
        // produce a fresh outer-vec entry.
        let num_rounds: usize = 2;
        let ood_samples: usize = 2;
        let per_round_queries: [usize; 2] = [8, 4];
        let folding_factor: usize = 4; // k = 4 preimages per shift query.
        // Round-`i` domain sizes from the domain-shrink formula
        // `log|L_i| = log|L_0| − i · log₂(k/2)`. With log|L_0| = 6 and
        // log₂(k/2) = 1 this gives |L_0| = 64, |L_1| = 32.
        let round_tree_sizes: [usize; 2] = [64, 32];

        // round_commitments: one StirCommitment per round.
        let round_commitments: Vec<StirCommitment> = (0..num_rounds)
            .map(|i| StirCommitment {
                root: MerkleRoot([0u8; 32]),
                tree_size: round_tree_sizes[i],
            })
            .collect();

        // ood_replies[i] has length `s` (= ood_samples).
        let ood_replies: Vec<Vec<Fp>> = (0..num_rounds)
            .map(|_| vec![Fp::zero(); ood_samples])
            .collect();

        // shift_answers[i][q] has length `k` (= folding_factor): one
        // Merkle-authenticated value per fold-axis preimage. Outer
        // length `num_rounds`, middle length `t_i`.
        let shift_answers: Vec<Vec<Vec<Fp>>> = (0..num_rounds)
            .map(|i| {
                (0..per_round_queries[i])
                    .map(|_| vec![Fp::zero(); folding_factor])
                    .collect()
            })
            .collect();

        // merkle_paths[i][q] has length `k`, one path per preimage. Each
        // path's siblings vector has length log₂(padded_tree_size); we
        // don't exercise verify() here so an empty siblings vec is fine
        // for the shape test.
        let merkle_paths: Vec<Vec<Vec<MerklePath>>> = (0..num_rounds)
            .map(|i| {
                (0..per_round_queries[i])
                    .map(|q| {
                        (0..folding_factor)
                            .map(|j| MerklePath {
                                siblings: Vec::new(),
                                leaf_index: q * folding_factor + j,
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // merkle_opened_leaves[i][q] has length `k`, mirroring merkle_paths.
        let merkle_opened_leaves: Vec<Vec<Vec<Fp>>> = (0..num_rounds)
            .map(|i| {
                (0..per_round_queries[i])
                    .map(|_| vec![Fp::zero(); folding_factor])
                    .collect()
            })
            .collect();

        // final_polynomial: zero polynomial trivially has degree < any
        // positive stopping_degree.
        let final_polynomial = UnivariatePoly::zero();

        // pow_nonces: always empty in this implementation.
        let pow_nonces: Vec<u64> = Vec::new();

        StirProof {
            round_commitments,
            ood_replies,
            shift_answers,
            merkle_paths,
            merkle_opened_leaves,
            final_polynomial,
            pow_nonces,
        }
    }

    /// **Proof-Shape Invariant test (original `todo!` slot 1).**
    ///
    /// The per-round outer vectors all have length `num_rounds`, the
    /// OOD inner length matches `s`, and each per-round inner length
    /// matches the demo `repetition_schedule`. This is the structural
    /// check the verifier runs before any cryptography.
    #[test]
    fn proof_round_count_matches_params() {
        // Demo params: M = 2, s = 2, t_0 = 8, t_1 = 4.
        let proof = demo_proof();

        // num_rounds() helper agrees with the canonical outer length.
        assert_eq!(proof.num_rounds(), 2);

        // Every per-round outer vec has length M.
        assert_eq!(proof.round_commitments.len(), 2);
        assert_eq!(proof.ood_replies.len(), 2);
        assert_eq!(proof.shift_answers.len(), 2);
        assert_eq!(proof.merkle_paths.len(), 2);
        assert_eq!(proof.merkle_opened_leaves.len(), 2);

        // OOD inner length = s = 2 in every round.
        for replies in &proof.ood_replies {
            assert_eq!(replies.len(), 2);
        }

        // Shift / path / leaf middle lengths follow the demo schedule,
        // and each per-query inner length equals k = 4 (the fold-axis
        // preimage count).
        let expected_t = [8usize, 4];
        let k: usize = 4;
        for i in 0..2 {
            assert_eq!(proof.shift_answers[i].len(), expected_t[i]);
            assert_eq!(proof.merkle_paths[i].len(), expected_t[i]);
            assert_eq!(proof.merkle_opened_leaves[i].len(), expected_t[i]);
            for q in 0..expected_t[i] {
                assert_eq!(proof.shift_answers[i][q].len(), k);
                assert_eq!(proof.merkle_paths[i][q].len(), k);
                assert_eq!(proof.merkle_opened_leaves[i][q].len(), k);
            }
        }

        // PoW is descoped — must be empty.
        assert!(
            proof.pow_nonces.is_empty(),
            "pow_nonces must be empty in this implementation",
        );
    }

    /// **Final-polynomial test (original `todo!` slot 2).**
    ///
    /// The proof carries exactly one final polynomial. The zero
    /// polynomial has `degree() == None`, which the verifier treats as
    /// satisfying `degree < stopping_degree` for any positive
    /// `stopping_degree`. A non-zero final polynomial must report
    /// `Some(d)` with `d < stopping_degree`.
    #[test]
    fn proof_has_one_final_polynomial() {
        let proof = demo_proof();
        let stopping_degree: usize = 1;

        match proof.final_polynomial.degree() {
            None => {
                // Zero polynomial — trivially degree-bounded.
            }
            Some(d) => assert!(
                d < stopping_degree,
                "final_polynomial degree {d} must be < stopping_degree {stopping_degree}",
            ),
        }
    }

    /// `Debug` formatting must produce non-empty output for every field.
    /// Smoke test: catches accidental `#[derive(Debug)]` removals or
    /// custom `Debug` impls that swallow useful state.
    #[test]
    fn proof_debug_output_is_non_empty() {
        let proof = demo_proof();
        let s = format!("{:?}", proof);
        assert!(!s.is_empty(), "Debug output is empty");
        // Sanity: the struct name should appear in the default derived
        // Debug output.
        assert!(
            s.contains("StirProof"),
            "Debug output missing struct name: {s}",
        );
    }

    /// `Clone` must preserve every field. We re-run the same shape
    /// checks against the clone to confirm no field is silently dropped
    /// or aliased by the clone implementation.
    #[test]
    fn proof_clone_roundtrip_preserves_fields() {
        let original = demo_proof();
        let cloned = original.clone();

        // Outer lengths match the original on every per-round field.
        assert_eq!(cloned.round_commitments.len(), original.round_commitments.len());
        assert_eq!(cloned.ood_replies.len(), original.ood_replies.len());
        assert_eq!(cloned.shift_answers.len(), original.shift_answers.len());
        assert_eq!(cloned.merkle_paths.len(), original.merkle_paths.len());
        assert_eq!(
            cloned.merkle_opened_leaves.len(),
            original.merkle_opened_leaves.len()
        );
        assert_eq!(cloned.pow_nonces.len(), original.pow_nonces.len());

        // Inner lengths agree round-by-round and (for the triple-nested
        // fields) preimage-by-preimage.
        for i in 0..original.num_rounds() {
            assert_eq!(cloned.ood_replies[i].len(), original.ood_replies[i].len());
            assert_eq!(
                cloned.shift_answers[i].len(),
                original.shift_answers[i].len()
            );
            assert_eq!(
                cloned.merkle_paths[i].len(),
                original.merkle_paths[i].len()
            );
            assert_eq!(
                cloned.merkle_opened_leaves[i].len(),
                original.merkle_opened_leaves[i].len()
            );
            for q in 0..original.shift_answers[i].len() {
                assert_eq!(
                    cloned.shift_answers[i][q].len(),
                    original.shift_answers[i][q].len()
                );
                assert_eq!(
                    cloned.merkle_paths[i][q].len(),
                    original.merkle_paths[i][q].len()
                );
                assert_eq!(
                    cloned.merkle_opened_leaves[i][q].len(),
                    original.merkle_opened_leaves[i][q].len()
                );
            }
        }

        // Commitments compare bitwise equal (StirCommitment: PartialEq).
        assert_eq!(cloned.round_commitments, original.round_commitments);

        // Final polynomial degree agrees.
        assert_eq!(
            cloned.final_polynomial.degree(),
            original.final_polynomial.degree(),
        );

        // num_rounds() agrees.
        assert_eq!(cloned.num_rounds(), original.num_rounds());
    }
}
