//! The STIR verifier state machine.
//!
//! ## What the verifier does, in one paragraph
//!
//! The verifier holds the public parameters `params` and the
//! round-by-round domain structure `stir_domain` — encoding the sequence
//! of evaluation domains `L_0, L_1, …, L_M` (each `L_i` a subset of the
//! Goldilocks field `F = F_p`) and the per-round degree bounds
//! `d_i = d_0 / k^i`. Given a [`crate::proof::StirProof`] from the
//! prover, it walks rounds in order: absorbs each round's commitment
//! into its own Fiat-Shamir transcript (faithfully recompiling the
//! prover's transcript), re-derives the round's OOD points and `t_i`
//! shift queries (the round-`i` shift-query count, set by the
//! repetition schedule) from the transcript state (so they match the
//! prover's), checks the prover's OOD replies and shift-answer Merkle
//! paths for the round-`i` committed function `f_i: L_i → F`, draws the
//! same round-`i` fold randomness `α_i`, and reconstructs the *expected*
//! next-round polynomial via [`crate::fold`] + [`crate::quotient`] +
//! [`crate::degree_correction`]. The proximity claim — that `f_0` is
//! δ-close in Hamming distance to some codeword of `RS[F, L_0, d_0]`
//! for the proximity parameter `δ` — is enforced through these
//! round-by-round checks. After `num_rounds`, it checks that the
//! prover's claimed `final_polynomial` (a) has degree
//! `< params.stopping_degree` and (b) agrees with the verifier's
//! independently-computed expected final polynomial on the shift queries
//! of the last round. Any disagreement → reject.
//!
//! ## Round-by-round (RBR) soundness
//!
//! STIR's soundness analysis is **round-by-round**: each round contributes
//! an additive sub-error to the total soundness loss, computed independently
//! and then union-bounded across rounds. Per round the error is the sum of:
//!
//! 1. **Fold-soundness term** (Theorem 4.2). Probability that the random
//!    `α_i` "rescues" a `δ`-far input: bounded by `(d_i - 1) / |F|` plus a
//!    list-decoding-radius adjustment.
//! 2. **Quotient distance term** (Lemma 4.4). Probability that the OOD
//!    quotient construction produces a deceptively-close-to-codeword
//!    function: bounded by `(s · k) / |F|` where `s = ood_samples`.
//! 3. **OOD list collapse** (Lemma 4.5). Probability that the OOD samples
//!    fail to pin the candidate list down to a single codeword: bounded by
//!    `(s · ε) / |F|` where `ε` is the list-decoding error margin.
//! 4. **Merkle binding** (collision-resistance of SHA3-256). Bounded by
//!    `2^{-256}` per query, negligible at any reasonable parameter setting.
//!
//! Sum these per round, then union-bound across `num_rounds` rounds, then
//! union-bound against the `repetition_schedule[i]` queries within each
//! round. The total soundness error is reported by
//! [`crate::params::StirParams::soundness_report`].
//!
//! ## Why RBR matters under Fiat-Shamir
//!
//! **Question for the reader.** Why is RBR soundness essential when
//! compiling STIR through Fiat-Shamir (BCS)? What goes wrong with monolithic
//! soundness alone?
//!
//! Under Fiat-Shamir, a cheating prover that already has a partially-formed
//! proof can **rewind and re-roll** the transcript: it tries different
//! prefixes (different first-round commitments, different OOD replies, ...)
//! until one produces favourable challenges deeper in the proof. With only a
//! monolithic soundness bound `ε_total`, the attacker effectively gets
//! `2^{security_bits} ≈ 1/ε_total` independent attempts at the *whole*
//! protocol, so `ε_total` blows up by exactly that factor — security
//! collapses. Round-by-round soundness fixes the error **per round**: at
//! each round, the conditional probability that the prover convinces the
//! verifier *given the previous rounds went through* is bounded by the
//! per-round error `ε_i`. Rewinding doesn't help because every fresh attempt
//! re-rolls every round's randomness independently. BCS's security proof
//! relies on RBR soundness, not monolithic soundness — so STIR (and FRI,
//! and any IOP that wants Fiat-Shamir compilation) must prove RBR. That's
//! the whole point of the per-round error decomposition above.
//!
//! ## Walk-through of one round's checks
//!
//! For round `i`:
//!
//! 1. **Absorb the commitment.** Pull `commitment_i = proof.round_commitments[i]`
//!    and absorb `commitment_i.root.0` into the transcript. *Must match*
//!    what the prover did at the same point.
//! 2. **Cross-check tree_size.** Reject if
//!    `commitment_i.tree_size != self.stir_domain.round_log_domain_size(i).pow(2)`.
//!    (See [`crate::commitment`] module docs for why we carry this size.)
//! 3. **Re-derive OOD points.** Call `ood::sample_ood_points(transcript,
//!    self.stir_domain.round_domain(i), self.params.ood_samples)` — the
//!    transcript state at this point is the same as the prover's, so the
//!    sampled points are identical.
//! 4. **Absorb OOD replies.** Push `proof.ood_replies[i]` into the
//!    transcript. (The prover *also* did this; we recompile faithfully.)
//! 5. **(Optional) Check PoW.** If `params.pow_bits[i] > 0`, verify the
//!    nonce in `proof.pow_nonces[i]` by re-grinding and checking leading
//!    zeros.
//! 6. **Re-derive shift indices.** Same trick as OOD: `transcript.sample_shift_indices(...)`.
//! 7. **Verify Merkle paths.** For each query `q ∈ 0..t_i`:
//!      - Let `leaf = proof.merkle_opened_leaves[i][q]` and
//!        `path  = proof.merkle_paths[i][q]`.
//!      - Reject if
//!        `!MerkleTree::verify(commitment_i.root, queries[q], leaf, &path, commitment_i.tree_size)`.
//!      - Reject if `leaf != proof.shift_answers[i][q]` (consistency
//!        between the opened leaf and the prover's claimed answer).
//! 8. **Absorb shift answers + paths.** Push into the transcript.
//! 9. **Draw `α_i = transcript.squeeze_field()`.** Same value the prover
//!    drew — Fiat-Shamir determinism.
//! 10. **Reconstruct expected `f_{i+1}`'s evaluation at next-round shift
//!     positions** by combining the OOD replies and shift answers through
//!     the fold/quotient/degcor machinery. Save the *expected* values
//!     for cross-check in round `i+1` (or against the final polynomial in
//!     the last round).
//!
//! After all rounds: check `proof.final_polynomial.degree() < params.stopping_degree`
//! and check that evaluating `final_polynomial` at the last round's
//! shift positions agrees with the verifier's expected values from step 10
//! of the previous iteration.
//!
//! `// CAUTION:` the verifier MUST check the Merkle paths against the
//! commitment root, NOT against a recomputed tree from the leaves it just
//! learned. The whole point of Merkle commitments is that the verifier
//! never sees the full leaf vector — checking against a recomputed tree
//! would be circular (the verifier would be checking its own re-hash, not
//! the prover's original commitment), and any adversary could trivially
//! make that succeed. Use [`crate::merkle::MerkleTree::verify`], which
//! takes only the root + path + claimed leaf — exactly what the verifier
//! is allowed to see.

use crate::domain::StirDomain;
use crate::params::StirParams;
use crate::proof::StirProof;
use crate::transcript::Transcript;

/// The STIR verifier.
///
/// Holds the static public-input state (the parameters and the cached
/// round-by-round domain structure) — exactly the inputs shared with
/// [`crate::prover::StirProver`]. The per-round state lives inside
/// [`Self::verify`] as local variables; the struct itself is constructed
/// once and consumed once.
pub struct StirVerifier {
    /// The fully-formed STIR parameters. Same value as the prover's;
    /// the verifier reads `repetition_schedule`, `ood_samples`,
    /// `folding_factor`, `num_rounds`, `stopping_degree`, and the per-round
    /// PoW configuration off this.
    pub params: StirParams,
    /// The round-by-round STIR domain structure
    /// `L_0 ⊃ L_1 ⊃ ... ⊃ L_M`. Precomputed once at construction and
    /// referenced by index during the verifier loop.
    pub stir_domain: StirDomain,
}

impl StirVerifier {
    /// Construct a verifier from STIR parameters.
    ///
    /// The verifier does not take the polynomial — it only sees the proof.
    /// The parameters are public and shared with the prover.
    ///
    /// # Paper reference
    /// STIR (eprint 2024/390), §3 "Protocol". The verifier's public input
    /// is the parameters; the proof is delivered via the transcript / proof
    /// object.
    pub fn new(params: StirParams) -> Self {
        // TODO:
        //   1. `params.validate()` (defensive double-check).
        //      WHY: a malformed StirParams gives a verifier that
        //      silently accepts or panics deep in the round loop. Better
        //      to fail at construction time with a clear message.
        //   2. Compute `stir_domain = StirDomain::new(&params)`.
        //      WHY: precompute once for O(1) per-round lookups.
        //   3. Return `Self { params, stir_domain }`.
        let _ = params;
        todo!()
    }

    /// Verify a STIR proof against a fresh Fiat-Shamir transcript.
    ///
    /// # Inputs
    /// - `proof`: the [`StirProof`] produced by the prover.
    /// - `transcript`: a freshly-initialised transcript (must match the
    ///   prover's initial state — typically created with the same domain
    ///   separator).
    ///
    /// # Outputs
    /// `Ok(())` if every check passes; `Err(&'static str)` describing the
    /// first failed check otherwise. The verifier returns early on the
    /// first failure — it does not exhaustively check downstream rounds
    /// once an early check has rejected.
    ///
    /// # Paper reference
    /// STIR (eprint 2024/390), §3 "Protocol" Figure 1 (verifier side); §4
    /// for the per-round soundness sub-errors that justify each check.
    pub fn verify(
        &self,
        proof: &StirProof,
        transcript: &mut Transcript,
    ) -> Result<(), &'static str> {
        // TODO: drive the STIR verifier state machine across `params.num_rounds` rounds.
        //
        //   Structural pre-checks (fast-fail):
        //     1. proof.round_commitments.len() == params.num_rounds
        //        else return Err("STIR: wrong round count in commitments").
        //        WHY: a length mismatch means a malformed proof. Catch it
        //        before doing any cryptographic work.
        //     2. proof.ood_replies.len() == params.num_rounds, similarly.
        //     3. proof.shift_answers.len() == params.num_rounds.
        //     4. proof.merkle_paths.len() == params.num_rounds.
        //     5. proof.merkle_opened_leaves.len() == params.num_rounds.
        //     6. For each round i: inner-length checks
        //          ood_replies[i].len()         == params.ood_samples
        //          shift_answers[i].len()       == params.repetition_schedule[i]
        //          merkle_paths[i].len()        == params.repetition_schedule[i]
        //          merkle_opened_leaves[i].len() == params.repetition_schedule[i].
        //        WHY: the inner-length mismatches are the second-cheapest
        //        structural failure mode.
        //
        //   Initialisation:
        //     - `expected_values: Vec<Fp> = vec![]` — holds, per round, the
        //       verifier's expected values for the next round's shift answers.
        //     - `prev_queries: Vec<usize> = vec![]` — last round's queries.
        //
        //   For `i in 0..self.params.num_rounds`:
        //     1. Let `commitment_i = &proof.round_commitments[i]`.
        //        Cross-check `commitment_i.tree_size` against
        //        `self.stir_domain.round_domain(i).size()`; reject if
        //        mismatched.
        //        WHY: per the commitment module's "Question", we want to
        //        catch round-misalignment bugs at the proof-object boundary.
        //     2. Absorb `commitment_i.root.0` into the transcript via
        //        `transcript.absorb_root(...)`.
        //     3. Re-derive OOD points: `let ood_points_i = ood::sample_ood_points(
        //          transcript, self.stir_domain.round_domain(i), self.params.ood_samples);`
        //     4. Absorb `proof.ood_replies[i]` into the transcript.
        //     5. (Optional) Re-derive shift indices:
        //        `let queries_i = transcript.sample_shift_indices(
        //          self.params.repetition_schedule[i], commitment_i.tree_size);`
        //     6. (Optional) PoW check: if `pow_bits[i] > 0`, verify
        //        `proof.pow_nonces[i]` by re-grinding the transcript state.
        //     7. For each query `q in 0..t_i`:
        //          let leaf = proof.merkle_opened_leaves[i][q];
        //          let path = &proof.merkle_paths[i][q];
        //          if !MerkleTree::verify(commitment_i.root.clone(),
        //                                  queries_i[q], leaf, path,
        //                                  commitment_i.tree_size) {
        //              return Err("STIR: Merkle path failed to verify");
        //          }
        //          if leaf != proof.shift_answers[i][q] {
        //              return Err("STIR: opened leaf does not match claimed shift answer");
        //          }
        //        WHY: each Merkle path check is a witness of "the prover
        //        committed to this value at this position". The leaf-vs-
        //        shift-answer consistency check is the link to the rest
        //        of the algebraic verification.
        //     8. Absorb `proof.shift_answers[i]` and the path siblings
        //        into the transcript.
        //     9. Draw `alpha_i = transcript.squeeze_field()` — same value
        //        the prover drew.
        //    10. Cross-check round i against round i-1's `expected_values`:
        //          if i > 0:
        //              for q in 0..t_i:
        //                  verify that proof.shift_answers[i][q] equals the
        //                  verifier's expected fold/quotient/degcor value
        //                  derived from prev round's outputs at queries_i[q].
        //              (Algorithmic detail: this is where Lemmas 4.2-4.5
        //              are applied as cross-round consistency checks.)
        //          Update `expected_values` for next round.
        //
        //   Final-polynomial check:
        //     11. Reject if `proof.final_polynomial.degree() >= Some(self.params.stopping_degree)`.
        //         WHY: the final polynomial is sent in the clear; checking
        //         its degree is the cheapest possible "is it actually
        //         low-degree" check and rules out an entire class of cheats.
        //     12. For each query of the last round, evaluate `final_polynomial`
        //         at the corresponding fold-domain point and compare against
        //         `expected_values[q]`.
        //         WHY: this is the protocol terminator — the final round
        //         agrees with the verifier's expectations only if every
        //         per-round fold/quotient/degcor was honest.
        //
        //   Return Ok(()).
        let _ = (proof, transcript);
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifier rejects a proof whose round count doesn't match params.
    #[test]
    fn verifier_rejects_proof_with_wrong_round_count() {
        // TODO:
        //   1. Build a StirParams with num_rounds = 2.
        //   2. Build a StirProof manually with round_commitments.len() = 1
        //      (mismatch).
        //   3. let verifier = StirVerifier::new(params).
        //   4. assert!(verifier.verify(&proof, &mut transcript).is_err()).
        // WHY: structural length checks are the verifier's first line of
        // defence. Cheap; must reject.
        todo!()
    }

    /// Verifier rejects a proof with a tampered Merkle path.
    #[test]
    fn verifier_rejects_tampered_merkle_path() {
        // TODO:
        //   1. Run an honest prove to get a valid proof.
        //   2. Mutate one byte of proof.merkle_paths[0][0].siblings[0].
        //   3. Reconstruct a fresh transcript and verify; assert Err.
        // WHY: SHA3-256 collision-resistance + correct verify logic should
        // catch this with overwhelming probability.
        todo!()
    }

    /// Verifier rejects a proof with a tampered final polynomial.
    #[test]
    fn verifier_rejects_wrong_final_polynomial() {
        // TODO:
        //   1. Run an honest prove to get a valid proof.
        //   2. Replace proof.final_polynomial with a different
        //      UnivariatePoly of the same degree (so the degree check
        //      doesn't fire — we test the *consistency* check).
        //   3. Verify; assert Err.
        // WHY: the final-polynomial cross-check is the protocol
        // terminator; if it doesn't fire on a wrong polynomial, the
        // whole proof would accept low-degree-like impostors.
        todo!()
    }
}
