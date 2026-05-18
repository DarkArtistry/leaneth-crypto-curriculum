//! The STIR prover state machine.
//!
//! ## What the prover does, in one paragraph
//!
//! The honest prover holds the public polynomial `f_0 ∈ F[X]` — over
//! the Goldilocks field `F = F_p` — of degree `< d_0` (the **initial
//! degree bound** `initial_degree_bound`), the input "low-degree
//! witness". The protocol runs `M = num_rounds` rounds at folding
//! factor `k = folding_factor`. Per round `i` the prover (a) computes
//! the evaluation table of the round-`i` committed function `f_i` on
//! the round-`i` evaluation domain `L_i` via FFT — this evaluation
//! table is the committed function `f_i: L_i → F` — (b) Merkle-commits
//! to that table and sends the root, (c) receives from the transcript a
//! set of OOD points and shift positions, (d) responds with OOD
//! evaluations + shift answers + Merkle opening paths, (e) draws the
//! round-`i` fold randomness `α_i` from the transcript and
//! folds/quotients/degree-corrects to produce `f_{i+1}`, a polynomial of
//! degree `< d_{i+1} = d_0 / k^{i+1}`. After `M` iterations the
//! remaining polynomial has degree `< stopping_degree` and is sent in
//! the clear as the proof's [`crate::proof::StirProof::final_polynomial`].
//!
//! ## One iteration, step by step
//!
//! For round `i ∈ 0..num_rounds`:
//!
//! 1. **Evaluate.** Compute `evals_i = FFT(f_i, L_i)` — the evaluation table
//!    of `f_i` on `L_i`, size `|L_i|` (for the honest prover, this is a
//!    Reed-Solomon codeword in `RS[F, L_i, d_i]`).
//! 2. **Commit.** Call [`crate::commitment::StirCommitment::commit`] on
//!    `&evals_i` to get `(commitment_i, tree_i)` — a Merkle commitment to
//!    the function `f_i: L_i → F` defined by `evals_i`. Absorb
//!    `commitment_i.root` into the transcript.
//! 3. **Derive OOD points.** Draw `s = params.ood_samples` field elements
//!    `z_{i,1}, ..., z_{i,s}` from the transcript (rejection-sampled out of
//!    `L_i`). Reply with `ood_replies_i[s] = f_i(z_{i,s})` — these are
//!    evaluations on the *coefficient form*, computed via Horner.
//! 4. **Absorb OOD replies.** Push `ood_replies_i` into the transcript.
//!    Order matters — see the `// CAUTION:` block below.
//! 5. **Derive shift queries.** Draw
//!    `t = params.repetition_schedule[i]` integer indices
//!    `q_{i,1}, ..., q_{i,t}` from the transcript (each pointing at a leaf
//!    of `tree_i`). Reply with `shift_answers_i[j] = evals_i[q_{i,j}]` and
//!    `merkle_paths_i[j] = tree_i.open(q_{i,j}).1` plus the opened leaves.
//! 6. **Absorb shift answers + Merkle paths.** Push them all into the
//!    transcript.
//! 7. **Derive fold randomness `α_i`.** Draw one field element from the
//!    post-absorption transcript state. Compute the next-round polynomial:
//!
//!    ```text
//!    f_{i+1}(X) = DegCor(Quotient(Fold(f_i, α_i, k), ood_points_i, ood_replies_i))
//!    ```
//!
//!    where `Fold` reduces degree by `k = folding_factor`, `Quotient` removes
//!    the OOD-induced low-weight error correlation, and `DegCor` (degree
//!    correction) multiplies by a fixed polynomial to land back in the
//!    exact next-round code `RS[F, L_{i+1}, d_{i+1}]`.
//!
//! After the loop, send `f_M = f_{num_rounds}` as `final_polynomial`. Its
//! degree is `< stopping_degree` by construction.
//!
//! ## Storage strategy: coefficient form + per-round evaluation form
//!
//! The prover keeps the **coefficient form** of `f_i` as the source-of-truth
//! state (it's what `Fold`, `Quotient`, and `DegCor` operate on directly).
//! Per round it materialises the **evaluation form** on `L_i` via FFT,
//! commits to it, and discards the table once the round is done (it is no
//! longer needed — the Merkle tree retains hashed leaves for opening, but the
//! field-element evaluations themselves are reconstructed from the
//! coefficient form when needed).
//!
//! This is a memory-vs-clarity trade. Production STIR implementations
//! incrementally update an evaluation table in place via in-place FFT,
//! halving memory usage. For an educational implementation we re-evaluate
//! every round to make the algorithm transparent.
//!
//! ## Named lemma — Soundness gap per iteration (Lemma 1, paper §4)
//!
//! STIR's round-by-round (RBR) soundness analysis bounds the per-round
//! cheating probability by the sum of three sub-errors:
//!
//! 1. **Fold soundness** (Theorem 4.2): if `f_i` is `δ`-far from
//!    `RS[F, L_i, d_i]`, then `Fold(f_i, α_i, k)` is `δ'`-far from
//!    `RS[F, L_i', d_i / k]` for *some* `δ' ≈ δ` except with probability
//!    `(d_i - 1) / |F|` over the choice of `α_i`. The fold step doesn't
//!    move the function much closer to a codeword.
//! 2. **Quotient distance lemma** (Lemma 4.4): if `Fold(f_i, α_i, k)`
//!    interpolates the OOD replies, the *quotient* polynomial
//!    `(Fold - I) / V_Z` (where `I` interpolates the OOD points/replies and
//!    `V_Z` vanishes on them) is `δ - s/|F|`-far from a lower-degree code,
//!    where `s = ood_samples`.
//! 3. **OOD list collapse** (Lemma 4.5): the OOD samples force the
//!    list-decoding candidate list of the folded function down to a single
//!    codeword (with the OOD replies treated as the consensus), except with
//!    probability `(s · ε) / |F|`.
//!
//! Summing across `num_rounds` rounds (union bound) gives the total
//! soundness error.
//!
//! **Question for the reader.** Why does the prover need to keep both
//! coefficient and evaluation forms? Couldn't we work in just one?
//!
//! Coefficient form is needed for `poly_fold` (which works on coefficients by
//! splitting them into `k` strides, see [`crate::fold`]), for `Quotient`
//! (polynomial division), and for `DegCor` (polynomial multiplication). The
//! evaluation form is needed for the *Merkle commit* (we hash the field-
//! element evaluations at the domain points) and for *shift answers* (we read
//! off `evals_i[q]` to respond to a query). FFT and inverse-FFT bridge the
//! two forms, but doing both per round trades memory for clarity. In a
//! production implementation you'd track only the evaluation form and use
//! inverse-FFT once per round (or, better, incremental in-place fold-on-
//! evaluations to skip the round-trip entirely). This educational version
//! keeps both for legibility.
//!
//! ## Worked params trace (small)
//!
//! With `log_initial_domain_size = 6`, `folding_factor = 4`, `num_rounds = 2`,
//! `initial_degree_bound = 16`:
//!
//! - Round 0: `|L_0| = 64`, `d_0 = 16`. Fold by 4 → `d_1 = 4`. Domain shrinks
//!   by `k/2 = 2` → `|L_1| = 32`. Rate `ρ_0 = 16/64 = 1/4`, `ρ_1 = 4/32 = 1/8`
//!   (smaller, by design).
//! - Round 1: `|L_1| = 32`, `d_1 = 4`. Fold by 4 → `d_2 = 1`. Domain shrinks
//!   by 2 → `|L_2| = 16`. The next "polynomial" is degree-`0` (a constant),
//!   so the protocol terminates and sends a constant as `final_polynomial`.
//!
//! `stopping_degree` is set so that the final polynomial fits cheaply in the
//! proof; here `stopping_degree = 4` suffices.
//!
//! `// CAUTION:` fold randomness MUST come from the transcript AFTER
//! absorbing the round's commitment + OOD replies + shift answers. Fiat-
//! Shamir order matters: if the prover could see `α_i` before committing to
//! `f_i` (or before pinning down its OOD replies), the prover could pick a
//! commitment to align with the resulting fold, breaking soundness. The
//! contract is "commit → reveal OOD → reveal shift → derive α" — every state
//! machine in this crate enforces that order.

use reed_solomon::UnivariatePoly;

use crate::domain::StirDomain;
use crate::params::StirParams;
use crate::proof::StirProof;
use crate::transcript::Transcript;

/// The STIR prover.
///
/// Wraps the public input (the polynomial `f_0` and the static parameters
/// `params`) together with the cached round-`0` domain `stir_domain`. The
/// per-round state — the current polynomial `f_i`, evaluation tables, Merkle
/// trees — lives inside [`Self::prove`] as local variables; the prover struct
/// itself is constructed once and consumed once.
///
/// This is the same shape as [`crate::verifier::StirVerifier`], on purpose —
/// the public-input state is identical between the two; what differs is the
/// per-round behaviour (the prover *produces* opens, the verifier *checks*
/// them).
pub struct StirProver {
    /// The fully-formed STIR parameters. Drives the per-round domain sizes,
    /// degree bounds, fold factor, OOD-sample count, repetition schedule,
    /// and stopping degree. Constructed once and treated as read-only by the
    /// prover.
    pub params: StirParams,
    /// The input polynomial `f_0` in coefficient form. Has degree
    /// `< params.initial_degree_bound` (a debug check verifies this at
    /// construction time).
    ///
    /// The prover internally folds this into `f_1, f_2, ..., f_M` during
    /// [`Self::prove`]; `initial_polynomial` itself is preserved unchanged
    /// for inspection (and for tests that re-run a proof with the same
    /// witness).
    pub initial_polynomial: UnivariatePoly,
    /// The round-by-round STIR domain structure
    /// `L_0 ⊃ L_1 ⊃ ... ⊃ L_M`. Precomputed once at construction and
    /// referenced by index (`stir_domain.round_domain(i)`) during the
    /// prover loop.
    ///
    /// Sharing the precomputed structure between prover and verifier is a
    /// minor optimisation; both could build their own from `params`.
    pub stir_domain: StirDomain,
}

impl StirProver {
    /// Construct a prover from STIR parameters and a polynomial to prove
    /// low-degree.
    ///
    /// Performs cheap construction-time validation:
    /// - `polynomial.degree() < params.initial_degree_bound` (debug assert).
    /// - `params.validate()` (typically already called by params'
    ///   constructor, but defensive double-check).
    ///
    /// Caches the [`StirDomain`] derived from `params`.
    ///
    /// # Paper reference
    /// STIR (eprint 2024/390), §3 "Protocol". The prover input is `f_0` and
    /// the public parameters — this constructor is the analogue.
    pub fn new(params: StirParams, polynomial: UnivariatePoly) -> Self {
        // TODO:
        //   1. (Debug) Assert `polynomial.degree() < params.initial_degree_bound`.
        //      WHY: the prover is only meaningful for inputs already inside
        //      the code; a degree overflow at construction means the caller
        //      mis-set parameters and any proof produced would be a soundness
        //      hole, not a slow failure later.
        //   2. Compute `stir_domain = StirDomain::new(&params)`.
        //      WHY: precomputing once lets every round look up its domain
        //      in O(1); rebuilding per round would re-do the root-of-unity
        //      arithmetic `num_rounds` times.
        //   3. Return `Self { params, initial_polynomial: polynomial, stir_domain }`.
        let _ = (params, polynomial);
        todo!()
    }

    /// Run the full STIR prover and produce a [`StirProof`].
    ///
    /// Consumes the transcript by appending the prover's messages and reading
    /// back challenges via Fiat-Shamir. The transcript should be freshly-
    /// initialised by the caller (see [`crate::protocol::run_stir`]) and not
    /// shared with the verifier — both parties build their own transcript
    /// independently and the transcripts should agree byte-for-byte by the
    /// end of an honest run.
    ///
    /// # Inputs
    /// - `transcript`: a Fiat-Shamir transcript. Must be empty (or seeded
    ///   with the public domain separator) at call time.
    ///
    /// # Outputs
    /// A [`StirProof`] populated with `num_rounds` rounds of data plus the
    /// final polynomial.
    ///
    /// # Panics
    /// May panic if the input polynomial's degree exceeds
    /// `params.initial_degree_bound` (caught by the constructor's debug
    /// assert; in release builds this surfaces as a soundness failure
    /// downstream).
    ///
    /// # Paper reference
    /// STIR (eprint 2024/390), §3 "Protocol" Figure 1 — the per-round
    /// commit / OOD / shift-query / fold loop. This implementation
    /// transcribes that figure step by step.
    pub fn prove(&self, transcript: &mut Transcript) -> StirProof {
        // TODO: drive the STIR prover state machine for `params.num_rounds` rounds.
        //
        //   Setup:
        //     - Let `current_poly = self.initial_polynomial.clone()` — the
        //       round-`i` polynomial. Updated at the end of each round.
        //     - Allocate empty `Vec`s for each `StirProof` field
        //       (round_commitments, ood_replies, shift_answers, merkle_paths,
        //       merkle_opened_leaves, pow_nonces). They'll be `push`ed to
        //       once per round.
        //
        //   For `i in 0..self.params.num_rounds`:
        //     1. Evaluate `current_poly` on `self.stir_domain.round_domain(i)`
        //        via FFT to get `evals_i` (length `|L_i|`).
        //        WHY: the commitment is to the function `f_i: L_i → F` via
        //        its evaluation table — never to coefficients.
        //     2. Commit: `(commitment_i, tree_i) = StirCommitment::commit(&evals_i)`.
        //        Absorb `commitment_i.root.0` into the transcript via
        //        `transcript.absorb_root(&commitment_i.root)`. Push the
        //        commitment into `round_commitments`.
        //        WHY: the commit-then-challenge ordering of Fiat-Shamir; any
        //        out-of-order absorption breaks soundness.
        //     3. Derive OOD points: call `ood::sample_ood_points(transcript,
        //        self.stir_domain.round_domain(i), self.params.ood_samples)`
        //        for `[z_{i,1}, ..., z_{i,s}]`. Compute
        //        `ood_replies_i = [current_poly.evaluate(z) for z in zs]`
        //        via Horner. Absorb the replies into the transcript and push
        //        into `ood_replies`.
        //        WHY: OOD = out-of-domain. These collapse the list of
        //        candidate codewords (Lemma 4.5).
        //     4. (Optional) PoW: if `self.params.pow_bits[i] > 0`, run the
        //        grind: `nonce_i = transcript.grind(pow_bits)`. Push to
        //        `pow_nonces`.
        //        WHY: PoW lowers per-round repetitions at the cost of prover
        //        work; harmless to skip if disabled in params.
        //     5. Derive shift queries: call
        //        `transcript.sample_shift_indices(self.params.repetition_schedule[i],
        //        |L_i|)` for `[q_{i,1}, ..., q_{i,t}]`. For each `q`, gather
        //        `shift_answers_i.push(evals_i[q])`,
        //        `merkle_paths_i.push(tree_i.open(q).1)`,
        //        `merkle_opened_leaves_i.push(evals_i[q])`.
        //        Absorb all answers + path siblings into the transcript.
        //        Push each into the corresponding `Vec<Vec<_>>`.
        //        WHY: the actual proximity test — Merkle opening at random
        //        positions. The repetition schedule is harmonic so the
        //        front rounds carry the most queries.
        //     6. Derive fold randomness: `alpha_i = transcript.squeeze_field()`.
        //        WHY: a fresh challenge sampled AFTER all prover messages
        //        for this round are absorbed — Fiat-Shamir order.
        //     7. Compute the next round's polynomial:
        //          let folded = fold::poly_fold(&current_poly, alpha_i,
        //              self.params.folding_factor);
        //          let quotiented = quotient::poly_quotient(&folded,
        //              &ood_points_i, &ood_replies_i);
        //          current_poly = degree_correction::poly_degree_correct(
        //              &quotiented, &self.params, i);
        //        WHY: these three operations together produce a polynomial
        //        guaranteed to live in `RS[F, L_{i+1}, d_{i+1}]` (Lemma 4.4
        //        + the degree-correction multiplier).
        //
        //   After the loop:
        //     8. The remaining `current_poly` is `f_M`, with degree
        //        `< self.params.stopping_degree`. Assert this (debug-only).
        //        Set `final_polynomial = current_poly`.
        //        WHY: this is the proof's terminator; the verifier checks
        //        its degree explicitly.
        //
        //   Return `StirProof { round_commitments, ood_replies, shift_answers,
        //                       merkle_paths, merkle_opened_leaves,
        //                       final_polynomial, pow_nonces }`.
        let _ = transcript;
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constructor accepts a polynomial within the degree bound.
    #[test]
    fn prover_constructs_with_valid_params() {
        // TODO:
        //   1. Build a StirParams with log_initial_domain_size = 6,
        //      folding_factor = 4, num_rounds = 2, initial_degree_bound = 16.
        //   2. Build a UnivariatePoly of degree 15 (within bound).
        //   3. let prover = StirProver::new(params.clone(), poly);
        //   4. Assert prover.params == params (or some inspectable field).
        // WHY: smoke test the happy-path construction; debug-asserts must
        // not fire on a valid input.
        todo!()
    }

    /// Honest prove produces a proof with the right number of rounds.
    #[test]
    fn prove_produces_proof_with_correct_round_count() {
        // TODO:
        //   1. Build params with num_rounds = 2.
        //   2. Build a low-degree polynomial.
        //   3. Build a Transcript with a fixed domain separator.
        //   4. let proof = StirProver::new(params, poly).prove(&mut transcript);
        //   5. Assert proof.round_commitments.len() == 2.
        //   6. Assert proof.ood_replies.len() == 2.
        //   7. Assert proof.shift_answers.len() == 2.
        // WHY: contract check between the prover loop and the proof struct.
        todo!()
    }

    /// Final polynomial has degree < stopping_degree.
    #[test]
    fn prove_final_polynomial_has_low_degree() {
        // TODO:
        //   1. Build params with stopping_degree = 4.
        //   2. Build a low-degree poly, run the prover.
        //   3. Assert proof.final_polynomial.degree() < Some(4) (or is
        //      None for the zero polynomial).
        // WHY: a wrong-degree final polynomial is one of the explicit
        // verifier-rejection conditions; the prover must enforce it on its
        // side.
        todo!()
    }
}
