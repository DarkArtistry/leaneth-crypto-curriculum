//! The STIR prover state machine — paper-faithful per-round update.
//!
//! ## Anchor: STIR's per-round state machine
//!
//! This module turns an input polynomial `p_0 ∈ F[X]` of degree `< d_0` into
//! a [`crate::proof::StirProof`] over `num_rounds = M` rounds. Each round
//! commits to the current function on its evaluation domain, samples OOD
//! points and folding randomness via Fiat-Shamir, applies the canonical
//! Quotient → Fold → DegCor update to produce the next round's function,
//! and opens `k` Merkle paths per shift query so the verifier can later
//! replay the fold locally. The output is a [`crate::proof::StirProof`]
//! that the verifier in [`crate::verifier`] can check against the same
//! params and the same transcript seed.
//!
//! ## What the prover does, in one paragraph
//!
//! The honest prover holds the public polynomial `f_0 ∈ F[X]` — over the
//! Goldilocks field `F = F_p` — of degree `< d_0` (the **initial degree
//! bound** `initial_degree_bound`). The protocol runs `M = num_rounds`
//! rounds at folding factor `k = folding_factor`. Per round `i` the
//! prover (a) evaluates the round-`i` committed function `f_i` on the
//! round-`i` evaluation domain `L_i` via FFT, (b) Merkle-commits to that
//! evaluation table and absorbs the root into the Fiat-Shamir transcript,
//! (c) receives `s = ood_samples` out-of-domain points from the post-
//! absorption transcript and responds with `f_i(z)` for each; (d)
//! squeezes the **degree-correction randomness** `r_comb_i` and then the
//! **fold randomness** `α_i`, (e) updates `current_poly` via the per-
//! paper **Quotient → Fold → DegCor** chain on the OOD set, (f) samples
//! `t_i` shift-query indices in `[0, n_i / k)` from the transcript, and
//! for each query opens `k` Merkle paths against `tree_i` at the fold-
//! axis preimage positions `q_idx + j · (n_i / k)`, absorbing each
//! opened leaf into the transcript. After `M` rounds the residual
//! polynomial has degree `< d_M = stopping_degree` and is shipped in the
//! clear as `final_polynomial`.
//!
//! ## Per-round transcript schedule (`stir-full-spec.md` §1.1)
//!
//! For each `i ∈ 0..num_rounds`, in this exact order:
//!
//! 1. `transcript.absorb_root(&commitment_i.root)`
//! 2. `ood_points_i ← sample_ood_points(transcript, s, L_{i+1})`
//! 3. `for r in ood_replies_i: transcript.absorb_field(r)`
//! 4. `r_comb_i ← transcript.squeeze_field()`   (**degree-correction randomness**)
//! 5. `α_i      ← transcript.squeeze_field()`   (**fold randomness**)
//! 6. `shift_indices_i ← transcript.sample_shift_indices(t_i, (n_i / k) as u32)`
//! 7. `for each query q, for each preimage j ∈ 0..k:
//!     transcript.absorb_field(leaf)` after opening the Merkle path.
//!
//! **Key ordering change vs the previous educational variant:**
//! `r_comb_i` and `α_i` are squeezed BEFORE shift indices are sampled,
//! not after. This matches Construction 5.2 of the STIR paper (eprint
//! 2024/390): the verifier needs `α_i` to fold the `k` preimages of a
//! shift query, and it needs `r_comb_i` to apply the DegCor multiplier
//! locally — both challenges must therefore be derived before the shift
//! queries the verifier will check. Fresh `r_comb_i` (not reused from
//! `α_i`) is required by the **DegCor Soundness theorem** in
//! [`crate::degree_correction`]: reusing `α_i` would compose two
//! Schwartz-Zippel events on the same scalar and weaken the per-round
//! bound.
//!
//! ## Per-round update with Quotient + Fold + DegCor
//!
//! The previous educational variant did `current_poly = poly_fold(current_poly, k, α_i)`
//! — pure fold, no quotient, no DegCor. That gave an FRI+OOD protocol
//! rather than paper-faithful STIR. The refactor restores the full
//! per-round chain:
//!
//! ```text
//! ood_pairs   = { (z, f_i(z)) : z ∈ ood_points_i }
//! q_i(X)      = (f_i(X) − p_{ood}(X)) / V_{ood}(X)    // [`crate::quotient::poly_quotient`]
//! g_i(Y)      = Fold(q_i, k, α_i)                     // [`crate::fold::poly_fold`]
//! current_poly = poly_deg_cor(&g_i, e_i, r_comb_i)    // [`crate::degree_correction::poly_deg_cor`]
//! ```
//!
//! where the degree bump `e_i` is computed by
//!
//! ```text
//! g_deg_bound = ⌈(d_i − s) / k⌉
//! e_i         = d_{i+1} − g_deg_bound        (saturating, so e_i ≥ 0)
//! ```
//!
//! and the OOD set `ood_pairs` is the round's "S" — the shift queries
//! against `L_{i+1}` (size `n_i / k` fibre representatives) are
//! **checks**, not constraints that the prover folds away. (See
//! `stir-full-spec.md` §2.2 decision 3 for why.) After DegCor the new
//! `current_poly` has degree `< d_{i+1}`, matching the Prover-Round
//! Invariant.
//!
//! ### Demo-params worked example (`d_0 = 16, k = 4, s = 2, M = 2`)
//!
//! With `params::StirParams::new(6, 16, 4)`: `|L_0| = 64`, `|L_1| = 32`,
//! `|L_2| = 16`. Degrees `d_0 = 16, d_1 = 4, d_2 = 1`. Round 0:
//!
//! ```text
//! deg(f_0)              < 16
//! deg(q_0) ≤ 16 − 2     = 14         (Quotient by s = 2 OOD points)
//! deg(g_0) ≤ ⌈14 / 4⌉   = 4
//! g_deg_bound           = ⌈(16 − 2) / 4⌉ = 4
//! e_0 = d_1 − g_deg_bound = 4 − 4 = 0   ⇒ DegCor is the identity.
//! ```
//!
//! Round 1:
//!
//! ```text
//! deg(f_1)              < 4
//! deg(q_1) ≤ 4 − 2      = 2
//! deg(g_1) ≤ ⌈2 / 4⌉    = 1
//! g_deg_bound           = ⌈(4 − 2) / 4⌉ = 1
//! e_1 = d_2 − g_deg_bound = 1 − 1 = 0   ⇒ DegCor is the identity.
//! ```
//!
//! For the demo parameters `e_i = 0` in every round (because `d_i / k =
//! ⌈(d_i − s) / k⌉` exactly), so the DegCor multiplier collapses to the
//! constant `1` and the demo prover effectively just does Quotient +
//! Fold. The refactor matters for parameters where `e_i > 0` (e.g. with
//! `s > k` per round, or with non-uniform `k`-vs-degree ratios); the
//! call path is exercised either way and the verifier mirrors it
//! identically (see `stir-full-spec.md` §6.3 / [`crate::degree_correction::eval_g_r`]).
//!
//! ## Shift queries on the **fibre representative** space
//!
//! Per `stir-full-spec.md` §5 (revising §1.1 step 6), shift indices are
//! sampled from `[0, n_i / k)`, NOT `[0, n_i)` or `[0, |L_{i+1}|)`. The
//! reason: each fibre representative `q_idx` corresponds to a fold-axis
//! coset `{ω_k^j · L_i.element(q_idx) : j = 0..k}` whose evaluations
//! live in `L_i` at positions
//!
//! ```text
//!     preimage_indices[j] = q_idx + j · (n_i / k)        for j ∈ 0..k.
//! ```
//!
//! The prover opens `k` Merkle paths per query — one per preimage — and
//! the verifier authenticates all `k` against `commitment_i.root` before
//! folding them by `α_i` to recover `g_i(z_k)` (where `z_k =
//! L_i.element(q_idx).pow(k)` is the corresponding point in `L_i^k`).
//! See [`crate::fold`] §"Why it works on evaluation tables" for the
//! fibre layout and `fold.rs` line 405–407 for the stride-`n_i/k`
//! sibling pattern.
//!
//! ## Named theorem: Prover-Round Invariant
//!
//! > **Prover-Round Invariant theorem.** Let `params: StirParams`,
//! > `stir_domain = StirDomain::new(&params)`. At the **start** of round
//! > `i ∈ 0..num_rounds`, the local variable `current_poly` in
//! > [`StirProver::prove`] satisfies the invariant
//! >
//! > ```text
//! > (I_i):  deg(current_poly) < d_i  =  params.round_degree_bound(i).
//! > ```
//! >
//! > At the **end** of round `i`, the proof's per-round vectors have each
//! > been extended by exactly one entry (one Merkle commitment, one
//! > `Vec<Fp>` of `s` OOD replies, one `Vec<Vec<Fp>>` of `t_i × k` shift
//! > answers/paths/leaves), the transcript state has absorbed (in order)
//! > `commitment_i.root`, the `s` OOD replies, and then for each of the
//! > `t_i` shift queries, the `k` preimage values, and `current_poly`
//! > has been updated to `poly_deg_cor(Fold(Quotient(current_poly,
//! > ood_pairs), k, α_i), e_i, r_comb_i)`, satisfying `(I_{i+1})`.
//!
//! **Proof.** Base case `(I_0)`: `current_poly = self.initial_polynomial`,
//! which the constructor debug-asserts has `degree < initial_degree_bound = d_0`.
//! Inductive step: assume `(I_i)`.
//!
//! - **Quotient.** By the *Polynomial Identity Lemma / Constraint
//!   Embedding theorem* (in [`crate::quotient`], Completeness clause),
//!   `poly_quotient(&current_poly, &ood_pairs)` returns a polynomial of
//!   degree `≤ deg(current_poly) − s < d_i − s` (using `s = ood_samples`).
//! - **Fold.** By the *Folding theorem* (in [`crate::fold`]),
//!   `poly_fold(&q_i, k, α_i)` reduces degree to
//!   `< ⌈(d_i − s) / k⌉ = g_deg_bound`.
//! - **DegCor.** By the *Degree-Correction theorem* (in
//!   [`crate::degree_correction`]), multiplying by `g_{r_comb_i}` of
//!   degree `e_i` bumps the bound up by exactly `e_i`. Hence
//!   `deg(current_poly) < g_deg_bound + e_i ≤ d_{i+1}` (by the
//!   `e_i = d_{i+1} − g_deg_bound` computation, saturating-`sub` if
//!   the formula somehow underflows; for honest inputs the inequality is
//!   tight).
//!
//! Hence `(I_{i+1})` holds. After `M` iterations, `(I_M)` gives
//! `deg(current_poly) < d_M = stopping_degree`, which is exactly the
//! final-polynomial degree bound the verifier explicitly checks. ∎
//!
//! ## Storage strategy: coefficient form + per-round evaluation form
//!
//! The prover keeps the **coefficient form** of `f_i` as the source-of-
//! truth state: Quotient, Fold and DegCor all operate naturally on
//! coefficients. Per round it materialises the **evaluation form** on
//! `L_i` via [`reed_solomon::fft::fft_on_domain`], commits to it, and
//! retains the Merkle tree locally for the round's shift-query opens.
//! Production STIR implementations would update an evaluation table in
//! place; we re-evaluate every round to keep the algorithm transparent.
//! See `stir-full-spec.md` §0 for why the prover **cannot** use the
//! evaluation-form `fold()` to step from `L_i` to `L_{i+1}` (the
//! codeword-side fold lands on `L_i^k`, not `L_{i+1}`).
//!
//! `// CAUTION:` fold randomness `α_i` MUST come from the transcript AFTER
//! the round's commitment and OOD replies have been absorbed. Likewise
//! `r_comb_i` MUST come AFTER OOD replies and BEFORE `α_i`. The order is
//! "commit → reveal OOD → derive r_comb → derive α → sample shifts →
//! reveal shift leaves" — every state machine in this crate enforces it.

use reed_solomon::fft::fft_on_domain;
use reed_solomon::{Fp, UnivariatePoly};

use crate::commitment::StirCommitment;
use crate::degree_correction::poly_deg_cor;
use crate::domain::StirDomain;
use crate::fold::poly_fold;
use crate::ood::{evaluate_at, sample_ood_points};
use crate::params::StirParams;
use crate::proof::StirProof;
use crate::quotient::poly_quotient;
use crate::transcript::Transcript;

/// Return the `k` preimage indices in `L_i` for a fibre representative
/// index `q_idx ∈ [0, n_i / k)`.
///
/// Per `stir-full-spec.md` §5: layout is natural order on
/// `L_i = c · ⟨g⟩`, so the primitive `k`-th root of unity
/// `ω_k = g^{n_i / k}` corresponds to walking the fibre at stride
/// `n_i / k` in the evaluation table. The `k` preimages of the fibre
/// representative `L_i.element(q_idx)` are therefore at indices
/// `q_idx + j · (n_i / k)` for `j ∈ 0..k`. This stride is exactly what
/// the codeword-side [`crate::fold::fold`] uses; see `fold.rs` lines
/// 403–407 for the matching natural-order sibling walk.
fn preimage_indices_in_l_i(q_idx: usize, n_i: usize, k: usize) -> Vec<usize> {
    debug_assert_eq!(n_i % k, 0, "n_i must be divisible by k");
    let stride = n_i / k;
    debug_assert!(q_idx < stride, "q_idx must be in [0, n_i / k)");
    (0..k).map(|j| q_idx + j * stride).collect()
}

/// The STIR prover.
///
/// Wraps the public input (the polynomial `f_0` and the static parameters
/// `params`) together with the cached round-`0` domain `stir_domain`. The
/// per-round state — the current polynomial `f_i`, evaluation tables, Merkle
/// trees — lives inside [`Self::prove`] as local variables; the prover struct
/// itself is constructed once and consumed once.
pub struct StirProver {
    /// The fully-formed STIR parameters.
    pub params: StirParams,
    /// The input polynomial `f_0` in coefficient form. Has degree
    /// `< params.initial_degree_bound`.
    pub initial_polynomial: UnivariatePoly,
    /// The round-by-round STIR domain structure `L_0 ⊃ L_1 ⊃ ... ⊃ L_M`.
    pub stir_domain: StirDomain,
}

impl StirProver {
    /// Construct a prover from STIR parameters and a polynomial to prove
    /// low-degree.
    ///
    /// Performs cheap construction-time validation:
    /// - `polynomial.degree() < params.initial_degree_bound` (debug assert).
    /// - `params.validate()`.
    pub fn new(params: StirParams, polynomial: UnivariatePoly) -> Self {
        debug_assert!(
            polynomial
                .degree()
                .map(|d| d < params.initial_degree_bound)
                .unwrap_or(true),
            "initial polynomial degree {:?} must be < initial_degree_bound {}",
            polynomial.degree(),
            params.initial_degree_bound,
        );

        params
            .validate()
            .expect("StirProver::new: invalid StirParams");

        let stir_domain = StirDomain::new(&params);

        Self {
            params,
            initial_polynomial: polynomial,
            stir_domain,
        }
    }

    /// Run the full STIR prover and produce a [`StirProof`].
    ///
    /// Consumes the transcript by appending the prover's messages and reading
    /// back challenges via Fiat-Shamir. Implements `stir-full-spec.md` §2.1
    /// (per-round Quotient + Fold + DegCor) and §1.1 (per-round transcript
    /// schedule with `r_comb_i` squeezed between OOD-reply absorption and
    /// `α_i`).
    pub fn prove(&self, transcript: &mut Transcript) -> StirProof {
        let num_rounds = self.params.num_rounds as usize;
        let k = self.params.folding_factor as usize;
        let s = self.params.ood_samples as usize;

        // Per-round outputs.
        let mut round_commitments: Vec<StirCommitment> = Vec::with_capacity(num_rounds);
        let mut ood_replies: Vec<Vec<Fp>> = Vec::with_capacity(num_rounds);
        let mut shift_answers: Vec<Vec<Vec<Fp>>> = Vec::with_capacity(num_rounds);
        let mut merkle_paths: Vec<Vec<Vec<crate::merkle::MerklePath>>> =
            Vec::with_capacity(num_rounds);
        let mut merkle_opened_leaves: Vec<Vec<Vec<Fp>>> = Vec::with_capacity(num_rounds);

        // Round-`i` polynomial in coefficient form. Updated at the end of
        // every iteration via DegCor(Fold(Quotient(·, ood), k, α), e, r_comb).
        //
        // Base case `(I_0)`: deg(current_poly) < d_0.
        let mut current_poly = self.initial_polynomial.clone();

        for i in 0..num_rounds {
            let domain_i = self.stir_domain.round_domain(i);
            let n_i = domain_i.size();
            let d_i = self.params.round_degree_bound(i as u32);
            let d_ip1 = self.params.round_degree_bound((i + 1) as u32);

            // ---- (1) Evaluate `current_poly` on L_i via FFT, then commit. ----
            //
            // Pad coeffs up to |L_i| (fft_on_domain requires equal lengths).
            // Prover-Round Invariant `(I_i)` guarantees no truncation.
            let mut padded_coeffs: Vec<Fp> = current_poly.coeffs().to_vec();
            assert!(
                padded_coeffs.len() <= n_i,
                "Prover-Round Invariant violated at round {i}: poly has {} \
                 coefficients but |L_{i}| = {n_i}",
                padded_coeffs.len(),
            );
            padded_coeffs.resize(n_i, Fp::zero());
            let evals_i: Vec<Fp> = fft_on_domain(&padded_coeffs, domain_i);

            let (commitment_i, tree_i) = StirCommitment::commit(&evals_i);
            transcript.absorb_root(&commitment_i.root);

            // ---- (2) OOD points + replies. ----
            //
            // Per `stir-full-spec.md` §1.1 step 2-3: sample s points
            // rejection-sampled against L_{i+1}, reply with f_i(z) for each,
            // absorb every reply.
            let next_domain = self.stir_domain.round_domain(i + 1);
            let ood_points_i: Vec<Fp> = sample_ood_points(
                transcript,
                self.params.ood_samples,
                next_domain,
            );
            let ood_replies_i: Vec<Fp> = ood_points_i
                .iter()
                .map(|&z| evaluate_at(&current_poly, z))
                .collect();
            for &reply in &ood_replies_i {
                transcript.absorb_field(reply);
            }

            // ---- (3) Squeeze r_comb_i, then α_i. ----
            //
            // ORDER IS LOAD-BEARING. `r_comb_i` is the DegCor randomness;
            // it must be a fresh squeeze (NOT reused from `α_i`) per the
            // DegCor Soundness theorem. `α_i` is the fold randomness. See
            // `stir-full-spec.md` §1.1 pinned ordering decisions.
            let r_comb_i: Fp = transcript.squeeze_field();
            let alpha_i: Fp = transcript.squeeze_field();

            // ---- (4) Per-round polynomial update: Quotient → Fold → DegCor. ----
            //
            // ood_pairs = { (z, f_i(z)) : z ∈ ood_points_i }. The shift
            // queries (sampled in step 5 below) are CHECKS, not constraints
            // — they are NOT in ood_pairs. See `stir-full-spec.md` §2.2
            // decision 3 for the algebraic reason.
            let ood_pairs: Vec<(Fp, Fp)> = ood_points_i
                .iter()
                .zip(ood_replies_i.iter())
                .map(|(&z, &b)| (z, b))
                .collect();

            // Quotient: deg(q_i) ≤ deg(current_poly) − s < d_i − s.
            // When s = 0 (no OOD step), poly_quotient is undefined; skip.
            let q_i: UnivariatePoly = if ood_pairs.is_empty() {
                current_poly.clone()
            } else {
                poly_quotient(&current_poly, &ood_pairs)
            };

            // Fold: deg(g_i) ≤ ⌈(d_i − s) / k⌉.
            let g_i: UnivariatePoly = poly_fold(&q_i, self.params.folding_factor, alpha_i);

            // DegCor: multiply by g_{r_comb_i} of degree e_i so the
            // result's degree bound is d_{i+1}. With `g_deg_bound`
            // defended via `saturating_sub`, the e_i = 0 path is the
            // identity (see `poly_deg_cor`'s early return).
            let g_deg_bound = if d_i >= s {
                (d_i - s + k - 1) / k
            } else {
                0
            };
            let e_i: usize = d_ip1.saturating_sub(g_deg_bound);

            current_poly = poly_deg_cor(&g_i, e_i, r_comb_i);
            // (I_{i+1}): deg(current_poly) < d_{i+1}.

            // ---- (5) Shift queries on the fibre-representative space [0, n_i / k). ----
            //
            // Per `stir-full-spec.md` §5: shift indices range over `n_i / k`
            // (NOT `n_i` and NOT `|L_{i+1}|`). Each q_idx corresponds to a
            // fibre with k preimages in L_i at stride n_i / k.
            assert!(
                n_i % k == 0,
                "round {i}: |L_i| = {n_i} not divisible by k = {k}",
            );
            let t_i = self.params.repetition_schedule[i];
            let shift_indices_i: Vec<u32> =
                transcript.sample_shift_indices(t_i, (n_i / k) as u32);

            let mut shift_answers_i: Vec<Vec<Fp>> = Vec::with_capacity(t_i as usize);
            let mut merkle_paths_i: Vec<Vec<crate::merkle::MerklePath>> =
                Vec::with_capacity(t_i as usize);
            let mut merkle_opened_leaves_i: Vec<Vec<Fp>> =
                Vec::with_capacity(t_i as usize);

            for &q_idx_u32 in &shift_indices_i {
                let q_idx = q_idx_u32 as usize;
                let preimage_idxs = preimage_indices_in_l_i(q_idx, n_i, k);

                let mut answers_q: Vec<Fp> = Vec::with_capacity(k);
                let mut paths_q: Vec<crate::merkle::MerklePath> = Vec::with_capacity(k);
                let mut leaves_q: Vec<Fp> = Vec::with_capacity(k);

                for &p_idx in &preimage_idxs {
                    let (leaf, path) = tree_i.open(p_idx);
                    debug_assert_eq!(
                        leaf, evals_i[p_idx],
                        "tree.open must agree with evals_i at preimage index",
                    );
                    answers_q.push(leaf);
                    paths_q.push(path);
                    leaves_q.push(leaf);

                    // Absorb the preimage value into the transcript so the
                    // verifier's reproduced state matches before drawing the
                    // next round's challenges.
                    transcript.absorb_field(leaf);
                }

                shift_answers_i.push(answers_q);
                merkle_paths_i.push(paths_q);
                merkle_opened_leaves_i.push(leaves_q);
            }

            // ---- (6) Stash per-round outputs. ----
            round_commitments.push(commitment_i);
            ood_replies.push(ood_replies_i);
            shift_answers.push(shift_answers_i);
            merkle_paths.push(merkle_paths_i);
            merkle_opened_leaves.push(merkle_opened_leaves_i);
        }

        // After the loop, `current_poly` is `f_M`. By the Prover-Round
        // Invariant applied `M` times, `deg(f_M) < d_M = stopping_degree`.
        debug_assert!(
            current_poly
                .degree()
                .map(|d| d < self.params.stopping_degree)
                .unwrap_or(true),
            "final polynomial degree {:?} must be < stopping_degree {}",
            current_poly.degree(),
            self.params.stopping_degree,
        );

        StirProof {
            round_commitments,
            ood_replies,
            shift_answers,
            merkle_paths,
            merkle_opened_leaves,
            final_polynomial: current_poly,
            // PoW descoped — see the field doc on `StirProof::pow_nonces`.
            pow_nonces: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Curriculum's demo parameters: `M = 2`, `s = 2`, `d_M = 1`,
    /// `k = 4`, `|L_0| = 64`.
    fn demo_params() -> StirParams {
        StirParams::new(6, 16, 4)
    }

    /// A low-degree input polynomial within `d_0 = 16` with distinct
    /// small coefficients so intermediate quotient/fold/degcor stages
    /// don't collapse to zero.
    fn demo_polynomial() -> UnivariatePoly {
        let coeffs: Vec<Fp> = (1u64..=15).map(Fp::new).collect();
        UnivariatePoly::new(coeffs)
    }

    /// Constructor accepts a polynomial within the degree bound.
    #[test]
    fn prover_constructs_with_valid_params() {
        let params = demo_params();
        let poly = demo_polynomial();

        let prover = StirProver::new(params.clone(), poly.clone());

        assert_eq!(prover.params.num_rounds, params.num_rounds);
        assert_eq!(
            prover.params.initial_degree_bound,
            params.initial_degree_bound
        );
        assert_eq!(prover.params.folding_factor, params.folding_factor);
        assert_eq!(
            prover.initial_polynomial.degree(),
            poly.degree(),
            "stored polynomial must match input",
        );

        assert_eq!(
            prover.stir_domain.num_rounds(),
            params.num_rounds as usize,
            "StirDomain caches num_rounds + 1 domains; num_rounds() must agree with params",
        );
    }

    /// Honest prove produces a proof with the right number of rounds and
    /// the triple-nested per-query inner shape (k = folding_factor
    /// preimages per shift query).
    #[test]
    fn prove_produces_proof_with_correct_round_count() {
        let params = demo_params();
        let poly = demo_polynomial();
        let prover = StirProver::new(params.clone(), poly);

        let mut transcript = Transcript::new(b"stir-prover-round-count-test");
        let proof = prover.prove(&mut transcript);

        let m = params.num_rounds as usize;
        let k = params.folding_factor as usize;

        // Outer per-round lengths = M.
        assert_eq!(proof.round_commitments.len(), m);
        assert_eq!(proof.ood_replies.len(), m);
        assert_eq!(proof.shift_answers.len(), m);
        assert_eq!(proof.merkle_paths.len(), m);
        assert_eq!(proof.merkle_opened_leaves.len(), m);

        // Per-round inner lengths match the schedule, and each query's
        // k-dimension equals folding_factor.
        for i in 0..m {
            assert_eq!(
                proof.ood_replies[i].len(),
                params.ood_samples as usize,
                "round {i}: OOD inner length must equal params.ood_samples",
            );
            let t_i = params.repetition_schedule[i] as usize;
            assert_eq!(
                proof.shift_answers[i].len(),
                t_i,
                "round {i}: shift_answers middle length must equal repetition_schedule[{i}]",
            );
            assert_eq!(
                proof.merkle_paths[i].len(),
                t_i,
                "round {i}: merkle_paths middle length must equal repetition_schedule[{i}]",
            );
            assert_eq!(
                proof.merkle_opened_leaves[i].len(),
                t_i,
                "round {i}: merkle_opened_leaves middle length must equal repetition_schedule[{i}]",
            );

            for q in 0..t_i {
                assert_eq!(
                    proof.shift_answers[i][q].len(),
                    k,
                    "round {i}, query {q}: shift_answers inner length must equal k",
                );
                assert_eq!(
                    proof.merkle_paths[i][q].len(),
                    k,
                    "round {i}, query {q}: merkle_paths inner length must equal k",
                );
                assert_eq!(
                    proof.merkle_opened_leaves[i][q].len(),
                    k,
                    "round {i}, query {q}: merkle_opened_leaves inner length must equal k",
                );
            }
        }

        // PoW descoped — must be empty per the Proof-Shape Invariant.
        assert!(proof.pow_nonces.is_empty());
    }

    /// Final polynomial has degree `< stopping_degree` — the inductive
    /// end-point of the Prover-Round Invariant theorem.
    #[test]
    fn prove_final_polynomial_has_low_degree() {
        let params = demo_params(); // stopping_degree = 1 by default
        let poly = demo_polynomial();
        let prover = StirProver::new(params.clone(), poly);

        let mut transcript = Transcript::new(b"stir-prover-final-degree-test");
        let proof = prover.prove(&mut transcript);

        let stopping_degree = params.stopping_degree;
        match proof.final_polynomial.degree() {
            None => {
                // Zero polynomial — trivially degree-bounded.
            }
            Some(d) => assert!(
                d < stopping_degree,
                "final polynomial degree {d} must be < stopping_degree {stopping_degree}",
            ),
        }
    }
}
