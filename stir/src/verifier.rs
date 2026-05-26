//! The STIR verifier state machine — paper-faithful per-round consistency.
//!
//! ## What the verifier does, in one paragraph
//!
//! The verifier holds the public parameters `params` and the
//! round-by-round domain structure `stir_domain` — encoding the sequence
//! of evaluation domains `L_0, L_1, …, L_M` (each `L_i` a subset of the
//! Goldilocks field `F = F_p`) and the per-round degree bounds
//! `d_i = d_0 / k^i`. Given a [`crate::proof::StirProof`] from the
//! prover, it walks rounds in order, faithfully recompiling the prover's
//! Fiat-Shamir transcript: it absorbs each round's commitment root,
//! re-derives the `s = ood_samples` OOD points and absorbs the matching
//! OOD replies, squeezes the **degree-correction randomness** `r_comb_i`
//! and then the **fold randomness** `α_i` (note: in that order, before
//! shift indices, matching the refactored prover), samples the round's
//! `t_i` shift queries on the fibre-representative space `[0, n_i / k)`,
//! and for each query verifies `k = folding_factor` Merkle paths against
//! `commitment_i.root` while absorbing each preimage leaf back into the
//! transcript. After all paths verify, it runs the per-query algebraic
//! consistency check (§"Per-query consistency check" below) to recover
//! the round-`(i+1)` polynomial value at the fibre's `k`-th-power point
//! and — at the last round — compares against
//! `proof.final_polynomial.evaluate(z_k)`. Any disagreement → reject.
//!
//! ## Anchor: the verifier's M-round mirror loop
//!
//! The verifier is a **bit-exact mirror of the prover loop**. The
//! prover absorbs a fixed sequence of bytes into its transcript and
//! reads challenges back out; the verifier rebuilds that same byte
//! sequence by walking the proof object in the same order, absorbing
//! the same bytes at the same points, and reading the same challenges
//! out at the same points. There is no shared transcript state on the
//! wire — both sides build their own transcripts from `b"stir-protocol-v0"`
//! and converge through the absorbs.
//!
//! The per-round absorb/squeeze schedule (which both sides must follow
//! identically, in this order; see `stir-full-spec.md` §1.1):
//!
//! ```text
//! for i in 0..M:
//!     absorb commitment_i.root                ← prover sent, verifier reads
//!     squeeze ood_points_i                    ← rejection-sampled from F \ L_{i+1}
//!     absorb each ood_replies_i[j]            ← prover sent, verifier reads
//!     squeeze r_comb_i                        ← degree-correction randomness
//!     squeeze α_i                             ← fold randomness  (NB: before shifts!)
//!     squeeze shift_indices_i on [0, n_i/k)   ← fibre representatives in L_i
//!     for each q in 0..t_i:
//!         for each j in 0..k:
//!             verify Merkle path at preimage index q_idx + j·(n_i/k)
//!             absorb leaf_i_q_j               ← bind preimage value into transcript
//! squeeze t_final shift indices on |L_M|      ← terminal queries (unchecked, see §3.4)
//! ```
//!
//! **Key ordering change vs the previous educational variant:** `r_comb_i`
//! and `α_i` are squeezed BEFORE shift indices are sampled, not after.
//! This matches Construction 5.2 of the STIR paper (eprint 2024/390):
//! the verifier needs `α_i` to fold the `k` preimages of a shift query,
//! and it needs `r_comb_i` to apply the DegCor multiplier locally
//! ([`crate::degree_correction::eval_g_r`]) — both challenges must
//! therefore be derived before the shift queries the verifier will check.
//! Fresh `r_comb_i` (not reused from `α_i`) is required by the **DegCor
//! Soundness theorem** in [`crate::degree_correction`].
//!
//! ## Per-query consistency check via Local Fold + Quotient + DegCor
//!
//! This is the load-bearing algebraic check that catches a cheating
//! prover at the round-`(M-1)` boundary. For each round `i ∈ 0..M`,
//! query `q ∈ 0..t_i` with `q_idx = shift_indices_i[q] ∈ [0, n_i/k)`,
//! the verifier runs the five-step recipe of `stir-full-spec.md` §3.2:
//!
//! **Step 1 — Merkle authentication of the `k` preimages.** For each
//! `j ∈ 0..k`, let `preimage_indices[j] = q_idx + j · (n_i/k)` (§5
//! formula). The verifier checks
//! [`crate::merkle::MerkleTree::verify`] for each `(leaf_j, path_j)`
//! against `commitment_i.root`, cross-checks `leaf_j ==
//! proof.shift_answers[i][q][j]`, and absorbs each `leaf_j` into the
//! transcript (mirror of prover §2.1 step 5).
//!
//! **Step 2 — Local Quotient at each of the `k` preimages.** The prover
//! computes the round update in the order Quotient → Fold → DegCor
//! ([`crate::prover`] §2.1), so the verifier MIRRORS that order: it
//! applies the OOD-pair quotient pointwise at each authenticated
//! preimage `x_j ∈ L_i` (where `f_i(x_j) =` the absorbed leaf) BEFORE
//! folding. The OOD set is
//! `ood_pairs = { (z_j, f_i(z_j)) : z_j ∈ ood_points_i, f_i(z_j) =
//! ood_replies_i[j] }`. For each `j ∈ 0..k`:
//!
//! ```text
//! p_ood(x_j)   = lagrange_eval(ood_points_i, ood_replies_i, x_j)
//! V_ood(x_j)   = Π_l (x_j − z_l)                       // over OOD points only
//! q_i(x_j)     = (f_i(x_j) − p_ood(x_j)) / V_ood(x_j)
//! ```
//!
//! **Shift answers are inputs to the consistency check, not constraints
//! in the quotient** — see `stir-full-spec.md` §2.2 decision 3. **Why
//! preimage-by-preimage and not pointwise at `z_k`?** Because polynomial
//! quotient and fold do NOT commute — `Fold((f − p)/V)(z_k) ≠
//! (Fold(f)(z_k) − p(z_k))/V(z_k)` in general, since multiplication by
//! the degree-`s` polynomial `V_ood(X)` mixes the `k` row components in
//! the polynomial-fold decomposition `f(X) = Σ_l X^l f_l(X^k)`. The
//! prover's `Fold(Quotient(f_i))` requires the verifier to Quotient
//! first, then Fold.
//!
//! **Step 3 — Local Fold to recover `g_i(z_k)`.** Apply the per-fibre
//! IDFT recipe from [`crate::fold`] §"Why it works on evaluation tables"
//! to the `k` quotient values:
//!
//! ```text
//! ω_k     = ω_i^{n_i / k}                              // primitive k-th root of unity
//! x_0     = L_i.element(q_idx)                          // first preimage
//! z_k     = x_0^k                                       // fibre's k-th-power image in L_i^k
//! twisted = ifft_subgroup_size_k_at_ω_k(q_at_preimages) // twisted[l] = q_l(z_k) · x_0^l
//! r_l     = twisted[l] / x_0^l                         // peel off the X^l factor
//! g_i(z_k) = Σ_l α_i^l · r_l                           // standard fold combination
//! ```
//!
//! This recovers `g_i(z_k) = Fold(q_i, k, α_i)(z_k)` for `z_k ∈ L_i^k`.
//!
//! **Step 4 — Local DegCor at `z_k`.** Bump the degree bound up by
//! `e_i = d_{i+1} − ⌈(d_i − s)/k⌉` via
//! [`crate::degree_correction::eval_g_r`]:
//!
//! ```text
//! g_r(z_k)        = eval_g_r(r_comb_i, z_k, e_i)
//! f_{i+1}(z_k)    = g_i(z_k) · g_r(z_k)
//! ```
//!
//! For the demo parameters `e_i = 0` in every round (the DegCor
//! multiplier collapses to the constant `1`), but the verifier evaluates
//! `g_r` defensively to bit-for-bit match the prover.
//!
//! **Step 5 — Terminal consistency at `i + 1 == M`.** If this is the
//! last round, compare `f_{i+1}(z_k)` against
//! `proof.final_polynomial.evaluate(z_k)`. Disagreement → reject. For
//! `i + 1 < M` no direct Merkle cross-check is performed — the chain
//! terminates at the final-polynomial comparison, with inter-round
//! consistency carried implicitly by the OOD-binding of the
//! round-`(i+1)` commitment (see `stir-full-spec.md` §3.4).
//!
//! ### Demo-params worked example (`d_0 = 16, k = 4, s = 2, M = 2`)
//!
//! Take the curriculum's demo parameters `StirParams::new(6, 16, 4)`:
//! `|L_0| = 64, |L_1| = 32, |L_2| = 16`. Round 0 fibre-representative
//! space size is `n_0 / k = 64 / 4 = 16`; round 1's is `32 / 4 = 8`.
//! With `t_0 = 8, t_1 = 4`, the verifier opens `8 · 4 = 32` Merkle
//! paths in round 0 against `tree_0` and `4 · 4 = 16` in round 1
//! against `tree_1`. Per query it folds the `k = 4` preimages to a
//! single field element `g_i(z_k)`, then locally quotients by the
//! `s = 2` OOD pairs and applies DegCor with `e_i = 0` (so the
//! multiplier is `1`). At round 1 (`i + 1 == M = 2`) it compares the
//! recovered `f_2(z_k)` against `proof.final_polynomial.evaluate(z_k)`,
//! where `final_polynomial` has degree `< stopping_degree = 1` (i.e. a
//! constant). Any byte tampered in the proof rotates the transcript
//! state by an effectively random 256-bit rotation (Fiat-Shamir),
//! so the resulting consistency check fails with probability `≈ 1 −
//! negl`.
//!
//! ## Educational residue: `t_final` queries are squeezed but unchecked
//!
//! After the main loop the verifier squeezes
//! `params.repetition_schedule[num_rounds]` terminal shift indices over
//! `|L_M|`. This is a transcript-fidelity squeeze: a future extension
//! may bind additional consistency checks to those indices, and the
//! prover's loop must end with the transcript in the same state so any
//! such extension stays Fiat-Shamir-consistent. **In this refactor the
//! squeezed indices are discarded.** The real cross-round binding is
//! enforced by the per-query `f_M(z_k) == final_polynomial.evaluate(z_k)`
//! check at round `M-1` (see Step 5 above). Per `stir-full-spec.md` §3.4.
//!
//! ## Named theorem: Symmetric Challenge theorem
//!
//! > **Symmetric Challenge theorem.** Fix a Fiat-Shamir transcript hash
//! > `H`, a domain separator `ds`, and let `(m_0, m_1, …, m_{L-1})` be
//! > the ordered sequence of prover messages absorbed during a run of
//! > STIR. Define the *post-message-`j`* transcript state recursively:
//! > `S_0 := H.init(ds)`, `S_{j+1} := H.update(S_j, m_j)`. Suppose
//! > prover and verifier start from byte-identical seed `S_0` and
//! > absorb the *same* sequence `(m_0, …, m_{L-1})` (per the schedule
//! > above). Then for every `j ≤ L` and every challenge-squeeze
//! > primitive `c ∈ {squeeze_field, sample_shift_indices,
//! > sample_ood_points}` invoked off `S_j` with the same arguments,
//! >
//! > ```text
//! >   c(prover_S_j, args)  ==  c(verifier_S_j, args).
//! > ```
//! >
//! > In particular, the prover's and verifier's tuples
//! >
//! > ```text
//! >   (ood_points_0, r_comb_0, α_0, shift_indices_0, leaves_0, …,
//! >    ood_points_{M-1}, r_comb_{M-1}, α_{M-1}, shift_indices_{M-1},
//! >    leaves_{M-1}, t_final_indices)
//! > ```
//! >
//! > coincide bit-for-bit. (Note: `r_comb_i` and `α_i` are squeezed
//! > BEFORE shift indices and BEFORE preimage leaves are absorbed —
//! > matching the refactored prover; see `stir-full-spec.md` §1.1.)
//! >
//! > **Proof (one line).** Squeeze primitives are pure functions of the
//! > transcript state plus their arguments (see
//! > [`crate::transcript::Transcript::squeeze_field`] and friends — they
//! > finalize a clone of the running BLAKE3 state, re-mix the accepted
//! > bytes, and read deterministic XOF output). If `S_j_prover ==
//! > S_j_verifier` byte-for-byte, the same XOF outputs the same bytes,
//! > the same rejection-sampling loop accepts the same values, and the
//! > squeezed outputs coincide. The recursive step is structural: equal
//! > states absorb equal messages to produce equal next states (BLAKE3
//! > `update` is deterministic). Induction on `j` closes the loop. ∎
//!
//! **Why this matters.** The verifier never needs to receive the
//! prover's challenges as part of the proof — it *re-derives* them from
//! the same transcript discipline. The proof carries only the prover's
//! *responses* (commitments, OOD replies, shift answers, paths, final
//! polynomial). The Symmetric Challenge theorem is the formal statement
//! that this works: it is *the* binding between the proof object and
//! the verifier's check sequence.
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
//! 4. **DegCor binding** (this module's `DegCor Soundness theorem`, see
//!    [`crate::degree_correction`]). Probability that the random `r_comb_i`
//!    fails to bind the degree-bumped polynomial to the prover's claim:
//!    bounded by `(d* − d) / |F|`.
//! 5. **Merkle binding** (collision-resistance of SHA3-256). Bounded by
//!    `2^{-256}` per query, negligible at any reasonable parameter setting.
//!
//! Sum these per round, then union-bound across `num_rounds` rounds, then
//! union-bound against the `repetition_schedule[i]` queries within each
//! round. The total soundness error is reported by
//! [`crate::params::StirParams::soundness_report`].
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

use reed_solomon::fft::ifft_subgroup;
use reed_solomon::interpolate::lagrange_interpolate;
use reed_solomon::Fp;
#[cfg(test)]
use reed_solomon::UnivariatePoly;

use crate::degree_correction::eval_g_r;
use crate::domain::StirDomain;
use crate::merkle::MerkleTree;
use crate::ood::sample_ood_points;
use crate::params::StirParams;
use crate::proof::StirProof;
use crate::transcript::Transcript;

/// Return the `k` preimage indices in `L_i` for fibre representative
/// `q_idx ∈ [0, n_i / k)`. Mirror of the prover's helper; see
/// [`crate::prover`] and `stir-full-spec.md` §5.
fn preimage_indices_in_l_i(q_idx: usize, n_i: usize, k: usize) -> Vec<usize> {
    debug_assert_eq!(n_i % k, 0, "n_i must be divisible by k");
    let stride = n_i / k;
    debug_assert!(q_idx < stride, "q_idx must be in [0, n_i / k)");
    (0..k).map(|j| q_idx + j * stride).collect()
}

/// Locally fold the `k` Merkle-authenticated preimage values of one
/// shift-query fibre to recover `g_i(z_k) = Fold(f_i, k, α)(z_k)` at
/// `z_k = x_0^k ∈ L_i^k`. This is the verifier's pointwise mirror of
/// [`crate::fold::fold`]; the algorithm follows fold.rs's
/// "Why it works on evaluation tables" recipe (IDFT-and-untwist).
///
/// # Inputs
/// - `leaves`: the `k` preimage values `f_i(ω_k^j · x_0)` for `j ∈ 0..k`.
/// - `alpha`: the round's fold randomness `α_i`.
/// - `x_0`: the first preimage `L_i.element(q_idx)`.
/// - `omega_k`: the primitive `k`-th root of unity `ω_i^{n_i / k}`.
/// - `k`: the folding factor (must be a power of two).
///
/// # Algorithm
/// 1. IDFT-`k` of the `k` siblings at root `ω_k` recovers the twisted
///    row evaluations `twisted[l] = f_l(z_k) · x_0^l` (see fold.rs).
/// 2. Peel off the `x_0^l` twist factor: `r_l = twisted[l] / x_0^l`
///    via a running multiplication by `x_0^{-1}` (one inverse needed).
/// 3. Linear-combine with the powers of `α`: `Σ_l α^l · r_l`.
fn local_fold_at_zk(leaves: &[Fp], alpha: Fp, x_0: Fp, omega_k: Fp, k: usize) -> Fp {
    debug_assert_eq!(leaves.len(), k, "expected k = {} leaves, got {}", k, leaves.len());

    // Step 1: IDFT-k at ω_k. twisted[l] = f_l(z_k) · x_0^l.
    let twisted = ifft_subgroup(leaves, omega_k);

    // Step 2 + Step 3: peel off x_0^l, linear-combine with α^l.
    // Running x_0^l via multiplication; alpha^l likewise.
    let x_0_inv = x_0.inverse().expect("x_0 ∈ L_i is non-zero");
    let mut x_inv_pow = Fp::one(); // x_0^{-l}, starting at l = 0
    let mut alpha_pow = Fp::one(); // α^l
    let mut acc = Fp::zero();
    for &t_l in twisted.iter() {
        // r_l = t_l · x_0^{-l}
        let r_l = t_l * x_inv_pow;
        acc = acc + alpha_pow * r_l;
        alpha_pow = alpha_pow * alpha;
        x_inv_pow = x_inv_pow * x_0_inv;
    }
    acc
}

/// Evaluate the Lagrange interpolant through `(points[i], values[i])`
/// at `at`. Implemented via [`reed_solomon::interpolate::lagrange_interpolate`]
/// + a single polynomial evaluation. For the demo parameters
/// `points.len() == s == 2`, so the underlying `O(n²)` cost is trivial.
fn lagrange_eval(points: &[Fp], values: &[Fp], at: Fp) -> Fp {
    debug_assert_eq!(points.len(), values.len());
    let pairs: Vec<(Fp, Fp)> = points
        .iter()
        .zip(values.iter())
        .map(|(&x, &y)| (x, y))
        .collect();
    let poly = lagrange_interpolate(&pairs);
    poly.evaluate(at)
}

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
    pub fn new(params: StirParams) -> Self {
        params
            .validate()
            .expect("StirVerifier::new: invalid StirParams");
        let stir_domain = StirDomain::new(&params);
        Self {
            params,
            stir_domain,
        }
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
    /// Implementation follows `stir-full-spec.md` §3.
    pub fn verify(
        &self,
        proof: &StirProof,
        transcript: &mut Transcript,
    ) -> Result<(), &'static str> {
        let num_rounds = self.params.num_rounds as usize;
        let s = self.params.ood_samples as usize;
        let k = self.params.folding_factor as usize;

        // -------------------------------------------------------------
        // Structural pre-checks (Proof-Shape Invariant theorem, see
        // `crate::proof` module docs). These are cheap and fail fast on
        // malformed proofs before any cryptographic work.
        // -------------------------------------------------------------
        if proof.round_commitments.len() != num_rounds {
            return Err("STIR: round_commitments.len() != params.num_rounds");
        }
        if proof.ood_replies.len() != num_rounds {
            return Err("STIR: ood_replies.len() != params.num_rounds");
        }
        if proof.shift_answers.len() != num_rounds {
            return Err("STIR: shift_answers.len() != params.num_rounds");
        }
        if proof.merkle_paths.len() != num_rounds {
            return Err("STIR: merkle_paths.len() != params.num_rounds");
        }
        if proof.merkle_opened_leaves.len() != num_rounds {
            return Err("STIR: merkle_opened_leaves.len() != params.num_rounds");
        }
        // PoW is descoped this iteration — see
        // `crate::proof::StirProof::pow_nonces` field doc.
        if !proof.pow_nonces.is_empty() {
            return Err("STIR: pow_nonces must be empty in this implementation");
        }
        // Per-round inner-length checks (triple-nested for the per-query
        // fields, since each shift query opens k = folding_factor preimages).
        for i in 0..num_rounds {
            if proof.ood_replies[i].len() != s {
                return Err("STIR: ood_replies[i].len() != params.ood_samples");
            }
            let t_i = self.params.repetition_schedule[i] as usize;
            if proof.shift_answers[i].len() != t_i {
                return Err("STIR: shift_answers[i].len() != repetition_schedule[i]");
            }
            if proof.merkle_paths[i].len() != t_i {
                return Err("STIR: merkle_paths[i].len() != repetition_schedule[i]");
            }
            if proof.merkle_opened_leaves[i].len() != t_i {
                return Err("STIR: merkle_opened_leaves[i].len() != repetition_schedule[i]");
            }
            for q in 0..t_i {
                if proof.shift_answers[i][q].len() != k {
                    return Err("STIR: shift_answers[i][q].len() != folding_factor");
                }
                if proof.merkle_paths[i][q].len() != k {
                    return Err("STIR: merkle_paths[i][q].len() != folding_factor");
                }
                if proof.merkle_opened_leaves[i][q].len() != k {
                    return Err("STIR: merkle_opened_leaves[i][q].len() != folding_factor");
                }
            }
        }

        // -------------------------------------------------------------
        // Per-round verification loop (the mirror of the prover loop).
        // -------------------------------------------------------------
        for i in 0..num_rounds {
            let commitment_i = &proof.round_commitments[i];
            let domain_i = self.stir_domain.round_domain(i);
            let n_i = domain_i.size();
            let d_i = self.params.round_degree_bound(i as u32);
            let d_ip1 = self.params.round_degree_bound((i + 1) as u32);

            // (1) Cross-check the prover's claimed tree_size against
            //     the params-derived expected size (Tree-Size Binding
            //     theorem; `crate::commitment` module doc).
            let expected_tree_size: usize = n_i;
            if commitment_i.tree_size != expected_tree_size {
                return Err("STIR: commitment.tree_size mismatch with params");
            }

            // (2) Absorb the round's Merkle root. Must precede every
            //     subsequent squeeze in this round.
            transcript.absorb_root(&commitment_i.root);

            // (3) Re-derive the round's OOD points, forbidden domain L_{i+1}.
            //     StirDomain stores M+1 domains, so round_domain(i+1) is
            //     always valid for i < M.
            let next_domain = self.stir_domain.round_domain(i + 1);
            let ood_points_i: Vec<Fp> = sample_ood_points(
                transcript,
                self.params.ood_samples,
                next_domain,
            );

            // (4) Absorb the prover's OOD replies in transcript order.
            for &reply in &proof.ood_replies[i] {
                transcript.absorb_field(reply);
            }

            // (5) Squeeze r_comb_i, then α_i.
            //
            //     ORDER IS LOAD-BEARING. Mirror of prover §2.1 step 3:
            //     r_comb_i is the DegCor randomness (fresh squeeze, not
            //     reused from α_i) per the DegCor Soundness theorem.
            //     α_i is the fold randomness. Both are squeezed BEFORE
            //     shift indices.
            let r_comb_i: Fp = transcript.squeeze_field();
            let alpha_i: Fp = transcript.squeeze_field();

            // (6) Sample the round's t_i shift indices over [0, n_i / k).
            //     Per `stir-full-spec.md` §5: indices range over the
            //     fibre-representative space, NOT |L_i| or |L_{i+1}|.
            assert!(
                n_i % k == 0,
                "round {i}: |L_i| = {n_i} not divisible by k = {k}",
            );
            let t_i = self.params.repetition_schedule[i];
            let shift_indices_i: Vec<u32> =
                transcript.sample_shift_indices(t_i, (n_i / k) as u32);

            // (7) Compute the degree bump e_i bit-for-bit identical to
            //     the prover (defensive `saturating_sub` and integer
            //     ceil). For demo params this is 0 in every round.
            let g_deg_bound = if d_i >= s {
                (d_i - s + k - 1) / k
            } else {
                0
            };
            let e_i: usize = d_ip1.saturating_sub(g_deg_bound);

            // Primitive k-th root of unity for the local IDFT.
            let omega_k = domain_i.generator().pow((n_i / k) as u64);

            // (8) Per-query consistency check (5-step recipe, §"Per-query
            //     consistency check" in the module doc).
            for q in 0..(t_i as usize) {
                let q_idx = shift_indices_i[q] as usize;
                let preimage_idxs = preimage_indices_in_l_i(q_idx, n_i, k);

                // Step 1: verify k Merkle paths, cross-check leaves, and
                // absorb each leaf into the transcript.
                let leaves_q = &proof.merkle_opened_leaves[i][q];
                let paths_q = &proof.merkle_paths[i][q];
                let answers_q = &proof.shift_answers[i][q];
                for j in 0..k {
                    let leaf = leaves_q[j];
                    let path = &paths_q[j];
                    if !MerkleTree::verify(
                        commitment_i.root.clone(),
                        preimage_idxs[j],
                        leaf,
                        path,
                        expected_tree_size,
                    ) {
                        return Err("STIR: Merkle path failed to verify");
                    }
                    if leaf != answers_q[j] {
                        return Err("STIR: opened leaf does not match shift answer");
                    }
                    // Mirror prover §2.1 step 5: absorb each preimage
                    // value into the transcript.
                    transcript.absorb_field(leaf);
                }

                // Step 2: local Quotient at each of the k preimages.
                //
                // The prover does Quotient → Fold (NOT Fold → Quotient),
                // so the verifier must MIRROR the order. At each preimage
                // x_j we compute
                //
                //   q_i(x_j) = (f_i(x_j) - p_ood(x_j)) / V_ood(x_j).
                //
                // For s = 0 (no OOD step), the quotient is the identity:
                // q_i(x_j) = f_i(x_j) and the loop below collapses.
                let x_0 = domain_i.element(preimage_idxs[0]);
                let z_k = x_0.pow(k as u64);

                let q_at_preimages: Vec<Fp> = if s == 0 {
                    answers_q.clone()
                } else {
                    let mut out = Vec::with_capacity(k);
                    for j in 0..k {
                        let x_j = domain_i.element(preimage_idxs[j]);
                        let p_at_xj =
                            lagrange_eval(&ood_points_i, &proof.ood_replies[i], x_j);
                        let v_at_xj: Fp = ood_points_i
                            .iter()
                            .map(|&zj| x_j - zj)
                            .fold(Fp::one(), |acc, factor| acc * factor);
                        let v_inv = v_at_xj.inverse().ok_or(
                            "STIR: V_ood(x_j) = 0; preimage collided with an OOD point",
                        )?;
                        out.push((answers_q[j] - p_at_xj) * v_inv);
                    }
                    out
                };

                // Step 3: local Fold of the per-preimage quotient values
                // to recover g_i(z_k) = Fold(q_i, k, α_i)(z_k) where
                // z_k = x_0^k ∈ L_i^k.
                let g_i_at_zk =
                    local_fold_at_zk(&q_at_preimages, alpha_i, x_0, omega_k, k);

                // Step 4: local DegCor by r_comb_i.
                let g_r_at_zk = eval_g_r(r_comb_i, z_k, e_i);
                let f_ip1_at_zk = g_i_at_zk * g_r_at_zk;

                // Step 5: terminal comparison at i + 1 == M.
                //
                // For i + 1 < M the cross-round binding is enforced
                // implicitly by the round-(i+1) OOD-binding; no direct
                // Merkle cross-check here (see `stir-full-spec.md` §3.4).
                if i + 1 == num_rounds {
                    let claimed = proof.final_polynomial.evaluate(z_k);
                    if claimed != f_ip1_at_zk {
                        return Err(
                            "STIR: final-polynomial consistency check failed",
                        );
                    }
                }
            }
        }

        // -------------------------------------------------------------
        // Final-polynomial degree check.
        //
        // The prover sends `final_polynomial` in the clear; degree-bounding
        // it is the cheapest possible check and rules out an entire class
        // of cheats. The zero polynomial's `degree()` returns `None`,
        // which we treat as trivially `< stopping_degree` (a zero codeword
        // is in every RS code).
        // -------------------------------------------------------------
        if let Some(d) = proof.final_polynomial.degree() {
            if d >= self.params.stopping_degree {
                return Err("STIR: final_polynomial degree >= stopping_degree");
            }
        }

        // -------------------------------------------------------------
        // Terminal t_final squeeze (transcript-schedule fidelity).
        //
        // The squeezed indices are UNCHECKED — see `stir-full-spec.md`
        // §3.4. The real cross-round binding is enforced by the per-query
        // f_M(z_k) == final_polynomial.evaluate(z_k) check inside the main
        // loop. We still perform the squeeze so the transcript state ends
        // bit-for-bit identical to the prover's — any future extension
        // that adds checks at `t_final_indices` will then be Fiat-Shamir
        // consistent without further protocol changes.
        let t_final = self.params.repetition_schedule[num_rounds];
        let final_domain = self.stir_domain.round_domain(num_rounds);
        let final_size = final_domain.size() as u32;
        let _t_final_indices: Vec<u32> =
            transcript.sample_shift_indices(t_final, final_size);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::StirCommitment;
    use crate::merkle::{MerklePath, MerkleRoot};
    use crate::params::StirParams;

    /// Build a small fixed-shape `StirProof` that matches `params` in
    /// outer/middle/inner lengths but is otherwise arbitrary (root =
    /// zeros, shift answers = zeros, empty Merkle paths). The only thing
    /// this helper guarantees is the **triple-nested shape** required by
    /// the Proof-Shape Invariant theorem.
    fn empty_shape_proof(params: &StirParams) -> StirProof {
        let num_rounds = params.num_rounds as usize;
        let s = params.ood_samples as usize;
        let k = params.folding_factor as usize;

        let round_commitments: Vec<StirCommitment> = (0..num_rounds)
            .map(|i| StirCommitment {
                root: MerkleRoot([0u8; 32]),
                tree_size: 1usize << params.round_log_domain_size(i as u32),
            })
            .collect();
        let ood_replies: Vec<Vec<Fp>> =
            (0..num_rounds).map(|_| vec![Fp::zero(); s]).collect();
        let shift_answers: Vec<Vec<Vec<Fp>>> = (0..num_rounds)
            .map(|i| {
                (0..params.repetition_schedule[i] as usize)
                    .map(|_| vec![Fp::zero(); k])
                    .collect()
            })
            .collect();
        let merkle_paths: Vec<Vec<Vec<MerklePath>>> = (0..num_rounds)
            .map(|i| {
                (0..params.repetition_schedule[i] as usize)
                    .map(|q| {
                        (0..k)
                            .map(|j| MerklePath {
                                siblings: Vec::new(),
                                leaf_index: q * k + j,
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let merkle_opened_leaves: Vec<Vec<Vec<Fp>>> = (0..num_rounds)
            .map(|i| {
                (0..params.repetition_schedule[i] as usize)
                    .map(|_| vec![Fp::zero(); k])
                    .collect()
            })
            .collect();

        StirProof {
            round_commitments,
            ood_replies,
            shift_answers,
            merkle_paths,
            merkle_opened_leaves,
            final_polynomial: UnivariatePoly::zero(),
            pow_nonces: Vec::new(),
        }
    }

    /// Verifier rejects a proof whose round count doesn't match params.
    ///
    /// Structural length checks are the verifier's first line of
    /// defence — cheap, fast-fail before any cryptography runs.
    #[test]
    fn verifier_rejects_proof_with_wrong_round_count() {
        // Demo params: `M = num_rounds = 2`.
        let params = StirParams::new(6, 16, 4);
        assert_eq!(params.num_rounds, 2);

        // Build a shape-valid proof, then *truncate* the
        // `round_commitments` to length 1 (mismatch).
        let mut proof = empty_shape_proof(&params);
        proof.round_commitments.truncate(1);

        let verifier = StirVerifier::new(params);
        let mut transcript = Transcript::new(b"stir-protocol-v0");
        let result = verifier.verify(&proof, &mut transcript);
        assert!(
            result.is_err(),
            "verifier must reject a proof with the wrong round count",
        );
        // Spot-check the error string mentions the right field.
        assert!(
            result.unwrap_err().contains("round_commitments"),
            "unexpected error message",
        );
    }

    /// Verifier rejects a proof with a tampered Merkle path.
    ///
    /// Tampers `proof.merkle_paths[0][0][0].siblings[0]` (the first
    /// preimage's first sibling of the first query's first round) — the
    /// triple-nested indexing matches the refactored proof shape. The
    /// verifier's correctness here reduces to `MerkleTree::verify`
    /// returning `false` on a sibling-byte flip, plus the per-query
    /// failure-propagation we just wired in.
    #[test]
    fn verifier_rejects_tampered_merkle_path() {
        // Build an honest proof end-to-end.
        let params = StirParams::new(6, 16, 4);
        let mut coeffs = vec![Fp::zero(); params.initial_degree_bound];
        for (i, c) in coeffs.iter_mut().enumerate() {
            // A non-trivial low-degree polynomial. Any concrete
            // coefficient choice works; we just need degree < 16.
            *c = Fp::new((i as u64) + 1);
        }
        let polynomial = UnivariatePoly::new(coeffs);

        let proof_result = crate::protocol::run_stir(
            params.clone(),
            polynomial,
        );
        let mut proof = proof_result.expect("honest run_stir must succeed");

        // Sanity: an unmodified proof should verify.
        {
            let verifier = StirVerifier::new(params.clone());
            let mut t = Transcript::new(b"stir-protocol-v0");
            verifier
                .verify(&proof, &mut t)
                .expect("honest proof must verify");
        }

        // Tamper: flip one byte of the first round's first query's first
        // preimage's first sibling. The Merkle path tree has at least
        // one level because |L_0| = 64 > 1.
        assert!(
            !proof.merkle_paths[0][0][0].siblings.is_empty(),
            "expected non-empty Merkle path for |L_0| > 1",
        );
        proof.merkle_paths[0][0][0].siblings[0][0] ^= 0xFF;

        let verifier = StirVerifier::new(params);
        let mut t = Transcript::new(b"stir-protocol-v0");
        let result = verifier.verify(&proof, &mut t);
        assert!(
            result.is_err(),
            "verifier must reject a proof with a tampered Merkle path",
        );
    }

    /// Verifier rejects a proof whose `final_polynomial` was substituted
    /// for a different polynomial of the same (low) degree. The
    /// degree-bound check then passes, and the **per-query final-
    /// polynomial consistency check** at round `M-1` must catch the
    /// substitution.
    #[test]
    fn verifier_rejects_wrong_final_polynomial() {
        let params = StirParams::new(6, 16, 4);
        let coeffs: Vec<Fp> = (0..params.initial_degree_bound)
            .map(|i| Fp::new((i as u64) + 1))
            .collect();
        let polynomial = UnivariatePoly::new(coeffs);

        let mut proof = crate::protocol::run_stir(
            params.clone(),
            polynomial,
        )
        .expect("honest run_stir must succeed");

        // Sanity: an unmodified proof should verify.
        {
            let verifier = StirVerifier::new(params.clone());
            let mut t = Transcript::new(b"stir-protocol-v0");
            verifier
                .verify(&proof, &mut t)
                .expect("honest proof must verify");
        }

        // Substitute a different final polynomial. Demo
        // `stopping_degree == 1`, so degree must be `< 1` — constants
        // only. Pick a value distinct from the honest one (guard with a
        // conditional swap to handle the unlikely adversarial collision).
        let new_final = if proof.final_polynomial.evaluate(Fp::one())
            == Fp::new(42)
        {
            UnivariatePoly::new(vec![Fp::new(43)])
        } else {
            UnivariatePoly::new(vec![Fp::new(42)])
        };
        proof.final_polynomial = new_final;

        let verifier = StirVerifier::new(params);
        let mut t = Transcript::new(b"stir-protocol-v0");
        let result = verifier.verify(&proof, &mut t);
        assert!(
            result.is_err(),
            "verifier must reject a proof with a substituted final polynomial",
        );
    }
}
