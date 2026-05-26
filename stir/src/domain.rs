//! The STIR per-round evaluation-domain progression `L_0, L_1, ..., L_M`.
//!
//! ## Anchor: what this module gives the rest of STIR
//!
//! Every other module in this crate — [`crate::fold`], [`crate::quotient`],
//! [`crate::ood`], [`crate::commitment`], the [`crate::prover`] and
//! [`crate::verifier`] — needs to know **exactly one thing about geometry**:
//! given a round index `i ∈ {0, 1, ..., M}`, what is the evaluation domain
//! `L_i ⊂ F_p^*` that the round-`i` committed function lives on?
//!
//! This module computes the answer **once**, deterministically from a
//! [`crate::params::StirParams`], and caches the full sequence
//! `[L_0, L_1, ..., L_M]` as a vector of [`reed_solomon::EvaluationDomain`].
//! Prover and verifier both call [`StirDomain::new`] on the same `params`
//! and therefore obtain *bit-identical* domain sequences — there is no
//! Fiat-Shamir randomness in this module; the domain schedule is part of
//! the **public protocol parameters**, fixed before the proof starts.
//!
//! Concretely, [`StirDomain`] exposes four operations the rest of STIR
//! uses:
//!
//! - [`StirDomain::round_domain`] `(i)` — index into the sequence; used by
//!   round-`i` folding, OOD sampling, and shift-query generation.
//! - [`StirDomain::initial_domain`] — alias for `round_domain(0)`; the RS
//!   codeword domain, where the prover's first Merkle commit lives.
//! - [`StirDomain::final_domain`] — alias for `round_domain(M)`; where the
//!   trailing low-degree codeword lives and is sent in plaintext.
//! - [`StirDomain::num_rounds`] — `M`; bound for round loops.
//!
//! Everything else (folding maps, quotient polynomials, OOD evaluation
//! checks, Merkle openings) reads coordinates *off* these domains rather
//! than recomputing root-of-unity exponentiations. This is purely a
//! caching optimization, but it also forces a single source of truth for
//! the schedule — eliminating a class of "prover and verifier disagree on
//! `L_i`" bugs.
//!
//! ## What STIR's per-round domain progression looks like
//!
//! STIR runs `M` folding rounds. In round `i`, the prover commits to a
//! function `g_i: L_i → F_p`. The next round folds the function (degree
//! drops by factor `k = folding_factor`) and **moves to a strictly smaller
//! shifted domain `L_{i+1}`**.
//!
//! The key structural fact — the one most readers miss when transitioning
//! from FRI to STIR:
//!
//! > **In STIR the evaluation domain shrinks by `k/2` per round; in FRI it
//! > shrinks by `k`. Decoupling shrink-rate from fold-rate is what gives
//! > STIR its variable per-round rate `ρ_i = ρ_0 / 2^i`, which in turn
//! > drives its `O(log² d)` query advantage.**
//!
//! For our default `k = 4`, `log₂(k/2) = 1`, so `|L_i|` halves per round —
//! see the [`crate::params`] domain-shrink theorem (cross-referenced as
//! `params::StirParams::round_log_domain_size`) for the algebra.
//!
//! ## The construction (formal)
//!
//! Let `n = log_initial_domain_size` and `M = num_rounds`. Let
//! `ω_0 = Fp::primitive_root_of_unity(n)`, the primitive `2^n`-th root of
//! unity in Goldilocks. Let `g = MULTIPLICATIVE_GENERATOR = 7`, a
//! primitive root of `F_p^*` (so `g` has order `p − 1`).
//!
//! For each round `i ∈ {0, 1, ..., M}`, define
//! `n_i = round_log_domain_size(i) = n − i · log₂(k/2)`. Then:
//!
//! ```text
//!   L_0 = ⟨ω_0⟩                        (a pure subgroup, offset = 1)
//!   L_i = c_i · ⟨ω_i⟩      (i ≥ 1)     where ω_i = primitive 2^{n_i}-th root,
//!                                       and c_i = g^{2^{i-1}}.
//! ```
//!
//! Concretely, `c_1 = g, c_2 = g^2, c_3 = g^4, ..., c_i = g^{2^{i-1}}`.
//! Each `c_i` is a successive power-of-two power of the multiplicative
//! generator `g` — a deterministic, public schedule that requires no
//! Fiat-Shamir randomness and is shared by prover and verifier.
//!
//! Note: `ω_i = ω_0^{2^{i · log₂(k/2)}}`. For `k = 4` (`log₂(k/2) = 1`)
//! this simplifies to `ω_i = ω_0^{2^i}`, i.e., **each round squares the
//! generator**.
//!
//! ## Named theorem: domain-disjointness
//!
//! > **Domain-disjointness theorem.** With the offset schedule above
//! > (`c_0 = 1`, `c_i = g^{2^{i-1}}` for `i ≥ 1`), the consecutive
//! > domains are pairwise disjoint:
//! >
//! > ```text
//! >   L_i ∩ L_{i+1} = ∅   for every i ∈ {0, 1, ..., M − 1}.
//! > ```
//!
//! **Proof sketch.** Suppose for contradiction that some `x ∈ L_i ∩ L_{i+1}`.
//! Then `x = c_i · ω_i^a = c_{i+1} · ω_{i+1}^b` for some integers `a, b`.
//! Rearranging, `c_{i+1} / c_i = ω_i^a · ω_{i+1}^{−b}`. Both `ω_i` and
//! `ω_{i+1}` are powers of `ω_0` of order dividing `2^n`, so the
//! right-hand side is a `2^n`-th root of unity — in particular it lies in
//! `⟨ω_0⟩`, the unique subgroup of `F_p^*` of order `2^n`. Now:
//!
//! - `⟨ω_0⟩` consists exactly of the `2^n`-th roots of unity, i.e., the
//!   elements `y ∈ F_p^*` with `y^{2^n} = 1`.
//! - `(c_{i+1} / c_i)^{2^n} = g^{(2^i − 2^{i−1}) · 2^n} = g^{2^{i−1} · 2^n}`
//!   (taking `c_0 := 1` so this expression is also `g^{2^n}` when `i = 0`).
//! - For this to equal `1`, we would need `2^{i−1} · 2^n` (or `2^n` when
//!   `i = 0`) to be a multiple of `ord(g) = p − 1`. But `p − 1 = 2^{32} · (2^{32} − 1)`
//!   in Goldilocks, and `(2^{32} − 1)` is an odd cofactor `> 1`. Since
//!   `2^{i−1} · 2^n` is a *pure* power of 2 and `n ≤ 32`, it cannot be a
//!   multiple of any odd number `> 1`, so `(c_{i+1} / c_i)^{2^n} ≠ 1`.
//!
//! Contradiction. Hence `L_i ∩ L_{i+1} = ∅`. ∎
//!
//! In one line: `c_{i+1} / c_i = g^{2^{i−1}}` is **not** a `2^{n_i}`-th
//! power in `F_p^*`, because `g` generates the full odd cofactor of
//! `p − 1` and our exponent is a pure power of 2.
//!
//! **Why this matters for STIR.** The out-of-domain (OOD) step
//! ([`crate::ood`]) samples points from `F_p \ L_{i+1}` and asks the
//! prover to evaluate the folded function there. If `L_i ∩ L_{i+1}` were
//! non-empty, an OOD sample drawn from `L_i \ L_{i+1}` could *accidentally
//! coincide* with an `L_i`-point the prover has already opened a Merkle
//! path to — giving the prover free degrees of freedom to adapt the OOD
//! response. The disjointness above rules that out: the verifier's OOD
//! queries hit points the prover could not have pre-computed from the
//! previous round's Merkle tree.
//!
//! (Soundness in the actual paper uses a stronger condition — disjointness
//! from the `k`-th-power image `L_i^k`, see eprint 2024/390 §3.2 — but
//! pairwise disjointness of the `L_i` themselves is the cleanest invariant
//! to enforce and is what this implementation guarantees.)
//!
//! ## Worked numeric example
//!
//! Take `log_initial_domain_size = 6`, `folding_factor = 4`, `num_rounds = 2`
//! (i.e., `StirParams::new(6, 16, 4)`). Then `log₂(k/2) = 1`, so:
//!
//! ```text
//! Round 0:  log|L_0| = 6.   |L_0| = 64.
//!           ω_0 = primitive 64th root of unity.
//!           c_0 = 1.                          (subgroup, offset = 1)
//!           L_0 = ⟨ω_0⟩  = {1, ω_0, ω_0², ..., ω_0^{63}}.
//!
//! Round 1:  log|L_1| = 5.   |L_1| = 32.
//!           ω_1 = ω_0^2   (primitive 32nd root of unity).
//!           c_1 = g^{2^0} = g^1 = 7.
//!           L_1 = 7 · ⟨ω_0²⟩  = {7, 7·ω_0², 7·ω_0^4, ..., 7·ω_0^{62}}.
//!
//! Round 2:  log|L_2| = 4.   |L_2| = 16.
//!           ω_2 = ω_0^4   (primitive 16th root of unity).
//!           c_2 = g^{2^1} = g^2 = 49.
//!           L_2 = 49 · ⟨ω_0^4⟩ = {49, 49·ω_0^4, 49·ω_0^8, ..., 49·ω_0^{60}}.
//! ```
//!
//! Domain sizes `[64, 32, 16]` halve per round (because `k = 4 ⇒ k/2 = 2`).
//! Generators `[ω_0, ω_0², ω_0^4]` square per round. Offsets
//! `[1, 7, 49]` are successive `2^{i−1}`-th powers of `g = 7`. The
//! disjointness theorem applied with `n = 6`: `(c_2/c_1)^{2^6} = (g)^{64}
//! = 7^{64} ≠ 1` since the order of `7` is `p − 1 = 2^{32}·(2^{32}−1)`
//! which has odd cofactor `> 1`, so `7^{64}` cannot equal `1`. ✓
//!
//! ## Caveats for implementers
//!
//! ```text
//! // CAUTION: the offset schedule c_i = g^{2^{i-1}} is part of the
//! //          PUBLIC PROTOCOL PARAMETERS, not Fiat-Shamir randomness. It
//! //          is fixed by `StirDomain::new(&params)` deterministically.
//! //          Treating it as random would over-rotate freedoms the
//! //          adversary already has; treating it as adversary-chosen
//! //          would break the disjointness theorem above.
//!
//! // CAUTION: prover and verifier MUST share `params` byte-for-byte so
//! //          their `StirDomain` sequences agree. This is the cheapest
//! //          invariant in the protocol to maintain — both sides just
//! //          call `StirDomain::new(&params)` — but the easiest to break
//! //          by accident if one side reconstructs `params` from a wire
//! //          format and gets a single bit wrong.
//! ```
//!
//! ## See also
//!
//! - [`crate::params::StirParams::round_log_domain_size`] for the
//!   domain-shrink formula `log|L_i| = log|L_0| − i · log₂(k/2)`.
//! - [`crate::ood`] for the OOD step that relies on `L_i ∩ L_{i+1} = ∅`.
//! - [`reed_solomon::EvaluationDomain`] for the underlying coset
//!   representation.

use reed_solomon::domain::EvaluationDomain;
use reed_solomon::field::MULTIPLICATIVE_GENERATOR;
use reed_solomon::Fp;

/// The sequence of STIR evaluation domains `[L_0, L_1, ..., L_M]`.
///
/// `round_domains[i]` is the evaluation domain *used as input* to round
/// `i`. `round_domains[0]` is the domain on which the prover originally
/// committed (the RS codeword domain). `round_domains[num_rounds()]` is
/// the *final* domain — i.e., the one the last folded codeword lives on,
/// which the verifier reads in plaintext.
pub struct StirDomain {
    /// The evaluation domains for every round, in order.
    ///
    /// - `round_domains[0]` is the subgroup `⟨ω_0⟩` of order
    ///    `2^{log_initial_domain_size}` (offset = 1).
    /// - `round_domains[i]` for `i ≥ 1` is the coset
    ///    `g^{2^{i-1}} · ⟨ω_i⟩` of order
    ///    `2^{round_log_domain_size(i)}`, where `g = MULTIPLICATIVE_GENERATOR`.
    /// - `round_domains.len() == num_rounds + 1` (one extra for the
    ///    final domain after the last fold).
    ///
    /// See the module-level "domain-disjointness theorem" for why
    /// consecutive entries here are pairwise disjoint.
    pub round_domains: Vec<EvaluationDomain>,
}

impl StirDomain {
    /// Build the per-round domain progression from STIR parameters.
    ///
    /// Reads `log_initial_domain_size`, `num_rounds`, and `folding_factor`
    /// (via [`crate::params::StirParams::round_log_domain_size`]) out of
    /// `params`, then constructs `[L_0, ..., L_M]` per the construction
    /// in the module docs:
    ///
    /// - `L_0 = ⟨ω_0⟩` (subgroup, offset 1).
    /// - `L_i = g^{2^{i-1}} · ⟨ω_i⟩` for `i ≥ 1`, where `ω_i` is the
    ///   primitive `2^{n_i}`-th root of unity.
    ///
    /// The offset schedule `c_i = g^{2^{i-1}}` is what makes the
    /// **domain-disjointness theorem** in the module docs go through —
    /// see that theorem for the proof that `L_i ∩ L_{i+1} = ∅`.
    ///
    /// # Panics
    ///
    /// Panics if `params.round_log_domain_size(num_rounds)` would
    /// underflow (the subgroup would shrink below size 1) — though
    /// [`crate::params::StirParams::validate`] should have caught this
    /// already.
    pub fn new(params: &crate::params::StirParams) -> Self {
        let num_rounds = params.num_rounds as usize;
        let mut round_domains: Vec<EvaluationDomain> = Vec::with_capacity(num_rounds + 1);

        // Round 0: pure subgroup ⟨ω_0⟩. Offset = 1. This matches the RS
        // encoder convention — the initial codeword domain in
        // [`reed_solomon::encode::ReedSolomonCode`] is also `new_subgroup`,
        // so the prover's first Merkle commit lives on the same set of
        // points the RS encoder produced.
        let n0 = params.round_log_domain_size(0);
        round_domains.push(EvaluationDomain::new_subgroup(n0));

        // Rounds 1..=M: each L_i = c_i · ⟨ω_i⟩ with c_i = g^{2^{i-1}}.
        //
        // We compute c_i incrementally by repeated squaring:
        //
        //     c_1 = g            (= g^{2^0})
        //     c_{i+1} = c_i^2    (since 2^i = 2 · 2^{i-1})
        //
        // No big exponentiations; each step is one multiplication.
        // See the module-level disjointness theorem for why this
        // schedule keeps consecutive L_i, L_{i+1} disjoint.
        let g = Fp::new(MULTIPLICATIVE_GENERATOR);
        let mut offset = g; // c_1 = g^{2^0} = g
        for i in 1..=num_rounds {
            let n_i = params.round_log_domain_size(i as u32);
            round_domains.push(EvaluationDomain::new_coset(n_i, offset));
            // Prepare c_{i+1} = c_i^2 for the next iteration (unused after
            // the last one — that's fine, the cost is one multiply).
            offset = offset * offset;
        }

        debug_assert_eq!(round_domains.len(), num_rounds + 1);
        Self { round_domains }
    }

    /// The evaluation domain used in round `round` (0-indexed).
    ///
    /// `round_domain(0)` is the initial domain `L_0`.
    /// `round_domain(self.num_rounds())` is the final domain `L_M`.
    ///
    /// # Panics
    ///
    /// Panics if `round > self.num_rounds()`.
    pub fn round_domain(&self, round: usize) -> &EvaluationDomain {
        assert!(
            round < self.round_domains.len(),
            "round {round} exceeds num_rounds ({})",
            self.num_rounds(),
        );
        &self.round_domains[round]
    }

    /// The number of folding rounds STIR runs.
    ///
    /// Equal to `round_domains.len() - 1` because we store one extra
    /// domain (the final domain `L_M` after the last fold).
    pub fn num_rounds(&self) -> usize {
        // By construction `round_domains` has length `num_rounds + 1`.
        self.round_domains.len() - 1
    }

    /// The initial evaluation domain `L_0`.
    ///
    /// Equivalent to `self.round_domain(0)`. Provided as a method because
    /// `L_0` is referenced in many places (transcript absorption, query
    /// generation) and a named accessor is clearer than a magic `0`.
    pub fn initial_domain(&self) -> &EvaluationDomain {
        &self.round_domains[0]
    }

    /// The final evaluation domain `L_M`.
    ///
    /// Equivalent to `self.round_domain(self.num_rounds())`. The final
    /// folded codeword lives here and is sent in plaintext.
    pub fn final_domain(&self) -> &EvaluationDomain {
        &self.round_domains[self.round_domains.len() - 1]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::StirParams;
    use std::collections::HashSet;

    /// The domain vector has length `num_rounds + 1`.
    #[test]
    fn domain_count_equals_num_rounds_plus_1() {
        // log|L_0| = 6, d_0 = 4, k = 4 → num_rounds = ceil(log_4(4)) = 1.
        // We override num_rounds to exercise the off-by-one explicitly.
        let params = StirParams::new(6, 16, 4).with_num_rounds(3);
        // Domain-shrink: |L_0|=64, |L_1|=32, |L_2|=16, |L_3|=8. All ≥ 1. OK.
        let sd = StirDomain::new(&params);
        assert_eq!(sd.round_domains.len(), 4);
        assert_eq!(sd.num_rounds(), 3);
    }

    /// Each successive domain is half the size of the previous one
    /// (the "shrink by k/2" property for k = 4).
    #[test]
    fn successive_domains_halve_in_size() {
        let params = StirParams::new(5, 4, 4).with_num_rounds(3);
        let sd = StirDomain::new(&params);
        for i in 0..sd.num_rounds() {
            assert_eq!(
                sd.round_domain(i).size(),
                2 * sd.round_domain(i + 1).size(),
                "round {i}: |L_{i}| should be twice |L_{}|",
                i + 1,
            );
        }
        assert_eq!(sd.round_domain(0).size(), 32);
        assert_eq!(sd.round_domain(3).size(), 4);
    }

    /// The initial domain is a subgroup (offset == 1), not a coset.
    #[test]
    fn initial_domain_is_subgroup_not_coset() {
        let params = StirParams::new(6, 16, 4);
        let sd = StirDomain::new(&params);
        assert_eq!(sd.initial_domain().offset(), Fp::one());
        assert_eq!(sd.initial_domain().element(0), Fp::one());
    }

    /// All non-initial domains are proper cosets (offset != 1).
    #[test]
    fn later_domains_are_cosets() {
        let params = StirParams::new(6, 16, 4);
        let sd = StirDomain::new(&params);
        assert!(sd.num_rounds() >= 1);
        for i in 1..=sd.num_rounds() {
            assert_ne!(
                sd.round_domain(i).offset(),
                Fp::one(),
                "round {i}: L_i should be a proper coset, offset != 1",
            );
        }
    }

    // ------------------------------------------------------------------
    // Task-added tests (a), (b), (c).
    // ------------------------------------------------------------------

    /// (a) `StirParams::new(6, 16, 4)` produces a 3-domain sequence with
    /// sizes `[64, 32, 16]`, matching the worked numeric example in the
    /// module docs.
    #[test]
    fn worked_example_sizes_match_module_doc() {
        let params = StirParams::new(6, 16, 4);
        // Sanity-check the params themselves: M = ceil(log_4(16)) = 2.
        assert_eq!(params.num_rounds, 2);

        let sd = StirDomain::new(&params);
        assert_eq!(sd.round_domains.len(), 3, "M + 1 = 3 domains");
        assert_eq!(sd.round_domain(0).size(), 64);
        assert_eq!(sd.round_domain(1).size(), 32);
        assert_eq!(sd.round_domain(2).size(), 16);
    }

    /// (b) Consecutive cosets are pairwise disjoint — the
    /// domain-disjointness theorem in the module docs, exhibited
    /// empirically by sampling all elements of each L_i and asserting
    /// that no point appears in two domains. (For sizes ≤ 64 we can
    /// afford to enumerate exhaustively, which is stronger than the
    /// "sample 100" the task description suggests.)
    #[test]
    fn consecutive_domains_are_disjoint() {
        let params = StirParams::new(6, 16, 4);
        let sd = StirDomain::new(&params);

        // Materialize each L_i as a HashSet of its elements (canonical
        // representatives — Fp's Eq/Hash already canonicalize).
        let sets: Vec<HashSet<Fp>> = (0..=sd.num_rounds())
            .map(|i| {
                let d = sd.round_domain(i);
                (0..d.size()).map(|j| d.element(j)).collect()
            })
            .collect();

        // Pairwise disjointness for all i, j (i < j). The theorem proves
        // only `L_i ∩ L_{i+1} = ∅`, but the same odd-cofactor argument
        // generalizes to any i ≠ j, so we check the stronger property.
        for i in 0..sets.len() {
            for j in (i + 1)..sets.len() {
                let overlap: Vec<&Fp> = sets[i].intersection(&sets[j]).collect();
                assert!(
                    overlap.is_empty(),
                    "L_{i} and L_{j} overlap on {} point(s); first witness: {:?}",
                    overlap.len(),
                    overlap.first(),
                );
            }
        }
    }

    /// (c) `round_domain(i).generator()` is `ω_0^{2^i}` for `k = 4`
    /// (where `log₂(k/2) = 1`, so each round squares the generator).
    /// This is the "generators square per round" property in the
    /// worked-example section of the module docs.
    #[test]
    fn round_generator_is_omega_zero_squared_per_round() {
        let params = StirParams::new(6, 16, 4);
        let sd = StirDomain::new(&params);

        let n0 = params.log_initial_domain_size;
        let omega_0 = Fp::primitive_root_of_unity(n0);

        // ω_i = ω_0^{2^i} for k = 4. (More generally,
        // ω_i = ω_0^{(k/2)^i}; here (k/2) = 2.)
        let mut expected = omega_0;
        for i in 0..=sd.num_rounds() {
            assert_eq!(
                sd.round_domain(i).generator(),
                expected,
                "round {i}: generator should be ω_0^{{2^{i}}}",
            );
            // Square for the next round: ω_{i+1} = ω_i^2.
            expected = expected * expected;
        }
    }

    /// Sanity: the offset schedule itself matches `c_i = g^{2^{i-1}}`
    /// for `i ≥ 1` and `c_0 = 1`. This is the inline invariant the
    /// disjointness proof depends on, so it's worth pinning down.
    #[test]
    fn offset_schedule_is_successive_squarings_of_g() {
        let params = StirParams::new(6, 16, 4).with_num_rounds(3);
        let sd = StirDomain::new(&params);

        let g = Fp::new(MULTIPLICATIVE_GENERATOR);

        assert_eq!(sd.round_domain(0).offset(), Fp::one(), "c_0 = 1");

        let mut expected = g; // c_1 = g^{2^0} = g
        for i in 1..=sd.num_rounds() {
            assert_eq!(
                sd.round_domain(i).offset(),
                expected,
                "round {i}: offset should be g^{{2^{}}}",
                i - 1,
            );
            expected = expected * expected;
        }
    }
}
