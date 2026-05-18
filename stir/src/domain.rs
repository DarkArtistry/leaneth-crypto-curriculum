//! The STIR per-round evaluation-domain progression `L_0, L_1, ..., L_M`.
//!
//! ## What this module does
//!
//! STIR runs `M` folding rounds. In round `i`, the prover has committed to
//! some function `g_i` on an evaluation domain `L_i ⊂ F_p^*`. The next round
//! folds the function (degree drops by factor `k = folding_factor`) and
//! **moves to a strictly smaller, *shifted* domain `L_{i+1}`**.
//!
//! This struct caches the full sequence `[L_0, L_1, ..., L_M]` so that the
//! prover and verifier can both index into it round-by-round without
//! recomputing root-of-unity exponentiations.
//!
//! The key structural fact — and the one most readers miss when transitioning
//! from FRI to STIR:
//!
//! > **In STIR the evaluation domain shrinks by 2 per round, regardless of
//! > the folding factor `k`. In FRI the domain shrinks by `k` per round.**
//!
//! Decoupling shrink-rate from fold-rate is what gives STIR its *variable
//! per-round rate*, which in turn drives its better query complexity.
//!
//! ## The shift relationship (key idea)
//!
//! Let `L_0 = ⟨ω⟩` be the smooth multiplicative subgroup of order `2^n`
//! (so `|L_0| = 2^n`). After one round, STIR uses
//!
//! ```text
//! L_1 = ω · ⟨ω²⟩
//! ```
//!
//! — a **coset** of the order-`2^{n-1}` subgroup `⟨ω²⟩`, shifted by `ω`.
//! In general,
//!
//! ```text
//! L_i = ω^{a_i} · ⟨ω^{2^i}⟩
//! ```
//!
//! where the **shift exponent** `a_i` is fixed by the public protocol
//! schedule (we use `a_i = 1` for every `i ≥ 1` in this implementation —
//! the simplest valid schedule). The order of the underlying subgroup
//! `⟨ω^{2^i}⟩` is `2^{n-i}` because squaring `i` times halves the order
//! `i` times.
//!
//! Two facts you can verify by hand:
//!
//! - **`L_i` and `L_0^k` are disjoint** when `a_i ≠ 0 mod 2^i`. The image
//!    of `L_{i-1}` under the `k`-folding map is `L_{i-1}^k` — a subset of
//!    `⟨ω^{2^i}⟩`. The shift `ω · ⟨ω^{2^i}⟩` deliberately sits in a
//!    different coset of `⟨ω^{2^i}⟩`. (See [`reed_solomon::domain`] for the
//!    coset-partition picture in `F_17`; same idea here.)
//! - **`|L_i| = 2^{n-i}`**, halving per round.
//!
//! ## Why shift, not just shrink?
//!
//! Suppose we *only* shrunk the subgroup, choosing `L_i = ⟨ω^{2^i}⟩` (no
//! shift). The folding map `g_{i-1}(x) → g_i(x^k)` sends `L_{i-1}` to its
//! `k`-th powers, which is a subgroup of `⟨ω^{2^{i-1}}⟩`. Without a shift,
//! the new evaluation domain `L_i = ⟨ω^{2^i}⟩` would *contain* the image
//! of `L_{i-1}`. A cheating prover could then derive the "next-round
//! committed values" deterministically from the *previous* commitment —
//! the verifier's queries would tell it nothing new.
//!
//! The shift `ω · ⟨ω^{2^i}⟩` breaks this correlation: the verifier's
//! queries at the new domain hit points the prover couldn't have
//! pre-computed from the previous Merkle tree, forcing a fresh
//! commitment. This is the **shift property** STIR's soundness analysis
//! depends on (eprint 2024/390 §3.2).
//!
//! ## Worked example
//!
//! Take `log_initial_domain_size = 4` (so `|L_0| = 16`) and
//! `folding_factor = 4`. Run two folds.
//!
//! ```text
//! Round 0:  L_0 = ⟨ω⟩       of order 16.
//!           |L_0| = 16, offset = 1 (subgroup).
//!
//! Round 1:  L_1 = ω · ⟨ω²⟩  of order 8.
//!           Subgroup squared (16 → 8), offset = ω (= primitive 16th root).
//!
//! Round 2:  L_2 = ω · ⟨ω⁴⟩  of order 4.
//!           Subgroup squared again (8 → 4), offset = ω.
//! ```
//!
//! Note that the **subgroup halves per round** (16 → 8 → 4), independent
//! of `folding_factor = 4`. With FRI's "domain shrinks by `k`" we'd have
//! `|L_0| → |L_0|/k → |L_0|/k²` = 16 → 4 → 1; STIR keeps more elements
//! around to preserve list-decoding capacity at higher rates.
//!
//! **Question for the reader.** Why does STIR shrink the domain by 2 per
//! round while folding the polynomial by `k`, instead of matching them
//! (shrink-by-`k`, the FRI choice)? What would happen if `|L_i|` shrank
//! by `k` like in FRI?
//!
//! Answer: matching shrink-rate and fold-rate keeps the **rate
//! `ρ_i = deg(g_i) / |L_i|`** *constant* across rounds — exactly the FRI
//! design. STIR instead lets `|L_i|` shrink slower than `deg(g_i)`, so
//! `ρ_i = ρ_0 · (k/2)^i` decreases each round. A lower rate gives a
//! *larger* code distance, hence cheaper soundness amplification per
//! query — that's where STIR's query-complexity improvement comes from
//! relative to FRI at the same security. If the domain shrank by `k`, we
//! would recover FRI's constant-rate, more-queries trade-off, and lose
//! the asymptotic advantage.
//!
//! ## In this module
//!
//! - [`StirDomain`] holds the precomputed `[L_0, ..., L_M]`.
//! - [`StirDomain::new`] builds them from a [`crate::params::StirParams`].
//! - The other methods are indexers for prover/verifier code.
//!
//! `// CAUTION:` the shift offset (`ω^{a_i}` in the formula above) is part
//! of the **public protocol parameters**, not Fiat-Shamir randomness. It
//! is fixed before the proof starts and known to both parties. Treating it
//! as random would over-rotate freedoms the adversary already has; treating
//! it as adversary-chosen would break the disjointness analysis.

use reed_solomon::domain::EvaluationDomain;

/// The sequence of STIR evaluation domains `[L_0, L_1, ..., L_M]`.
///
/// `round_domains[i]` is the evaluation domain *used as input* to round `i`.
/// `round_domains[0]` is the domain on which the prover originally committed
/// (the RS codeword domain). `round_domains[num_rounds()]` is the *final*
/// domain — i.e., the one the last folded codeword lives on, which the
/// verifier reads in plaintext.
pub struct StirDomain {
    /// The evaluation domains for every round, in order.
    ///
    /// - `round_domains[0]` is the subgroup `⟨ω⟩` of order
    ///    `2^{log_initial_domain_size}`.
    /// - `round_domains[i]` for `i ≥ 1` is a coset `ω · ⟨ω^{2^i}⟩` of
    ///    order `2^{log_initial_domain_size - i}`.
    /// - `round_domains.len() == num_rounds + 1` (one extra for the final
    ///    domain after the last fold).
    pub round_domains: Vec<EvaluationDomain>,
}

impl StirDomain {
    /// Build the per-round domain progression from STIR parameters.
    ///
    /// Reads `log_initial_domain_size` and `num_rounds` (and possibly
    /// `folding_factor`, depending on the schedule) out of `params`, then
    /// constructs the `[L_0, ..., L_M]` sequence via the recipe in the
    /// module docs.
    ///
    /// Panics if `params.num_rounds + 1 > params.log_initial_domain_size`
    /// (the subgroup would shrink to size `< 2`) or if any required root
    /// of unity is unavailable in the Goldilocks 2-adic tower.
    pub fn new(params: &crate::params::StirParams) -> Self {
        // TODO:
        //   1. Pull `log_n = params.log_initial_domain_size` and
        //      `num_rounds = params.num_rounds` out of `params`.
        //      WHY: keeps `StirDomain` decoupled from how params are sourced.
        //   2. Build `L_0 = EvaluationDomain::new_subgroup(log_n)`.
        //      WHY: the initial codeword lives on a pure subgroup containing
        //      1 — same convention as reed_solomon's encoder.
        //   3. Compute the shift offset `omega = Fp::primitive_root_of_unity(log_n)`.
        //      WHY: ω is the generator of L_0. Using `ω^1 = ω` as the coset
        //      offset for every L_i (i ≥ 1) is the simplest valid schedule
        //      that keeps each L_i in a non-trivial coset of ⟨ω^{2^i}⟩.
        //   4. Loop `i in 1..=num_rounds`:
        //        let sub_log = log_n - i;
        //        let domain_i = EvaluationDomain::new_coset(sub_log, omega);
        //        round_domains.push(domain_i);
        //      WHY: subgroup of order 2^{log_n - i} (squaring `i` times halves
        //      the order `i` times). Offset = ω → disjoint from ⟨ω^{2^i}⟩.
        //   5. Return `Self { round_domains }`.
        //      WHY: `round_domains.len() == num_rounds + 1` by construction.
        let _ = params;
        todo!()
    }

    /// The evaluation domain used in round `round` (0-indexed).
    ///
    /// `round_domain(0)` is the initial domain `L_0`.
    /// `round_domain(self.num_rounds())` is the final domain `L_M`.
    ///
    /// Panics if `round > self.num_rounds()`.
    pub fn round_domain(&self, round: usize) -> &EvaluationDomain {
        // TODO:
        //   1. Assert `round < self.round_domains.len()`.
        //      WHY: panic with a clear message beats an opaque indexing
        //      panic from `Vec`.
        //   2. Return `&self.round_domains[round]`.
        let _ = round;
        todo!()
    }

    /// The number of folding rounds STIR runs.
    ///
    /// Equal to `round_domains.len() - 1` because we store one extra
    /// domain (the final domain `L_M` after the last fold).
    pub fn num_rounds(&self) -> usize {
        // TODO: return `self.round_domains.len() - 1`.
        // WHY: by construction `round_domains` has length `num_rounds + 1`.
        todo!()
    }

    /// The initial evaluation domain `L_0`.
    ///
    /// Equivalent to `self.round_domain(0)`. Provided as a method because
    /// `L_0` is referenced in many places (transcript absorption, query
    /// generation) and a named accessor is clearer than a magic `0`.
    pub fn initial_domain(&self) -> &EvaluationDomain {
        // TODO: return `&self.round_domains[0]`.
        // WHY: just an aliased accessor for `round_domain(0)`.
        todo!()
    }

    /// The final evaluation domain `L_M`.
    ///
    /// Equivalent to `self.round_domain(self.num_rounds())`. The final
    /// folded codeword lives here and is sent in plaintext.
    pub fn final_domain(&self) -> &EvaluationDomain {
        // TODO: return `&self.round_domains[self.round_domains.len() - 1]`.
        // WHY: last domain in the sequence; used by the verifier to check
        // the trailing low-degree codeword.
        todo!()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The domain vector has length `num_rounds + 1`.
    #[test]
    fn domain_count_equals_num_rounds_plus_1() {
        // TODO:
        //   1. Construct StirParams with log_initial_domain_size = 6, num_rounds = 3.
        //   2. let sd = StirDomain::new(&params).
        //   3. Assert sd.round_domains.len() == 4 and sd.num_rounds() == 3.
        // WHY: pins down the off-by-one between rounds and stored domains —
        // the most common bug in this kind of code.
        todo!()
    }

    /// Each successive domain is half the size of the previous one.
    #[test]
    fn successive_domains_halve_in_size() {
        // TODO:
        //   1. log_initial_domain_size = 5, num_rounds = 3.
        //   2. Construct StirDomain.
        //   3. For i in 0..num_rounds, assert
        //         sd.round_domain(i).size() == 2 * sd.round_domain(i + 1).size().
        //   4. Assert sd.round_domain(0).size() == 32 and sd.round_domain(3).size() == 4.
        // WHY: the central "shrink by 2 per round, not by k" property.
        todo!()
    }

    /// The initial domain is a subgroup (offset == 1), not a coset.
    #[test]
    fn initial_domain_is_subgroup_not_coset() {
        // TODO:
        //   1. Construct StirDomain with any reasonable params.
        //   2. Assert sd.initial_domain().offset() == Fp::one().
        //   3. Assert sd.initial_domain().element(0) == Fp::one().
        // WHY: pins down the L_0 = ⟨ω⟩ convention. Production STARKs may
        // start with a coset for trace-disjointness, but STIR's pure
        // low-degree-test view doesn't need that — keep it simple.
        todo!()
    }

    /// All non-initial domains are proper cosets (offset != 1).
    #[test]
    fn later_domains_are_cosets() {
        // TODO:
        //   1. Construct StirDomain with num_rounds >= 1.
        //   2. For i in 1..=num_rounds:
        //        assert_ne!(sd.round_domain(i).offset(), Fp::one());
        // WHY: confirms the shift property — disjointness from previous
        // round's folded image.
        todo!()
    }
}
