//! Configuration parameters for one run of STIR.
//!
//! ## What this module does, in one paragraph
//!
//! A STIR instance is determined by a handful of integers: the size of
//! the initial evaluation domain `|L_0|`, the initial degree bound
//! `d_0`, the folding factor `k`, the number of rounds `M`, the rate
//! `ρ_0 = d_0 / |L_0|`, the security target `λ`, the per-round
//! repetition counts `(t_0, t_1, ..., t_{M-1}, t_final)`, the number of
//! out-of-domain samples `s` per round, and the stopping degree at
//! which the prover sends the final folded polynomial in plain. Every
//! other quantity (per-round degree bound, per-round domain size,
//! per-round soundness error) is **derived** from these. This module
//! defines the [`StirParams`] struct that holds these knobs, the
//! constructors / builders to assemble them, a [`StirParams::validate`] check that
//! the knobs are consistent, and a [`RbrSoundnessReport`] that
//! aggregates the per-round soundness errors into a total `log₂`
//! cheating probability.
//!
//! ## Worked numeric example
//!
//! Pick a small instance that fits in hand calculations:
//!
//! ```text
//!   log_initial_domain_size = 6        →  |L_0| = 64
//!   initial_degree_bound    = 16       →  d_0   = 16
//!   folding_factor          = 4        →  k     = 4
//!   rate_log_inv            = 2        →  ρ_0   = d_0 / |L_0| = 16/64 = 1/4
//!   num_rounds              = 2        →  M     = 2
//!   stopping_degree         = 1        →  d_M = d_0 / k^M = 16 / 4² = 1
//!   ood_samples             = 2        →  s     = 2 per round
//!   repetition_schedule     = [8, 4, 4] (length M + 1)
//! ```
//!
//! From these, the derived per-round shape is:
//!
//! | Round `i` | `log|L_i|`              | `|L_i|` | `d_i`        | `ρ_i`        |
//! |-----------|-------------------------|---------|--------------|--------------|
//! | 0         | 6                       | 64      | 16           | 1/4          |
//! | 1         | 6 - log₂(k/2) = 6 - 1 = 5 | 32      | 16 / 4 = 4   | 4/32 = 1/8   |
//! | 2         | 5 - 1 = 4               | 16      | 4 / 4 = 1    | 1/16         |
//!
//! The rate drops geometrically (`1/4 → 1/8 → 1/16`) — that's STIR's
//! soundness juice. FRI keeps the rate constant.
//!
//! ## Named theorems & derivations
//!
//! ### Round-count formula
//!
//! > **Round-count theorem.** With folding factor `k` and stopping
//! > degree `d_M`, the number of rounds is
//! >
//! > ```text
//! > M = ⌈ log_k(d_0 / d_M) ⌉.
//! > ```
//! >
//! > Proof: each round divides the degree bound by `k`, so after `M`
//! > rounds the residual degree is `d_0 / k^M`. Setting `d_0 / k^M ≤ d_M`
//! > and solving for `M` gives the ceiling. ∎
//!
//! Numerical check on the worked example: `d_0 = 16, d_M = 1, k = 4`
//! gives `M = ⌈ log_4(16) ⌉ = 2`. ✓
//!
//! ### Domain-shrink formula
//!
//! > **Domain-shrink theorem.** With STIR's "domain shrinks by `k/2`
//! > per round" (in contrast to FRI's "by `k`"), the round-`i` domain
//! > satisfies
//! >
//! > ```text
//! > log₂|L_i| = log₂|L_0| − i · log₂(k/2).
//! > ```
//! >
//! > Proof: the construction in §3.2 of the paper maps `L_i` to a
//! > coset of half its `k`-th power image — a smooth multiplicative
//! > coset of size `|L_i| / (k/2)`. Iterating `i` times gives the
//! > formula. ∎
//!
//! For `k = 4` (`log₂(k/2) = 1`) and `log₂|L_0| = 6`: `log|L_1| = 5`,
//! `log|L_2| = 4`. ✓
//!
//! ### Rate-drop formula
//!
//! Combining the two theorems above:
//!
//! ```text
//! ρ_i = d_i / |L_i| = (d_0 / k^i) / (|L_0| / (k/2)^i)
//!     = ρ_0 · (k/2)^i / k^i
//!     = ρ_0 · (1 / 2)^i      ← yes really, the rate halves each round
//! ```
//!
//! Half-per-round rate drop is what powers STIR's improvement over FRI.
//!
//! ## Why the bounds: K_MIN, harmonic schedule, OOD sample count
//!
//! **Question for the reader.** Why must `folding_factor` be a power of
//! 2 and at least 4?
//!
//! Power-of-2 is structural: the FFT machinery in [`reed_solomon`]
//! requires power-of-2 domain sizes, and folding by `k` partitions
//! `|L_i|` into `|L_i| / k` cosets of size `k` — so `k` must divide
//! `|L_i|` cleanly. Once `k = 2^j`, this divisibility is automatic at
//! every round. The lower bound `k ≥ 4` is **not** structural — `k = 2`
//! would work mechanically — but the **soundness** analysis in §4.2 of
//! the paper requires `k ≥ 4` for the per-round error to be small
//! enough that the harmonic-`t_i` schedule sums to a `O(log d)` total.
//! For `k = 2`, the per-round error inflates by `~ρ^{-1/2}` and the
//! total query count would degrade to `O(log² d · poly(λ))` — losing
//! STIR's headline advantage. Pick `k ∈ {4, 8, 16}` in production;
//! `k = 4` is a popular sweet spot (small Merkle openings, good
//! soundness).
//!
//! **Question for the reader.** Why is `repetition_schedule` declining
//! harmonically rather than geometrically?
//!
//! The per-round soundness error decreases roughly as `ρ_i^{Θ(t_i)}`.
//! Setting all `M` round errors equal to `2^{-λ/M}` and solving for
//! `t_i` gives the formula in §4.3 of the paper — `t_i ∼ λ / (M · log(1/ρ_i))`.
//! Since `ρ_i = ρ_0 / 2^i` (rate halves each round), `log(1/ρ_i) = i + log(1/ρ_0)`
//! which grows **linearly** in `i`, so `t_i` decreases like `1 / (i + c)`
//! — the classic harmonic schedule. Sum: `Σ 1/(i + c) ≈ ln M`, giving
//! total queries `Σ t_i · k ≈ λ · k · ln M / log(1/ρ_0)` — the `log²`
//! complexity. A *geometric* schedule like `t_i = t_0 / 2^i` would have
//! `Σ t_i → 2 · t_0`, a constant — but the per-round error in late
//! rounds would blow up because `t_i` shrinks too fast, breaking the
//! union bound.
//!
//! ## Caveats for implementers
//!
//! ```text
//! // CAUTION: `repetition_schedule` declines HARMONICALLY in i, not
//! //          geometrically. The sum is O(log M), which is precisely
//! //          what buys STIR its log-log query advantage over FRI.
//! //          If you accidentally pick a geometric schedule, the
//! //          total queries look smaller in benchmarks BUT the
//! //          soundness analysis breaks in late rounds (where t_i is
//! //          too small to drive the per-round error below 2^{-λ/M}).
//!
//! // CAUTION: stopping_degree must satisfy 1 ≤ d_M ≤ d_0 / k^M.
//! //          If d_M is too large, the prover sends a long final
//! //          polynomial in plain (defeating the point); if too small,
//! //          a later round would have d_i < 1 which makes no sense.
//!
//! // CAUTION: ood_samples ≥ 1 is mandatory. s = 0 disables the OOD
//! //          collapse and reverts STIR's analysis to a FRI-shaped
//! //          one — losing the headline soundness improvement.
//! //          Paper recommends s ∈ {1, 2, 3} for typical λ ∈ [80, 128].
//! ```
//!
//! ## See also
//!
//! - [`crate::transcript`] for the Fiat-Shamir transcript that derives
//!   round randomness from these parameters at runtime.
//! - [`crate::protocol::run_stir`] for the top-level entry point that
//!   consumes a [`StirParams`].

use std::result::Result;

/// Minimum allowed folding factor.
///
/// STIR's soundness analysis (§4.2 of the paper) requires `k ≥ 4` for
/// the per-round error bound to compose into a tight `log²`-query
/// total. Smaller `k` (e.g. `k = 2`) would mechanically work but the
/// per-round error inflates by `~ρ^{-1/2}` and the total query count
/// degrades — losing STIR's advantage over FRI. `K_MIN = 4` is enforced
/// in [`StirParams::validate`].
pub const K_MIN: u32 = 4;

/// All parameters that determine a single run of STIR.
///
/// Pass to [`crate::protocol::run_stir`], or construct via
/// [`StirParams::new`] + [`StirParams::with_security_bits`].
///
/// Field invariants (checked by [`StirParams::validate`]):
///
/// - `folding_factor` is a power of 2 and `≥ K_MIN`.
/// - `repetition_schedule.len() == num_rounds + 1` (one entry per
///   round, plus one for the final round's index-query count).
/// - `num_rounds ≥ 1`.
/// - `initial_degree_bound ≤ 2^log_initial_domain_size` (rate `≤ 1`).
/// - `stopping_degree ≥ 1` and `stopping_degree ≤ initial_degree_bound`.
/// - `rate_log_inv ≥ 1` (rate strictly less than 1).
/// - `security_bits ≥ 1`.
#[derive(Clone, Debug)]
pub struct StirParams {
    /// `log₂|L_0|` — the binary log of the **initial evaluation
    /// domain** size.
    ///
    /// Concretely, `|L_0| = 2^log_initial_domain_size`. The initial
    /// domain is a smooth multiplicative coset of `F_p^*` (see
    /// [`reed_solomon::EvaluationDomain`]). Goldilocks supports
    /// `log_initial_domain_size ≤ 32` (its 2-adicity).
    pub log_initial_domain_size: u32,

    /// `d_0` — the **initial degree bound**.
    ///
    /// The prover claims its committed function `f_0: L_0 → F` is
    /// δ-close in Hamming distance to the evaluation table of some
    /// polynomial `p` of degree `< d_0` (equivalently, to a codeword
    /// in `RS[F, L_0, d_0]`). Satisfies `d_0 ≤ |L_0|`
    /// (rate `ρ_0 = d_0 / |L_0| ≤ 1`); for non-trivial proximity we
    /// want `d_0 ≪ |L_0|`, i.e., rate strictly less than 1.
    pub initial_degree_bound: usize,

    /// `k` — the **folding factor**.
    ///
    /// Per round, the prover folds `k` consecutive evaluations of `f_i`
    /// into one evaluation of `f_{i+1}`. Must be a power of 2 and at
    /// least [`K_MIN`]. Typical: `k ∈ {4, 8, 16}`. Larger `k` means
    /// fewer rounds but more Merkle openings per round.
    pub folding_factor: u32,

    /// `M` — the **number of rounds**.
    ///
    /// After `M` rounds the prover sends the residual function in
    /// plain (no further folding). Derived from
    /// `M = ⌈ log_k(d_0 / stopping_degree) ⌉` — see the round-count
    /// theorem in the module docs.
    pub num_rounds: u32,

    /// `log₂(1/ρ_0)` — the binary log of the **inverse initial rate**.
    ///
    /// Equivalently: `log₂(|L_0| / d_0)`. Typical STARK family values:
    /// `2` (`ρ_0 = 1/4`), `3` (`ρ_0 = 1/8`), `4` (`ρ_0 = 1/16`).
    /// Stored separately from `log_initial_domain_size`/`initial_degree_bound`
    /// for convenience; [`StirParams::validate`] cross-checks
    /// consistency.
    pub rate_log_inv: u32,

    /// `λ` — the target **security bits**.
    ///
    /// The verifier's cheating probability must be `≤ 2^{-λ}`. Used by
    /// [`StirParams::soundness_report`] to pick the repetition counts
    /// that hit this target. Typical: `λ ∈ {80, 100, 128}`.
    pub security_bits: u32,

    /// `(t_0, t_1, ..., t_{M-1}, t_final)` — the **per-round Merkle
    /// repetition counts**.
    ///
    /// Length is exactly `num_rounds + 1`. Entry `t_i` is the number of
    /// Merkle openings the verifier asks for in round `i`. Declines
    /// harmonically — see the module docs for why a geometric schedule
    /// would break soundness.
    pub repetition_schedule: Vec<u32>,

    /// `s` — the number of **out-of-domain (OOD) samples** per round.
    ///
    /// Each round the verifier samples `s` field elements from
    /// `F \ L_{i+1}` and asks the prover for the folded function's
    /// evaluation there. This is STIR's "list-decoding collapse" —
    /// see [`crate::ood`] and Lemma 4.4 of the paper. Must be `≥ 1`
    /// for the soundness analysis to apply (`s = 0` reverts to a
    /// FRI-style argument).
    pub ood_samples: u32,

    /// `d_M` — the **stopping degree**.
    ///
    /// The prover stops folding once the round-`i` degree bound `d_i`
    /// drops to `stopping_degree`, at which point the residual
    /// polynomial is sent in plain. Typical `d_M ∈ {1, 2, 4}` — small
    /// enough that the final transmission cost is negligible.
    pub stopping_degree: usize,
}

/// Round-by-round (RBR) soundness report.
///
/// Per the RBR-soundness theorem (§4 of the paper), the total cheating
/// probability of STIR is bounded by the sum of per-round errors,
/// independent of how the prover splits its strategy across rounds.
/// This struct exposes the breakdown so the implementer can sanity-check
/// the parameter choice — e.g., verify no single round dominates the
/// total error.
///
/// All errors are stored as **`log₂` of the per-round soundness error**
/// (negative numbers; closer-to-zero means *worse* soundness). A
/// well-tuned schedule has per-round errors all clustered around
/// `-λ / num_rounds`.
#[derive(Clone, Debug)]
pub struct RbrSoundnessReport {
    /// `log₂(err_i)` for each round `i ∈ {0, 1, ..., M}`. Length is
    /// `num_rounds + 1` (the extra entry is for the final-round index
    /// query phase).
    pub per_round_errors: Vec<f64>,

    /// `log₂(Σ err_i)` — total cheating probability. Must be `≤ -λ`
    /// for the parameter choice to hit the security target.
    pub total_error_log2: f64,
}

impl StirParams {
    /// Construct a default-ish [`StirParams`] from the three
    /// most-load-bearing knobs.
    ///
    /// Derives `num_rounds`, `rate_log_inv`, `repetition_schedule`,
    /// `ood_samples`, and `stopping_degree` from sensible defaults.
    /// Override any of them by field assignment after construction, or
    /// use the [`StirParams::with_security_bits`] builder to retune
    /// the schedule for a given `λ`.
    ///
    /// # Panics
    ///
    /// Panics if `folding_factor` is not a power of 2 or is below
    /// [`K_MIN`]; if `initial_degree_bound` exceeds
    /// `2^log_initial_domain_size`; or if the derived `num_rounds`
    /// would be zero.
    ///
    /// # Paper reference
    ///
    /// Construction follows the parameter-tuning guidance in §5 of
    /// eprint 2024/390 with `λ = 128` and `ood_samples = 2`.
    pub fn new(
        log_initial_domain_size: u32,
        initial_degree_bound: usize,
        folding_factor: u32,
    ) -> Self {
        // TODO:
        //   1. Panic if `folding_factor < K_MIN` or `folding_factor` is not
        //      a power of 2 — these are hard structural requirements (see
        //      module docs §"Why the bounds").
        //   2. Compute `domain_size = 1 << log_initial_domain_size` and
        //      panic if `initial_degree_bound > domain_size`.
        //   3. Compute `rate_log_inv = log_initial_domain_size - log2(initial_degree_bound)`
        //      (panic if the rate is not a clean power of 2 — the soundness
        //      analysis assumes power-of-2 rates).
        //   4. Compute `num_rounds = ceil(log_k(d_0))` for the default
        //      `stopping_degree = 1`. Use the round-count theorem from the
        //      module docs.
        //   5. Default `ood_samples = 2` and `security_bits = 128` per §5
        //      of the paper.
        //   6. Default `repetition_schedule` to a harmonic schedule fit to
        //      `security_bits = 128`: `t_i = ceil(λ / (num_rounds · log(1/ρ_i)))`
        //      for each `i`, plus a final `t_final` of similar size.
        //   7. Cross-check `validate()` would accept the result; panic with
        //      a clear message if not.
        let _ = (log_initial_domain_size, initial_degree_bound, folding_factor);
        todo!()
    }

    /// Retune the parameter set for a target security level `λ`.
    ///
    /// Replaces `repetition_schedule` with one fit to the new `λ`,
    /// holding all other knobs fixed. Builder-style: returns `self`.
    ///
    /// # Paper reference
    ///
    /// The formula for `t_i` given `λ` is in §4.3 of eprint 2024/390.
    pub fn with_security_bits(self, lambda: u32) -> Self {
        // TODO:
        //   1. Recompute `repetition_schedule`: for each round `i`,
        //      `t_i = ceil(lambda / (num_rounds · log(1/ρ_i)))` where
        //      `ρ_i = ρ_0 · (1/2)^i` (rate-drop formula from module docs).
        //   2. Set the final entry `t_final` similarly using the final
        //      domain's rate.
        //   3. Update `self.security_bits = lambda`.
        //   4. Return `self`.
        let _ = lambda;
        todo!()
    }

    /// Override `num_rounds` directly. Builder-style.
    ///
    /// Useful for the demo and for unit tests that want to drive the
    /// protocol over a non-default round count. Production callers should
    /// usually trust the default derived from `log_initial_domain_size /
    /// log_2(folding_factor)`.
    pub fn with_num_rounds(self, num_rounds: u32) -> Self {
        // TODO: set `self.num_rounds = num_rounds` and return `self`.
        // Caller is responsible for keeping `repetition_schedule.len() ==
        // num_rounds + 1` consistent.
        let _ = num_rounds;
        todo!()
    }

    /// Override `rate_log_inv` (the initial rate as `1 / 2^rate_log_inv`).
    /// Builder-style.
    pub fn with_rate_log_inv(self, rate_log_inv: u32) -> Self {
        // TODO: set `self.rate_log_inv = rate_log_inv` and return `self`.
        let _ = rate_log_inv;
        todo!()
    }

    /// Override `ood_samples` (the OOD sample count `s`). Builder-style.
    ///
    /// Provable soundness uses `s = 1`; the conjectured tighter analysis
    /// from §4.3 of eprint 2024/390 allows `s = 2`.
    pub fn with_ood_samples(self, ood_samples: u32) -> Self {
        // TODO: set `self.ood_samples = ood_samples` and return `self`.
        let _ = ood_samples;
        todo!()
    }

    /// Override `stopping_degree`. Builder-style.
    pub fn with_stopping_degree(self, stopping_degree: usize) -> Self {
        // TODO: set `self.stopping_degree = stopping_degree` and return
        // `self`.
        let _ = stopping_degree;
        todo!()
    }

    /// Override `repetition_schedule` directly. Builder-style.
    ///
    /// The slice should have length `num_rounds + 1` (one entry per round
    /// plus a final-round entry).
    pub fn with_repetition_schedule(self, schedule: Vec<u32>) -> Self {
        // TODO: set `self.repetition_schedule = schedule` and return
        // `self`. Caller is responsible for `schedule.len() == num_rounds + 1`.
        let _ = schedule;
        todo!()
    }

    /// `d_i = d_0 / k^round` — the degree bound at round `round`.
    ///
    /// `round = 0` returns the initial bound `initial_degree_bound`.
    /// `round = num_rounds` returns `stopping_degree`.
    ///
    /// # Panics
    ///
    /// Panics if `round > num_rounds`.
    pub fn round_degree_bound(&self, round: u32) -> usize {
        // TODO:
        //   1. Panic if `round > self.num_rounds`.
        //   2. Compute `k_pow = folding_factor.pow(round)`.
        //   3. Return `self.initial_degree_bound / (k_pow as usize)` (integer
        //      division — exact under our power-of-2 conventions).
        //   4. Cross-reference §"Round-count formula" in the module docs.
        let _ = round;
        todo!()
    }

    /// `log₂|L_round|` — the binary log of the round-`round` domain
    /// size.
    ///
    /// Uses the domain-shrink formula
    /// `log|L_i| = log|L_0| − i · log₂(k/2)` from the module docs.
    ///
    /// # Panics
    ///
    /// Panics if `round > num_rounds` or if the formula would produce
    /// a negative result (parameter mis-tune).
    pub fn round_log_domain_size(&self, round: u32) -> u32 {
        // TODO:
        //   1. Panic if `round > self.num_rounds`.
        //   2. Compute `log_k_over_2 = log2(folding_factor / 2)` —
        //      well-defined since `folding_factor` is a power of 2 ≥ 4,
        //      so `k/2` is a power of 2 ≥ 2.
        //   3. Compute `log_size = self.log_initial_domain_size - round * log_k_over_2`.
        //      Panic if underflow (caller has mis-tuned `num_rounds`).
        //   4. Return `log_size`.
        //   5. Cross-reference §"Domain-shrink formula" in the module docs.
        let _ = round;
        todo!()
    }

    /// Verify the parameter invariants documented on each field.
    ///
    /// Returns `Ok(())` if every field invariant holds; otherwise
    /// `Err(message)` describing the first violation. Callers should
    /// run `validate()` after any field assignment and before invoking
    /// [`crate::protocol::run_stir`].
    pub fn validate(&self) -> Result<(), &'static str> {
        // TODO:
        //   1. Reject `folding_factor < K_MIN` — "STIR requires k ≥ 4
        //      (paper §4.2)".
        //   2. Reject `folding_factor` not a power of 2 — "FFT requires
        //      power-of-2 fold size".
        //   3. Reject `repetition_schedule.len() != (num_rounds + 1) as usize`.
        //   4. Reject `num_rounds < 1`.
        //   5. Reject `initial_degree_bound > 1 << log_initial_domain_size`
        //      — "rate must be ≤ 1".
        //   6. Reject `stopping_degree < 1 || stopping_degree > initial_degree_bound`.
        //   7. Reject `rate_log_inv < 1` — "rate must be strictly < 1".
        //   8. Reject `security_bits < 1` and `ood_samples < 1` — see
        //      module-doc CAUTION block.
        //   9. Return Ok(()).
        todo!()
    }

    /// Produce a round-by-round soundness report for this parameter
    /// set.
    ///
    /// Computes `log₂(err_i)` for each round per the RBR-soundness
    /// theorem in §4 of the paper, sums into a total, and reports.
    /// Use this to verify the parameter choice hits the desired `λ` —
    /// the total should be ≤ `-self.security_bits`.
    pub fn soundness_report(&self) -> RbrSoundnessReport {
        // TODO:
        //   1. For each round `i ∈ 0..=num_rounds`:
        //        a. compute `ρ_i = round_degree_bound(i) / 2^round_log_domain_size(i)`,
        //        b. compute per-round error `err_i = ρ_i^{t_i}` (in log₂
        //           form, `log2(err_i) = t_i · log2(ρ_i)`),
        //        c. push `log2(err_i)` into `per_round_errors`.
        //   2. Sum the per-round errors in linear space (i.e. exp-sum then
        //      log) to get `total_error_log2`. Use the formula
        //      `log2(Σ 2^{e_i}) = max(e_i) + log2(Σ 2^{e_i - max(e_i)})`
        //      to avoid float overflow when one round dominates.
        //   3. Return the `RbrSoundnessReport`.
        //   4. Cross-reference §"Why the bounds" — the harmonic schedule
        //      should produce nearly-uniform per-round errors.
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `validate()` must reject any `folding_factor < K_MIN`, since the
    /// soundness analysis assumes `k ≥ 4`.
    #[test]
    fn params_validate_rejects_k_less_than_4() {
        // TODO:
        //   1. Build a `StirParams` with `folding_factor = 2` (otherwise
        //      sensible: log_initial_domain_size = 6, initial_degree_bound = 16,
        //      etc.).
        //   2. Call `validate()` and assert it returns `Err(_)` with a
        //      message mentioning "k ≥ 4" or "folding_factor".
        //   3. Repeat for `folding_factor = 3` (non-power-of-2).
        todo!()
    }

    /// `round_degree_bound(round)` must equal `d_0 / k^round`.
    #[test]
    fn round_degree_bound_is_d_divided_by_k_to_round() {
        // TODO:
        //   1. Build `StirParams::new(6, 16, 4)` (the module-doc worked
        //      example).
        //   2. Assert `round_degree_bound(0) == 16`.
        //   3. Assert `round_degree_bound(1) == 4`.
        //   4. Assert `round_degree_bound(2) == 1`.
        todo!()
    }

    /// `soundness_report().per_round_errors` must have length
    /// `num_rounds + 1`.
    #[test]
    fn soundness_report_has_M_entries() {
        // TODO:
        //   1. Build a `StirParams` with `num_rounds = 3`.
        //   2. Call `soundness_report()`.
        //   3. Assert `report.per_round_errors.len() == 4`
        //      (`num_rounds + 1` per the field doc).
        //   4. Assert each entry is strictly negative (a probability < 1
        //      has negative log).
        todo!()
    }
}
