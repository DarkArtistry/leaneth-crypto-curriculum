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
//! ## Anchor: STIR is a proximity test for a Reed-Solomon code
//!
//! Every variable in this struct only makes sense if you remember what
//! the protocol is actually doing. Fix a finite field `F`, an evaluation
//! domain `L_0 ⊂ F` of size `N = |L_0|`, and a degree bound `d_0`. The
//! **Reed-Solomon code** at these parameters is
//!
//! ```text
//!   RS[F, L_0, d_0] = { (p(x))_{x ∈ L_0} : p ∈ F[X], deg(p) < d_0 } ⊂ F^N.
//! ```
//!
//! The prover commits to a function `f_0 : L_0 → F` and claims it is
//! δ-close (in Hamming distance) to some codeword in `RS[F, L_0, d_0]`.
//! STIR is a low-query, low-randomness IOP that lets a verifier **test
//! that proximity claim**. Every per-round soundness number, every
//! query-count formula in this module, every parameter tweak, exists to
//! make it hard for a prover to fool that one test.
//!
//! ### The rate `ρ_0 = d_0 / |L_0|` — the load-bearing knob
//!
//! `RS[F, L_0, d_0]` is a linear code with block length `n = |L_0|` and
//! dimension `k = d_0` (a polynomial of degree `< d_0` is determined by
//! `d_0` coefficients, so there are `d_0` degrees of freedom in a
//! codeword). Its **rate** is the textbook coding-theory rate
//!
//! ```text
//!   ρ_0 := dimension / block-length = d_0 / |L_0|.
//! ```
//!
//! Two equivalent intuitions, both worth keeping in the head:
//!
//! 1. **Information density on the wire.** Out of `|L_0|` symbols
//!    transmitted, only `d_0` carry independent information; the
//!    remaining `|L_0| − d_0` are redundancy. `ρ_0` is the fraction
//!    that "matters" — and `1 − ρ_0` is the redundancy that makes the
//!    code error-correcting.
//!
//! 2. **How rare a codeword is among all functions `L_0 → F`.** Count:
//!    `|RS| = |F|^{d_0}`, total `|F|^{|L_0|}`, so codewords occupy a
//!    `|F|^{-(1 − ρ_0) · |L_0|}` fraction of `F^{|L_0|}`. Low `ρ_0` ⇒
//!    codewords are an astronomically tiny sliver of all possible
//!    tables ⇒ a uniformly random function is far from `RS` with
//!    overwhelming probability ⇒ the verifier can catch a cheating
//!    prover from very few spot checks.
//!
//! Concrete numerical extremes (`|L_0| = 64` fixed):
//!
//! - `d_0 = 16` → `ρ_0 = 1/4`. Singleton relative distance ≈ 3/4: any
//!   two distinct codewords disagree on ≥ 3/4 of positions. A single
//!   spot-check catches a non-codeword with prob ≈ 3/4. **Strong.**
//! - `d_0 = 63` → `ρ_0 ≈ 63/64`. Relative distance ≈ 1/64. A cheating
//!   function can agree with a codeword on ~63 of every 64 positions;
//!   you would need dozens of queries to catch it. **Protocol
//!   degenerates.**
//!
//! Slogan to memorise: **low rate ⇒ large minimum distance ⇒ strong
//! soundness per query**. The link is the Singleton bound, met with
//! equality by Reed-Solomon: `δ_min = 1 − ρ_0 + 1/|L_0| ≈ 1 − ρ_0`.
//! FRI/STIR soundness analyses lean on this (and on Johnson-bound
//! list-decoding up to radius `1 − √ρ_0`).
//!
//! ### Why STIR ≡ "Shift To Improve Rate"
//!
//! The acronym is literal. FRI keeps `ρ_i = ρ_0` constant across
//! rounds: it folds the polynomial by `k` (degree `d_i → d_i / k`) and
//! shrinks the domain by `k` (size `|L_i| → |L_i| / k`), so their
//! ratio is unchanged. STIR's twist: fold the degree by `k` but shrink
//! the domain by only `k/2`. Then
//!
//! ```text
//!   ρ_{i+1} = d_{i+1} / |L_{i+1}|
//!           = (d_i / k) / (|L_i| / (k/2))
//!           = ρ_i · (k/2) / k
//!           = ρ_i / 2.
//! ```
//!
//! **The rate halves every round, independent of `k`.** Each round the
//! code gets sparser, each round soundness per query gets stronger, so
//! STIR's late rounds buy more security per query than its early
//! rounds. That improving rate is exactly what makes the *declining*
//! harmonic query schedule `(t_0, t_1, ..., t_M)` work — see the
//! repetition-schedule discussion below.
//!
//! Mental model to keep: **every fold tries to make the code sparser;
//! STIR just does it more aggressively than FRI** by decoupling the
//! domain-shrink rate from the degree-shrink rate. The full algebra of
//! the rate drop is restated in §"Rate-drop formula" below, and the
//! domain-shrink half of it is proved in §"Domain-shrink formula".
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
    ///
    /// **What `ρ_0` *means*** — and why it is the single most
    /// load-bearing knob in this struct — is in the module-level
    /// §"Anchor: STIR is a proximity test for a Reed-Solomon code".
    /// Short version: `ρ_0` is the rate of the Reed-Solomon code
    /// `RS[F, L_0, d_0]` whose proximity STIR is testing; low rate ⇒
    /// large minimum distance ⇒ strong soundness per query. The whole
    /// protocol is engineered to drive `ρ_i` down by a factor of 2 each
    /// round (see §"Why STIR ≡ Shift To Improve Rate"), so `ρ_0` is
    /// also the starting point of that geometric decay.
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
        // (1) Structural checks on `k` — module doc §"Why the bounds".
        assert!(
            folding_factor >= K_MIN,
            "folding_factor must be ≥ K_MIN ({K_MIN}); got {folding_factor}",
        );
        assert!(
            folding_factor.is_power_of_two(),
            "folding_factor must be a power of 2; got {folding_factor}",
        );

        // (2) Rate ≤ 1 — i.e. d_0 ≤ |L_0|. We further insist `d_0` is a
        // power of 2 so that ρ_0 = 1 / 2^{rate_log_inv} is a clean
        // dyadic rate (the soundness analysis assumes this).
        let domain_size: usize = 1usize << log_initial_domain_size;
        assert!(
            initial_degree_bound <= domain_size,
            "initial_degree_bound ({initial_degree_bound}) must be ≤ \
             |L_0| = 2^{log_initial_domain_size} = {domain_size}",
        );
        assert!(
            initial_degree_bound.is_power_of_two(),
            "initial_degree_bound must be a power of 2 (clean dyadic \
             rate); got {initial_degree_bound}",
        );

        // (3) ρ_0 = d_0 / |L_0| ⇒ rate_log_inv = log|L_0| − log d_0.
        //     `trailing_zeros` is log₂ for any power of 2 — no float ops,
        //     no precision issues.
        //     Beware: a naive `initial_degree_bound / domain_size`
        //     truncates to 0 for any non-trivial rate; do NOT compute
        //     the rate that way.
        let log_initial_degree_bound: u32 = initial_degree_bound.trailing_zeros();
        let rate_log_inv: u32 = log_initial_domain_size - log_initial_degree_bound;
        assert!(
            rate_log_inv >= 1,
            "rate must be strictly < 1 (need d_0 < |L_0|)",
        );

        // (4) Round-count theorem (module doc §"Round-count formula"):
        //         M = ⌈ log_k(d_0 / d_M) ⌉.
        //     With default stopping_degree d_M = 1 this is
        //         M = ⌈ log_k(d_0) ⌉ = ⌈ log₂(d_0) / log₂(k) ⌉.
        let log_folding_factor: u32 = folding_factor.trailing_zeros();
        let num_rounds: u32 = log_initial_degree_bound.div_ceil(log_folding_factor);
        assert!(
            num_rounds >= 1,
            "derived num_rounds = 0 (d_0 too small relative to k); \
             increase d_0 or shrink k",
        );

        // (5) Paper §5 defaults.
        let stopping_degree: usize = 1;
        let ood_samples: u32 = 2;
        let security_bits: u32 = 128;

        // (6) Harmonic schedule per §4.3:
        //         t_i = ⌈ λ / (M · log₂(1/ρ_i)) ⌉.
        //     Rate-drop formula (module doc) gives ρ_i = ρ_0 / 2^i, so
        //     log₂(1/ρ_i) = rate_log_inv + i. The (M+1)-th entry covers
        //     the final-round index queries on |L_M|.
        let repetition_schedule: Vec<u32> = (0..=num_rounds)
            .map(|i| {
                let log_inv_rho_i = rate_log_inv + i;
                let denom = num_rounds * log_inv_rho_i;
                security_bits.div_ceil(denom)
            })
            .collect();

        // (7) Self-check before handing the params out.
        let params = Self {
            log_initial_domain_size,
            initial_degree_bound,
            folding_factor,
            num_rounds,
            rate_log_inv,
            security_bits,
            repetition_schedule,
            ood_samples,
            stopping_degree,
        };
        params
            .validate()
            .expect("StirParams::new produced an invalid configuration");
        params
    }

    /// Retune the parameter set for a target security level `λ`.
    ///
    /// Replaces `repetition_schedule` with one fit to the new `λ`,
    /// holding all other knobs fixed. Builder-style: returns `self`.
    ///
    /// # Paper reference
    ///
    /// The formula for `t_i` given `λ` is in §4.3 of eprint 2024/390.
    pub fn with_security_bits(mut self, lambda: u32) -> Self {
        // Harmonic schedule per §4.3:
        //     t_i = ⌈ λ / (M · log₂(1/ρ_i)) ⌉,
        // with log₂(1/ρ_i) = rate_log_inv + i (rate-drop formula).
        // The (M+1)-th entry covers the final-round index queries.
        self.repetition_schedule = (0..=self.num_rounds)
            .map(|i| {
                let log_inv_rho_i = self.rate_log_inv + i;
                let denom = self.num_rounds * log_inv_rho_i;
                lambda.div_ceil(denom)
            })
            .collect();
        self.security_bits = lambda;
        self
    }

    /// Override `num_rounds` directly. Builder-style.
    ///
    /// Useful for the demo and for unit tests that want to drive the
    /// protocol over a non-default round count. Production callers should
    /// usually trust the default derived from `log_initial_domain_size /
    /// log_2(folding_factor)`.
    pub fn with_num_rounds(mut self, num_rounds: u32) -> Self {
        self.num_rounds = num_rounds;
        self
    }

    /// Override `rate_log_inv` (the initial rate as `1 / 2^rate_log_inv`).
    /// Builder-style.
    pub fn with_rate_log_inv(mut self, rate_log_inv: u32) -> Self {
        self.rate_log_inv = rate_log_inv;
        self
    }

    /// Override `ood_samples` (the OOD sample count `s`). Builder-style.
    ///
    /// Provable soundness uses `s = 1`; the conjectured tighter analysis
    /// from §4.3 of eprint 2024/390 allows `s = 2`.
    pub fn with_ood_samples(mut self, ood_samples: u32) -> Self {
        self.ood_samples = ood_samples;
        self
    }

    /// Override `stopping_degree`. Builder-style.
    pub fn with_stopping_degree(mut self, stopping_degree: usize) -> Self {
        self.stopping_degree = stopping_degree;
        self
    }

    /// Override `repetition_schedule` directly. Builder-style.
    ///
    /// The slice should have length `num_rounds + 1` (one entry per round
    /// plus a final-round entry).
    pub fn with_repetition_schedule(mut self, schedule: Vec<u32>) -> Self {
        self.repetition_schedule = schedule;
        self
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
        assert!(
            round <= self.num_rounds,
            "round {round} exceeds num_rounds ({})",
            self.num_rounds,
        );
        // d_i = d_0 / k^round — module doc §"Round-count formula".
        // Exact integer division under our power-of-2 conventions.
        let k_pow = (self.folding_factor as usize).pow(round);
        self.initial_degree_bound / k_pow
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
        assert!(
            round <= self.num_rounds,
            "round {round} exceeds num_rounds ({})",
            self.num_rounds,
        );
        // log|L_i| = log|L_0| − i · log₂(k/2) — module doc §"Domain-shrink
        // formula". Safe: folding_factor ≥ K_MIN = 4 ⇒ k/2 ≥ 2 is a
        // power of 2, so log₂(k/2) = log₂(k) − 1 ≥ 1.
        let log_k_over_2 = self.folding_factor.trailing_zeros() - 1;
        let decrement = round * log_k_over_2;
        assert!(
            self.log_initial_domain_size >= decrement,
            "domain underflow at round {round}: log|L_0| = {} < decrement = {decrement}",
            self.log_initial_domain_size,
        );
        self.log_initial_domain_size - decrement
    }

    /// Verify the parameter invariants documented on each field.
    ///
    /// Returns `Ok(())` if every field invariant holds; otherwise
    /// `Err(message)` describing the first violation. Callers should
    /// run `validate()` after any field assignment and before invoking
    /// [`crate::protocol::run_stir`].
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.folding_factor < K_MIN {
            return Err("folding_factor < K_MIN (STIR requires k ≥ 4, paper §4.2)");
        }
        if !self.folding_factor.is_power_of_two() {
            return Err("folding_factor must be a power of 2 (FFT requires power-of-2 fold size)");
        }
        if self.num_rounds < 1 {
            return Err("num_rounds must be ≥ 1");
        }
        if self.repetition_schedule.len() != (self.num_rounds as usize) + 1 {
            return Err("repetition_schedule.len() must equal num_rounds + 1");
        }
        let domain_size: usize = 1usize << self.log_initial_domain_size;
        if self.initial_degree_bound > domain_size {
            return Err("initial_degree_bound > |L_0| (rate must be ≤ 1)");
        }
        if self.stopping_degree < 1 {
            return Err("stopping_degree must be ≥ 1");
        }
        if self.stopping_degree > self.initial_degree_bound {
            return Err("stopping_degree must be ≤ initial_degree_bound");
        }
        if self.rate_log_inv < 1 {
            return Err("rate_log_inv must be ≥ 1 (rate strictly < 1)");
        }
        if self.security_bits < 1 {
            return Err("security_bits must be ≥ 1");
        }
        if self.ood_samples < 1 {
            return Err("ood_samples must be ≥ 1 (s = 0 disables OOD collapse)");
        }
        Ok(())
    }

    /// Produce a round-by-round soundness report for this parameter
    /// set.
    ///
    /// Computes `log₂(err_i)` for each round per the RBR-soundness
    /// theorem in §4 of the paper, sums into a total, and reports.
    /// Use this to verify the parameter choice hits the desired `λ` —
    /// the total should be ≤ `-self.security_bits`.
    pub fn soundness_report(&self) -> RbrSoundnessReport {
        // Per-round error: err_i = ρ_i^{t_i}, so
        //     log₂(err_i) = t_i · log₂(ρ_i) = − t_i · (rate_log_inv + i).
        // (Rate-drop formula: ρ_i = ρ_0 / 2^i ⇒ log₂(1/ρ_i) = rate_log_inv + i.)
        // A well-tuned harmonic schedule produces nearly-uniform errors;
        // see module doc §"Why the bounds".
        let per_round_errors: Vec<f64> = self
            .repetition_schedule
            .iter()
            .enumerate()
            .map(|(i, &t_i)| {
                let log_inv_rho_i = self.rate_log_inv + i as u32;
                -(t_i as f64) * (log_inv_rho_i as f64)
            })
            .collect();

        // Log-sum-exp in base 2 to avoid float underflow when one round
        // dominates the cheating probability:
        //     log₂(Σ 2^{e_i}) = m + log₂(Σ 2^{e_i − m}), m = max_i e_i.
        let max_e = per_round_errors
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_shifted: f64 = per_round_errors
            .iter()
            .map(|&e| (e - max_e).exp2())
            .sum();
        let total_error_log2 = max_e + sum_shifted.log2();

        RbrSoundnessReport {
            per_round_errors,
            total_error_log2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `validate()` must reject any `folding_factor < K_MIN`, since the
    /// soundness analysis assumes `k ≥ 4`. Build the struct manually
    /// (bypassing `new`, which would panic on these inputs before
    /// `validate` could run).
    #[test]
    fn params_validate_rejects_k_less_than_4() {
        let too_small_k = StirParams {
            log_initial_domain_size: 6,
            initial_degree_bound: 16,
            folding_factor: 2, // ← violation: k < K_MIN
            num_rounds: 2,
            rate_log_inv: 2,
            security_bits: 128,
            repetition_schedule: vec![1, 1, 1],
            ood_samples: 2,
            stopping_degree: 1,
        };
        let err = too_small_k.validate().unwrap_err();
        assert!(
            err.contains("k ≥ 4") || err.contains("K_MIN"),
            "unexpected error: {err}",
        );

        // k = 5 satisfies `k ≥ K_MIN` so the first check passes; the
        // power-of-2 check is what should fire.
        let non_power_of_two = StirParams {
            folding_factor: 5,
            ..too_small_k
        };
        let err = non_power_of_two.validate().unwrap_err();
        assert!(
            err.contains("power of 2"),
            "unexpected error: {err}",
        );
    }

    /// `round_degree_bound(round)` must equal `d_0 / k^round`. Uses the
    /// module-doc worked example.
    #[test]
    fn round_degree_bound_is_d_divided_by_k_to_round() {
        let params = StirParams::new(6, 16, 4);
        assert_eq!(params.round_degree_bound(0), 16);
        assert_eq!(params.round_degree_bound(1), 4);
        assert_eq!(params.round_degree_bound(2), 1);
    }

    /// `soundness_report().per_round_errors` must have length
    /// `num_rounds + 1`, and every entry must be strictly negative
    /// (each `err_i ∈ (0, 1)` so `log₂(err_i) < 0`).
    #[test]
    fn soundness_report_has_m_plus_one_entries() {
        // StirParams::new(8, 64, 4) → ρ_0 = 1/4, num_rounds = 3.
        let params = StirParams::new(8, 64, 4);
        assert_eq!(params.num_rounds, 3);

        let report = params.soundness_report();
        assert_eq!(report.per_round_errors.len(), 4);
        for (i, &e) in report.per_round_errors.iter().enumerate() {
            assert!(e < 0.0, "round {i}: log₂(err_i) should be < 0, got {e}");
        }
        // Total cheating prob should also be a real negative number.
        assert!(
            report.total_error_log2 < 0.0 && report.total_error_log2.is_finite(),
            "total_error_log2 should be finite and negative; got {}",
            report.total_error_log2,
        );
    }
}
