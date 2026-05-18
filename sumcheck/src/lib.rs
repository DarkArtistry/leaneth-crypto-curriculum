//! # Sumcheck Protocol
//!
//! Educational implementation of the sumcheck protocol of
//! Lund-Fortnow-Karloff-Nisan (1992), as presented in Thaler §4.1.
//!
//! ## What sumcheck proves — the general statement
//!
//! Sumcheck is an interactive proof for claims of the form:
//!
//! > Given a multivariate polynomial `g: F^n → F` (public, known to both
//! > parties), a finite **summation set** `S ⊆ F`, and a claimed value `H`,
//! > prove that
//! >
//! > ```text
//! > H = sum over (b_1, b_2, ..., b_n) in S^n of g(b_1, b_2, ..., b_n).
//! > ```
//!
//! The two letters worth pinning down before anything else:
//!
//! - **`n`** = the **number of variables** of `g`. Also the dimension of the
//!   summation cube `S^n`. *Not* the size of `S`, *not* the size of `F`. Just
//!   "how many slots does an input to `g` have". A polynomial like
//!   `g(x_1, x_2, x_3)` has `n = 3`.
//! - **`k = |S|`** = the **size of the summation set** along each axis. The
//!   total number of points in `S^n` is `k^n`.
//!
//! The naive way to verify the claim is to compute the sum directly: evaluate
//! `g` at every one of the **`k^n` points** in `S^n` and add them up. That's
//! exponential in `n`. For `n = 20, k = 2` it's already `~10^6` evaluations;
//! for `n = 40` it's `~10^12`.
//!
//! Sumcheck reduces this to:
//!
//! - **`n` rounds** of communication between prover and verifier.
//! - Per round `i`: the prover sends a `k`-vector
//!   `[s_i(elements[0]), ..., s_i(elements[k-1])]` — the univariate slice
//!   `s_i(X)` evaluated at every element of `S`. Since storing `g` as `k^n`
//!   evaluations on `S^n` forces per-variable degree `< k`, each `s_i` is
//!   determined uniquely by these `k` values via Lagrange interpolation.
//! - The verifier checks `sum over j in 0..k of msg[j] == previous_claim`,
//!   samples a random `r_i ∈ F`, and updates the running claim to `s_i(r_i)`
//!   via `k`-point Lagrange interpolation.
//! - **One final evaluation of `g`** at the random point `(r_1, ..., r_n)`.
//!
//! **Verifier work collapses from `O(k^n)` to `O(n · k)`** field ops plus one
//! evaluation of `g`. For the Boolean cube (`k = 2`), that's `O(n)` plus one
//! evaluation — an exponential speedup on the verifier side. (The prover
//! still does `O(k^n)` total work.)
//!
//! ## What this crate implements: generic-over-`S` sumcheck
//!
//! The protocol code is **generic over any [`domain::SumDomain`]**
//! `D: SumDomain`. Two concrete instances are exposed:
//!
//! - [`domain::BooleanHypercube`] (`S = {0, 1}`, `k = 2`) — the case Lean
//!   Ethereum and every modern multilinear SNARK (GKR, Spartan, HyperPlonk,
//!   Jolt, Lasso, …) use. The demo and most tests instantiate this case.
//! - [`domain::Interval3`] (`S = {0, 1, 2}`, `k = 3`) — exists to exercise
//!   the generic `|S| > 2` Lagrange path end-to-end. Used by two integration
//!   tests.
//!
//! The mechanics:
//!
//! - [`polynomial::MultivariatePoly<D>`] stores `g` as `k^n` evaluations on
//!   the grid `S^n`, indexed in **mixed-radix LSB-first** order (the base-`k`
//!   little-endian generalisation of the Boolean-bit indexing — see
//!   [`polynomial`] for the worked numeric example).
//! - The round message is `Vec<Fp>` of length `k = |S|`.
//! - The per-round verifier check is `sum over j in 0..k of msg[j] ==
//!   prev_claim`, and the running claim is updated to `s_i(r_i)` by
//!   `k`-point Lagrange interpolation.
//!
//! For `|S| = 2` this collapses exactly to the original Boolean-only
//! behaviour: round message `[s_i(0), s_i(1)]`, check `s_i(0) + s_i(1) ==
//! prev_claim`, update `s_i(r) = (1 - r)·s_i(0) + r·s_i(1)`. The Boolean
//! case is a strict special case of the generic code, not a separate path.
//!
//! Concrete numeric anchors live next to the code that needs them:
//! [`domain`] works out `H = 44` for `n = 3, S = {0,1}` and `H = 18` for
//! `n = 2, S = {0,1,2}`. The crate's [`polynomial::MultilinearPoly`] type
//! alias is just `MultivariatePoly<BooleanHypercube>` — the Boolean case
//! by another name.
//!
//! ## Naive vs sumcheck — what the demo shows
//!
//! See `src/bin/demo.rs` — it computes the hypercube sum for the
//! same multilinear polynomial **two ways** and prints the operation counts
//! side by side:
//!
//! 1. **Naive:** call [`polynomial::MultilinearPoly::sum_over_hypercube`],
//!    which sums all `2^n` precomputed evaluations. **`2^n - 1` field
//!    additions, zero `g` evaluations** (the polynomial is in evaluation
//!    form already).
//! 2. **Sumcheck:** run [`protocol::run_sumcheck`]. The **verifier's** total
//!    work is `~4n` field operations across the `n` rounds, plus **one**
//!    evaluation of `g` at a random point (formula-predicted; the actual demo
//!    measures higher per-round cost because each Lagrange-basis call invokes
//!    one Fermat inverse via `pow(p−2)`. See `src/bin/demo.rs` for the
//!    measured numbers.). The prover's work is still `~2^n` (sumcheck doesn't
//!    save anything for the prover), but in a real protocol the prover is
//!    doing the proving and the verifier is the resource-constrained party —
//!    so the verifier's exponential speedup is the practical win.
//!
//! The demo prints both totals concretely. For `n = 10`, predicted:
//! `2^10 = 1024` naive ops vs `~40 + 1` formula-counted sumcheck verifier ops
//! — a `~25×` ideal speedup, ignoring inverse cost. The demo measures the
//! inverse overhead explicitly.
//!
//! ## Soundness
//!
//! Per round, a cheating prover passes the check with probability at most
//! `(k - 1) / |F|` by the **polynomial root bound** (a nonzero univariate
//! polynomial of degree `< k` over a field has at most `k - 1` roots; apply
//! to `hat_s_i - s_i`). Union bound across `n` rounds:
//!
//! ```text
//! Pr[verifier fooled] ≤ n · (k - 1) / |F|.
//! ```
//!
//! For `k = 2` (Boolean, multilinear case) and our `F_p` with `p = 2^61 - 1`,
//! `n = 20`: total error `≤ 20 / 2^61 ≈ 2^-56.5`. For `k = 3` (`Interval3`)
//! the bound doubles to `≤ 40 / 2^61 ≈ 2^-55.5`. Either way:
//! cryptographically sound by a wide margin.
//!
//! ## Architecture
//!
//! - [`field`]: a toy prime field `F_p` over the Mersenne prime `2^61 - 1`.
//! - [`domain`]: the [`domain::SumDomain`] trait + the [`domain::BooleanHypercube`]
//!   and [`domain::Interval3`] instances.
//! - [`polynomial`]: [`polynomial::MultivariatePoly<D>`] — evaluation form on
//!   `S^n`, mixed-radix indexed, with `evaluate`, `sum_over_domain`, and
//!   `fix_first_variable` (the generic Lagrange step).
//! - [`prover`]: the [`prover::SumcheckProver`] type — keeps state across rounds.
//! - [`verifier`]: the [`verifier::SumcheckVerifier`] type — randomized challenger.
//! - [`protocol`]: the [`protocol::run_sumcheck`] orchestrator.
//!
//! ## What's generic now, what's still future work
//!
//! **Now generic.** The trait abstraction over `D: SumDomain` is in place,
//! so the following all just work for any finite `S ⊆ F`:
//!
//! - **Per-variable summation domain.** Any `D: SumDomain` plugs into
//!   [`protocol::run_sumcheck`] unchanged.
//! - **Per-variable degree (implicit).** Storing `g` as `|S|^n` evaluations
//!   on `S^n` is faithful iff `g` has per-variable degree `< |S|` — see the
//!   multivariate Lagrange-interpolation theorem in [`polynomial`]. Sumcheck
//!   then sends `|S|`-length round messages and the verifier interpolates.
//! - **Mixed-radix storage** of `MultivariatePoly`. Base-`k`, little-endian,
//!   with `x_1` as the least-significant digit.
//! - **Lagrange-based round message and verifier check.** Both the prover's
//!   `compute_round_message` and the verifier's "update claim to `s_i(r)`"
//!   step use `|S|`-point Lagrange interpolation.
//!
//! **Still future work.**
//!
//! - **Higher-per-variable-degree with a *separate* degree parameter from
//!   `|S|`.** The production AIR-constraint case sends degree-`d` round
//!   polynomials as `d + 1` *coefficients* (or `d + 1` evaluations at
//!   prover-chosen points) regardless of `|S|`. The current encoding bakes
//!   `d = |S| - 1` into the storage layout; the AIR case would decouple
//!   them, since AIR constraints over `S = {0, 1}` can have degree much
//!   higher than `1`.
//! - **Univariate sumcheck over a multiplicative coset.** Aurora-style
//!   `H ⊆ F^*`, where the polynomial is *univariate* of high degree and the
//!   sum is over a single coset, not a product cube. Different protocol
//!   shape; not just a `D` swap.
//! - **Batched / parallel sumcheck.** Proving `r` distinct sumcheck
//!   instances at once, sharing randomness across them. Reduces verifier
//!   work and message size by `r`× in amortised settings.
//!
//! ## Usage
//!
//! ```ignore
//! use sumcheck::{
//!     domain::BooleanHypercube,
//!     field::Fp,
//!     polynomial::MultivariatePoly,
//!     protocol::run_sumcheck,
//! };
//! use rand::SeedableRng;
//!
//! let evals = vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)];
//! let poly = MultivariatePoly::new(BooleanHypercube, 2, evals);
//! let rng = rand::rngs::StdRng::seed_from_u64(42);
//! let result = run_sumcheck(&poly, rng);
//! assert!(result.is_ok());
//! ```

#![warn(missing_docs)]

pub mod field;
pub mod domain;
pub mod polynomial;
pub mod prover;
pub mod verifier;
pub mod protocol;
