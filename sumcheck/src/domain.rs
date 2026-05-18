//! Per-variable **summation domains** for the sumcheck protocol.
//!
//! ## The role of `S` in the sumcheck statement
//!
//! The sumcheck protocol proves claims of the form
//!
//! ```text
//! H = sum over (x_1, ..., x_n) in S^n of g(x_1, ..., x_n)
//! ```
//!
//! where `g: F^n -> F` is a public multivariate polynomial and `S ⊆ F` is a
//! finite **per-variable summation set**. Every variable ranges over the same
//! `S`, so the total number of summed points is `|S|^n`. This module gives
//! `S` a name: a [`SumDomain`] is just "the set of points each variable runs
//! over" — nothing more, nothing less.
//!
//! ## Two concrete `S`s, with numbers
//!
//! ### `S = {0, 1}` (the [`BooleanHypercube`])
//!
//! Take `n = 3` and `g(x_1, x_2, x_3) = 1 + 2·x_1 + 3·x_2 + 4·x_3`. Then
//! `S^n = {0, 1}^3` has `2^3 = 8` points, and
//!
//! ```text
//! H = g(0,0,0) + g(1,0,0) + g(0,1,0) + g(1,1,0)
//!   + g(0,0,1) + g(1,0,1) + g(0,1,1) + g(1,1,1)
//!   = 1 + 3 + 4 + 6 + 5 + 7 + 8 + 10
//!   = 44.
//! ```
//!
//! Per sumcheck round the prover sends two field elements `[s_i(0), s_i(1)]`
//! (a degree-1 line through two known points), and the verifier checks
//! `s_i(0) + s_i(1) == prev_claim` — **two additions**, the cheapest possible.
//!
//! ### `S = {0, 1, 2}` (the [`Interval3`])
//!
//! Take `n = 2` and `g(x_1, x_2) = x_1 + x_2`. Then `S^n = {0,1,2}^2` has
//! `3^2 = 9` points, and
//!
//! ```text
//! H = g(0,0) + g(1,0) + g(2,0)
//!   + g(0,1) + g(1,1) + g(2,1)
//!   + g(0,2) + g(1,2) + g(2,2)
//!   = 0 + 1 + 2 + 1 + 2 + 3 + 2 + 3 + 4
//!   = 18.
//! ```
//!
//! Each round message now carries `|S| = 3` values, and reducing the round
//! message at a random `r ∉ S` requires Lagrange interpolation through three
//! points instead of a two-point line. That's the path this crate's generic
//! code has to exercise — see [`MultivariatePoly::fix_first_variable`].
//!
//! ## Why every modern SNARK uses `S = {0, 1}`
//!
//! The per-round prover message has length `|S| · (d_i + 1)`-ish, and the
//! verifier check is a `|S|`-term sum followed by Lagrange interpolation
//! through `|S|` points. **Both are minimised at `|S| = 2`**:
//!
//! - Round message: 2 field elements (a line).
//! - Verifier sum check: `s_i(0) + s_i(1) == prev_claim` — 1 add, 1 compare.
//! - Interpolation to a random `r`: `(1 - r)·s_i(0) + r·s_i(1)` — 1 sub, 2 muls,
//!   1 add. (Compare to the 3-point Lagrange formula needed for `|S| = 3`.)
//!
//! That's why GKR, Spartan, HyperPlonk, Jolt, Lasso, and every other modern
//! multilinear SNARK fix `S = {0, 1}`. The Boolean hypercube isn't a deep
//! mathematical choice — it's just the smallest possible `S` you can sum
//! over without losing the polynomial-identity-testing trick.
//!
//! ## Why we still expose `Interval3`
//!
//! If the only `S` we ever use is `{0, 1}`, the "generic over `S`" abstraction
//! is decorative — the code path for general Lagrange interpolation never
//! gets exercised and could rot or be silently incorrect. [`Interval3`]
//! (`S = {0, 1, 2}`) is the smallest `S` with `|S| > 2`, so it exercises the
//! generic Lagrange path with the least extra arithmetic. Tests use it to
//! catch bugs that the Boolean special-case (`L_0(r) = 1 - r`, `L_1(r) = r`)
//! would hide.
//!
//! Variants like [Aurora]-style univariate sumcheck (`S` = a multiplicative
//! coset) or higher-degree STARK constraint sumchecks live downstream of
//! exactly this abstraction.
//!
//! [Aurora]: https://eprint.iacr.org/2018/828
//!
//! ## Implementation note: ZSTs + `OnceLock`
//!
//! [`BooleanHypercube`] and [`Interval3`] carry no data — they're zero-sized
//! marker types whose `elements()` method returns a `&'static [Fp]` cached
//! in a process-wide [`OnceLock`]. Cloning is free; equality between two
//! `BooleanHypercube` values is trivially true. The actual `Vec<Fp>` is
//! allocated once per program run, on first call.

use crate::field::Fp;
use std::sync::OnceLock;

/// A finite per-variable **summation domain** `S ⊆ F`.
///
/// Implementors expose the elements of `S` as a slice of [`Fp`]. All elements
/// must be **pairwise distinct** — Lagrange interpolation through `S` divides
/// by `(elements[i] - elements[j])` for `i ≠ j`, which would be zero if two
/// elements coincided. This invariant is the caller's responsibility; it is
/// not checked at construction.
///
/// A [`SumDomain`] is purely "what set does each variable range over"; it
/// has nothing to say about the *number* of variables or the polynomial
/// itself. See [`MultivariatePoly`](crate::polynomial::MultivariatePoly)
/// for how `n_vars` and `D: SumDomain` combine.
pub trait SumDomain: Clone {
    /// The elements of `S`, in a fixed order. The index of an element in
    /// this slice is the digit value used by the mixed-radix indexing of
    /// [`MultivariatePoly`](crate::polynomial::MultivariatePoly): if
    /// `elements()[j] = c`, then `c` corresponds to digit value `j`.
    fn elements(&self) -> &[Fp];

    /// `|S|` — the size of the summation domain along each axis.
    ///
    /// Default implementation forwards to `elements().len()`.
    fn size(&self) -> usize {
        self.elements().len()
    }
}

/// The Boolean hypercube domain `S = {0, 1}`.
///
/// This is the `S` used by GKR, Spartan, HyperPlonk, Jolt, Lasso, and every
/// other modern multilinear SNARK — `|S| = 2` minimises the per-round
/// message size (two values) and the verifier check (one addition).
///
/// Zero-sized: cloning is free, and there is no per-instance state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BooleanHypercube;

impl SumDomain for BooleanHypercube {
    fn elements(&self) -> &[Fp] {
        // One static slice, initialised on first call. All
        // `BooleanHypercube` values across the process share it.
        static ELS: OnceLock<Vec<Fp>> = OnceLock::new();
        ELS.get_or_init(|| vec![Fp::new(0), Fp::new(1)])
    }
}

/// The three-point domain `S = {0, 1, 2}`.
///
/// Exists to exercise the generic Lagrange-interpolation path in
/// [`MultivariatePoly::fix_first_variable`](
/// crate::polynomial::MultivariatePoly::fix_first_variable) with `|S| > 2`.
/// The Boolean special-case `L_0(r) = 1 - r`, `L_1(r) = r` is too simple to
/// catch bugs in the general-`|S|` formula; `Interval3` triggers a real
/// 3-point Lagrange computation.
///
/// Zero-sized: cloning is free, and there is no per-instance state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Interval3;

impl SumDomain for Interval3 {
    fn elements(&self) -> &[Fp] {
        static ELS: OnceLock<Vec<Fp>> = OnceLock::new();
        ELS.get_or_init(|| vec![Fp::new(0), Fp::new(1), Fp::new(2)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_hypercube_has_two_elements_zero_and_one() {
        let d = BooleanHypercube;
        assert_eq!(d.size(), 2);
        assert_eq!(d.elements(), &[Fp::new(0), Fp::new(1)]);
    }

    #[test]
    fn interval3_has_three_elements_zero_one_two() {
        let d = Interval3;
        assert_eq!(d.size(), 3);
        assert_eq!(d.elements(), &[Fp::new(0), Fp::new(1), Fp::new(2)]);
    }

    #[test]
    fn elements_slice_is_stable_across_calls() {
        // The OnceLock guarantees a single allocation; the returned slice
        // pointer should be identical across calls.
        let d = BooleanHypercube;
        let a = d.elements().as_ptr();
        let b = d.elements().as_ptr();
        assert_eq!(a, b);
    }
}
