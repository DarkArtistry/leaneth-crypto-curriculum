//! Multivariate polynomials in evaluation form on a product domain `S^n`.
//!
//! A [`MultivariatePoly<D>`] is `n` variables stored as `|S|^n` field
//! evaluations on the grid `S^n`, where `S` is a [`SumDomain`] (see
//! [`crate::domain`]). The Boolean hypercube (`S = {0, 1}`) is just one
//! particular `D`; the storage layout, the "fix one variable" operation,
//! and `evaluate` all generalise to any finite `S ⊆ F`.
//!
//! ## What is "evaluation form on a product domain"?
//!
//! Let `g: F^n -> F` be a multivariate polynomial whose per-variable degree
//! is strictly less than `k = |S|`. By the
//! **multivariate Lagrange-interpolation theorem** — a degree-`< k`-per-variable
//! polynomial in `n` variables is uniquely determined by its values at any
//! grid `T_1 × ... × T_n` of pairwise-distinct points with `|T_i| = k` — `g`
//! is uniquely pinned down by the `k^n` numbers `g(s_1, ..., s_n)` for
//! `(s_1, ..., s_n) ∈ S^n`. Storing those `k^n` numbers *is* the polynomial,
//! in **evaluation form on `S^n`**.
//!
//! When `S = {0, 1}` and `k = 2`, the constraint "degree `< k` per variable"
//! is just "**multilinear**" (per-variable degree `≤ 1`), so this generalises
//! the classical Multilinear Extension (MLE) story to general `S`.
//!
//! ## Why the MLE (and its `k > 2` generalisation) is unique
//!
//! Claim: among all polynomials `p(x_1, ..., x_n) ∈ F[x_1, ..., x_n]` with
//! **per-variable degree `< k`**, the one agreeing with a target function
//! `f: S^n -> F` on all `k^n` points of `S^n` is **unique**.
//!
//! Proof (constructive, by dimension count + injectivity).
//!
//! - The space `V` of polynomials with per-variable degree `< k` in each of
//!   `n` variables has the monomial basis
//!   `{ x_1^{i_1} · x_2^{i_2} · ... · x_n^{i_n} : 0 ≤ i_v < k }`, hence
//!   `dim V = k^n`.
//! - The evaluation map `ev: V -> F^{S^n}` sending `p` to
//!   `(p(s))_{s ∈ S^n}` is linear, between two `k^n`-dimensional spaces.
//! - **`ev` is injective.** By induction on `n`. Base `n = 1`: a nonzero
//!   univariate `p` of degree `< k` has at most `k - 1` roots (factor
//!   theorem: if a polynomial `p(X) ∈ F[X]` has `p(α) = 0`, then
//!   `p(X) = (X - α) · q(X)` for some `q ∈ F[X]` of degree `deg p - 1`), so
//!   it can't vanish on all `k` points of `S`. Inductive step `n -> n + 1`: write
//!   `p(x_1, ..., x_{n+1}) = sum_{i = 0}^{k - 1} p_i(x_2, ..., x_{n+1}) · x_1^i`.
//!   If `p` vanishes on `S^{n+1}` then for every `rest ∈ S^n` the univariate
//!   `x_1 -> p(x_1, rest)` vanishes on `S`, so by the base case it's
//!   identically zero, so each `p_i(rest) = 0`. That means every `p_i`
//!   vanishes on `S^n`, so by induction every `p_i = 0`, so `p = 0`. ∎
//! - A linear map between equal-dimensional spaces that is injective is a
//!   **bijection**. So for every target function `f: S^n -> F` there is
//!   exactly one `p ∈ V` with `ev(p) = f`. That `p` is what we store as `evals`.
//!
//! ## Worked numeric example with `|D| = 3`, `n = 2`
//!
//! Take `D = Interval3` (so `S = {0, 1, 2}`, `k = 3`) and the polynomial
//!
//! ```text
//! g(x_1, x_2) = x_1 + x_2.
//! ```
//!
//! Per-variable degree of `g` is `1 < k`, so storing `g` as its `3^2 = 9`
//! evaluations on `S^2` is faithful — no information loss. The evaluations:
//!
//! ```text
//! g(0, 0) = 0    g(1, 0) = 1    g(2, 0) = 2
//! g(0, 1) = 1    g(1, 1) = 2    g(2, 1) = 3
//! g(0, 2) = 2    g(1, 2) = 3    g(2, 2) = 4
//! ```
//!
//! Laid out in **mixed-radix LSB-first** order with `x_1` least significant:
//!
//! ```text
//! evals = [0, 1, 2,   1, 2, 3,   2, 3, 4]
//!          ^^^^^^^^   ^^^^^^^^   ^^^^^^^^
//!          x_2 = 0    x_2 = 1    x_2 = 2
//! ```
//!
//! - [`sum_over_domain`](MultivariatePoly::sum_over_domain) returns
//!   `0+1+2+1+2+3+2+3+4 = 18`, simply by adding the nine stored values.
//! - [`evaluate(&[7, 0])`](MultivariatePoly::evaluate) returns `Fp(7)`, since
//!   `g(7, 0) = 7 + 0`. The implementation gets there by interpolation, not
//!   by re-evaluating the symbolic expression.
//!
//! ## Mixed-radix indexing primer
//!
//! "Mixed-radix" sounds fancy; for our case (every digit shares the same
//! radix `k = |S|`) it's just **base-`k`, little-endian**. An index
//! `i ∈ [0, k^n)` decomposes into `n` digits
//!
//! ```text
//! i = i_1 + i_2 · k + i_3 · k^2 + ... + i_n · k^{n-1},     i_j ∈ [0, k)
//! ```
//!
//! and `evals[i]` stores `g(elements[i_1], elements[i_2], ..., elements[i_n])`.
//! Equivalently, digit `j` is `i_j = (i / k^{j-1}) mod k`. Variable `x_1` is
//! the **least significant digit** — adjacent indices `(0, 1, ..., k-1)`
//! sweep `x_1` through `S` while holding `x_2, ..., x_n` fixed. That's what
//! makes [`fix_first_variable`](MultivariatePoly::fix_first_variable) cache-
//! friendly: the `k` evaluations needed to interpolate along the `x_1`-axis
//! are stored contiguously.
//!
//! Two concrete specialisations:
//!
//! - **`k = 2` (Boolean cube).** Digits are bits, mixed-radix collapses to
//!   ordinary LSB-first binary. `evals[5]` for `n = 3` is `g(1, 0, 1)`:
//!   `5 = 101_2 = 1 + 0·2 + 1·4`.
//! - **`k = 3` (Interval3).** Digits are base-3. `evals[5]` for `n = 2` is
//!   `g(elements[2], elements[1]) = g(2, 1)`: `5 = 1·3 + 2 = "12"_3`, so
//!   `i_1 = 2, i_2 = 1`. In the table above, `evals[5] = g(2, 1) = 3` ✓.
//!
//! ## Sum over the domain — `H = sum over x in S^n of g(x)`
//!
//! Trivially `evals.iter().sum()`, because `evals` *already* lists `g` at
//! every point of `S^n`. The whole point of evaluation form is that this
//! sum (the quantity sumcheck proves) is free to compute up-front: zero
//! polynomial evaluations, just `|S|^n - 1` field additions.
//!
//! ## Fixing `x_1 = r` — the generalised "step C"
//!
//! "Fix `x_1 = r`" means: take the `n`-variable polynomial `g` and replace
//! `x_1` with a specific field element `r`, getting an `(n-1)`-variable
//! polynomial `g'(x_2, ..., x_n) = g(r, x_2, ..., x_n)`. Sumcheck drives
//! itself forward one round at a time using exactly this move. **`r` is an
//! arbitrary field element** — sumcheck's verifier picks it uniformly at
//! random from all of `F`, not just from `S`.
//!
//! ### The pedagogical bridge: lines, generalised
//!
//! For `D = {0, 1}`, `g` is multilinear, so along the `x_1`-axis it's a
//! **straight line**, and two known endpoints `g(0, rest), g(1, rest)`
//! determine it. Slide along the line by parameter `r` to get
//! `g(r, rest) = (1 - r)·g(0, rest) + r·g(1, rest)`.
//!
//! For general `D` with `|D| = k`, the same picture, one notch more
//! powerful: `g` has per-variable degree `< k`, so along the `x_1`-axis it
//! is a **degree-`< k` curve**, and `k` known points
//! `g(elements[0], rest), ..., g(elements[k-1], rest)` determine it
//! uniquely. The named tool is the **Lagrange interpolation theorem**:
//!
//! > A polynomial of degree `< k` is uniquely determined by its values
//! > at any `k` distinct points, and that polynomial is given by the
//! > Lagrange interpolation formula.
//!
//! ### The Lagrange basis polynomial
//!
//! Define the **Lagrange basis polynomial** `L_j(x)` for `j ∈ {0, ..., k-1}`:
//!
//! ```text
//! L_j(x) = prod_{i ≠ j} (x - elements[i]) / (elements[j] - elements[i]).
//! ```
//!
//! By construction, `L_j` has degree `k - 1` and satisfies
//! `L_j(elements[j]) = 1` and `L_j(elements[i]) = 0` for `i ≠ j`. The
//! Lagrange interpolant of any function on `S` is then the sum
//!
//! ```text
//! g(r, rest) = sum over j in 0..k of L_j(r) · g(elements[j], rest).
//! ```
//!
//! For arbitrary `r ∈ F` this evaluates `g` at the **new** input `(r, rest)`
//! using only the `k` stored values along the corresponding `x_1`-slice.
//!
//! ### Boolean collapse — why this **is** the old `(1 - r)·A + r·B`
//!
//! Specialise to `D = {0, 1}` (`k = 2`, `elements = [0, 1]`):
//!
//! ```text
//! L_0(r) = (r - elements[1]) / (elements[0] - elements[1])
//!        = (r - 1) / (0 - 1)
//!        = (r - 1) / (-1)
//!        = 1 - r.
//!
//! L_1(r) = (r - elements[0]) / (elements[1] - elements[0])
//!        = (r - 0) / (1 - 0)
//!        = r.
//! ```
//!
//! Plug those back in:
//!
//! ```text
//! g(r, rest) = L_0(r)·g(0, rest) + L_1(r)·g(1, rest)
//!            = (1 - r)·g(0, rest) + r·g(1, rest).
//! ```
//!
//! That is **exactly** the Boolean line-interpolation formula
//! `(1 - r)·A + r·B`. The generic Lagrange path collapses to it as a
//! strict special case — no extra arithmetic, just `k = 2` plugged in.
//! This is the central pedagogical point of the generalisation: the
//! Boolean line is `|D| = 2` of a more general story.
//!
//! ### Storage-layout pairing (now: `k`-tuples, not pairs)
//!
//! Under mixed-radix LSB-first storage, the `k` evaluations sharing a fixed
//! `(x_2, ..., x_n)` "rest" but differing only in `x_1` are at the
//! **contiguous** indices
//!
//! ```text
//! { j + rest_idx · k : j ∈ 0..k }     for each rest_idx ∈ [0, k^{n-1}).
//! ```
//!
//! (For `k = 2` this reduces to the adjacent pair `(2·rest_idx, 2·rest_idx+1)`.)
//! [`fix_first_variable`](MultivariatePoly::fix_first_variable)
//! precomputes the `k` weights `L_0(r), ..., L_{k-1}(r)` once, then for each
//! `rest_idx` builds the new evaluation as a single `k`-term dot product
//! against that contiguous slice of stored values.
//!
//! **Question for the reader.** Why do we fold via `fix_first_variable` n times
//! instead of summing the full `k^n`-term symbolic Lagrange expression? Count
//! the operations.
//! Try to answer before reading on.
//!
//! The DP fold does `k^n + k^{n-1} + ... + k = k(k^n - 1)/(k - 1) = O(k^{n+1}/(k-1))`
//! field ops — geometric in `k`, dominated by the first fold. Direct symbolic
//! Lagrange is `O(n · k^n)`: every one of the `k^n` grid points contributes a
//! product of `n` Lagrange weights to the sum. For `k = 2`, the DP is `O(2^n)`
//! (the geometric series compresses to `2^{n+1} - 2`), while the direct
//! symbolic Lagrange sum is `O(n · 2^n)` — `n` Lagrange weights per grid point,
//! `k^n` grid points. Different big-O; the DP wins by a factor of `n`, not just
//! a constant. For larger `k` the DP wins cleanly, and as a bonus storage stays
//! in evaluation form throughout, so we never materialise `g` in coefficient form.
//!
//! ## Why DP for `evaluate` instead of summing the full Lagrange formula
//!
//! The direct multivariate-Lagrange evaluation at an arbitrary point
//! `(r_1, ..., r_n) ∈ F^n` is
//!
//! ```text
//! g(r_1, ..., r_n) = sum over (j_1, ..., j_n) in [0,k)^n of
//!                       g(elements[j_1], ..., elements[j_n])
//!                       · prod_{l=1..n} L_{j_l}(r_l).
//! ```
//!
//! That's `k^n` terms, each with `n` Lagrange-basis multiplications —
//! `O(n · k^n)` total. We can do better: **fix one variable at a time**.
//! Each call to [`fix_first_variable`](MultivariatePoly::fix_first_variable)
//! reduces the eval count from `k^m` to `k^{m-1}` and costs `k^m` field
//! ops. Summing the geometric series:
//!
//! ```text
//! k^n + k^{n-1} + ... + k = k · (k^n - 1) / (k - 1)  =  O(k^{n+1}).
//! ```
//!
//! For `k = 2` that's `O(2^{n+1}) = O(2^n)` — linear in the input size `2^n`.
//! For larger `k` it's still polynomial in `k^n`, vs the `O(n · k^n)` of the
//! direct symbolic sum. And crucially, the implementation is **shared** with
//! `fix_first_variable`: `evaluate` is just `n` applications of the same DP
//! step that sumcheck itself uses, then read the single remaining value.
//!
//! ## Why evaluation form?
//!
//! Sumcheck's hot paths all consume `g`-values on `S^n`, not symbolic
//! coefficients. The verifier needs `H = sum over x in S^n of g(x)` (free
//! given the eval table — just add the stored values) and round messages
//! built from `g`-values along slices of `S^n` (each just a contiguous
//! `k`-tuple of stored values, no re-evaluation of `g`). The only point
//! where `g` is "really" evaluated at non-`S` inputs is the verifier's
//! final check at `(r_1, ..., r_n) ∈ F^n` — and even that goes through
//! the same DP, not through symbolic substitution into a coefficient form
//! we never materialise.

use crate::domain::{BooleanHypercube, SumDomain};
use crate::field::Fp;

/// A multivariate polynomial in `n_vars` variables, stored as its evaluations
/// on the product domain `D^n_vars`.
///
/// Evaluations are laid out in **mixed-radix LSB-first** order: index
/// `i = i_1 + i_2 · k + ... + i_n · k^{n-1}` (with `k = |D|`) corresponds to
/// the input `(elements[i_1], ..., elements[i_n])`, so the first variable
/// `x_1` is least significant and the `k` evaluations sharing a fixed
/// `(x_2, ..., x_n)` "rest" sit at contiguous indices.
/// [`Self::fix_first_variable`] performs the Lagrange step
/// `g(r, rest) = sum_j L_j(r) · g(elements[j], rest)` along that contiguous
/// `k`-slice; for `D = BooleanHypercube` (`k = 2`) this collapses to the
/// familiar `(1 - r)·g(0, rest) + r·g(1, rest)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultivariatePoly<D: SumDomain> {
    /// The per-variable summation domain `S ⊆ F`.
    pub domain: D,
    /// Number of variables `n`.
    pub n_vars: usize,
    /// Evaluations on `D^n_vars`. Length must equal `|D|^n_vars`.
    /// Mixed-radix LSB-first indexed (see module docs).
    pub evals: Vec<Fp>,
}

/// Backward-compatible alias for the Boolean-hypercube case.
///
/// The historical [`MultilinearPoly`] (Boolean MLE) is exactly
/// `MultivariatePoly<BooleanHypercube>`. Existing call sites and narrative
/// in tests/demo can keep using the familiar name.
pub type MultilinearPoly = MultivariatePoly<BooleanHypercube>;

impl<D: SumDomain> MultivariatePoly<D> {
    /// Construct a [`MultivariatePoly`] from its evaluations on `D^n_vars`.
    ///
    /// `evals` must be laid out in **mixed-radix LSB-first** order: index
    /// `i = i_1 + i_2·k + ... + i_n·k^{n-1}` with `k = |D|` corresponds to
    /// `g(elements[i_1], elements[i_2], ..., elements[i_n])`. See the module
    /// docs for the full layout.
    ///
    /// Panics if `evals.len() != |D|^n_vars`.
    pub fn new(domain: D, n_vars: usize, evals: Vec<Fp>) -> Self {
        let k = domain.size();
        let expected_len = pow_usize(k, n_vars);
        assert_eq!(
            evals.len(),
            expected_len,
            "evaluations length must be |D|^n_vars = {}^{} = {}",
            k,
            n_vars,
            expected_len,
        );
        Self { domain, n_vars, evals }
    }

    /// Compute `H = sum over x in D^n_vars of g(x)`.
    ///
    /// `O(|D|^n_vars)` — just sum the stored evaluations. This is the
    /// quantity sumcheck proves; in evaluation form it's already in hand.
    pub fn sum_over_domain(&self) -> Fp {
        self.evals.iter().copied().sum()
    }

    /// Replace `x_1` with the field element `r`, returning the resulting
    /// polynomial in `n_vars - 1` variables on the same domain.
    ///
    /// # What it does
    ///
    /// Specialises `g` along the `x_1` axis:
    ///
    /// ```text
    /// g'(x_2, ..., x_n) = g(r, x_2, ..., x_n).
    /// ```
    ///
    /// `r` may be *any* field element, not just an element of `D` — sumcheck's
    /// verifier picks it uniformly at random from all of `F`.
    ///
    /// # Why and how (Lagrange along one axis)
    ///
    /// For fixed `(x_2, ..., x_n) = rest`, the function `x_1 -> g(x_1, rest)`
    /// is a polynomial of degree `< k` in one variable (per-variable degree
    /// bound). By the Lagrange interpolation theorem, the `k` values
    /// `g(elements[j], rest)` for `j ∈ 0..k` determine that univariate
    /// polynomial uniquely, and we can read off its value at `r` as
    ///
    /// ```text
    /// g(r, rest) = sum over j in 0..k of L_j(r) · g(elements[j], rest)
    /// ```
    ///
    /// where `L_j(x) = prod_{i ≠ j} (x - elements[i]) / (elements[j] - elements[i])`
    /// is the Lagrange basis polynomial that is `1` at `elements[j]` and `0`
    /// at every other element of `D`. For the Boolean case `D = {0, 1}` this
    /// collapses to `(1 - r)·g(0, rest) + r·g(1, rest)` — the familiar line
    /// interpolation.
    ///
    /// # Implementation
    ///
    /// We precompute the `k` Lagrange weights `L_j(r)` **once** before the
    /// outer loop, since they don't depend on `rest`. Under the mixed-radix
    /// LSB-first layout, the `k` stored values that share the same `rest`
    /// are at contiguous indices `rest_idx·k + j` for `j ∈ 0..k` — one
    /// `k`-element slice per output index. The output of length `k^{n-1}`
    /// is built by `k^{n-1}` independent `k`-term dot products.
    ///
    /// Cost: `O(k^n)` field operations, plus `O(k^2)` for the basis weights.
    ///
    /// Panics if `self.n_vars == 0`.
    pub fn fix_first_variable(&self, r: Fp) -> Self {
        // TODO: produce a polynomial in `n_vars - 1` variables whose
        //   evals on `S^{n-1}` are `g(r, rest)` for each `rest ∈ S^{n-1}`.
        //   1. Precompute the `|D|` Lagrange weights `L_j(r)` once (they don't
        //      depend on `rest`, so factor them out of the loop).
        //   2. For each `rest_idx` in `0..block_size`, take the contiguous
        //      `k`-slice `evals[rest_idx*k .. rest_idx*k + k]` (the `x_1`-axis
        //      values at this `rest`) and dot it against `weights`.
        //   3. Push the dot-product result into `new_evals`.
        //   See the "Fixing x_1 = r" and "Mixed-radix indexing primer" sections
        //   above for the derivation and the contiguous-slice layout.
        //
        //   Reference implementation below.

        assert!(
            self.n_vars > 0,
            "cannot fix a variable in a zero-variable polynomial",
        );

        let elements = self.domain.elements();
        let k = elements.len();

        // Precompute Lagrange weights L_j(r) for j ∈ 0..k. They depend only
        // on `r` and the domain, not on the `rest` slice.
        let weights: Vec<Fp> = (0..k)
            .map(|j| lagrange_basis_at(elements, j, r))
            .collect();

        // Output has |D|^{n - 1} evaluations.
        let new_n_vars = self.n_vars - 1;
        let new_len = pow_usize(k, new_n_vars);
        let mut new_evals = Vec::with_capacity(new_len);

        for rest_idx in 0..new_len {
            // The `k` evaluations sharing this `rest_idx` lie at indices
            // `rest_idx · k + j` for j ∈ 0..k — a contiguous slice, since
            // x_1 is the least significant mixed-radix digit.
            let base = rest_idx * k;
            let slice = &self.evals[base..base + k];

            // f'(rest) = sum_j L_j(r) · f(elements[j], rest).
            let mut acc = Fp::zero();
            for j in 0..k {
                acc += weights[j] * slice[j];
            }
            new_evals.push(acc);
        }

        Self {
            domain: self.domain.clone(),
            n_vars: new_n_vars,
            evals: new_evals,
        }
    }

    /// Evaluate `g` at an arbitrary point `(r_1, ..., r_n) ∈ F^n` via repeated
    /// [`fix_first_variable`](Self::fix_first_variable).
    ///
    /// Each call shrinks the polynomial by one variable; after `n_vars`
    /// calls we have a 0-variable polynomial with a single stored value,
    /// which is `g(r_1, ..., r_n)`.
    ///
    /// Cost: `O(|D|^{n_vars + 1})` — geometric sum of `|D|^m` for
    /// `m = 1..n_vars`. For `|D| = 2` this is `O(2^n)`, sharing its
    /// implementation with the sumcheck inner step (`fix_first_variable`).
    ///
    /// Panics if `point.len() != self.n_vars`.
    pub fn evaluate(&self, point: &[Fp]) -> Fp {
        // TODO: return `g(point[0], ..., point[n_vars-1])` via the DP
        //   fold rather than expanding the full symbolic Lagrange sum.
        //   1. Clone `self` into a mutable `current` (we'll mutate it round-by-round).
        //   2. Loop over `r_i` in `point`, calling `fix_first_variable(r_i)`;
        //      each call drops one variable and rebuilds evals using Lagrange
        //      weights along the first axis (same step sumcheck uses).
        //   3. After `n_vars` folds, `current` has 0 variables and a single
        //      stored value — that is `g(r_1, ..., r_n)`. Return it.
        //   Cost is `O(k^{n+1})` field ops vs `O(n · k^n)` for the direct
        //   k^n-term symbolic Lagrange expansion; see the "Why DP for `evaluate`
        //   instead of summing the full Lagrange formula" section above.
        //
        //   Reference implementation below.

        assert_eq!(point.len(), self.n_vars);
        let mut current = self.clone();
        for &p in point {
            current = current.fix_first_variable(p);
        }
        // After n_vars folds, current is the unique 0-variable poly: one entry.
        current.evals[0]
    }
}

// Boolean-hypercube convenience: the test and demo narrative talks about
// "sum over the Boolean hypercube". Expose that as a thin alias of the
// generic `sum_over_domain`.
impl MultivariatePoly<BooleanHypercube> {
    /// Alias of [`MultivariatePoly::sum_over_domain`] specialised to the
    /// Boolean case. Provided so the historic terminology
    /// ("sum over the Boolean hypercube") still has a clear name in the
    /// `D = BooleanHypercube` setting.
    pub fn sum_over_hypercube(&self) -> Fp {
        self.sum_over_domain()
    }
}

/// Evaluate the Lagrange basis polynomial `L_i` at `target`, where the
/// basis is built over `points`:
///
/// ```text
/// L_i(target) = prod_{j ≠ i} (target - points[j]) / (points[i] - points[j]).
/// ```
///
/// `L_i` is the unique degree-`< points.len()` polynomial that is `1` at
/// `points[i]` and `0` at every other `points[j]`.
///
/// The caller must ensure that the elements of `points` are **pairwise
/// distinct**; otherwise the denominator `(points[i] - points[j])` for some
/// `j ≠ i` would be zero and have no field inverse. In this crate `points`
/// is always [`SumDomain::elements`], which is required to be distinct by
/// the [`SumDomain`] contract.
fn lagrange_basis_at(points: &[Fp], i: usize, target: Fp) -> Fp {
    // TODO: evaluate `L_i(target)` for the basis built over `points`.
    //   1. numerator   = prod over j ≠ i of (target - points[j])   — the part
    //      that vanishes whenever `target` is some other `points[j]`.
    //   2. denominator = prod over j ≠ i of (points[i] - points[j]) — the
    //      normalising constant that makes `L_i(points[i]) = 1`.
    //   3. Return `numerator * denominator.inverse()`. Inverting is safe
    //      because the `points` are pairwise distinct (SumDomain contract),
    //      so every factor `(points[i] - points[j])` for `j ≠ i` is nonzero
    //      and thus has a field inverse.
    //
    //   Reference implementation below.

    let mut numerator = Fp::one();
    let mut denominator = Fp::one();
    let xi = points[i];
    for (j, &xj) in points.iter().enumerate() {
        if j == i {
            continue;
        }
        numerator *= target - xj;
        denominator *= xi - xj;
    }
    // SAFE: denominator is a product of nonzero terms because `points` is
    // pairwise distinct (SumDomain contract / caller's responsibility).
    numerator
        * denominator
            .inverse()
            .expect("Lagrange denominator is zero — domain elements must be distinct")
}

/// `base^exp` for `usize`. Used only for the length check `|D|^n_vars`.
/// Returns 1 when `exp == 0` (matching `pow(0) == 1` for our domain sizes).
fn pow_usize(base: usize, exp: usize) -> usize {
    let mut acc: usize = 1;
    for _ in 0..exp {
        acc *= base;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Interval3;

    // -------------------------------------------------------------------
    // Boolean-hypercube tests carried over from the pre-refactor API,
    // now constructed via the generic constructor.
    // -------------------------------------------------------------------

    #[test]
    fn evaluate_at_boolean_input_returns_eval() {
        // For 2 variables, the mixed-radix LSB-first layout (k = 2) reads:
        //
        //   evals[0] = f(0, 0)   because 0 = 00
        //   evals[1] = f(1, 0)   because 1 = 01
        //   evals[2] = f(0, 1)   because 2 = 10
        //   evals[3] = f(1, 1)   because 3 = 11
        let poly = MultivariatePoly::new(
            BooleanHypercube,
            2,
            vec![Fp::new(10), Fp::new(20), Fp::new(30), Fp::new(40)],
        );

        assert_eq!(poly.evaluate(&[Fp::zero(), Fp::zero()]), Fp::new(10));
        assert_eq!(poly.evaluate(&[Fp::one(), Fp::zero()]), Fp::new(20));
        assert_eq!(poly.evaluate(&[Fp::zero(), Fp::one()]), Fp::new(30));
        assert_eq!(poly.evaluate(&[Fp::one(), Fp::one()]), Fp::new(40));
    }

    #[test]
    fn fix_then_evaluate_matches_full_evaluate() {
        // 3-variable multilinear with 8 evaluations.
        let poly = MultivariatePoly::new(
            BooleanHypercube,
            3,
            vec![
                Fp::new(1),
                Fp::new(2),
                Fp::new(3),
                Fp::new(4),
                Fp::new(5),
                Fp::new(6),
                Fp::new(7),
                Fp::new(8),
            ],
        );

        let r = Fp::new(5);
        let r2 = Fp::new(7);
        let r3 = Fp::new(11);

        // First fix x_1 = r → 2-variable poly, then evaluate at (r2, r3).
        let fixed = poly.fix_first_variable(r);
        let fixed_eval = fixed.evaluate(&[r2, r3]);

        // Must match a full evaluation at (r, r2, r3).
        let full_eval = poly.evaluate(&[r, r2, r3]);

        assert_eq!(fixed_eval, full_eval);
    }

    #[test]
    fn sum_over_hypercube_matches_naive() {
        let evals = vec![Fp::new(3), Fp::new(5), Fp::new(7), Fp::new(9)];
        let poly = MultivariatePoly::new(BooleanHypercube, 2, evals.clone());

        let expected = evals[0] + evals[1] + evals[2] + evals[3];

        assert_eq!(poly.sum_over_hypercube(), expected);
    }

    // -------------------------------------------------------------------
    // New tests on Interval3 (|S| = 3) — these exercise the generic
    // Lagrange-interpolation path that the Boolean special-case wouldn't
    // catch a bug in.
    // -------------------------------------------------------------------

    #[test]
    fn fix_first_variable_on_interval3_lagrange_collapse() {
        // 1-var polynomial g(x) = x on S = {0, 1, 2}.
        // Stored evals = [g(0), g(1), g(2)] = [0, 1, 2].
        let poly = MultivariatePoly::new(
            Interval3,
            1,
            vec![Fp::new(0), Fp::new(1), Fp::new(2)],
        );

        // fix x = 5  →  g'(rest) = g(5) = 5. New poly has 0 variables and
        // a single stored evaluation equal to 5.
        let fixed = poly.fix_first_variable(Fp::new(5));
        assert_eq!(fixed.n_vars, 0);
        assert_eq!(fixed.evals, vec![Fp::new(5)]);
    }

    #[test]
    fn evaluate_matches_sum_for_interval3() {
        // 2-var polynomial g(x, y) = x + y on S = {0, 1, 2}.
        // Mixed-radix LSB-first order with x_1 = x, x_2 = y:
        //
        //   evals[0] = g(0, 0) = 0
        //   evals[1] = g(1, 0) = 1
        //   evals[2] = g(2, 0) = 2
        //   evals[3] = g(0, 1) = 1
        //   evals[4] = g(1, 1) = 2
        //   evals[5] = g(2, 1) = 3
        //   evals[6] = g(0, 2) = 2
        //   evals[7] = g(1, 2) = 3
        //   evals[8] = g(2, 2) = 4
        let poly = MultivariatePoly::new(
            Interval3,
            2,
            vec![
                Fp::new(0),
                Fp::new(1),
                Fp::new(2),
                Fp::new(1),
                Fp::new(2),
                Fp::new(3),
                Fp::new(2),
                Fp::new(3),
                Fp::new(4),
            ],
        );

        // evaluate at (7, 0) — Lagrange interp gives g(7, 0) = 7 + 0 = 7.
        assert_eq!(poly.evaluate(&[Fp::new(7), Fp::new(0)]), Fp::new(7));

        // Sum over S^2 = {0,1,2}^2: 0+1+2+1+2+3+2+3+4 = 18.
        assert_eq!(poly.sum_over_domain(), Fp::new(18));
    }
}
