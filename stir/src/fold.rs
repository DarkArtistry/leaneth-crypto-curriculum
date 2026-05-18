//! The **Fold** operation — STIR's core round-reduction step.
//!
//! ## What this module does
//!
//! Folding takes a polynomial / codeword `f` of degree `< d` over the
//! Goldilocks field `F = F_p` and produces a new polynomial / codeword
//! of degree `< d / k`, where `k` (the **folding factor**, a fixed
//! power of two — at least 4 in STIR) is the round's reduction factor.
//! A single random scalar `r ∈ F_p` — the **fold randomness**, the
//! verifier's per-round challenge — drives the reduction; the verifier
//! sends `r`, both prover and verifier apply the same fold, and the
//! protocol moves on to a problem of `1/k` the original size. The
//! `k` **row functions** `f_0, ..., f_{k-1}` introduced just below are
//! the coefficient-stride pieces of `f` that the fold mixes together
//! using powers of `r`.
//!
//! ## What folding does, in one paragraph
//!
//! Every polynomial `f(X)` of degree `< d` decomposes **uniquely** as
//!
//! ```text
//! f(X) = sum_{i=0..k-1} f_i(X^k) · X^i,        with deg(f_i) < d/k.
//! ```
//!
//! Read coefficient-by-coefficient: the `j`-th coefficient of `f` becomes the
//! `(j / k)`-th coefficient of `f_{j mod k}`. So `f_0` collects coefficients
//! at positions `0, k, 2k, ...`; `f_1` collects positions `1, k+1, 2k+1, ...`;
//! and so on. Conversely, `f_i` is the **`i`-th row** in the `k × (d/k)`
//! coefficient matrix of `f` (with row index `= position mod k`).
//!
//! The **fold by `r`** operation is then
//!
//! ```text
//! Fold(f, r)(Y)  :=  sum_{i=0..k-1} r^i · f_i(Y),
//! ```
//!
//! a single polynomial of degree `< d/k`. The `f_i(X^k) · X^i` decomposition
//! becomes `f_i(Y) · r^i` after substituting `Y = X^k` (which collapses the
//! `X^k` argument to `Y`) and `r` (which collapses the `X^i` factor to a
//! scalar `r^i`).
//!
//! ## Equivalent `k`-row view
//!
//! Think of `f(X)` as a `k × (d/k)` matrix `M` where row `i` lists the
//! coefficients of `f_i`. Then the fold is the matrix-vector product
//!
//! ```text
//! (r^0, r^1, r^2, ..., r^{k-1}) · M,
//! ```
//!
//! producing a length-`d/k` row vector — the coefficients of `Fold(f, r)`.
//! "Coefficient form fold" really is one matrix multiply on the coefficient
//! grid.
//!
//! ## Why it works on evaluation tables, not just polynomials
//!
//! Folding on evaluation tables is the operationally important case in STIR:
//! the prover commits to `f` as an evaluation table on `L`, never as
//! coefficients, and the verifier needs to derive the folded codeword on `L^k`
//! using only the original table.
//!
//! The trick: pick the evaluation domain `L` to be **closed under
//! multiplication by `ω`**, where `ω` is a primitive `k`-th root of unity.
//! For smooth multiplicative domains `L = c · <g>` with `|L|` a multiple of
//! `k`, we can take `ω = g^{|L|/k}` and the closure holds automatically. Then
//! for any `x ∈ L`, the `k` points `x, ω·x, ω²·x, ..., ω^{k-1}·x` are also in
//! `L` — and they share the same `k`-th power: `(ω^i · x)^k = ω^{ik} · x^k =
//! x^k` (using `ω^k = 1`). So the `k`-fold "fibre" over `x^k` is exactly
//! `{ω^i · x}_{i=0..k-1}`.
//!
//! Now evaluate the decomposition at `ω^i · x`:
//!
//! ```text
//! f(ω^i · x)  =  sum_{j=0..k-1} f_j((ω^i · x)^k) · (ω^i · x)^j
//!             =  sum_{j=0..k-1} f_j(x^k) · ω^{ij} · x^j.
//! ```
//!
//! Treat the `k` values `(f(x), f(ω·x), ..., f(ω^{k-1}·x))` as a vector and
//! the `k` row-evaluations `(f_0(x^k) · x^0, f_1(x^k) · x^1, ..., f_{k-1}(x^k)
//! · x^{k-1})` as another. The matrix relating them is the DFT matrix
//! `V[i, j] = ω^{ij}` — a Vandermonde at `ω` (see `reed_solomon::fft`). Invert
//! the DFT (a single `k`-point IDFT, cheap for small `k`), divide each entry
//! by the appropriate `x^j`, and you recover `(f_0(x^k), ..., f_{k-1}(x^k))`.
//! Linear-combine with `(r^0, r^1, ..., r^{k-1})` to produce the folded
//! evaluation `Fold(f, r)(x^k)`.
//!
//! Repeat for every `x` in a representative set (one per fibre — equivalently,
//! one per coset of `<ω>` in `L`) and you've built the entire folded
//! evaluation table on the squared/`k`-th-power domain `L^k`.
//!
//! ## Worked example (`F_17`, `k = 2`, `r = 4`)
//!
//! Take `f(X) = 3 + 5X + 7X² + 11X³` of degree 3 (4 coefficients). The
//! decomposition into `k = 2` rows splits coefficients by parity of position:
//!
//! ```text
//! f_0(Y) = 3 + 7Y       (positions 0 and 2)
//! f_1(Y) = 5 + 11Y      (positions 1 and 3)
//! ```
//!
//! With `r = 4`:
//!
//! ```text
//! Fold(f, 4)(Y) = f_0(Y) + 4 · f_1(Y)
//!               = (3 + 7Y) + 4·(5 + 11Y)
//!               = (3 + 20) + (7 + 44) · Y
//!               = 23 + 51 · Y
//!               ≡ 6 + 0 · Y      (mod 17)
//!               = 6.
//! ```
//!
//! So `poly_fold(f, 2, 4) = UnivariatePoly::new(vec![Fp::new(6)])` — a
//! degree-0 constant, well within the `< d/k = 2` bound. The `Y` coefficient
//! happens to vanish mod 17, and `UnivariatePoly::new` strips the trailing
//! zero, leaving a length-1 coefficient vector.
//!
//! ## Named theorem
//!
//! > **Fold Soundness Lemma (STIR §4).** Let `f` be a function on the
//! > evaluation domain `L`, and let `RS[F, L, d]` be the Reed-Solomon code of
//! > degree bound `< d`. If `f` is δ-far (in relative Hamming distance) from
//! > every codeword of `RS[F, L, d]`, then for a uniformly random `r ∈ F`,
//! >
//! > ```text
//! > Pr_r[ Fold(f, r) is < δ-far from RS[F, L^k, d/k] ]  ≤  poly(k, d) / |F|.
//! > ```
//!
//! Intuitively: a malicious prover who tries to "cancel" the corrupted
//! coefficients across rows would have to engineer `f` so that the specific
//! linear combination dictated by `r` happens to land back in the code. For
//! `r` chosen *after* `f` is committed and over a large field, this happens
//! with negligible probability — the same Schwartz-Zippel argument that
//! underpins every interactive low-degree test.
//!
//! ## Socratic prompt
//!
//! **Question for the reader.** Why is the random `r` essential here? What
//! would happen if we just used `r = 1`?
//!
//! Answer: with `r = 1` the fold collapses to `f_0(Y) + f_1(Y) + ... +
//! f_{k-1}(Y)` — a fixed deterministic linear combination of the rows. A
//! malicious prover who knows `r = 1` in advance can construct `f` whose rows
//! sum to a low-degree polynomial *despite* `f` itself being arbitrarily
//! corrupted: e.g., for `k = 2`, take `f_0` correct and `f_1 = -f_0 + g` for a
//! tiny `g` of degree `< d/k` — then `Fold(f, 1) = g` is in the next-round
//! code while `f` is not in the current-round code. Randomness over a large
//! field forces the adversary to commit to a single error before the linear
//! combination is fixed, and the Schwartz-Zippel bound kicks in: a non-zero
//! polynomial of low degree in `r` vanishes on a vanishingly small fraction
//! of the field.
//!
//! ## Cross-module interface
//!
//! - [`fold`] operates on evaluation tables on a smooth domain `L`.
//! - [`poly_fold`] operates on coefficient-form polynomials — useful for
//!   correctness tests against [`fold`] (round-trip via `reed_solomon::fft`).
//!
//! See `reed_solomon::polynomial::UnivariatePoly`,
//! `reed_solomon::domain::EvaluationDomain`, and `reed_solomon::fft` for the
//! underlying types and DFT machinery this module relies on.

use reed_solomon::{Fp, UnivariatePoly};

/// Fold a codeword (evaluation table) by random linear combination of its
/// `k = folding_factor` "rows".
///
/// Given a codeword `evals` of `f(X)` evaluated on a domain `L` closed under
/// multiplication by `omega` (a primitive `k`-th root of unity), returns the
/// evaluations of `Fold(f, r)(Y)` on `L^k` — the image of `L` under
/// `x → x^k`, which has `|L|/k` elements.
///
/// # Inputs
///
/// - `evals`: `f(L[0]), f(L[1]), ..., f(L[n-1])`, with `n = |L|`.
/// - `folding_factor`: `k`. Must divide `n` and equal `omega`'s order.
/// - `randomness`: `r ∈ F_p`. The verifier's challenge.
/// - `omega`: a primitive `k`-th root of unity.
///
/// # Output
///
/// A `Vec<Fp>` of length `n / k`, with entry `j` equal to
/// `Fold(f, r)(L[j]^k)` — that is, the folded codeword on `L^k`, indexed by
/// the fibre representative `L[j]`.
///
/// # Layout convention
///
/// `evals` is assumed to be in **natural order** on `L = c · <g>`: entry `j`
/// is `f(c · g^j)`. Under that ordering, the `k`-fold fibre `{ω^i · L[j]}`
/// over `L[j]^k` corresponds to indices `j, j + n/k, j + 2n/k, ..., j +
/// (k-1)n/k` of `evals` (i.e., a stride of `n/k`), because we set
/// `omega = g^{n/k}` to walk through the fibre.
///
/// # Panics
///
/// Panics if `folding_factor == 0`, if `folding_factor` does not divide
/// `evals.len()`, or if `omega.pow(folding_factor) != Fp::one()` (caller
/// passed a non-`k`-th-root, which would break the IDFT step).
///
/// (STIR §4, Fold definition; reduction-step soundness in Lemma 4.4.)
pub fn fold(
    evals: &[Fp],
    folding_factor: u32,
    randomness: Fp,
    omega: Fp,
) -> Vec<Fp> {
    // TODO: build the folded evaluation table on the k-th-power domain.
    //   1. Validate: `folding_factor >= 1`, `k := folding_factor as usize`
    //      divides `evals.len()`, and `omega^k == 1` (caller really did pass
    //      a k-th root of unity — otherwise the IDFT step below is nonsense).
    //   2. Let `m = evals.len() / k`. The output has length `m`.
    //   3. Precompute powers `(r^0, r^1, ..., r^{k-1})` and the inverse DFT
    //      matrix at `omega` of size `k × k`. For small `k` (typically 2-16
    //      in STIR), an O(k²) explicit matrix suffices; no recursion needed.
    //   4. For each fibre representative index `j ∈ 0..m`:
    //        a. Collect the k "siblings" `evals[j], evals[j + m], evals[j + 2m],
    //           ..., evals[j + (k-1)m]` — these are `f(ω^i · L[j])` by the
    //           layout convention above.
    //        b. Apply the k-point inverse DFT to recover the *normalised*
    //           row-evaluations `f_i(L[j]^k) · L[j]^i` for `i = 0..k-1`.
    //        c. Divide entry `i` by `L[j]^i` to peel off the `X^i` factor
    //           (cheap: walk a running product of `L[j]`).
    //        d. Linear-combine with `(r^0, ..., r^{k-1})` to get
    //           `Fold(f, r)(L[j]^k)`. Push to the output vector.
    //   5. Return the length-`m` output.
    //
    // Reference: the "Why it works on evaluation tables, not just polynomials"
    // and "Worked example" sections of the module docs.
    //
    // CAUTION: the evaluation domain `L` MUST be closed under `x → ω·x` where
    // ω is a primitive `k`-th root. For STIR this holds by construction
    // (smooth domains of size a multiple of `k`, with ω chosen as the
    // appropriate power of the domain generator); for arbitrary domains it
    // would fail — neither the fibre structure nor the IDFT recovery applies.
    let _ = (evals, folding_factor, randomness, omega);
    todo!()
}

/// Fold a coefficient-form polynomial by the row-randomization rule:
///
/// ```text
/// Fold(f, r)(Y) = sum_{i=0..k-1} r^i · f_i(Y),
/// ```
///
/// where `f(X) = sum_i f_i(X^k) · X^i` is the unique `k`-row decomposition
/// (with `deg(f_i) < deg(f) / k`).
///
/// # Inputs
///
/// - `poly`: the input polynomial in coefficient form (ascending degree).
/// - `folding_factor`: `k`. Must be `≥ 1`.
/// - `randomness`: `r ∈ F_p`.
///
/// # Output
///
/// `UnivariatePoly` of degree `< ceil(deg(poly)+1) / k`.
///
/// # Panics
///
/// Panics if `folding_factor == 0`.
///
/// # Example
///
/// `poly = 3 + 5X + 7X² + 11X³`, `k = 2`, `r = 4` (in F_17):
///
/// ```text
/// f_0 = 3 + 7Y,   f_1 = 5 + 11Y
/// Fold = f_0 + 4·f_1 = 23 + 51Y ≡ 6 (mod 17).
/// ```
///
/// (STIR §4, coefficient-form Fold.)
pub fn poly_fold(
    poly: &UnivariatePoly,
    folding_factor: u32,
    randomness: Fp,
) -> UnivariatePoly {
    // TODO: coefficient-form fold via row extraction.
    //   1. Validate `folding_factor >= 1`. Let `k = folding_factor as usize`.
    //   2. Read `c = poly.coeffs()`. Empty (zero polynomial) → return
    //      UnivariatePoly::zero().
    //   3. Each row `i ∈ 0..k` has coefficients `c[i], c[i+k], c[i+2k], ...`
    //      (the every-k-th-starting-at-i slice). Build `f_i` as a
    //      UnivariatePoly with that coefficient vector — call
    //      UnivariatePoly::new so trailing zeros are stripped.
    //   4. Accumulate `result = 0`, walk a running `r^i` (start at Fp::one()),
    //      and for each row do `result = result + scalar_mul(&f_i, r_power);
    //      r_power = r_power * randomness`. Return `result`.
    //
    // Equivalence with `fold`: if `evals = (f(L[0]), ..., f(L[n-1]))` is the
    // evaluation table of `poly`, then `fold(evals, k, r, omega)` (with
    // omega = the k-th root used by the domain) is the evaluation table of
    // `poly_fold(poly, k, r)` on `L^k`. The `fold_matches_poly_fold_on_evaluations`
    // test below pins this down.
    let _ = (poly, folding_factor, randomness);
    todo!()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// `poly_fold` of a constant polynomial: every "row" is the constant
    /// itself (in row 0) or zero (rows 1..k), so the fold equals `c · r^0 = c`.
    #[test]
    fn poly_fold_of_constant_equals_constant() {
        // TODO:
        //   1. Build `c = UnivariatePoly::new(vec![Fp::new(42)])`.
        //   2. Fold with k = 4, randomness = Fp::new(99).
        //   3. Assert the result equals `c` (the only non-zero row is row 0,
        //      which contributes 42 · r^0 = 42 to the fold).
        todo!()
    }

    /// `fold` on evaluation tables must match `poly_fold` on coefficients:
    /// take a polynomial, encode it on a domain, fold the codeword, decode the
    /// folded codeword on the squared domain — should equal `poly_fold(poly, k, r)`
    /// evaluated on the squared domain.
    #[test]
    fn fold_matches_poly_fold_on_evaluations() {
        // TODO:
        //   1. Pick a domain `L` of log_size = 4 (size 16) and a random
        //      polynomial of degree < 8 (8 coefficients) so that d/k = 4 fits
        //      in the squared domain `L^k` of size 8.
        //   2. Compute `evals = fft_on_domain(coeffs, &L)`.
        //   3. Pick k = 2, r = Fp::new(13), omega = primitive_root_of_unity(1)
        //      (the size-2-th root, i.e. -1).
        //   4. Compute `folded_evals = fold(&evals, k, r, omega)`.
        //   5. Compute `folded_poly = poly_fold(&poly, k, r)` and evaluate it
        //      pointwise on `L^k` (the size-8 squared domain).
        //   6. Assert the two length-8 vectors are equal entry by entry.
        todo!()
    }

    /// With `r = 0`, `Fold(f, 0) = f_0` — only the row-0 contribution survives,
    /// since `r^0 = 1` and `r^i = 0` for `i ≥ 1`.
    #[test]
    fn fold_with_r_zero_returns_first_row_only() {
        // TODO:
        //   1. Build `f(X) = 1 + 2X + 3X² + 4X³` (so `f_0 = 1 + 3Y`, `f_1 = 2 + 4Y`).
        //   2. `poly_fold(f, 2, Fp::zero())` should equal `UnivariatePoly::new([1, 3])`.
        //   3. Optionally repeat for k = 4: `poly_fold(f, 4, Fp::zero())` returns
        //      `UnivariatePoly::new([1])` (just `f_0` constant).
        todo!()
    }

    /// `fold` must reject inputs whose length is not divisible by the folding
    /// factor — there's no consistent way to bundle them into k-element fibres.
    #[test]
    #[should_panic]
    fn fold_factor_must_divide_evals_length() {
        // TODO:
        //   1. Build `evals = vec![Fp::new(1); 5]` (length 5, not divisible by 2).
        //   2. Call `fold(&evals, 2, Fp::one(), Fp::primitive_root_of_unity(1))`.
        //   3. Should panic via the divisibility assertion inside `fold`.
        todo!()
    }
}
