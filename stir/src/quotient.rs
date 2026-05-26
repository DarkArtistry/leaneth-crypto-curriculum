//! The **Quotient** operation — divide a codeword (or polynomial) by a
//! vanishing polynomial whose roots are the constraint points.
//!
//! ## What this module does, in one paragraph
//!
//! Each STIR round collects a **constraint set** `S = {(x_i, y_i)}_{i=1..s}`
//! — the verifier's OOD ("out-of-domain") replies and shift answers,
//! `s` pairs of a point in the field `F = F_p` and a claimed value — and
//! enforces that the prover's polynomial `f` of degree `< d` passes
//! through those points. The **vanishing polynomial** `V_S(X) :=
//! prod (X − x_i)` has degree `s` and vanishes exactly on `S`, and the
//! **quotient** `q(X) := (f(X) − p_S(X)) / V_S(X)` (where `p_S` is the
//! degree-`< s` Lagrange interpolant of `S`, defined formally below)
//! turns those `s` "passes-through" constraints into a degree
//! reduction: an `s`-element constraint set quotients a degree-`< d`
//! polynomial down to a degree-`< d − s` polynomial.
//!
//! ## Anchor: how STIR pins down out-of-domain values into the proof
//!
//! STIR's headline improvement over FRI comes from **out-of-domain (OOD)
//! sampling** (see [`crate::ood`]): after the prover commits to a
//! folded codeword on the round-`i` domain `L_i`, the verifier draws
//! random points `r ∈ F_p \ L_i` and forces the prover to reveal claimed
//! values `g(r)`. The list-decoding analysis (Johnson bound, see the
//! intro to [`crate::ood`]) shows that **a single such reveal already
//! collapses the prover's freedom from a list of `ℓ ≈ 1/(2η√ρ)` candidate
//! polynomials down to one**.
//!
//! But a reveal-and-trust model is too weak: nothing yet *binds* the
//! prover to the value it announces. The quotient construction in this
//! module is what does the binding. Given the constraint set
//! `S = {(r_1, g(r_1)), ..., (r_s, g(r_s))}` ∪ (shift queries), the
//! prover-and-verifier protocol replaces the committed function `g` with
//!
//! ```text
//!   q(X) := (g(X) − p_S(X)) / V_S(X),
//! ```
//!
//! the **quotient of `g` by `S`**, and continues the low-degree test on
//! `q` instead of `g`. The next-round low-degree test only accepts if
//! `q` is itself close to a code of degree `< d − s` — which, by the
//! theorem below, can only happen if `g` *truly* satisfied every
//! constraint `g(x_i) = y_i` on `S`. **A lying prover that committed to
//! a `g` disagreeing with its announced OOD values gets a `q` outside
//! the next-round code; the verifier rejects.** That is how the OOD
//! reveals stop being a trust-me handshake and start being algebraic
//! certificates pinned into the proof.
//!
//! ## What quotienting does, mathematically
//!
//! Given a polynomial `f(X)` and a constraint set `S = {(x_i, y_i)}_{i=1..s}`
//! with all `x_i` distinct, write the **vanishing polynomial**
//!
//! ```text
//! V_S(X)  :=  prod_{i=1..s} (X − x_i),     deg V_S = s,
//! ```
//!
//! and the **Lagrange interpolant** `p_S(X)` of degree `< s` through `S` — the
//! unique polynomial of degree `< s` with `p_S(x_i) = y_i` for all `i`. The
//! **quotient** of `f` by `S` is the unique polynomial
//!
//! ```text
//! q(X)  :=  ( f(X) − p_S(X) ) / V_S(X),
//! ```
//!
//! **provided** `f` satisfies all the constraints — `f(x_i) = y_i` for every
//! `i`. (Equivalently, `f − p_S` vanishes on `S`, so `V_S | (f − p_S)` and the
//! division is exact.) On evaluation form: at any point `L[j] ∈ L` disjoint
//! from `S`,
//!
//! ```text
//! q(L[j])  =  ( f(L[j]) − p_S(L[j]) ) / V_S(L[j]).
//! ```
//!
//! Pointwise subtraction, pointwise division. No polynomial arithmetic in
//! the inner loop.
//!
//! ## Where STIR uses it
//!
//! After collecting the round's OOD replies (`s` points) and shift answers
//! (`t_{i-1}` points sampled from the previous round's domain), the verifier
//! has a constraint set of size `|G_i| = s + t_{i-1}`. Both prover and
//! verifier compute the quotient on this combined constraint set, which:
//!
//! - **Enforces the constraints algebraically.** Wherever `f` disagrees with
//!   `p_S` on the constraint points, the numerator `f − p_S` does not vanish
//!   on `S` — but the denominator does — so the quotient blows up at those
//!   points. Detecting such blow-up at a constraint point would be ill-defined,
//!   which is why STIR's protocol guarantees `L_i ∩ S = ∅` (constraint and
//!   evaluation domains are disjoint by design — see [`quotient`] panic
//!   contract below).
//! - **Reduces the degree bound by exactly `|S|`.** This is the round-by-round
//!   shrinkage that lets STIR climb down from `d_0` to a constant in
//!   logarithmically many rounds.
//!
//! ## Named theorem
//!
//! > **Polynomial Identity Lemma (Constraint Embedding theorem).**
//! > Let `f ∈ F[X]` with `deg f < d`, and let `S = {(x_1, y_1), ..., (x_s, y_s)}`
//! > with the `x_i` pairwise distinct. Let `p_S` be the Lagrange interpolant
//! > of degree `< s` through `S` and `V_S(X) := prod_{i=1..s} (X − x_i)` of
//! > degree exactly `s`. Then:
//! >
//! > 1. **(Soundness, contrapositive form.)** If `f(x_j) ≠ y_j` for some `j`,
//! >    then `V_S` does **not** divide `f − p_S`: the polynomial division
//! >    `(f − p_S) / V_S` has a non-zero remainder.
//! > 2. **(Completeness.)** If `f(x_i) = y_i` for every `i`, then
//! >    `V_S | (f − p_S)`, the quotient `q := (f − p_S) / V_S` is a polynomial,
//! >    and `deg q < d − s`.
//!
//! ### Constructive proof
//!
//! Write `g := f − p_S`. Since `deg p_S < s ≤ d`, the degree of `g` is at
//! most `max(deg f, deg p_S) ≤ d − 1`.
//!
//! - **(2) Completeness.** Suppose `f(x_i) = y_i` for all `i`. Then for every
//!   `i`, `g(x_i) = f(x_i) − p_S(x_i) = y_i − y_i = 0`. So `g` has `s`
//!   distinct roots `x_1, ..., x_s`. By the **factor theorem** (a polynomial
//!   `g` is divisible by `X − a` iff `g(a) = 0`), iterated `s` times on
//!   distinct `x_i`'s, the product `V_S = prod (X − x_i)` divides `g`.
//!   The quotient `q := g / V_S` is a polynomial of degree
//!   `deg g − deg V_S ≤ (d − 1) − s = d − s − 1 < d − s`, as claimed.
//!
//!   *Why "iterate the factor theorem" is valid here.* The factor theorem
//!   gives us `g = (X − x_1) · h_1` for some polynomial `h_1`. Evaluate at
//!   `x_2`: `0 = g(x_2) = (x_2 − x_1) · h_1(x_2)`. Since `x_2 ≠ x_1`,
//!   `(x_2 − x_1)` is a unit (non-zero in a field), so `h_1(x_2) = 0` and
//!   the factor theorem applies again: `h_1 = (X − x_2) · h_2`. Inducting
//!   on the `s` distinct roots gives `g = V_S · h_s`, exactly the desired
//!   factorisation.
//!
//! - **(1) Soundness, contrapositive form.** Suppose `f(x_j) ≠ y_j` for some
//!   `j`. Then `g(x_j) = f(x_j) − p_S(x_j) = f(x_j) − y_j ≠ 0`. If `V_S`
//!   divided `g`, then `g = V_S · h` for some polynomial `h`, and
//!   `g(x_j) = V_S(x_j) · h(x_j) = 0 · h(x_j) = 0` — contradicting
//!   `g(x_j) ≠ 0`. So `V_S ∤ g`, and the polynomial division leaves a
//!   non-zero remainder. ∎
//!
//! **Why this is the load-bearing fact for STIR.** A cheating prover wants
//! to commit to a `g` that disagrees with its announced OOD value at some
//! `x_j`. After quotienting, by (1) above, the result `(g − p_S) / V_S`
//! produces a non-zero remainder — equivalently, the natural extension to
//! a rational function lies *outside* the polynomial code of degree
//! `< d − s`. The next round's low-degree test catches the resulting
//! function-vs-codeword distance with probability `≥ 1 − (1 − δ')^t` where
//! `δ'` is the relative distance and `t` is the next-round shift-query
//! count. Re-evaluation at a random fresh point post-folding is what
//! converts the algebraic certificate into a probabilistic catch.
//!
//! ## Worked example
//!
//! Take `p(X) = X² + 1` and the constraint set `S = {(0, 1), (1, 2)}`.
//! (Sanity-check the constraints are consistent with `p`: `p(0) = 1` ✓,
//! `p(1) = 1 + 1 = 2` ✓.)
//!
//! 1. **Lagrange interpolant** through `(0, 1)` and `(1, 2)`: the unique
//!    line `L_S(X) = a + b·X` with `L_S(0) = 1` and `L_S(1) = 2` is
//!    `a = 1, b = 1`, so
//!
//!    ```text
//!    L_S(X) = 1 + X.
//!    ```
//!
//! 2. **Vanishing polynomial**:
//!
//!    ```text
//!    V_S(X) = (X − 0)(X − 1) = X · (X − 1) = X² − X.
//!    ```
//!
//! 3. **Numerator `p − L_S`**:
//!
//!    ```text
//!    p(X) − L_S(X)
//!        = (X² + 1) − (1 + X)
//!        = X² − X.
//!    ```
//!
//!    Sanity check: `(p − L_S)(0) = 0` ✓, `(p − L_S)(1) = 1 − 1 = 0` ✓
//!    — the numerator vanishes on `S`, exactly as Theorem (2) predicts.
//!
//! 4. **Quotient**:
//!
//!    ```text
//!    q(X) = (X² − X) / (X² − X) = 1.
//!    ```
//!
//!    A constant polynomial. Check the degree drop: `deg p = 2`, `|S| = 2`,
//!    so `deg q < 2 − 2 = 0`, i.e. `q` is the constant polynomial.
//!    Matches.
//!
//! 5. **Verify the defining identity** `q · V_S + L_S = p`:
//!
//!    ```text
//!    1 · (X² − X) + (1 + X) = X² − X + 1 + X = X² + 1 = p(X).  ✓
//!    ```
//!
//! Compare with the second, larger worked example below for the
//! one-constraint case.
//!
//! ## Second worked example (one constraint, asymmetric `|S|`)
//!
//! Take `f(X) = 1 + 2X + 3X² + 4X³` (degree 3) and a single constraint
//! `S = {(2, f(2))}`. Evaluate `f` at `X = 2`:
//!
//! ```text
//! f(2) = 1 + 4 + 12 + 32 = 49.
//! ```
//!
//! So `S = {(2, 49)}`. The vanishing polynomial is `V_S(X) = X − 2` (a
//! single root), and the Lagrange interpolant through a single point is
//! the constant `p_S(X) = 49`. The quotient is
//!
//! ```text
//! q(X) = ( f(X) − 49 ) / (X − 2)
//!      = ( 4X³ + 3X² + 2X + (1 − 49) ) / (X − 2)
//!      = ( 4X³ + 3X² + 2X − 48 ) / (X − 2).
//! ```
//!
//! Synthetic division by `(X − 2)` (top-down, multiplier `a = 2`):
//!
//! ```text
//!   carry = 4                              ← leading coeff
//!   carry = 3 + 2·4   = 11                 ← coefficient of X²
//!   carry = 2 + 2·11  = 24                 ← coefficient of X¹
//!   carry = −48 + 2·24 = 0   ← remainder, must be zero ✓
//! ```
//!
//! Reading the carries from top to bottom of the quotient: `q(X) = 4X² + 11X + 24`,
//! degree 2 = (deg f) − |S| = 3 − 1. The exact degree drop matches the
//! theorem above (Completeness clause).
//!
//! ## Socratic prompt
//!
//! **Question for the reader.** Why does the quotient `(f − p_S) / V_S` have
//! *strictly lower* degree than `f`, and what bound does it satisfy?
//!
//! Answer: write `g := f − p_S`. The interpolant `p_S` has `deg p_S < s ≤ deg f`
//! (whenever `s ≤ deg f`), so `deg g = deg f` (the high-degree terms of `f`
//! are unaffected by subtracting a lower-degree `p_S`). By construction
//! `g(x_i) = f(x_i) − y_i = 0` for every `i = 1..s` — so `g` vanishes on the
//! `s` distinct points of `S`. By the factor-theorem iteration argument in
//! the proof above, `V_S = prod (X − x_i)` divides `g`, and the quotient
//! `q = g / V_S` is a polynomial (no remainder). Its degree is exactly
//! `deg g − s = deg f − s` (when there's no remainder). So `deg q = deg f − s`
//! and `q` lies in the degree-`< deg f − s + 1` polynomial space — which
//! translates to `q ∈ RS[F, L, d − s]` on the codeword side whenever
//! `f ∈ RS[F, L, d]`. That `s`-step degree drop is the per-round shrinkage
//! that powers STIR's logarithmic round complexity.
//!
//! ## Cross-module interface
//!
//! - [`quotient`] operates on evaluation tables — the prover's and verifier's
//!   common path. Used by [`crate::prover`] when folding the committed
//!   evaluation table through each round, and by [`crate::verifier`] when
//!   re-evaluating consistency checks at sampled shift points.
//! - [`poly_quotient`] operates on coefficient-form polynomials — useful for
//!   correctness tests, reasoning, and the post-stopping reconstruction of
//!   the final small polynomial.
//!
//! Both rely on [`reed_solomon::interpolate::lagrange_interpolate`] for
//! `p_S` and on `UnivariatePoly`'s `Mul/Sub` for `V_S` construction.

use reed_solomon::interpolate::lagrange_interpolate;
use reed_solomon::{EvaluationDomain, Fp, UnivariatePoly};

/// Quotient a codeword (evaluation table) by the vanishing polynomial of a
/// constraint set.
///
/// Computes `q(L[j]) = (f(L[j]) - p_S(L[j])) / V_S(L[j])` for each `j`, where
/// `p_S` is the Lagrange interpolant through `points` and `V_S` is their
/// vanishing polynomial. See the module-level **Polynomial Identity Lemma /
/// Constraint Embedding theorem** for the algebraic guarantee that makes this
/// well-defined when the constraints are consistent.
///
/// # Inputs
///
/// - `evals`: `f(L[0]), ..., f(L[n-1])`.
/// - `domain`: the evaluation domain `L` — used to enumerate the `L[j]`.
/// - `points`: the constraint set `S = {(x_i, y_i)}`. The `x_i` must be
///   distinct and *disjoint* from `L` (STIR guarantees this by construction).
///
/// # Output
///
/// A `Vec<Fp>` of length `n = |L|`, with entry `j` equal to `q(L[j])`.
///
/// # Panics
///
/// Panics if `evals.len() != domain.size()`, if `points` is empty (no
/// constraint, no quotient — caller should use `evals` directly), if `points`
/// has a duplicate `x_i`, or — most importantly — if any `L[j] == x_i` for
/// some constraint `(x_i, y_i)` (which would force a division by zero via
/// `V_S(L[j]) = 0`).
///
/// (STIR §4, Lemma 4.4; Quotient definition. See module docs for proof.)
pub fn quotient(
    evals: &[Fp],
    domain: &EvaluationDomain,
    points: &[(Fp, Fp)],
) -> Vec<Fp> {
    assert_eq!(
        evals.len(),
        domain.size(),
        "evals length ({}) must equal domain size ({})",
        evals.len(),
        domain.size(),
    );
    assert!(
        !points.is_empty(),
        "quotient: constraint set must be non-empty (otherwise use `evals` directly)",
    );

    // Build the Lagrange interpolant p_S through the constraint points.
    // `lagrange_interpolate` panics on duplicate x-coordinates — we rely on
    // that as our distinctness check.
    //
    // Cross-reference: the algebraic role of `p_S` is "the unique low-degree
    // certificate of the prover's claimed values on S" — see the module-level
    // Polynomial Identity Lemma, Completeness clause: `f − p_S` vanishes on
    // S precisely when the prover's announced values are honest.
    let p_s = lagrange_interpolate(points);

    let mut out = Vec::with_capacity(domain.size());
    for (j, l_j) in domain.iter().enumerate() {
        // Compute V_S(L[j]) = prod_i (L[j] − x_i). Inline product avoids
        // building V_S as a polynomial when we only need its value at one
        // point.
        let mut denom = Fp::one();
        for &(x_i, _y_i) in points {
            let factor = l_j - x_i;
            assert!(
                factor != Fp::zero(),
                "quotient: domain point L[{}] = {:?} coincides with constraint x_i = {:?}; \
                 STIR requires L ∩ S = ∅",
                j,
                l_j,
                x_i,
            );
            denom = denom * factor;
        }

        // denom != 0 since each factor was checked non-zero. Inverse is safe.
        let denom_inv = denom
            .inverse()
            .expect("denom is a product of non-zero field elements, must be invertible");

        let numer = evals[j] - p_s.evaluate(l_j);
        out.push(numer * denom_inv);
    }
    out
}

/// Quotient a coefficient-form polynomial by the vanishing polynomial of a
/// constraint set.
///
/// Returns `q(X) := (f(X) − p_S(X)) / V_S(X)`. Panics if the constraints
/// aren't consistent with `f` — i.e., if `f(x_i) != y_i` for some `i` —
/// because then the polynomial division leaves a non-zero remainder
/// (see the module-level **Polynomial Identity Lemma**, soundness clause).
///
/// # Implementation note: iterated synthetic division
///
/// `V_S = prod_i (X − x_i)` factors completely over `F_p` with known roots,
/// so instead of running schoolbook long division we **iterate synthetic
/// division** by each `(X − x_i)`. One synthetic-division step has cost
/// linear in the polynomial degree; iterating `|S|` times gives `O(|S| · deg f)`
/// field operations — same complexity as one long-division pass, but
/// substantially simpler code and (importantly) zero schoolbook bookkeeping
/// to get wrong.
///
/// At each step the remainder must be zero (the factor theorem, repeatedly).
/// We assert this; an inconsistent constraint set is a caller bug and a panic
/// is easier to debug than a silently-wrong quotient.
///
/// # Inputs
///
/// - `poly`: the input polynomial `f` in coefficient form.
/// - `points`: the constraint set `S`. The `x_i` must be distinct.
///
/// # Output
///
/// A `UnivariatePoly` of degree `deg f − |S|` (or zero if cancellation occurs).
///
/// # Panics
///
/// Panics if `points` is empty (no constraint, no quotient — caller should
/// use `f` directly), if `points` contains a duplicate `x_i` (via
/// `lagrange_interpolate`), or if synthetic division leaves a non-zero
/// remainder at any step (caller passed inconsistent constraints).
///
/// (STIR §4, coefficient-form Quotient. Proof in module docs.)
pub fn poly_quotient(
    poly: &UnivariatePoly,
    points: &[(Fp, Fp)],
) -> UnivariatePoly {
    assert!(
        !points.is_empty(),
        "poly_quotient: constraint set must be non-empty (otherwise return `poly` itself)",
    );

    // Step 1: build the Lagrange interpolant p_S. `lagrange_interpolate`
    // panics on duplicate x-coordinates, giving us distinctness checking
    // for free.
    let p_s = lagrange_interpolate(points);

    // Step 2: form g := poly − p_S. By the Polynomial Identity Lemma
    // (Completeness clause, module docs), g vanishes on every x_i in S iff
    // the constraints are consistent with `poly`. We do not check this
    // up front — the synthetic-division remainder check below will catch
    // any violation.
    let g = poly.clone() - p_s;

    // Step 3: iteratively divide `g` by (X − x_i) for each i. Each step
    // returns a polynomial of one lower degree and asserts the synthetic-
    // division remainder is zero.
    let mut current_coeffs: Vec<Fp> = g.coeffs().to_vec();
    for (i, &(x_i, _y_i)) in points.iter().enumerate() {
        current_coeffs = synthetic_divide(&current_coeffs, x_i).unwrap_or_else(|rem| {
            panic!(
                "poly_quotient: constraints inconsistent with polynomial at step {} \
                 (x_i = {:?}); synthetic-division remainder = {:?}. This means \
                 the caller's `points` does not agree with `poly` on all x_i.",
                i, x_i, rem,
            )
        });
    }

    UnivariatePoly::new(current_coeffs)
}

/// Synthetic division of a polynomial `g(X)` (given in ascending-coefficient
/// form `[c_0, c_1, ..., c_n]`) by `(X − a)`.
///
/// Returns `Ok(quotient_coeffs)` if the remainder is zero, `Err(remainder)`
/// otherwise. The quotient has length `n` (degree `n − 1`) when the input
/// has length `n + 1` (degree `n`); on `g = 0` (empty input) returns
/// `Ok(vec![])` immediately.
///
/// **Algorithm.** Standard synthetic division, top-down:
///
/// ```text
///   carry := c_n
///   q_{n-1} := carry
///   for i = n - 1 down to 1:
///       carry := c_i + a · carry
///       q_{i-1} := carry
///   remainder := c_0 + a · carry
/// ```
///
/// Reads `n + 1` coefficients, writes `n` quotient coefficients, performs
/// `n` mults and `n` adds. Cost is linear in the degree.
fn synthetic_divide(coeffs: &[Fp], a: Fp) -> Result<Vec<Fp>, Fp> {
    if coeffs.is_empty() {
        // Zero polynomial divides by anything with zero quotient and zero
        // remainder.
        return Ok(Vec::new());
    }

    let n = coeffs.len() - 1; // degree
    if n == 0 {
        // A non-zero constant divided by (X − a) has zero quotient and
        // the constant itself as remainder.
        return Err(coeffs[0]);
    }

    // Quotient coefficients in ascending order: q_0, q_1, ..., q_{n−1}.
    // We compute them top-down (q_{n−1} first), so we fill `quot` in
    // reverse and then it ends up in ascending order naturally.
    let mut quot = vec![Fp::zero(); n];

    // Walk `coeffs` from the high end (c_n) down to c_1.
    let mut carry = coeffs[n];
    quot[n - 1] = carry;
    for i in (1..n).rev() {
        carry = coeffs[i] + a * carry;
        quot[i - 1] = carry;
    }

    // Final remainder = c_0 + a · carry. By the factor theorem, this equals
    // g(a). If non-zero, the division is not exact.
    let remainder = coeffs[0] + a * carry;
    if remainder == Fp::zero() {
        Ok(quot)
    } else {
        Err(remainder)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the vanishing polynomial `V_S(X) = prod_i (X − x_i)` as a
    /// `UnivariatePoly`. Test helper.
    fn vanishing_poly(points: &[(Fp, Fp)]) -> UnivariatePoly {
        let mut v = UnivariatePoly::one();
        for &(x_i, _) in points {
            // (X − x_i) has coeffs [−x_i, 1].
            v = v * UnivariatePoly::new(vec![-x_i, Fp::one()]);
        }
        v
    }

    /// The doc's worked example, end to end: `p(X) = X² + 1`,
    /// `S = {(0, 1), (1, 2)}`, expect `q(X) = 1`.
    #[test]
    fn worked_example_x_squared_plus_one() {
        let p = UnivariatePoly::new(vec![Fp::one(), Fp::zero(), Fp::one()]);
        let points = [
            (Fp::new(0), Fp::new(1)),
            (Fp::new(1), Fp::new(2)),
        ];

        let q = poly_quotient(&p, &points);

        // Expected: q(X) = 1, the constant polynomial.
        assert_eq!(q, UnivariatePoly::new(vec![Fp::one()]));
        assert_eq!(q.degree(), Some(0));
    }

    /// Quotient a codeword by a *consistent* single-point constraint: the
    /// pointwise division should agree with `poly_quotient` evaluated on the
    /// domain.
    ///
    /// This exercises both `quotient` (evaluation form) and `poly_quotient`
    /// (coefficient form) on the same input and checks they agree at every
    /// domain point — the consistency contract the prover/verifier depend on.
    #[test]
    fn quotient_of_codeword_through_one_point() {
        // f(X) = 1 + 2X + 3X² + 4X³.
        let f = UnivariatePoly::new(vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)]);

        // Domain of size 8 (a subgroup of order 8 ⊂ F_p^*). Its elements are
        // 8th roots of unity, all of which are in F_p^* — none equal to 99.
        let domain = EvaluationDomain::new_subgroup(3);

        // Pick x_0 outside the domain. The 8th roots of unity in F_p are
        // determined by `primitive_root_of_unity(3)`; Fp::new(99) is not a
        // power of that root (cheap to confirm: 99^8 ≠ 1 in F_p, but we
        // don't bother proving it here — `quotient` itself will panic if
        // we got it wrong, so the test will fail loudly).
        let x_0 = Fp::new(99);
        let y_0 = f.evaluate(x_0);
        let points = [(x_0, y_0)];

        // Evaluate f on the domain — using polynomial `evaluate` is fine
        // for size 8 (we don't strictly need FFT here).
        let evals: Vec<Fp> = (0..domain.size())
            .map(|i| f.evaluate(domain.element(i)))
            .collect();

        // Evaluation-form quotient.
        let evals_quot = quotient(&evals, &domain, &points);

        // Coefficient-form quotient, then evaluate on the domain.
        let poly_quot = poly_quotient(&f, &points);
        let poly_quot_on_domain: Vec<Fp> = (0..domain.size())
            .map(|i| poly_quot.evaluate(domain.element(i)))
            .collect();

        assert_eq!(evals_quot, poly_quot_on_domain);

        // And quotient has expected degree (3 − 1 = 2).
        assert_eq!(poly_quot.degree(), Some(2));
    }

    /// `poly_quotient` of a polynomial through points it actually passes
    /// through returns a degree-`(deg f − |S|)` polynomial, and the defining
    /// identity `q · V_S + p_S == f` holds.
    #[test]
    fn poly_quotient_no_remainder_when_constraint_consistent() {
        // f(X) = 1 + 2X + 3X² + 4X³  (degree 3).
        let f = UnivariatePoly::new(vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)]);

        // Two distinct, generic constraint x-coordinates.
        let x_1 = Fp::new(5);
        let x_2 = Fp::new(7);
        let points = [(x_1, f.evaluate(x_1)), (x_2, f.evaluate(x_2))];

        let q = poly_quotient(&f, &points);

        // Theorem (Completeness): deg q = deg f − |S| = 3 − 2 = 1.
        assert_eq!(q.degree(), Some(1));

        // Strong identity check: q · V_S + p_S == f.
        let v_s = vanishing_poly(&points);
        let p_s = lagrange_interpolate(&points);
        let reconstructed = q.clone() * v_s + p_s;
        assert_eq!(reconstructed, f);
    }

    /// Quotienting by `|S| = s` constraint points drops the polynomial's
    /// degree by exactly `s`. This is what powers STIR's per-round degree
    /// shrinkage. Uses a degree-7 polynomial and three constraints.
    #[test]
    fn quotient_lowers_degree_by_constraint_count() {
        // Build a fixed degree-7 polynomial — 8 ascending-degree coefficients,
        // all non-zero so the degree is exactly 7.
        let f = UnivariatePoly::new(vec![
            Fp::new(1),
            Fp::new(2),
            Fp::new(3),
            Fp::new(5),
            Fp::new(7),
            Fp::new(11),
            Fp::new(13),
            Fp::new(17),
        ]);
        assert_eq!(f.degree(), Some(7));

        // Three distinct constraint x-coordinates with consistent y's.
        let xs = [Fp::new(20), Fp::new(21), Fp::new(22)];
        let points: Vec<(Fp, Fp)> = xs.iter().map(|&x| (x, f.evaluate(x))).collect();

        let q = poly_quotient(&f, &points);

        assert_eq!(q.degree(), Some(4)); // 7 − 3 = 4.
    }

    /// Soundness clause of the Polynomial Identity Lemma: when the prover's
    /// claimed `y_j` disagrees with `f(x_j)`, the synthetic-division
    /// remainder at step `j` is non-zero, and `poly_quotient` panics. This
    /// is the algebraic check that catches a cheating prover.
    #[test]
    #[should_panic(expected = "constraints inconsistent with polynomial")]
    fn poly_quotient_panics_on_inconsistent_constraints() {
        // f(X) = X² + 1, but lie about f(2): truth is 5, we'll claim 99.
        let f = UnivariatePoly::new(vec![Fp::one(), Fp::zero(), Fp::one()]);
        let points = [(Fp::new(2), Fp::new(99))];
        let _ = poly_quotient(&f, &points);
    }

    /// Helper-level check: the worked example's `q · V_S + L_S == p`
    /// identity, plus a manual confirmation that `(p − L_S)` vanishes on `S`.
    /// This is the constructive-proof check for the Completeness clause.
    #[test]
    fn worked_example_identity_holds() {
        let p = UnivariatePoly::new(vec![Fp::one(), Fp::zero(), Fp::one()]);
        let points = [
            (Fp::new(0), Fp::new(1)),
            (Fp::new(1), Fp::new(2)),
        ];

        let l_s = lagrange_interpolate(&points);
        let v_s = vanishing_poly(&points);

        // p − L_S vanishes on S.
        let diff = p.clone() - l_s.clone();
        for &(x_i, _) in &points {
            assert_eq!(diff.evaluate(x_i), Fp::zero());
        }

        // q · V_S + L_S = p.
        let q = poly_quotient(&p, &points);
        let reconstructed = q * v_s + l_s;
        assert_eq!(reconstructed, p);
    }
}
