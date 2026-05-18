//! Lagrange interpolation: recover a polynomial from `(x_i, y_i)` evaluation pairs.
//!
//! Given `n` distinct points `(x_0, y_0), (x_1, y_1), ..., (x_{n-1}, y_{n-1})`
//! with all `x_i` distinct, there is a **unique** polynomial of degree `< n`
//! that passes through all of them. Lagrange's formula constructs it
//! explicitly:
//!
//! ```text
//! p(X) = sum_{i=0..n} y_i · L_i(X)
//! ```
//!
//! where
//!
//! ```text
//!            prod_{j != i} (X - x_j)
//! L_i(X) = -------------------------
//!            prod_{j != i} (x_i - x_j)
//! ```
//!
//! `L_i(x_i) = 1` (numerator and denominator are equal at `X = x_i`) and
//! `L_i(x_k) = 0` for `k != i` (numerator has a `(X - x_k)` factor that
//! vanishes). So `p(x_i) = y_i` for every `i`. Uniqueness comes from the
//! degree bound: any two polynomials of degree `< n` agreeing at `n` points
//! must be equal (their difference has too many roots).
//!
//! ## Two recovery paths in this crate
//!
//! 1. **Arbitrary points** ([`lagrange_interpolate`]): the formula above,
//!    `O(n^2)` field operations. Works on any `n` distinct points; this is
//!    what you want when the evaluation points aren't a smooth coset (e.g.,
//!    when decoding from a subset of received values).
//!
//! 2. **Smooth domain** (use `fft::ifft_on_domain`): if the points are a
//!    smooth coset `L`, the inverse FFT recovers coefficients in
//!    `O(n log n)`. This is the fast path for "decode the codeword" — you
//!    have all `n` values on `L` and want the underlying polynomial.
//!
//! The decoder in `decode.rs` uses (2) when the codeword is intact and
//! switches to (1) in the partial-evaluation case.
//!
//! ## Worked example
//!
//! Interpolate the points `(1, 1), (2, 4), (3, 9)` (which lie on `p(X) = X^2`).
//! Lagrange basis:
//!
//! ```text
//! L_0(X) = (X-2)(X-3) / ((1-2)(1-3)) = (X-2)(X-3) / 2
//! L_1(X) = (X-1)(X-3) / ((2-1)(2-3)) = (X-1)(X-3) / -1
//! L_2(X) = (X-1)(X-2) / ((3-1)(3-2)) = (X-1)(X-2) / 2
//! ```
//!
//! `p(X) = 1·L_0(X) + 4·L_1(X) + 9·L_2(X)`. Expanding:
//!
//! ```text
//! L_0(X) = (X^2 - 5X + 6) / 2
//! L_1(X) = (X^2 - 4X + 3) · -1 = -X^2 + 4X - 3
//! L_2(X) = (X^2 - 3X + 2) / 2
//! ```
//!
//! `p(X) = (X^2 - 5X + 6)/2 - 4·(X^2 - 4X + 3) + 9·(X^2 - 3X + 2)/2`
//!       = `(1/2 - 4 + 9/2)·X^2 + (-5/2 + 16 - 27/2)·X + (3 - 12 + 9)`
//!       = `1·X^2 + 0·X + 0 = X^2` ✓
//!
//! In a prime field, "1/2" means `Fp::new(2).inverse()`. The arithmetic is
//! the same; the only translation is replacing `1/k` with the modular
//! inverse.

use crate::field::Fp;
use crate::polynomial::UnivariatePoly;

/// Construct the unique polynomial of degree `< points.len()` that passes
/// through all the given `(x, y)` pairs.
///
/// Preconditions:
/// - All `x` coordinates are distinct (else the denominators below vanish).
/// - `points.len() >= 1`.
///
/// Cost: `O(n^2)` field operations. Fine for the educational sizes we use
/// here (`n <= 64` or so). For `n` much larger and structured points, use
/// `fft::ifft_on_domain`.
///
/// Panics if the input is empty or contains a duplicate `x` coordinate.
pub fn lagrange_interpolate(points: &[(Fp, Fp)]) -> UnivariatePoly {
    // TODO: build the unique degree-`< n` polynomial through all points.
    //   1. For each i, build the i-th Lagrange basis poly `L_i(X)`:
    //      numerator `∏_{j≠i} (X - x_j)` divided by scalar `∏_{j≠i} (x_i - x_j)`.
    //      `L_i(x_i) = 1` and `L_i(x_k) = 0` for k ≠ i by construction.
    //   2. Accumulate `result += y_i · L_i(X)` so `result(x_i) = y_i` for every i.
    // See the worked example with `(1,1), (2,4), (3,9) → X²` in the module docs.
    //
    // Reference implementation below.
    assert!(!points.is_empty(), "points must not be empty");

    // Distinctness check on x coordinates. O(n²) but n is small for our use.
    let n = points.len();
    for i in 0..n {
        for j in (i + 1)..n {
            assert_ne!(
                points[i].0, points[j].0,
                "duplicate x coordinate at indices {} and {}",
                i, j
            );
        }
    }

    let mut result = UnivariatePoly::zero();
    for i in 0..n {
        let mut numerator = UnivariatePoly::one(); // polynomial accumulator, starts at 1
        let mut denominator = Fp::one(); // scalar accumulator, starts at 1

        for j in 0..n {
            if j == i {
                continue;
            }
            let xj = points[j].0;
            
            numerator = numerator * UnivariatePoly::new(vec![-xj, Fp::one()]);
            denominator = denominator * (points[i].0 - xj);
        }


        let denom_inv = denominator
            .inverse()
            .expect("distinct x_i implies non-zero denominator");

        let scaled = scalar_mul(&numerator, points[i].1 * denom_inv);
        result = result + scaled;
    }
    result
}

/// Multiply every coefficient of `poly` by `scalar`. Returns a new polynomial.
///
/// Useful inside [`lagrange_interpolate`]; exposed publicly because the
/// decoder also wants it.
pub fn scalar_mul(poly: &UnivariatePoly, scalar: Fp) -> UnivariatePoly {
    // TODO: map each coefficient through `* scalar`, then wrap with
    // UnivariatePoly::new (which strips trailing zeros — important when
    // scalar == zero!).

    // new: `lagrange_interpolate` calls this, so it needs a real body.
    // Wrap with `UnivariatePoly::new` (not `from_coeffs_unstripped`) so that
    // `scalar = Fp::zero()` correctly canonicalises to the zero polynomial
    // instead of "a polynomial with n trailing zeros".
    let scaled: Vec<Fp> = poly.coeffs().iter().map(|&c| c * scalar).collect();
    UnivariatePoly::new(scaled)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use super::*;

    #[test]
    fn interpolate_constant() {
        // TODO: one point (5, 7). The interpolating polynomial is the constant 7.
        // Assert poly == UnivariatePoly::new(vec![Fp::new(7)]).
        assert_eq!(lagrange_interpolate(&[(Fp::new(5), Fp::new(7))]), UnivariatePoly::new(vec![Fp::new(7)]));
    }

    #[test]
    fn interpolate_linear() {
        // TODO: two points (0, 1) and (1, 3). The interpolating polynomial is
        // p(X) = 1 + 2X.
        assert_eq!(lagrange_interpolate(&[(Fp::new(0), Fp::new(1)), (Fp::new(1), Fp::new(3))]), UnivariatePoly::new(vec![Fp::new(1), Fp::new(2)]));
    }

    #[test]
    fn interpolate_x_squared() {
        // TODO: three points (1, 1), (2, 4), (3, 9). Result must be X^2,
        // i.e., UnivariatePoly::new(vec![Fp::zero(), Fp::zero(), Fp::one()]).
        assert_eq!(lagrange_interpolate(&[(Fp::new(1), Fp::new(1)), (Fp::new(2), Fp::new(4)), (Fp::new(3), Fp::new(9))]), UnivariatePoly::new(vec![Fp::zero(), Fp::zero(), Fp::one()]));
    }

    #[test]
    fn interpolate_recovers_random_polynomial() {
        // TODO:
        //   1. Build a random degree-5 polynomial p (6 coefficients).
        //   2. Pick 6 distinct x's.
        //   3. Compute y_i = p.evaluate(x_i).
        //   4. lagrange_interpolate(&points) should equal p.
        // Use a seeded RNG.
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        // 1. Random degree-5 polynomial (6 coefficients).
        let mut rng = StdRng::seed_from_u64(42);
        let coefficients: Vec<Fp> = (0..6).map(|_| Fp::random(&mut rng)).collect();
        let p = UnivariatePoly::new(coefficients);

        // 2. Pick 6 distinct x's: x_i = 1, 2, 3, 4, 5, 6.
        // 3. Compute y_i = p(x_i) by EVALUATING the polynomial — this is the
        //    step that connects the points to the polynomial. (The original
        //    bug was pairing x_i with a coefficient, which had no relation
        //    to p's value at x_i.)
        let points: Vec<(Fp, Fp)> = (1..=6u64)
            .map(|i| {
                let x = Fp::new(i);
                (x, p.evaluate(x))
            })
            .collect();

        // 4. Interpolating those (x_i, p(x_i)) pairs must recover p exactly,
        //    since p is the unique polynomial of degree < 6 passing through
        //    6 distinct points (Polynomial Interpolation Theorem).
        assert_eq!(lagrange_interpolate(&points), p);
    }

    #[test]
    #[should_panic]
    fn interpolate_with_duplicate_x_panics() {
        // TODO: two points with the same x but different y. Should panic.
        let _ = lagrange_interpolate(&[(Fp::new(1), Fp::new(2)), (Fp::new(1), Fp::new(3))]);
    }

    #[test]
    fn scalar_mul_by_zero_is_zero() {
        // TODO: scalar_mul of any poly by Fp::zero() returns the zero polynomial.
        // (This is why scalar_mul has to call UnivariatePoly::new rather than
        // from_coeffs_unstripped — to canonicalize.)
        scalar_mul(&UnivariatePoly::new(vec![Fp::new(1), Fp::new(2)]), Fp::zero());
    }
}
