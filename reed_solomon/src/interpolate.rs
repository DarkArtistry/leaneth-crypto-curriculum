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
    // TODO:
    //   1. Assert points.is_empty() == false.
    //   2. (Optional, but recommended.) Sanity-check that all x's are distinct.
    //      Since duplicates would otherwise produce a divide-by-zero in step 4,
    //      catching them early is friendlier. O(n^2) check is fine for our sizes.
    //   3. Build the result by accumulating term by term:
    //        result = UnivariatePoly::zero();
    //        for i in 0..n:
    //            // Numerator: prod_{j != i} (X - x_j)
    //            // Start with the constant polynomial 1, and multiply in (X - x_j) for each j != i.
    //            // The polynomial (X - x_j) is UnivariatePoly::new(vec![-x_j, Fp::one()]).
    //            let mut numerator = UnivariatePoly::one();
    //            let mut denominator = Fp::one();
    //            for j in 0..n if j != i:
    //                let (xj, _) = points[j];
    //                numerator = numerator * UnivariatePoly::new(vec![-xj, Fp::one()]);
    //                denominator = denominator * (points[i].0 - xj);
    //            // L_i(X) = numerator * (denominator^-1)
    //            let denom_inv = denominator.inverse().expect("distinct x_i implies non-zero denominator");
    //            let scaled_numerator = scalar_mul(&numerator, points[i].1 * denom_inv);
    //            result = result + scaled_numerator;
    //        result
    //
    // Helper `scalar_mul(poly, scalar)`: multiply every coefficient by scalar.
    // You can write this as a free function in this module, or inline the loop.
    let _ = points;
    todo!()
}

/// Multiply every coefficient of `poly` by `scalar`. Returns a new polynomial.
///
/// Useful inside [`lagrange_interpolate`]; exposed publicly because the
/// decoder also wants it.
pub fn scalar_mul(poly: &UnivariatePoly, scalar: Fp) -> UnivariatePoly {
    // TODO: map each coefficient through `* scalar`, then wrap with
    // UnivariatePoly::new (which strips trailing zeros — important when
    // scalar == zero!).
    let _ = (poly, scalar);
    todo!()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolate_constant() {
        // TODO: one point (5, 7). The interpolating polynomial is the constant 7.
        // Assert poly == UnivariatePoly::new(vec![Fp::new(7)]).
        todo!()
    }

    #[test]
    fn interpolate_linear() {
        // TODO: two points (0, 1) and (1, 3). The interpolating polynomial is
        // p(X) = 1 + 2X.
        todo!()
    }

    #[test]
    fn interpolate_x_squared() {
        // TODO: three points (1, 1), (2, 4), (3, 9). Result must be X^2,
        // i.e., UnivariatePoly::new(vec![Fp::zero(), Fp::zero(), Fp::one()]).
        todo!()
    }

    #[test]
    fn interpolate_recovers_random_polynomial() {
        // TODO:
        //   1. Build a random degree-5 polynomial p (6 coefficients).
        //   2. Pick 6 distinct x's.
        //   3. Compute y_i = p.evaluate(x_i).
        //   4. lagrange_interpolate(&points) should equal p.
        // Use a seeded RNG.
        todo!()
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
        todo!()
    }
}
