//! Univariate polynomials over `F_p`, in **coefficient form**.
//!
//! ## Coefficient form vs evaluation form
//!
//! In sumcheck (objective 1) we stored multilinear polynomials in **evaluation
//! form** — a vector of length `2^v` listing the polynomial's value at every
//! corner of the Boolean hypercube. That representation makes the
//! `fix_first_variable` operation a one-line linear interpolation.
//!
//! For Reed-Solomon we want to **evaluate the polynomial at lots of different
//! points on a smooth domain `L`**, so coefficient form is the right primary
//! representation:
//!
//! ```text
//! p(X) = a_0 + a_1·X + a_2·X^2 + ... + a_{d-1}·X^{d-1}
//! ```
//!
//! stored as `coeffs = [a_0, a_1, ..., a_{d-1}]` (ascending degree, `a_0` first).
//! Evaluation at a point `x` is one Horner's-rule pass; FFT (in `fft.rs`)
//! takes coefficients to evaluations on `L` in `O(n log n)`.
//!
//! ## Canonicalization (degree)
//!
//! `coeffs` always has its trailing zeros stripped, so the **last entry is
//! non-zero or the vector is empty**. The empty vector represents the zero
//! polynomial. This keeps `degree()` honest: a length-3 `coeffs` always means
//! a degree-2 polynomial, never "degree 2 with the leading coefficient set
//! to zero by accident".
//!
//! ## Worked example: `p(X) = 1 + 2X + 3X^2`
//!
//! ```text
//! coeffs   = [1, 2, 3]
//! degree() = Some(2)
//! evaluate at X = 5 (Horner from the top down):
//!   acc = 3
//!   acc = acc * 5 + 2 = 17
//!   acc = acc * 5 + 1 = 86
//! check: 1 + 2·5 + 3·25 = 1 + 10 + 75 = 86 ✓
//! ```
//!
//! Horner's rule is `O(d)` multiplications and `O(d)` additions, vs `O(d^2)`
//! if you compute powers of `X` independently and sum.

use crate::field::Fp;
use std::ops::{Add, Mul, Sub};

/// A univariate polynomial over `F_p`, stored in coefficient form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnivariatePoly {
    /// `coeffs[i]` is the coefficient of `X^i`. Trailing zeros are stripped:
    /// either `coeffs.is_empty()` (the zero polynomial) or `*coeffs.last().unwrap() != Fp::zero()`.
    coeffs: Vec<Fp>,
}

impl UnivariatePoly {
    /// Construct a polynomial from its coefficients, ascending degree.
    /// Trailing zeros are stripped to keep the representation canonical.
    pub fn new(mut coeffs: Vec<Fp>) -> Self {
        // TODO:
        //   1. Strip trailing zeros: while the last element exists and equals Fp::zero(), pop.
        //   2. Return Self { coeffs }.
        while coeffs.last() == Some(&Fp::zero()) {
            coeffs.pop();
        }
        Self{coeffs}
    }

    /// Construct from coefficients without stripping trailing zeros.
    ///
    /// Useful in FFT, where we want to keep length `n` even if some leading
    /// coefficients happened to be zero. **Caller is responsible for the
    /// invariant** if they bypass this method's stripping — and `degree()`
    /// may report a higher-than-true degree as a result.
    ///
    /// Prefer [`UnivariatePoly::new`] in normal use.
    pub fn from_coeffs_unstripped(coeffs: Vec<Fp>) -> Self {
        Self { coeffs }
    }

    /// The zero polynomial.
    pub fn zero() -> Self {
        // TODO: empty coeffs vector.
        Self{coeffs: Vec::new()}
    }

    /// The constant polynomial `1`.
    pub fn one() -> Self {
        // TODO: coeffs = [Fp::one()].
        Self{coeffs: vec![Fp::one()]}
    }

    /// True iff this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        // TODO: trivially, `self.coeffs.is_empty()` (given the canonical form).
        self.coeffs.is_empty()
    }

    /// Degree of the polynomial.
    ///
    /// `None` for the zero polynomial (its degree is conventionally `-∞`).
    /// `Some(d)` for a polynomial whose leading non-zero coefficient is `a_d`.
    pub fn degree(&self) -> Option<usize> {
        // TODO: if coeffs is empty, None. Else Some(coeffs.len() - 1).
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Read-only access to the coefficient vector.
    pub fn coeffs(&self) -> &[Fp] {
        &self.coeffs
    }

    /// Evaluate the polynomial at a single point `x`, using Horner's method.
    ///
    /// Horner's rule rewrites `a_0 + a_1·X + a_2·X^2 + ... + a_d·X^d` as
    /// `a_0 + X·(a_1 + X·(a_2 + ... + X·a_d))` and computes the inner
    /// expression first — `d` multiplications, `d` additions. No precomputed
    /// powers of `X`.
    pub fn evaluate(&self, x: Fp) -> Fp {
        // TODO:
        //   - For the zero polynomial (empty coeffs), return Fp::zero().
        //   - Otherwise, fold from the highest-degree coefficient down:
        //       result = 0
        //       for c in coeffs.iter().rev():
        //           result = result * x + c
        //       return result
        //
        //   Use `iter().rev()` for the reverse-order traversal.
        if self.coeffs.is_empty() {
            Fp::zero()
        } else {
            let mut result = Fp::zero();
            for c in self.coeffs.iter().rev() {
                result = result * x + *c;
            }
            result
        }

    }
}

// ============================================================================
// Polynomial arithmetic. We need add and sub for the decoder, and mul for the
// encoder's correctness tests (and for Berlekamp-Welch if you tackle it).
// ============================================================================

impl Add for UnivariatePoly {
    type Output = UnivariatePoly;

    fn add(self, rhs: UnivariatePoly) -> UnivariatePoly {
        // TODO: pointwise add coefficients up to the longer length, then re-canonicalize
        // via UnivariatePoly::new.
        //
        // Pseudocode:
        //   let len = max(self.coeffs.len(), rhs.coeffs.len());
        //   let mut out = Vec::with_capacity(len);
        //   for i in 0..len:
        //       a = self.coeffs.get(i).copied().unwrap_or(Fp::zero());
        //       b = rhs.coeffs.get(i).copied().unwrap_or(Fp::zero());
        //       out.push(a + b);
        //   UnivariatePoly::new(out)

        let len = std::cmp::max(self.coeffs.len(), rhs.coeffs.len());
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).copied().unwrap_or(Fp::zero());
            let b = rhs.coeffs.get(i).copied().unwrap_or(Fp::zero());
            out.push(a + b);
        }
        UnivariatePoly::new(out)
    }
}

impl Sub for UnivariatePoly {
    type Output = UnivariatePoly;

    fn sub(self, rhs: UnivariatePoly) -> UnivariatePoly {
        // TODO: same pattern as Add but with `a - b`.
        let len = std::cmp::max(self.coeffs.len(), rhs.coeffs.len());
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).copied().unwrap_or(Fp::zero());
            let b = rhs.coeffs.get(i).copied().unwrap_or(Fp::zero());
            out.push(a - b);
        }
        UnivariatePoly::new(out)
    }
}

impl Mul for UnivariatePoly {
    type Output = UnivariatePoly;

    fn mul(self, rhs: UnivariatePoly) -> UnivariatePoly {
        // TODO: schoolbook multiplication.
        //
        // If either is zero, return zero (the result has empty coeffs).
        // Otherwise: result has degree (self.deg + rhs.deg). Allocate
        // `vec![Fp::zero(); self_len + rhs_len - 1]` and accumulate
        //   out[i + j] += self.coeffs[i] * rhs.coeffs[j]
        // for all i, j. Wrap in UnivariatePoly::new to canonicalize.
        //
        // (For RS encoding we don't need fast multiplication; schoolbook
        // is `O(d^2)`, fine for educational sizes. FFT-based multiply is
        // an optional polish.)
        if self.is_zero() || rhs.is_zero() {
            UnivariatePoly::zero()
        } else {
            let mut out = vec![Fp::zero(); self.coeffs.len() + rhs.coeffs.len() - 1];

            for i in 0..self.coeffs.len() {
                for j in 0..rhs.coeffs.len() {
                    out[i + j] += self.coeffs[i] * rhs.coeffs[j];
                }
            }

            UnivariatePoly::new(out)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_strips_trailing_zeros() {
        // TODO: build with `[1, 2, 0, 0]`; assert coeffs() returns `[1, 2]`
        // and degree() == Some(1).
        UnivariatePoly::new(vec![Fp::new(1), Fp::new(2), Fp::zero(), Fp::zero()]);
        assert_eq!(UnivariatePoly::new(vec![Fp::new(1), Fp::new(2), Fp::zero(), Fp::zero()]).coeffs(), vec![Fp::new(1), Fp::new(2)]);
        assert_eq!(UnivariatePoly::new(vec![Fp::new(1), Fp::new(2), Fp::zero(), Fp::zero()]).degree(), Some(1));
    }

    #[test]
    fn zero_polynomial_has_no_degree() {
        // TODO: UnivariatePoly::zero().degree() == None and is_zero() == true.
        assert_eq!(UnivariatePoly::zero().degree(), None);
        assert!(UnivariatePoly::zero().is_zero());

    }

    #[test]
    fn evaluate_constant() {
        // TODO: p = UnivariatePoly::new(vec![Fp::new(7)]); evaluate at any x → 7.
        assert_eq!(UnivariatePoly::new(vec![Fp::new(7)]).evaluate(Fp::new(1)), Fp::new(7));
        assert_eq!(UnivariatePoly::new(vec![Fp::new(7)]).evaluate(Fp::new(2)), Fp::new(7));
    }

    #[test]
    fn evaluate_linear() {
        // TODO: p(X) = 1 + 2X; p.evaluate(Fp::new(5)) == Fp::new(11).
        assert_eq!(UnivariatePoly::new(vec![Fp::one(), Fp::new(2)]).evaluate(Fp::new(5)), Fp::new(11));
    }

    #[test]
    fn evaluate_matches_horner_worked_example() {
        // TODO: p(X) = 1 + 2X + 3X^2; p.evaluate(Fp::new(5)) == Fp::new(86).
        // (See the worked example in the module docstring.)
        assert_eq!(UnivariatePoly::new(vec![Fp::one(), Fp::new(2), Fp::new(3)]).evaluate(Fp::new(5)), Fp::new(86));
    }

    #[test]
    fn evaluate_at_zero_is_constant_term() {
        // TODO: any non-zero p, p.evaluate(Fp::zero()) == p.coeffs()[0].
        assert_eq!(UnivariatePoly::new(vec![Fp::one(), Fp::new(2), Fp::new(3)]).evaluate(Fp::zero()), Fp::one());
        assert_eq!(UnivariatePoly::new(vec![Fp::one(), Fp::new(10), Fp::new(11)]).evaluate(Fp::zero()), Fp::one());
    }

    #[test]
    fn add_then_sub_is_identity() {
        // TODO: p = some non-trivial poly, q = some other; (p + q) - q == p.
        // Equality is structural — both sides should have the same canonical coeffs.
        let p = UnivariatePoly::new(vec![Fp::one(), Fp::new(2), Fp::new(3)]);
        let q = UnivariatePoly::new(vec![Fp::one(), Fp::new(10), Fp::new(11)]);

        let result = (p.clone() + q.clone()) - q;

        assert_eq!(result, p);
        assert_eq!(result.coeffs(), &[Fp::one(), Fp::new(2), Fp::new(3)]);
    }

    #[test]
    fn mul_degrees_add() {
        // TODO: p of degree 2, q of degree 3; (p * q).degree() == Some(5)
        // (assuming non-zero leading coefficients).
        let p = UnivariatePoly::new(vec![Fp::one(), Fp::new(2)]);
        let q = UnivariatePoly::new(vec![Fp::one(), Fp::new(3), Fp::new(4)]);

        let result = p.clone() * q.clone();

        assert_eq!(result.degree(), Some(3));
    }

    #[test]
    fn mul_evaluates_pointwise() {
        // TODO: pick p, q, x. Assert (p * q).evaluate(x) == p.evaluate(x) * q.evaluate(x).
        // This is a strong correctness check for `mul`.
        let p = UnivariatePoly::new(vec![Fp::one(), Fp::new(2)]);
        let q = UnivariatePoly::new(vec![Fp::one(), Fp::new(3), Fp::new(4)]);
        let x = Fp::new(5);

        let product = p.clone() * q.clone();

        assert_eq!(product.evaluate(x), p.evaluate(x) * q.evaluate(x));
    }
}
