//! Multilinear polynomials in evaluation form over the boolean hypercube `{0,1}^v`.
//!
//! ## What is a multilinear polynomial?
//!
//! A polynomial `p(X_1, ..., X_n)` over a field `F` is **multilinear** if every
//! variable appears with degree at most 1. No `x_1^2`, no `x_3^7`, just monomials
//! that mention each variable zero or one times.
//!
//! Example for n = 2:
//!
//! ```text
//! p(x_1, x_2) = a + b·x_1 + c·x_2 + d·x_1·x_2
//! ```
//!
//! Total degree can be up to n, but **per-variable degree exactly 1** is what
//! makes it well-behaved for interactive-proof protocols (each variable, viewed
//! alone, is a straight line — see Step C of "Why evaluation form?" below).
//!
//! ## What is the Multilinear Extension (MLE)?
//!
//! Let `f : {0,1}^n → F` be any function — think of it as a lookup table of
//! `2^n` field values, one per boolean tuple.
//!
//! The **Multilinear Extension** of f, written `f~`, is the unique multilinear
//! polynomial over `F^n` that agrees with f on the boolean hypercube:
//!
//! ```text
//! f~(b) = f(b)    for every b in {0,1}^n
//! ```
//!
//! The explicit formula is Lagrange interpolation specialized to the cube:
//!
//! ```text
//! f~(x) = sum over b in {0,1}^n of  f(b) · prod_i (b_i·x_i + (1 - b_i)·(1 - x_i))
//! ```
//!
//! The factor `prod_i (b_i·x_i + (1 - b_i)·(1 - x_i))` is the multilinear basis
//! polynomial `chi_b(x)` — it equals 1 at `b` and 0 at every other boolean point.
//! This is exactly the formula `evaluate` implements.
//!
//! **Uniqueness:** any two multilinear polynomials agreeing on all `2^n` boolean
//! points are identical. Their difference would be a multilinear polynomial that
//! vanishes on the entire hypercube — which forces all coefficients to zero.
//!
//! ## Why MLEs show up everywhere in cryptography
//!
//! The **sum-check protocol** (Lund-Fortnow-Karloff-Nisan, 1990) lets a prover
//! convince a verifier that
//!
//! ```text
//! sum over x in {0,1}^n of g(x) = C
//! ```
//!
//! for a low-degree multivariate polynomial `g`, using only `O(n)` rounds. Most
//! statements we actually care about — circuit satisfaction, matrix multiplication,
//! R1CS, lookups — are naturally sums (or products) over the boolean hypercube.
//! By replacing the boolean function with its MLE `f~`, the statement becomes
//! a low-degree polynomial identity that sum-check can verify.
//!
//! This is why MLE-centric designs include:
//!
//! - **GKR** (circuit verification)
//! - **Spartan**, **HyperPlonk**, **Jolt** (modern multilinear SNARKs)
//! - many polynomial IOPs and SNARKs more broadly
//!
//! The "extension" step is what lets algebraic tools (Schwartz-Zippel, sum-check)
//! talk to combinatorial objects defined on the boolean cube.
//!
//! ## Storage convention
//!
//! Stored as a `Vec<Fp>` of length `2^v`. **Read each index as binary, with
//! `x_1` as the least significant bit.**
//!
//! Quick example: `evals[5]` for `v = 3`:
//!
//! ```text
//!     5 = 1 0 1  (binary)
//!         |  |  |
//!         x_3 x_2 x_1     →   evals[5] = f(1, 0, 1)
//! ```
//!
//! For `v = 2`, the order is:
//! - `evals[0] = f(0, 0)`         (i = 00, x_1 = 0, x_2 = 0)
//! - `evals[1] = f(1, 0)`         (i = 01, x_1 = 1, x_2 = 0)
//! - `evals[2] = f(0, 1)`         (i = 10, x_1 = 0, x_2 = 1)
//! - `evals[3] = f(1, 1)`         (i = 11, x_1 = 1, x_2 = 1)
//!
//! For `v = 3`, the order is:
//! - `evals[0] = f(0, 0, 0)`      (i = 000)
//! - `evals[1] = f(1, 0, 0)`      (i = 001)
//! - `evals[2] = f(0, 1, 0)`      (i = 010)
//! - `evals[3] = f(1, 1, 0)`      (i = 011)
//! - `evals[4] = f(0, 0, 1)`      (i = 100)
//! - `evals[5] = f(1, 0, 1)`      (i = 101)
//! - `evals[6] = f(0, 1, 1)`      (i = 110)
//! - `evals[7] = f(1, 1, 1)`      (i = 111)
//!
//! This LSB-first convention matches Plonky3 and is consistent with how
//! `fix_first_variable` reduces adjacent pairs of evaluations.
//!
//! ## Why evaluation form? — worked example with a real polynomial
//!
//! Take this specific multilinear polynomial:
//!
//! ```text
//! f(x_1, x_2, x_3) = 1 + 2·x_1 + 3·x_2 + 4·x_3
//! ```
//!
//! ### Step A — Where does `evals` come from?
//!
//! Plug each of the 8 boolean corners into `f`:
//!
//! ```text
//! f(0, 0, 0) = 1 + 0 + 0 + 0 = 1
//! f(1, 0, 0) = 1 + 2 + 0 + 0 = 3
//! f(0, 1, 0) = 1 + 0 + 3 + 0 = 4
//! f(1, 1, 0) = 1 + 2 + 3 + 0 = 6
//! f(0, 0, 1) = 1 + 0 + 0 + 4 = 5
//! f(1, 0, 1) = 1 + 2 + 0 + 4 = 7
//! f(0, 1, 1) = 1 + 0 + 3 + 4 = 8
//! f(1, 1, 1) = 1 + 2 + 3 + 4 = 10
//! ```
//!
//! Lay them out in LSB-first order — index `i` interpreted as binary with
//! `x_1` being the LSB:
//!
//! ```text
//! evals = [1, 3, 4, 6, 5, 7, 8, 10]
//!
//! evals[0] = f(0,0,0) = 1     (i = 000)
//! evals[1] = f(1,0,0) = 3     (i = 001)
//! evals[2] = f(0,1,0) = 4     (i = 010)
//! evals[3] = f(1,1,0) = 6     (i = 011)
//! evals[4] = f(0,0,1) = 5     (i = 100)
//! evals[5] = f(1,0,1) = 7     (i = 101)
//! evals[6] = f(0,1,1) = 8     (i = 110)
//! evals[7] = f(1,1,1) = 10    (i = 111)
//! ```
//!
//! That's it. **From here on, we work with `evals`, not the polynomial expression.**
//!
//! ### Step B — Sum over the cube, two ways
//!
//! **Eval form:** add the 8 numbers in the vector.
//!
//! ```text
//! H = 1 + 3 + 4 + 6 + 5 + 7 + 8 + 10 = 44
//! ```
//!
//! **Coefficient form:** plug all 8 corners into the expression `1 + 2x_1 + 3x_2 + 4x_3`,
//! get back the 8 numbers, then sum. That's exactly the work in Step A. Eval form
//! skips it because we already did it once and stored the results.
//!
//! ### Step C — Fix x_1 = r, two ways
//!
//! "Fix x_1 = r" means: take the v-variable polynomial and replace x_1 with a
//! specific value r, getting a (v-1)-variable polynomial. This is the move that
//! drives sumcheck forward — each round shrinks the polynomial by one variable.
//!
//! **r can be any field element** — not just 0 or 1. Sumcheck's verifier picks
//! r uniformly at random from the entire field for soundness.
//!
//! #### Why the (1 - r)·A + r·B formula?
//!
//! Multilinear means **the polynomial is a straight line along each axis**.
//! Pick any "rest" — say (x_2, x_3) = (0, 0). Then `f(x_1, 0, 0) = 1 + 2·x_1`
//! is just a line in x_1, passing through two known points:
//!
//! ```text
//! at x_1 = 0:  f(0, 0, 0) = 1     (this is evals[0])
//! at x_1 = 1:  f(1, 0, 0) = 3     (this is evals[1])
//! ```
//!
//! The line equation `(1 - r)·A + r·B` slides between A and B by fraction r —
//! at r=0 you get A, at r=1 you get B, and for any other r you extrapolate
//! along the same line.
//!
//! #### Sanity check with r = 5
//!
//! Direct (using the polynomial):
//!
//! ```text
//! f(5, 0, 0) = 1 + 2·5 = 11
//! ```
//!
//! Via the interpolation formula on the eval pair:
//!
//! ```text
//! f(5, 0, 0) = (1 - 5)·evals[0] + 5·evals[1]
//!            = (-4)·1 + 5·3
//!            = -4 + 15
//!            = 11
//! ```
//!
//! **Same answer, 11.** The formula is just the line equation, valid for any r.
//!
//! #### Why pairs?
//!
//! Each pair `(evals[2k], evals[2k+1])` shares the same "rest" — same
//! `(x_2, ..., x_v)` — but differs in x_1 (LSB). So each pair gives the two
//! endpoints of one line:
//!
//! ```text
//! (evals[0], evals[1]) = (1, 3)   → line at (x_2, x_3) = (0, 0)
//! (evals[2], evals[3]) = (4, 6)   → line at (x_2, x_3) = (1, 0)
//! (evals[4], evals[5]) = (5, 7)   → line at (x_2, x_3) = (0, 1)
//! (evals[6], evals[7]) = (8, 10)  → line at (x_2, x_3) = (1, 1)
//! ```
//!
//! Apply `(1 - r)·left + r·right` to each pair to slide along the corresponding
//! line at value r:
//!
//! ```text
//! new_evals[0] = (1 - r)·1 + r·3  = 1 + 2r       → f(r, 0, 0)
//! new_evals[1] = (1 - r)·4 + r·6  = 4 + 2r       → f(r, 1, 0)
//! new_evals[2] = (1 - r)·5 + r·7  = 5 + 2r       → f(r, 0, 1)
//! new_evals[3] = (1 - r)·8 + r·10 = 8 + 2r       → f(r, 1, 1)
//! ```
//!
//! That's the new polynomial in evaluation form: `[1+2r, 4+2r, 5+2r, 8+2r]`.
//!
//! #### Cross-check via coefficient form
//!
//! Substitute x_1 = r directly in `1 + 2x_1 + 3x_2 + 4x_3`:
//!
//! ```text
//! f(r, x_2, x_3) = (1 + 2r) + 3·x_2 + 4·x_3
//! ```
//!
//! Then evaluate at the 4 corners of {0,1}^2:
//!
//! ```text
//! f(r, 0, 0) = 1 + 2r
//! f(r, 1, 0) = 4 + 2r
//! f(r, 0, 1) = 5 + 2r
//! f(r, 1, 1) = 8 + 2r
//! ```
//!
//! **Same answers as the eval form.** Eval form did it with 4 pair-interpolations
//! over a vector. Coefficient form did it via symbolic substitution + 4 corner
//! evaluations. Both correct, eval form is mechanically simpler.
//!
//! ### Step D — What about non-boolean inputs?
//!
//! At the very end of sumcheck, the verifier needs `f(r_1, r_2, r_3)` for some
//! random field elements (not 0 or 1). The [`MultilinearPoly::evaluate`] method
//! reconstructs that on demand using the MLE formula — see the formula on its
//! docstring. We never need to materialize the polynomial expression
//! `1 + 2x_1 + 3x_2 + 4x_3` to do this.

use crate::field::Fp;

/// A multilinear polynomial in `num_vars` variables, stored as evaluations on `{0,1}^num_vars`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultilinearPoly {
    /// Number of variables `v`.
    pub num_vars: usize,
    /// Evaluations on `{0,1}^v` in the convention described in the module docs.
    pub evals: Vec<Fp>,
}

impl MultilinearPoly {
    /// Construct a multilinear polynomial from its evaluations on `{0,1}^num_vars`.
    ///
    /// Panics if `evals.len() != 2^num_vars`.
    pub fn new(num_vars: usize, evals: Vec<Fp>) -> Self {
        // TODO: validate that evals.len() == 1 << num_vars (i.e., 2^num_vars).
        // Use assert_eq! with a clear message.
        assert_eq!(evals.len(), 1 << num_vars, "evaluations length must be 2^num_vars");
        MultilinearPoly{ num_vars, evals }
    }

    /// Evaluate the multilinear extension at a point in `F^v`.
    ///
    /// Uses the standard MLE formula:
    /// `f(r) = sum over b in {0,1}^v of f(b) * chi_b(r)` where
    /// `chi_b(r) = product over i of (b_i * r_i + (1 - b_i) * (1 - r_i))`.
    ///
    /// Panics if `point.len() != self.num_vars`.
    pub fn evaluate(&self, point: &[Fp]) -> Fp {
        // TODO:
        //   1. Validate point length.
        //   2. For each b in 0..2^num_vars:
        //        bit_decompose b into [b_0, ..., b_{num_vars-1}]
        //        compute chi_b(point) = prod_i (b_i * point[i] + (1-b_i) * (1 - point[i]))
        //        accumulate self.evals[b] * chi_b(point) into result
        //   3. Return result.
        //
        // Hint: this is O(v · 2^v). Fine for our toy sizes (v <= 10).
        // For large v we'd use a "fix one variable at a time" approach with O(2^v) total work.
        assert_eq!(point.len(), self.num_vars);

        let mut result = Fp::zero();

        for b in 0..self.evals.len() {
            // `b` indexes one Boolean hypercube point.
            //
            // Example for 3 variables:
            //   b = 0 = 000 means point (0, 0, 0)
            //   b = 1 = 001 means point (1, 0, 0)
            //   b = 2 = 010 means point (0, 1, 0)
            //   b = 3 = 011 means point (1, 1, 0)
            //
            // `chi_b` is the selector/weight for this Boolean point.
            // It starts at 1 because we are about to multiply one factor per variable.
            let mut chi_b = Fp::one();

            for i in 0..self.num_vars {
                // Extract the i-th bit of b.
                //
                // If this bit is 1, then this Boolean point has coordinate x_i = 1.
                // If this bit is 0, then this Boolean point has coordinate x_i = 0.
                let bit_is_one = ((b >> i) & 1) == 1;

                if bit_is_one {
                    // For a Boolean coordinate equal to 1, use point[i].
                    //
                    // This factor is:
                    //   0 when point[i] = 0
                    //   1 when point[i] = 1
                    chi_b *= point[i];
                } else {
                    // For a Boolean coordinate equal to 0, use 1 - point[i].
                    //
                    // This factor is:
                    //   1 when point[i] = 0
                    //   0 when point[i] = 1
                    chi_b *= Fp::one() - point[i];
                }
            }

            // Add this Boolean value's contribution to the final evaluation:
            //
            //   contribution = f(b) * chi_b(point)
            //
            // where self.evals[b] is f(b).
            result += self.evals[b] * chi_b;
        }

        result
    }

    /// Compute the sum over the boolean hypercube: `H = sum over b in {0,1}^v of f(b)`.
    pub fn sum_over_hypercube(&self) -> Fp {
        // TODO: just sum self.evals. One line via the Sum trait.
        self.evals.iter().copied().sum()
    }

    /// Fix the first variable to a value `r`, returning a polynomial in `v-1` variables.
    ///
    /// # What it does
    ///
    /// Plugs `r` into the first slot of `f`, leaving the remaining variables free:
    ///
    /// ```text
    /// f'(x_2, x_3, ..., x_v) = f(r, x_2, x_3, ..., x_v)
    /// ```
    ///
    /// The output polynomial has one fewer variable than the input.
    ///
    /// # Why the interpolation formula works
    ///
    /// `f` is multilinear, so along the `x_1` axis it's a **straight line**
    /// (degree 1 in `x_1`). For any fixed values of the other variables, the
    /// graph of `f` against `x_1` is just a line passing through two known
    /// points: `f(0, rest)` and `f(1, rest)`.
    ///
    /// Linear interpolation between those two points gives:
    ///
    /// ```text
    /// f(r, rest) = (1 - r) * f(0, rest) + r * f(1, rest)
    /// ```
    ///
    /// (The same `(1 - t) * A + t * B` formula you'd use to slide between any
    /// two points on a line.)
    ///
    /// # In terms of `evals`
    ///
    /// Under the LSB-first convention (see module docs), pairs of evaluations
    /// that differ only in `x_1` are at **adjacent indices**: `(0, 1)`, `(2, 3)`,
    /// `(4, 5)`, etc. So the new evaluation vector is built as:
    ///
    /// ```text
    /// new_evals[i] = (1 - r) * evals[2*i] + r * evals[2*i + 1]
    /// ```
    ///
    /// for `i = 0, 1, ..., 2^(v-1) - 1`.
    ///
    /// # Concrete example with v = 3
    ///
    /// The original `f` has 8 evaluations. Fixing `x_1 = r` produces an `f'`
    /// with 4 evaluations. For instance, the new `f'(0, 1)` (with `x_2 = 0`,
    /// `x_3 = 1`) is computed from the old `evals[4]` and `evals[5]`, which
    /// store `f(0, 0, 1)` and `f(1, 0, 1)` respectively:
    ///
    /// ```text
    /// new_evals[2] = (1 - r) * evals[4] + r * evals[5]
    ///              = (1 - r) * f(0, 0, 1) + r * f(1, 0, 1)
    ///              = f(r, 0, 1)
    /// ```
    ///
    /// Panics if `self.num_vars == 0`.
    pub fn fix_first_variable(&self, r: Fp) -> Self {
        // Steps:
        //   1. Assert num_vars > 0.
        //   2. Allocate a new evals vector of length 2^(num_vars - 1).
        //   3. For each new_index in 0..new_len:
        //        old_pair_x1_zero = new_index << 1       (bit 0 = 0)
        //        old_pair_x1_one  = (new_index << 1) | 1 (bit 0 = 1)
        //        new_eval[new_index] = (1 - r) * evals[old_pair_x1_zero]
        //                                  + r * evals[old_pair_x1_one]
        //   4. Return MultilinearPoly with num_vars - 1.
        //
        // Hint: use Fp::one() - r for (1 - r).
        assert!(
            self.num_vars > 0,
            "cannot fix a variable in a zero-variable polynomial"
        );

        let new_num_vars = self.num_vars - 1;
        let new_len = 1 << new_num_vars;
        let mut new_evals = Vec::with_capacity(new_len);

        for new_index in 0..new_len {
            // Because the first variable is stored in bit 0, the two old indices
            // that differ only in the first variable are:
            //
            //   old_index_with_x1_0 = new_index shifted left by 1, with bit 0 = 0
            //   old_index_with_x1_1 = same index, but with bit 0 = 1
            //
            // Example:
            //   new_index = 2 = 10
            //   old_zero = 100
            //   old_one  = 101
            let old_index_with_x1_0 = new_index << 1;
            let old_index_with_x1_1 = old_index_with_x1_0 | 1;

            let eval_at_zero = self.evals[old_index_with_x1_0];
            let eval_at_one = self.evals[old_index_with_x1_1];

            // Interpolate between f(0, rest) and f(1, rest):
            //
            //   f(r, rest) = (1-r) * f(0, rest) + r * f(1, rest)
            let fixed_eval = (Fp::one() - r) * eval_at_zero + r * eval_at_one;

            new_evals.push(fixed_eval);
        }

        Self::new(new_num_vars, new_evals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_at_boolean_input_returns_eval() {
        // TODO: for a small poly, evaluate at a point in {0, 1}^v and assert
        // it equals the corresponding evals entry.
        // For 2 variables, indices correspond to bits of the index:
        //
        //   evals[0] = f(0, 0) because 0 = 00
        //   evals[1] = f(1, 0) because 1 = 01
        //   evals[2] = f(0, 1) because 2 = 10
        //   evals[3] = f(1, 1) because 3 = 11
        let poly = MultilinearPoly::new(
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
        // TODO: fix x_1 = r, then evaluate at (r_2, ..., r_v); compare against
        // the full evaluation at (r, r_2, ..., r_v). They must be equal.
        // A 3-variable polynomial with 8 evaluations.
        let poly = MultilinearPoly::new(
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

        // First fix x_1 = r, producing a 2-variable polynomial.
        let fixed = poly.fix_first_variable(r);

        // Then evaluate the reduced polynomial at (r2, r3).
        let fixed_eval = fixed.evaluate(&[r2, r3]);

        // This should match evaluating the original polynomial at (r, r2, r3).
        let full_eval = poly.evaluate(&[r, r2, r3]);

        assert_eq!(fixed_eval, full_eval);
    }

    #[test]
    fn sum_over_hypercube_matches_naive() {
        // TODO: a tautology test, but write it to exercise the API.
        let evals = vec![Fp::new(3), Fp::new(5), Fp::new(7), Fp::new(9)];
        let poly = MultilinearPoly::new(2, evals.clone());

        let expected = evals[0] + evals[1] + evals[2] + evals[3];

        assert_eq!(poly.sum_over_hypercube(), expected);
    }
}
