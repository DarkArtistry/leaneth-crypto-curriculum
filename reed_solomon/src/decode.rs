//! Reed-Solomon decoding.
//!
//! Two decoders here:
//!
//! 1. **Interpolation-based decoder** ([`decode_via_interpolation`]) — required.
//!    Given an *uncorrupted* codeword on the smooth domain `L`, recover the
//!    message polynomial in `O(n log n)` via inverse FFT. This is just "go
//!    back from evaluations to coefficients". No errors handled.
//!
//! 2. **Berlekamp-Welch decoder** ([`decode_berlekamp_welch`]) — stretch goal.
//!    Given a *possibly corrupted* received word `y_0, y_1, ..., y_{n-1}` at
//!    domain points `x_0, ..., x_{n-1}`, find the unique polynomial of degree
//!    `< d` that agrees with at least `n - e` of the received values, where
//!    `e <= (n - d) / 2` is the unique-decoding radius.
//!
//! ## Why both?
//!
//! In SNARKs (FRI / STIR / WHIR) we **don't actually decode** — we just
//! commit to a function and run a low-degree test. So the only decoder you
//! truly need for the rest of the curriculum is (1).
//!
//! But coding-theory fluency is a separate good. Berlekamp-Welch is the
//! cleanest classical algorithm for unique decoding, and it gives you the
//! tools to reason about list-decoding and the Johnson bound — concepts
//! that *do* appear in the STIR / WHIR analyses (the proximity radius and
//! how soundness scales with `δ`). If you're game, do (2).
//!
//! ## Quick guide to Berlekamp-Welch
//!
//! Given received word `(y_0, ..., y_{n-1})` at points `(x_0, ..., x_{n-1})`,
//! we want polynomials:
//!
//! - `E(X)` ("error locator"): degree at most `e`, monic, with roots exactly
//!   at the corrupted positions: `E(x_i) = 0` iff position `i` is in error.
//! - `Q(X) = P(X) · E(X)`: degree at most `e + d - 1`, where `P` is the
//!   message polynomial we want.
//!
//! For every position `i`:
//!
//! ```text
//! Q(x_i) = y_i · E(x_i)         (this holds at non-error positions trivially because y_i = P(x_i),
//!                                and at error positions because both sides are 0).
//! ```
//!
//! Treating the coefficients of `Q` and `E` as unknowns, this is a linear
//! system in `d + 2e` unknowns (coefficients of `Q`: `d + e`, coefficients
//! of `E` minus the leading 1: `e`). With `n` equations, it has a solution
//! whenever `n >= d + 2e`, i.e., `e <= (n - d) / 2` — the unique-decoding
//! radius.
//!
//! After solving, recover `P = Q / E` via polynomial long division. If the
//! division is not exact, decoding fails (input was beyond the radius).
//!
//! Implementation sketch:
//! 1. Build the `n × (d + 2e + 1)` coefficient matrix and the length-`n`
//!    right-hand side, encoding `Q(x_i) - y_i · E(x_i) = 0` row by row.
//! 2. Gaussian-eliminate over `F_p`. Watch for the leading `1` of `E`:
//!    you can either (a) fix it and solve a reduced system, or (b) solve
//!    the full homogeneous system and renormalize.
//! 3. Reconstruct `Q(X)` and `E(X)` from the solution vector.
//! 4. Polynomial-divide `Q / E`. If the remainder is non-zero, return Err.
//!
//! See textbook references: Welch & Berlekamp's original 1986 patent,
//! Sudan's lecture notes, or any modern coding-theory text.
//!
//! **Question for the reader.** Without the `E(X)` trick, `P(x_i) = y_i`
//! is nonlinear in the unknowns because we don't know *which* `i` are errors.
//! Why does multiplying through by an unknown `E(X)` — *adding more
//! unknowns!* — make the system *linear* and *solvable*?
//! Try to answer before reading on.
//!
//! Answer: the bilinear product `P(X) · E(X)` is *hidden inside* `Q(X)`,
//! which is treated as one unknown polynomial. The equation
//! `Q(x_i) = y_i · E(x_i)` is jointly linear in the coefficients of `Q` and
//! `E` (their degrees are bounded), and it holds at *every* `x_i` —
//! including error positions, where both sides vanish because `E(x_i) = 0`.
//! The "which positions are errors" information now lives in the roots of
//! the unknown polynomial `E`, which the linear solver discovers without
//! us ever having to choose a candidate error set.
//!
//! ## Why Berlekamp-Welch's `E·Q` trick linearises the problem
//!
//! The natural decoding equation is `P(x_i) = y_i` for every `i`. If *any*
//! `y_i` is corrupted, this is **nonlinear in the unknowns** in a sneaky
//! way: the coefficients of `P` are unknown, but so is the *set of error
//! positions* — and there's no way to pick the error set without also
//! knowing `P`. The two unknowns are entangled.
//!
//! Berlekamp-Welch breaks the entanglement with a fresh unknown polynomial
//! `E(X)` (the *error locator*) of degree `≤ e`, whose roots are exactly
//! the error positions. At a non-error point `i`, `E(x_i) ≠ 0` and
//! `P(x_i) = y_i`. At an error point, `E(x_i) = 0`, so *both sides of any
//! equation involving the product `P(x_i) · E(x_i)` vanish automatically*.
//!
//! Multiplying through gives `Q(X) := P(X) · E(X)`, and the key equation
//!
//! ```text
//! Q(x_i) = y_i · E(x_i)               for every i = 0, ..., n - 1
//! ```
//!
//! is **simultaneously linear** in the coefficients of `Q` (degrees
//! `0..d+e`) and `E` (degrees `0..e`, with the leading coefficient fixed
//! at 1 to break the homogeneous degree of freedom). The bilinear product
//! `P · E` is hidden inside `Q`, restoring linearity. An `O(n³)` Gaussian
//! elimination solves for `(Q, E)`, and `P = Q / E` falls out by one
//! polynomial division.
//!
//! ## Distance and unique-decoding radius
//!
//! The **minimum distance** of an RS code with `n` evaluation points and
//! message-space dimension `d` (degree-`< d` polynomials) is
//!
//! ```text
//! Δ = n - d + 1.
//! ```
//!
//! *Proof.* Two distinct polynomials of degree `< d` differ in a non-zero
//! degree-`< d` polynomial, which by the **polynomial root bound** has at
//! most `d - 1` roots. So the two evaluation tables agree on at most
//! `d - 1` of the `n` points and disagree on at least `n - (d - 1) = n - d + 1`.
//! ∎
//!
//! Hence the **unique-decoding radius** is `⌊(n - d) / 2⌋`. A received
//! word at Hamming distance `≤ ⌊(n - d) / 2⌋` from some codeword is
//! *strictly* closer to it than to any other codeword (by the triangle
//! inequality applied to the minimum distance `n - d + 1`), so the closest
//! codeword is unique. Beyond that radius, ties become possible — we enter
//! **list-decoding** territory, where the decoder returns a small list of
//! candidate codewords instead of a single answer.

use crate::domain::EvaluationDomain;
use crate::fft::ifft_on_domain;
use crate::field::Fp;
use crate::polynomial::UnivariatePoly;

/// Decode an uncorrupted codeword: recover the message polynomial.
///
/// Given a codeword `evals` of length `domain.size()` known to lie inside
/// the RS code with the given degree bound, returns the unique polynomial
/// of degree `< degree_bound` whose evaluations on `domain` match `evals`.
///
/// **Does not check for errors.** If the input has been corrupted, this
/// returns garbage. Use [`decode_berlekamp_welch`] (when implemented) if
/// you need error correction.
///
/// Cost: `O(n log n)` via inverse FFT.
///
/// Panics if `evals.len() != domain.size()`.
pub fn decode_via_interpolation(
    evals: &[Fp],
    domain: &EvaluationDomain,
    degree_bound: usize,
) -> UnivariatePoly {
    // TODO: recover `p` from a clean codeword `(p(x))_{x ∈ L}`.
    //   1. Assert `evals.len() == n` (one value per domain point).
    //   2. Call `ifft_on_domain(evals, domain)` to invert the encoder's FFT
    //      — iFFT on a smooth coset is just Lagrange interpolation specialised
    //      to roots-of-unity, so this returns the coefficient vector.
    //   3. Truncate to `degree_bound`; on a valid codeword the high coefficients
    //      are already zero, but truncation canonicalises the polynomial.
    // See "Why both?" in the module docs.
    //
    // Reference implementation below.
    assert_eq!(evals.len(), domain.size());
    let mut coeffs = ifft_on_domain(evals, domain);
    coeffs.truncate(degree_bound);
    UnivariatePoly::new(coeffs)
}

/// Decode a possibly-corrupted received word via Berlekamp-Welch.
///
/// Given a received word `received` (length `domain.size()`) and the assumed
/// degree bound, returns the unique low-degree polynomial agreeing with the
/// received word at all but at most `(n - degree_bound) / 2` positions.
///
/// Returns `Err` if no such polynomial exists (the input is beyond the
/// unique-decoding radius).
///
/// **Stretch goal.** You can leave this as `unimplemented!()` for the first
/// pass of objective 2 and ship the rest. Come back to it before objective 3
/// (STIR), where you'll need fluency with the proximity-radius parameter.
///
/// See the module docstring for the algorithm sketch.
pub fn decode_berlekamp_welch(
    received: &[Fp],
    domain: &EvaluationDomain,
    degree_bound: usize,
) -> Result<UnivariatePoly, &'static str> {
    // TODO: unique-decode a received word with up to `e` errors.
    //   1. Set max correctable errors `e = (n - degree_bound) / 2`.
    //   2. Set up the linear system asserting `Q(x_i) = y_i · E(x_i)` for all i,
    //      with `deg Q < degree_bound + e` and `deg E ≤ e` (monic, leading 1
    //      fixed). The `E·Q` trick linearises the otherwise-nonlinear problem.
    //   3. Solve for the coefficients of `Q` and `E` via Gauss-Jordan over F_p.
    //   4. Recover `P = Q / E` by polynomial division; non-zero remainder ⇒
    //      input is beyond the unique-decoding radius, return Err.
    // See "Quick guide to Berlekamp-Welch" and "Why Berlekamp-Welch's `E·Q`
    // trick linearises the problem" above.
    //
    // Reference implementation below.
    let n = domain.size();
    let e = (n - degree_bound) / 2;

    let d = degree_bound;
    assert_eq!(
        received.len(),
        n,
        "received word length must match domain size"
    );

    // Collect the domain elements (x_0, x_1, ..., x_{n-1}).
    let x: Vec<Fp> = domain.iter().collect();

    // Build the linear system.
    let unknowns = d + 2 * e;
    let mut matrix: Vec<Vec<Fp>> = Vec::with_capacity(n);
    let mut rhs: Vec<Fp> = Vec::with_capacity(n);

    for i in 0..n {
        let xi = x[i];
        let yi = received[i];

        let mut row = vec![Fp::zero(); unknowns];

        // Columns 0..(d+e): coefficients of Q. Coefficient of q_j is x_i^j.
        let mut pow = Fp::one();
        for j in 0..(d + e) {
            row[j] = pow;
            pow = pow * xi;
        }

        // Columns (d+e)..(d+2e): coefficients of E (the non-leading ones).
        // Coefficient of e_k is -y_i · x_i^k. Restart the power accumulator.
        let mut pow = Fp::one();
        for k in 0..e {
            row[d + e + k] = -yi * pow;
            pow = pow * xi;
        }

        // RHS: y_i · x_i^e (from moving the X^e coefficient of E, fixed at 1).
        // `pow` is now x_i^e because the loop above advanced it e times.
        let rhs_val = yi * pow;

        matrix.push(row);
        rhs.push(rhs_val);
    }

    // Solve. If the system is inconsistent or singular, the input is beyond
    // the unique-decoding radius (or otherwise undecodable).
    let solution = solve_linear_system(&matrix, &rhs)
        .ok_or("could not solve BW linear system — input is likely beyond the unique-decoding radius")?;

    // Reconstruct Q(X) and E(X) from the solution vector.
    let q_coeffs: Vec<Fp> = solution[0..(d + e)].to_vec();
    let mut e_coeffs: Vec<Fp> = solution[(d + e)..(d + 2 * e)].to_vec();
    e_coeffs.push(Fp::one()); // the X^e coefficient of E that we held fixed

    let q_poly = UnivariatePoly::new(q_coeffs);
    let e_poly = UnivariatePoly::new(e_coeffs);

    // Recover P = Q / E. If the division leaves a non-zero remainder, the
    // factorization Q = P · E doesn't actually hold for any low-degree P — so
    // the received word isn't within the decoding radius of any codeword.
    let (quotient, remainder) = poly_divmod(&q_poly, &e_poly);
    if !remainder.is_zero() {
        return Err("Q is not divisible by E — beyond unique-decoding radius");
    }

    Ok(quotient)
}

// ============================================================================
// new: BW supporting utilities (polynomial division + Gaussian elimination)
// ============================================================================

/// new: Polynomial long division.
///
/// Returns `(quotient, remainder)` such that `dividend = quotient · divisor + remainder`,
/// with `deg(remainder) < deg(divisor)`. Standard schoolbook long division.
///
/// Panics if `divisor` is the zero polynomial.
fn poly_divmod(
    dividend: &UnivariatePoly,
    divisor: &UnivariatePoly,
) -> (UnivariatePoly, UnivariatePoly) {
    assert!(!divisor.is_zero(), "cannot divide by zero polynomial");

    let divisor_coeffs = divisor.coeffs();
    let divisor_deg = divisor.degree().unwrap();
    let divisor_lead_inv = divisor_coeffs
        .last()
        .copied()
        .unwrap()
        .inverse()
        .expect("divisor leading coeff is non-zero by canonical form");

    // If dividend has lower degree than divisor, quotient is zero and remainder is dividend.
    let Some(dividend_deg) = dividend.degree() else {
        return (UnivariatePoly::zero(), UnivariatePoly::zero());
    };
    if dividend_deg < divisor_deg {
        return (UnivariatePoly::zero(), dividend.clone());
    }

    // Working remainder (mutable, ascending-degree coefficients).
    let mut remainder: Vec<Fp> = dividend.coeffs().to_vec();
    // Strip any trailing zeros up front (defensive).
    while remainder.last() == Some(&Fp::zero()) {
        remainder.pop();
    }

    // Quotient: at most (dividend_deg - divisor_deg + 1) coefficients.
    let mut quotient = vec![Fp::zero(); dividend_deg - divisor_deg + 1];

    while !remainder.is_empty() && remainder.len() - 1 >= divisor_deg {
        let rem_deg = remainder.len() - 1;
        let lead_r = *remainder.last().unwrap();

        // Quotient monomial: (lead_r / divisor_lead) · X^(rem_deg - divisor_deg).
        let q_coeff = lead_r * divisor_lead_inv;
        let q_deg = rem_deg - divisor_deg;
        quotient[q_deg] = q_coeff;

        // Subtract q_coeff · X^q_deg · divisor from remainder.
        for j in 0..=divisor_deg {
            let idx = q_deg + j;
            let d_coeff = divisor_coeffs[j];
            remainder[idx] = remainder[idx] - q_coeff * d_coeff;
        }

        // Strip newly-introduced trailing zeros.
        while remainder.last() == Some(&Fp::zero()) {
            remainder.pop();
        }
    }

    (
        UnivariatePoly::new(quotient),
        UnivariatePoly::new(remainder),
    )
}

/// new: Solve the linear system `M · x = b` over `F_p`.
///
/// Returns `Some(x)` if a solution exists, `None` if the system is
/// inconsistent. When the system is **under-determined** (free variables
/// exist), this picks the specific solution where every free variable is `0`.
/// That's fine for the BW use case: when the actual error count is less than
/// the assumed max `e`, the system has multiple `(Q, E)` solutions, but they
/// ALL give the same `P` after `Q / E`. Any one will do.
///
/// `matrix` is `n × m` and `rhs` is length `n`. We expect `n ≥ m`. Uses
/// Gauss-Jordan elimination.
fn solve_linear_system(matrix: &[Vec<Fp>], rhs: &[Fp]) -> Option<Vec<Fp>> {
    let n = matrix.len();
    if n == 0 {
        return Some(vec![]);
    }
    let m = matrix[0].len();

    // Build augmented matrix [M | b] of size n × (m + 1).
    let mut aug: Vec<Vec<Fp>> = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, &b)| {
            let mut r = row.clone();
            r.push(b);
            r
        })
        .collect();

    // Track which column each pivot row "owns" so we can read off the
    // solution properly when there are free variables (columns without pivots).
    let mut pivot_col_for_row: Vec<Option<usize>> = vec![None; n];

    let mut pivot_row = 0;
    for col in 0..m {
        if pivot_row >= n {
            break;
        }

        // Find a pivot: first row at or below `pivot_row` with a non-zero entry in column `col`.
        let pivot = (pivot_row..n).find(|&r| aug[r][col] != Fp::zero());
        let Some(pivot) = pivot else {
            // No pivot in this column → it's a free variable. Skip to the next column.
            continue;
        };
        aug.swap(pivot_row, pivot);

        // Normalise the pivot row so the pivot entry becomes 1.
        let pivot_val = aug[pivot_row][col];
        let pivot_inv = pivot_val.inverse()?;
        for c in 0..=m {
            aug[pivot_row][c] = aug[pivot_row][c] * pivot_inv;
        }

        // Eliminate this column in every other row (above and below).
        for r in 0..n {
            if r == pivot_row {
                continue;
            }
            let factor = aug[r][col];
            if factor == Fp::zero() {
                continue;
            }
            for c in 0..=m {
                aug[r][c] = aug[r][c] - factor * aug[pivot_row][c];
            }
        }

        pivot_col_for_row[pivot_row] = Some(col);
        pivot_row += 1;
    }

    // Consistency check: any row with all-zero matrix part but non-zero RHS
    // means the system is inconsistent (no solution exists at all).
    for r in 0..n {
        let all_zero = (0..m).all(|c| aug[r][c] == Fp::zero());
        if all_zero && aug[r][m] != Fp::zero() {
            return None;
        }
    }

    // Build the solution. Free variables (columns without pivots) get set to 0;
    // pivot variables get read from the corresponding RHS entry.
    let mut solution = vec![Fp::zero(); m];
    for r in 0..n {
        if let Some(col) = pivot_col_for_row[r] {
            solution[col] = aug[r][m];
        }
    }
    Some(solution)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::ReedSolomonCode;

    #[test]
    fn decode_round_trip() {
        // TODO:
        //   1. Build a domain of size 8, code with degree_bound 3.
        //   2. Build a polynomial of degree 2 (3 coeffs).
        //   3. Encode → codeword.
        //   4. decode_via_interpolation(codeword, domain, 3) should equal the original poly.

        // new: filled in test body.
        let domain = EvaluationDomain::new_subgroup(3); // size 8
        let code = ReedSolomonCode::new(domain.clone(), 3);

        // p(X) = 1 + 2X + 3X² (degree 2, 3 coefficients).
        let original = UnivariatePoly::new(vec![Fp::new(1), Fp::new(2), Fp::new(3)]);
        let codeword = code.encode(&original);

        let recovered = decode_via_interpolation(&codeword, &domain, 3);

        assert_eq!(recovered, original);
    }

    #[test]
    fn decode_after_zero_polynomial() {
        // TODO: encode the zero polynomial; decode should return UnivariatePoly::zero().

        // new: filled in test body.
        let domain = EvaluationDomain::new_subgroup(3); // size 8
        let code = ReedSolomonCode::new(domain.clone(), 3);

        let codeword = code.encode(&UnivariatePoly::zero());
        let recovered = decode_via_interpolation(&codeword, &domain, 3);

        assert_eq!(recovered, UnivariatePoly::zero());
    }

    #[test]
    fn berlekamp_welch_corrects_one_error() {
        // TODO (stretch): build a code with degree_bound 4, n = 16 (rate 1/4).
        // Unique-decoding radius = (16 - 4) / 2 = 6 errors. Inject 1 error,
        // decode, recover the original polynomial.

        // new: filled in test body. Un-ignored since BW is now implemented.
        let domain = EvaluationDomain::new_subgroup(4); // size 16
        let code = ReedSolomonCode::new(domain.clone(), 4);

        // p(X) = 7 + 11X + 13X² + 17X³ (degree 3, 4 coefficients).
        let original = UnivariatePoly::new(vec![
            Fp::new(7),
            Fp::new(11),
            Fp::new(13),
            Fp::new(17),
        ]);
        let codeword = code.encode(&original);

        // Corrupt position 5 (any position within the codeword).
        let mut received = codeword.clone();
        received[5] = received[5] + Fp::new(42);

        let recovered = decode_berlekamp_welch(&received, &domain, 4)
            .expect("BW should succeed with 1 error (within radius 6)");

        assert_eq!(recovered, original);
    }

    #[test]
    fn berlekamp_welch_fails_beyond_radius() {
        // TODO (stretch): inject (n - d)/2 + 1 errors. Decoder should return Err.

        // new: filled in test body. Un-ignored since BW is now implemented.
        // n = 16, d = 4, radius = (16 - 4)/2 = 6, so 7 errors is beyond.
        let domain = EvaluationDomain::new_subgroup(4); // size 16
        let code = ReedSolomonCode::new(domain.clone(), 4);

        let original = UnivariatePoly::new(vec![
            Fp::new(1),
            Fp::new(2),
            Fp::new(3),
            Fp::new(4),
        ]);
        let codeword = code.encode(&original);

        // Inject 7 errors at the first 7 positions (indices 0 through 6
        // inclusive). Each gets a different perturbation so they don't
        // accidentally cancel.
        let mut received = codeword;
        for i in 0..7 {
            received[i] = received[i] + Fp::new((i + 1) as u64 * 100);
        }

        let result = decode_berlekamp_welch(&received, &domain, 4);
        assert!(
            result.is_err(),
            "BW should fail when 7 errors are injected (radius is 6)"
        );
    }
}
