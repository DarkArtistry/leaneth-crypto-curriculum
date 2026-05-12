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
    // TODO:
    //   1. Assert evals.len() == domain.size().
    //   2. Run ifft_on_domain(evals, domain) to get the length-n coefficient vector.
    //   3. Truncate to length `degree_bound` if you want — the trailing
    //      coefficients SHOULD be zero for a real codeword, but the truncation
    //      makes the polynomial canonical regardless.
    //   4. Wrap with UnivariatePoly::new (which strips trailing zeros).
    //
    // Note: in practice, after iFFT on a real codeword, coeffs[degree_bound..]
    // will all be Fp::zero() — so step 3 isn't strictly necessary; UnivariatePoly::new
    // strips them. Adding a debug-only assertion that those high coefficients
    // are indeed zero would catch the "user passed me a non-codeword" case.
    let _ = (evals, domain, degree_bound);
    todo!()
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
    // TODO (stretch):
    //   - Compute n = domain.size(); set e = (n - degree_bound) / 2 (max errors).
    //   - Build and solve the BW linear system over F_p (Gaussian elimination).
    //   - Reconstruct Q(X) and E(X), then divide Q / E.
    //   - If the division is not exact, return Err("beyond unique-decoding radius").
    //   - Otherwise return Ok(P).
    //
    // For the first pass, you can leave this as:
    //
    //   unimplemented!("Berlekamp-Welch — stretch goal for objective 2")
    //
    // and the rest of the crate's tests will still pass.
    let _ = (received, domain, degree_bound);
    todo!()
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
        let _ = ReedSolomonCode::new;
        todo!()
    }

    #[test]
    fn decode_after_zero_polynomial() {
        // TODO: encode the zero polynomial; decode should return UnivariatePoly::zero().
        todo!()
    }

    #[test]
    #[ignore = "stretch goal — Berlekamp-Welch"]
    fn berlekamp_welch_corrects_one_error() {
        // TODO (stretch): build a code with degree_bound 4, n = 16 (rate 1/4).
        // Unique-decoding radius = (16 - 4) / 2 = 6 errors. Inject 1 error,
        // decode, recover the original polynomial.
        todo!()
    }

    #[test]
    #[ignore = "stretch goal — Berlekamp-Welch"]
    fn berlekamp_welch_fails_beyond_radius() {
        // TODO (stretch): inject (n - d)/2 + 1 errors. Decoder should return Err.
        todo!()
    }
}
