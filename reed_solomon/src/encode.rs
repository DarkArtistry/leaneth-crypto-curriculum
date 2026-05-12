//! The Reed-Solomon encoder.
//!
//! ## What encoding does
//!
//! Pick:
//! - A field `F`.
//! - An evaluation domain `L ⊆ F` of size `n` (must be a smooth coset for FFT).
//! - A degree bound `d`. The "message" is a polynomial of degree `< d`.
//!
//! The encoder takes a message (polynomial `p` with `d` coefficients) and
//! outputs the codeword
//!
//! ```text
//! c = (p(x))_{x ∈ L} ∈ F^n.
//! ```
//!
//! This is a vector of `n` field elements — the polynomial's evaluation
//! table on `L`. **The codeword IS the evaluation table**; there is no
//! separate "encoded" representation.
//!
//! ## Rate
//!
//! `ρ = d / n`. For RS-based SNARKs the typical choice is
//! `ρ ∈ {1/2, 1/4, 1/8, 1/16}` — smaller `ρ` = more redundancy = larger
//! decoding radius = bigger codeword. Most STARK systems use `ρ = 1/8`.
//!
//! ## Two implementations
//!
//! [`ReedSolomonCode::encode`] uses the FFT (`O(n log n)`) — this is what
//! you'd actually call.
//!
//! [`ReedSolomonCode::encode_naive`] just calls `poly.evaluate(x)` for every
//! `x ∈ L` (`O(n · d)`). Slow, but useful as a correctness oracle: the FFT
//! and naive paths must agree on the same input.
//!
//! ## Worked numbers
//!
//! `d = 4`, `n = 16`, so `ρ = 1/4`. A degree-3 polynomial like
//! `p(X) = 1 + 2X + 3X^2 + 4X^3` becomes a 16-element codeword
//! `[p(x_0), p(x_1), ..., p(x_{15})]`. Twelve of those values are "redundant"
//! in the sense that any 4 of them suffice to recover `p` — that's the
//! minimum-distance bound from coding theory.

use crate::domain::EvaluationDomain;
use crate::fft::fft_on_domain;
use crate::field::Fp;
use crate::polynomial::UnivariatePoly;

/// A Reed-Solomon code: the set of polynomial evaluation tables
/// `{ (p(x))_{x ∈ domain} : deg(p) < degree_bound }`.
#[derive(Clone, Debug)]
pub struct ReedSolomonCode {
    /// Evaluation domain `L`. The codeword has length `domain.size()`.
    domain: EvaluationDomain,
    /// `d`: the maximum number of coefficients in a message polynomial.
    /// Equivalently, `deg(p) < degree_bound`.
    degree_bound: usize,
}

impl ReedSolomonCode {
    /// Construct a Reed-Solomon code with the given evaluation domain and degree bound.
    ///
    /// Panics if `degree_bound > domain.size()`. (We need `d <= n` for the
    /// code to be sensible — `d = n` is the trivial "no redundancy" case;
    /// `d > n` means the encoder has fewer evaluations than coefficients,
    /// which can't define a unique polynomial.)
    pub fn new(domain: EvaluationDomain, degree_bound: usize) -> Self {
        // TODO: assert degree_bound <= domain.size(); store fields.
        let _ = (domain, degree_bound);
        todo!()
    }

    /// Length of every codeword.
    pub fn codeword_len(&self) -> usize {
        // TODO: just self.domain.size().
        todo!()
    }

    /// Maximum number of coefficients in a message polynomial.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// The evaluation domain.
    pub fn domain(&self) -> &EvaluationDomain {
        &self.domain
    }

    /// Encoding rate `ρ = d / n`.
    pub fn rate(&self) -> (usize, usize) {
        // TODO: return the unreduced fraction (degree_bound, codeword_len).
        // Reducing it is overkill — the caller can format however they want.
        todo!()
    }

    /// Encode a message polynomial: evaluate it on the entire domain.
    ///
    /// This is the **fast path**, using the FFT. The polynomial's coefficients
    /// are zero-padded to `domain.size()` before the FFT (since the FFT works
    /// on length-`n` inputs even when the polynomial has degree `< d < n`).
    ///
    /// Panics if `message.degree()` is `Some(deg)` with `deg >= self.degree_bound`.
    /// The encoder is only defined for messages within the degree bound.
    pub fn encode(&self, message: &UnivariatePoly) -> Vec<Fp> {
        // TODO:
        //   1. Validate the degree:
        //        if let Some(d) = message.degree() {
        //            assert!(d < self.degree_bound, "message exceeds degree bound");
        //        }
        //   2. Build a length-n coefficient vector padded with zeros:
        //        let mut padded = message.coeffs().to_vec();
        //        padded.resize(self.codeword_len(), Fp::zero());
        //   3. Run fft_on_domain(&padded, &self.domain).
        let _ = message;
        todo!()
    }

    /// Encode by naive Horner evaluation at every domain point.
    ///
    /// `O(n · d)` — slower than [`ReedSolomonCode::encode`], but a useful
    /// reference for correctness tests. The two methods must agree.
    pub fn encode_naive(&self, message: &UnivariatePoly) -> Vec<Fp> {
        // TODO:
        //   1. Validate the degree (same as encode).
        //   2. For each domain element x in self.domain.iter(): collect message.evaluate(x).
        //
        // Idiomatic: self.domain.iter().map(|x| message.evaluate(x)).collect()
        let _ = message;
        todo!()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_zero_polynomial_gives_zero_codeword() {
        // TODO: encode of UnivariatePoly::zero() → vec of n zeros.
        todo!()
    }

    #[test]
    fn encode_constant_polynomial_gives_constant_codeword() {
        // TODO: encode of constant `c` → vec of n copies of c.
        todo!()
    }

    #[test]
    fn encode_matches_encode_naive() {
        // TODO: domain of size 16, degree_bound 4.
        //   - Build a random message polynomial with 4 random coeffs.
        //   - Compare encode and encode_naive.
        todo!()
    }

    #[test]
    fn encode_codeword_has_expected_length() {
        // TODO: codeword.len() == domain.size() for any message.
        todo!()
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_too_high_degree() {
        // TODO: degree_bound = 4 but a message of degree 5 (length 6).
        // Should panic.
        todo!()
    }
}
