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
        // new: friendlier panic message showing both values, so a failing
        // assert tells you the actual numbers instead of "assertion failed".
        assert!(
            degree_bound <= domain.size(),
            "degree_bound ({}) cannot exceed domain size ({})",
            degree_bound,
            domain.size()
        );
        Self { domain, degree_bound }
    }

    /// Length of every codeword.
    pub fn codeword_len(&self) -> usize {
        // TODO: just self.domain.size().
        self.domain.size()
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
        (self.degree_bound, self.codeword_len())
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
        // TODO: produce the codeword `(p(x))_{x ∈ L}` via FFT.
        //   1. Validate `deg(message) < degree_bound` (the message must fit the code).
        //   2. Zero-pad the coefficient vector to `n = codeword_len`; FFT works on
        //      length-n inputs even when the polynomial has fewer real coefficients.
        //   3. Run `fft_on_domain(&padded, &self.domain)`. This **is** RS encoding:
        //      evaluating the message polynomial on every point of the domain.
        // See "What encoding does" in the module docs.
        //
        // Reference implementation below.
        if let Some(d) = message.degree() {
            assert!(d < self.degree_bound, "message exceeds degree bound");
        }
        let mut padded = message.coeffs().to_vec();
        padded.resize(self.codeword_len(), Fp::zero());
        // new: was `fft_on_domain(&mut padded, &self.domain); padded` — that
        // called the FFT, threw the result away, and returned the un-FFTed
        // padded coefficient vector. `fft_on_domain` *returns* the codeword;
        // it doesn't mutate in place. Just return the function's output.
        fft_on_domain(&padded, &self.domain)
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
        if let Some(d) = message.degree() {
            assert!(d < self.degree_bound, "message exceeds degree bound");
        }
        self.domain.iter().map(|x| message.evaluate(x)).collect()
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
        //
        // The zero polynomial p(X) = 0 evaluates to 0 at every point, so its
        // codeword is the all-zero vector of length n regardless of the
        // choice of domain or degree bound. Cheapest possible correctness
        // test for the encoder.
        let log_n = 3;
        let n = 1 << log_n; // 8
        let domain = EvaluationDomain::new_subgroup(log_n);
        let code = ReedSolomonCode::new(domain, 4); // any d ≤ n is fine

        let codeword = code.encode(&UnivariatePoly::zero());

        assert_eq!(codeword, vec![Fp::zero(); n]);
    }

    #[test]
    fn encode_constant_polynomial_gives_constant_codeword() {
        // TODO: encode of constant `c` → vec of n copies of c.
        //
        // The constant polynomial p(X) = c evaluates to c at every domain
        // point, so its codeword is [c, c, c, ..., c]. Unlike the
        // zero-polynomial test, this one actually exercises the FFT: the
        // input coefficient vector is [c, 0, 0, ..., 0] (one non-zero
        // entry), but the output codeword is [c, c, ..., c] (all entries
        // equal). A no-op encoder that returned its input untouched
        // would fail this test.
        let log_n = 3;
        let n = 1 << log_n; // 8
        let domain = EvaluationDomain::new_subgroup(log_n);
        let code = ReedSolomonCode::new(domain, 4); // any d ≥ 1 works

        let c = Fp::new(42);
        let message = UnivariatePoly::new(vec![c]); // p(X) = 42

        let codeword = code.encode(&message);

        assert_eq!(codeword, vec![c; n]);
    }

    #[test]
    fn encode_matches_encode_naive() {
        // TODO: domain of size 16, degree_bound 4.
        //   - Build a random message polynomial with 4 random coeffs.
        //   - Compare encode and encode_naive.
        //
        // new: two fixes —
        //   1. Swapped `thread_rng()` for a seeded `StdRng` so a failure
        //      can be replayed with the same inputs.
        //   2. Dropped the unused `let n = 1 << log_n;` line.
        //
        // This is the strongest encode test: it cross-checks the FFT path
        // against the independent Horner-based naive evaluation. A
        // single-side bug in either method shows up here.
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let log_n = 3;
        let mut rng = StdRng::seed_from_u64(0xE9C0DE_5EED);
        let coeffs: Vec<Fp> = (0..4).map(|_| Fp::random(&mut rng)).collect();
        let message = UnivariatePoly::new(coeffs);
        let domain = EvaluationDomain::new_subgroup(log_n);
        let code = ReedSolomonCode::new(domain, 4);

        let codeword = code.encode(&message);
        let naive_codeword = code.encode_naive(&message);

        assert_eq!(codeword, naive_codeword);
    }

    #[test]
    fn encode_codeword_has_expected_length() {
        // TODO: codeword.len() == domain.size() for any message.
        //
        // new: three fixes —
        //   1. The assertion was `codeword.len() == domain.size()`, but
        //      `domain` had already been moved into `ReedSolomonCode::new`
        //      so the line wouldn't compile (E0382, use of moved value).
        //      Now read the size back through the code's own accessor:
        //      `code.domain().size()`. That's also structurally the right
        //      assertion — "the codeword length matches THE CODE's domain
        //      size", not some separately-tracked variable.
        //   2. Dropped the unused `let naive_codeword = ...` line — this
        //      test is about length, not the naive cross-check.
        //   3. Seeded RNG for reproducibility (and dropped unused `n`).
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let log_n = 3;
        let mut rng = StdRng::seed_from_u64(0xCAFE_DEAD);
        let coeffs: Vec<Fp> = (0..4).map(|_| Fp::random(&mut rng)).collect();
        let message = UnivariatePoly::new(coeffs);
        let domain = EvaluationDomain::new_subgroup(log_n);
        let code = ReedSolomonCode::new(domain, 4);

        let codeword = code.encode(&message);

        assert_eq!(codeword.len(), code.domain().size());
    }

    #[test]
    // new: tightened `#[should_panic]` to require the SPECIFIC panic message.
    // Without `expected = "..."`, any panic anywhere in the body counts as
    // "passing", which would hide bugs that crash before reaching the
    // intended guard. With `expected`, only the actual degree-bound assert
    // satisfies the test.
    #[should_panic(expected = "message exceeds degree bound")]
    fn encode_panics_on_too_high_degree() {
        // TODO: degree_bound = 4 but a message of degree 5 (length 6).
        // Should panic.
        //
        // The guard we're exercising lives in `encode()`:
        //
        //     if let Some(d) = message.degree() {
        //         assert!(d < self.degree_bound, "message exceeds degree bound");
        //     }
        //
        // With degree_bound = 4 and a degree-5 message, `5 < 4` is false and
        // the `assert!` fires. The `#[should_panic]` attribute on this test
        // catches the panic and turns it into a pass.
        let log_n = 3;
        let domain = EvaluationDomain::new_subgroup(log_n);
        let code = ReedSolomonCode::new(domain, 4); // degree_bound = 4

        // 6 coefficients with a non-zero leading term => degree exactly 5.
        // (UnivariatePoly::new strips trailing zeros, so the X^5 coefficient
        // must be non-zero for the polynomial's degree to be 5.)
        let message = UnivariatePoly::new(vec![
            Fp::new(1),
            Fp::new(2),
            Fp::new(3),
            Fp::new(4),
            Fp::new(5),
            Fp::new(6),
        ]);

        // Panics: message.degree() = 5 is not strictly less than degree_bound = 4.
        code.encode(&message);
    }
}
