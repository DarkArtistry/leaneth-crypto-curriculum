//! The sumcheck prover, generic over a per-variable summation domain.
//!
//! ## What the prover does, in one paragraph
//!
//! The prover holds the public polynomial `g: F^n -> F` and the per-variable
//! domain `D` (so `S = D.elements()`, `k = |S|`). It walks through `n` rounds,
//! in each round (a) sending the verifier a **univariate slice** of the current
//! polynomial along its first remaining variable, and (b) receiving back a
//! random challenge `r` that fixes that variable. The "univariate slice" is the
//! function `X -> sum over the remaining cube of g(..., X, ...)` — by the
//! per-variable degree bound this is a polynomial of degree `< k`, so it's
//! determined by its `k` values on `S`.
//!
//! ## The round message: `[s_i(elements[0]), ..., s_i(elements[k-1])]`
//!
//! At the start of round `i`, the prover's internal state is a polynomial in
//! `n_remaining = n - i` variables (variables `x_1, ..., x_i` have already been
//! fixed to the verifier's challenges `r_1, ..., r_i`). Its evaluations on
//! `S^{n_remaining}` are stored in `self.polynomial.evals`, length
//! `k^{n_remaining}`, mixed-radix LSB-first (see
//! [`crate::polynomial`] module docs).
//!
//! Define
//!
//! ```text
//! s_i(X) = sum over (b_2, ..., b_{n_remaining}) in S^{n_remaining - 1}
//!            of self.polynomial(X, b_2, ..., b_{n_remaining}).
//! ```
//!
//! [`compute_round_message`](SumcheckProver::compute_round_message) returns the
//! `k`-vector `[s_i(elements[0]), s_i(elements[1]), ..., s_i(elements[k-1])]`
//! — the values of `s_i` on `S`. The verifier will Lagrange-interpolate these
//! `k` points to recover `s_i(r)` at the random challenge `r`.
//!
//! ## Deriving the partition from mixed-radix indexing
//!
//! The mixed-radix LSB-first convention puts variable `x_1` in the **least
//! significant** base-`k` digit. So if we let `block_size = k^{n_remaining - 1}`
//! be the number of points in `S^{n_remaining - 1}` (one per "rest" tuple
//! `(b_2, ..., b_{n_remaining})`), then for each fixed `j ∈ {0, ..., k-1}` the
//! `k`-th evaluation
//!
//! ```text
//! self.polynomial(elements[j], elements[rest_1], ..., elements[rest_{n-1}])
//! ```
//!
//! lives at index `j + rest * k`, where `rest ∈ [0, block_size)` enumerates the
//! mixed-radix-base-`k` encoding of `(rest_1, ..., rest_{n-1})`. Therefore
//!
//! ```text
//! s_i(elements[j]) = sum over rest in 0..block_size of evals[j + rest * k].
//! ```
//!
//! The full message is the `k`-vector obtained by doing this sum for each
//! `j ∈ {0, ..., k-1}`.
//!
//! ## Boolean collapse — `k = 2`
//!
//! For `D = BooleanHypercube`, `k = 2` and `block_size = 2^{n_remaining - 1}`.
//! The two sums above expand to:
//!
//! - `j = 0` (so `s_i(elements[0]) = s_i(0)`): sum over `rest` of
//!   `evals[0 + rest * 2]` — i.e. evals at indices `0, 2, 4, ..., 2^n - 2`,
//!   the **even-indexed** evals.
//! - `j = 1` (so `s_i(elements[1]) = s_i(1)`): sum over `rest` of
//!   `evals[1 + rest * 2]` — i.e. evals at indices `1, 3, 5, ..., 2^n - 1`,
//!   the **odd-indexed** evals.
//!
//! That recovers exactly the `step_by(2)` / `skip(1).step_by(2)` partition that
//! the original Boolean-only implementation hard-coded. The generic formula
//! `j + rest * k` gives it for free — we never special-case `k = 2`.
//!
//! ## `Interval3` worked example — `k = 3`
//!
//! For `D = Interval3` with `n_remaining = 2`, `block_size = 3^1 = 3`. The
//! evals are laid out as `[g(0,0), g(1,0), g(2,0), g(0,1), g(1,1), g(2,1),
//! g(0,2), g(1,2), g(2,2)]` (mixed-radix base-3, `x_1` least significant).
//! The message has `k = 3` entries:
//!
//! ```text
//! s_i(0) = evals[0 + 0*3] + evals[0 + 1*3] + evals[0 + 2*3]
//!        = g(0,0) + g(0,1) + g(0,2)
//! s_i(1) = evals[1 + 0*3] + evals[1 + 1*3] + evals[1 + 2*3]
//!        = g(1,0) + g(1,1) + g(1,2)
//! s_i(2) = evals[2 + 0*3] + evals[2 + 1*3] + evals[2 + 2*3]
//!        = g(2,0) + g(2,1) + g(2,2)
//! ```
//!
//! Three groups under base-3 LSB, exactly as the formula predicts.
//!
//! ## Receiving the challenge
//!
//! After sending the message, the prover gets back a challenge `r ∈ F` from
//! the verifier. It updates its state via
//! [`MultivariatePoly::fix_first_variable(r)`](
//! crate::polynomial::MultivariatePoly::fix_first_variable), which Lagrange-
//! interpolates each `k`-tuple of evaluations along the first axis and
//! collapses the polynomial to one fewer variable. After `n` such moves, the
//! polynomial has zero variables and a single stored value `g(r_1, ..., r_n)`
//! — but the prover doesn't need that; the verifier evaluates `g` at
//! `(r_1, ..., r_n)` directly in the final check.

use crate::domain::SumDomain;
use crate::field::Fp;
use crate::polynomial::MultivariatePoly;

/// Sumcheck prover state, generic over a per-variable summation domain `D`.
///
/// Round-by-round the prover's `polynomial` shrinks by one variable; the
/// `challenges` accumulated drive that reduction (see
/// [`receive_challenge`](Self::receive_challenge)). The original variable
/// count `n` is saved as `initial_n_vars` so [`is_done`](Self::is_done) knows
/// when all `n` rounds have been processed.
pub struct SumcheckProver<D: SumDomain> {
    /// Current polynomial. Starts as the input polynomial; loses one variable
    /// after each `receive_challenge`.
    polynomial: MultivariatePoly<D>,
    /// Number of variables in the *original* polynomial (before any fixing).
    /// Equal to the total number of rounds the protocol will run.
    initial_n_vars: usize,
    /// Verifier challenges accumulated so far. Length equals
    /// [`current_round`](Self::current_round).
    challenges: Vec<Fp>,
}

impl<D: SumDomain> SumcheckProver<D> {
    /// Construct a prover for the given multivariate polynomial.
    ///
    /// The prover takes ownership; it'll mutate the polynomial in place
    /// (shrinking it round by round) as challenges arrive.
    pub fn new(polynomial: MultivariatePoly<D>) -> Self {
        let initial_n_vars = polynomial.n_vars;
        Self {
            polynomial,
            initial_n_vars,
            challenges: vec![],
        }
    }

    /// The 0-indexed round we are about to send a message for.
    /// Equals `self.challenges.len()`.
    pub fn current_round(&self) -> usize {
        self.challenges.len()
    }

    /// True iff the protocol is complete (all `initial_n_vars` rounds have
    /// been processed).
    pub fn is_done(&self) -> bool {
        self.current_round() == self.initial_n_vars
    }

    /// Compute the round-`i` univariate message: the `|D|`-vector
    /// `[s_i(elements[0]), ..., s_i(elements[k-1])]`.
    ///
    /// Concretely, with `k = self.polynomial.domain.size()` and
    /// `block_size = k^{n_remaining - 1}`, the implementation is
    ///
    /// ```text
    /// for j in 0..k:
    ///   s_i(elements[j]) = sum over rest in 0..block_size of evals[j + rest * k]
    /// ```
    ///
    /// For `D = BooleanHypercube` this is the familiar even/odd partition of
    /// `evals`; for `D = Interval3` this is three groups under base-3 LSB.
    ///
    /// Returns a `Vec<Fp>` of length `|D|`. The companion verifier in
    /// [`crate::verifier`] will Lagrange-interpolate these points to recover
    /// `s_i(r)` at a random challenge `r ∈ F`.
    ///
    /// Panics if the protocol is already done.
    pub fn compute_round_message(&self) -> Vec<Fp> {
        // TODO: build the length-`k` vector
        //   `[s_i(elements[0]), ..., s_i(elements[k-1])]`.
        //   1. Let `k = self.polynomial.domain.size()` and
        //      `block_size = self.polynomial.evals.len() / k = k^{n_remaining - 1}`
        //      — the number of `rest` tuples we sum over per `x_1`-value.
        //   2. For each `j` in `0..k`, accumulate
        //      `sum over rest in 0..block_size of evals[j + rest * k]` — the
        //      contiguous LSB-first stride that holds `x_1 = elements[j]` fixed.
        //   3. Push the accumulator into `msg`. Length-`k` result, no more.
        //   See the "Deriving the partition from mixed-radix indexing" section
        //   above for why `j + rest * k` is the right partition.
        //
        //   Reference implementation below.

        assert!(
            !self.is_done(),
            "SumcheckProver::compute_round_message: protocol is already done",
        );

        let k = self.polynomial.domain.size();
        // block_size = k^{n_remaining - 1} = |evals| / k. The number of "rest"
        // tuples we sum over for each fixed value of x_1.
        let block_size = self.polynomial.evals.len() / k;

        let mut msg = Vec::with_capacity(k);
        for j in 0..k {
            // Sum evals[j + rest * k] for rest = 0..block_size.
            // For k = 2 this is the even (j=0) / odd (j=1) partition; for k = 3
            // it's the three base-3 LSB groups. See the module docs.
            let mut acc = Fp::zero();
            for rest in 0..block_size {
                acc += self.polynomial.evals[j + rest * k];
            }
            msg.push(acc);
        }
        msg
    }

    /// Receive verifier's challenge `r` for the current variable, and reduce
    /// the polynomial to one fewer variable.
    ///
    /// Internally calls
    /// [`MultivariatePoly::fix_first_variable`](
    /// crate::polynomial::MultivariatePoly::fix_first_variable), which is the
    /// generic Lagrange-along-one-axis step.
    pub fn receive_challenge(&mut self, r: Fp) {
        assert!(
            !self.is_done(),
            "SumcheckProver::receive_challenge: protocol is already done",
        );
        self.challenges.push(r);
        self.polynomial = self.polynomial.fix_first_variable(r);
    }

    /// The challenges accumulated so far. Useful for tests and the demo.
    pub fn challenges(&self) -> &[Fp] {
        &self.challenges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::BooleanHypercube;

    #[test]
    fn round_message_for_known_input() {
        // 2-var poly on the Boolean cube with evals = [1, 2, 3, 4] under the
        // mixed-radix LSB-first convention (k = 2):
        //
        //   evals[0] = f(0, 0) = 1
        //   evals[1] = f(1, 0) = 2
        //   evals[2] = f(0, 1) = 3
        //   evals[3] = f(1, 1) = 4
        //
        // First-round message = [s_0(0), s_0(1)] = [1+3, 2+4] = [4, 6].
        // Generic derivation (k = 2, block_size = 2):
        //   s_0(elements[0]) = evals[0 + 0*2] + evals[0 + 1*2] = 1 + 3 = 4
        //   s_0(elements[1]) = evals[1 + 0*2] + evals[1 + 1*2] = 2 + 4 = 6
        let polynomial = MultivariatePoly::new(
            BooleanHypercube,
            2,
            vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)],
        );
        let prover = SumcheckProver::new(polynomial);
        let message = prover.compute_round_message();
        assert_eq!(message, vec![Fp::new(4), Fp::new(6)]);
    }

    #[test]
    fn after_all_rounds_is_done() {
        // Drive the state machine through all n_vars rounds with arbitrary
        // challenges and confirm `is_done` flips to true.
        let polynomial = MultivariatePoly::new(
            BooleanHypercube,
            2,
            vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)],
        );
        let mut prover = SumcheckProver::new(polynomial);
        for _ in 0..2 {
            prover.receive_challenge(Fp::new(1));
        }
        assert!(prover.is_done());
    }
}
