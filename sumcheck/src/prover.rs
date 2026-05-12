//! The sumcheck prover.
//!
//! ## State machine
//!
//! The prover holds the polynomial `g` and progressively reduces it as the
//! verifier sends challenges. After round `i`, the polynomial has been fixed
//! at variables `x_1, ..., x_i = r_1, ..., r_i`, leaving a polynomial in
//! `v - i` variables.
//!
//! ## Round protocol
//!
//! In round `i` (0-indexed):
//!   1. Prover computes `s_i(X) = sum over b in {0,1}^{v-i-1} of g(r_1, ..., r_i, X, b)`.
//!   2. Prover sends `[s_i(0), s_i(1)]` to the verifier (since `s_i` is degree-1, two values suffice).
//!   3. Verifier sends `r_{i+1}` back.
//!   4. Prover updates state: `g := g.fix_first_variable(r_{i+1})`.
//!
//! After `v` rounds, `g` has been reduced to a constant — namely `g(r_1, ..., r_v)`.

use crate::field::Fp;
use crate::polynomial::MultilinearPoly;

/// Sumcheck prover state.
pub struct SumcheckProver {
    /// Current polynomial. Starts as the input polynomial; after each `receive_challenge`,
    /// loses one variable.
    polynomial: MultilinearPoly,
    /// Number of variables in the *original* polynomial (before any fixing).
    initial_num_vars: usize,
    /// Verifier challenges accumulated so far.
    challenges: Vec<Fp>,
}

impl SumcheckProver {
    /// Construct a prover for the given multilinear polynomial.
    pub fn new(polynomial: MultilinearPoly) -> Self {
        // TODO: store the polynomial, record its initial num_vars, init challenges = vec![].
        let initial_num_vars = polynomial.num_vars;

        Self{
            polynomial,
            initial_num_vars,
            challenges: vec![],
        }
    }

    /// The 0-indexed round we are about to send a message for.
    /// Equals `self.challenges.len()`.
    pub fn current_round(&self) -> usize {
        // TODO
        self.challenges.len()
    }

    /// True iff the protocol is complete (all `v` rounds have been processed).
    pub fn is_done(&self) -> bool {
        // TODO: current_round == initial_num_vars (all variables have been fixed).

        self.current_round() == self.initial_num_vars
    }

    /// Compute the round-`i` univariate message: `[s_i(0), s_i(1)]`.
    ///
    /// `s_i(X)` is the partial sum of `g` over the trailing variables, with
    /// the leading variables already fixed to challenges `r_1, ..., r_i`.
    ///
    /// Concretely, after `i` rounds the polynomial `self.polynomial` has
    /// `v - i` variables. We need `s_i(X)` for the *current first* variable
    /// of `self.polynomial`. So:
    ///
    /// - `s_i(0) = sum over b in {0,1}^{v-i-1} of self.polynomial(0, b)`
    /// - `s_i(1) = sum over b in {0,1}^{v-i-1} of self.polynomial(1, b)`
    ///
    /// Implementation hint (LSB-first convention — see `polynomial.rs` module docs):
    /// `x_1` is bit 0 of the index, so:
    /// - **even-indexed** evaluations (indices `0, 2, 4, ...`) have `x_1 = 0` → contribute to `s_i(0)`.
    /// - **odd-indexed** evaluations (indices `1, 3, 5, ...`) have `x_1 = 1` → contribute to `s_i(1)`.
    ///
    /// Panics if the protocol is already done.
    pub fn compute_round_message(&self) -> [Fp; 2] {
        // Steps:
        //   1. Panic if is_done().
        //   2. Sum the even-indexed evaluations to get s_i(0).
        //   3. Sum the odd-indexed evaluations to get s_i(1).
        //   4. Return [s_i(0), s_i(1)].
        //
        // Idiomatic Rust:
        //   let s0: Fp = self.polynomial.evals.iter().step_by(2).copied().sum();
        //   let s1: Fp = self.polynomial.evals.iter().skip(1).step_by(2).copied().sum();
        //
        // Or with an explicit loop:
        //   for k in 0..(self.polynomial.evals.len() / 2) {
        //       s0 += self.polynomial.evals[2 * k];
        //       s1 += self.polynomial.evals[2 * k + 1];
        //   }
        assert!(!self.is_done(), "SumcheckProver::compute_round_message: protocol is already done");

        let s0: Fp = self.polynomial.evals.iter().step_by(2).copied().sum();
        let s1: Fp = self.polynomial.evals.iter().skip(1).step_by(2).copied().sum();

        [s0, s1]
    }

    /// Receive verifier's challenge `r` for the current variable, and reduce
    /// the polynomial to one fewer variable.
    pub fn receive_challenge(&mut self, r: Fp) {
        // TODO:
        //   1. Panic if is_done().
        //   2. Push `r` onto self.challenges.
        //   3. Update self.polynomial = self.polynomial.fix_first_variable(r).
        assert!(!self.is_done(), "SumcheckProver::receive_challenge: protocol is already done");
        self.challenges.push(r);
        self.polynomial = self.polynomial.fix_first_variable(r);
    }

    /// Get the accumulated challenges (for testing / inspection).
    pub fn challenges(&self) -> &[Fp] {
        &self.challenges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_message_for_known_input() {
        // Construct a v=2 poly with evals = [1, 2, 3, 4] under the LSB-first convention
        // (see polynomial.rs module docs):
        //
        //   evals[0] = f(0, 0) = 1   (i = 00)
        //   evals[1] = f(1, 0) = 2   (i = 01)
        //   evals[2] = f(0, 1) = 3   (i = 10)
        //   evals[3] = f(1, 1) = 4   (i = 11)
        //
        // The first round message [s_0(0), s_0(1)]:
        //   s_0(0) = sum over b in {0,1} of f(0, b) = f(0, 0) + f(0, 1) = 1 + 3 = 4
        //   s_0(1) = sum over b in {0,1} of f(1, b) = f(1, 0) + f(1, 1) = 2 + 4 = 6
        //
        // Note: even-indexed evals (0, 2) → s_0(0); odd-indexed evals (1, 3) → s_0(1).
        //
        // TODO: build the polynomial and prover, call compute_round_message, assert [4, 6].
        let polynomial = MultilinearPoly::new(2, vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)]);
        let mut prover = SumcheckProver::new(polynomial);
        let message = prover.compute_round_message();
        assert_eq!(message, [Fp::new(4), Fp::new(6)]);
    }

    #[test]
    fn after_all_rounds_is_done() {
        // TODO: build a small poly, run v rounds with arbitrary challenges, assert is_done().
        let polynomial = MultilinearPoly::new(2, vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)]);
        let mut prover = SumcheckProver::new(polynomial);
        for _ in 0..2 {
            prover.receive_challenge(Fp::new(1));
        }
        assert!(prover.is_done());
    }
}
