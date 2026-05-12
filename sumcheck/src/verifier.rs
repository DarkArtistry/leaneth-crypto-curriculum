//! The sumcheck verifier.
//!
//! ## Invariants
//!
//! - `claimed_sum` is the prover's claim about `H = sum over hypercube of g(b)`.
//!   Set once, at construction; never modified.
//! - `current_claim` tracks the verifier's "running expectation":
//!   - Before round 0: `current_claim = claimed_sum`.
//!   - After round `i` (and challenge `r_{i+1}` is sampled):
//!     `current_claim = s_i(r_{i+1})`.
//! - Per-round check: `s_i(0) + s_i(1) == current_claim` (where `current_claim`
//!   was set in the previous round, or to `claimed_sum` for round 0).
//! - Final check: `g(r_1, ..., r_v) == current_claim` after the last round.
//!
//! ## Soundness
//!
//! Per round, a cheating prover passes the check with probability at most
//! `d / |F|` (Schwartz-Zippel; `d=1` for multilinear). Union bound across
//! `v` rounds: total error is at most `v / |F|` ~ `2^-58` for our `F_p`.

use crate::field::Fp;
use crate::polynomial::MultilinearPoly;
use rand::Rng;

/// Sumcheck verifier state.
pub struct SumcheckVerifier<R: Rng> {
    /// The prover's claim about the hypercube sum.
    claimed_sum: Fp,
    /// Total number of variables; equals the number of rounds we expect.
    num_vars: usize,
    /// Running expectation, updated each round.
    current_claim: Fp,
    /// Challenges sampled so far.
    challenges: Vec<Fp>,
    /// RNG used to sample challenges.
    rng: R,
}

impl<R: Rng> SumcheckVerifier<R> {
    /// Construct a verifier given the prover's claimed sum and the number of variables.
    pub fn new(claimed_sum: Fp, num_vars: usize, rng: R) -> Self {
        // TODO: store params; initialize current_claim = claimed_sum, challenges = vec![].
        Self {
            claimed_sum,
            num_vars,
            current_claim: claimed_sum,
            challenges: vec![],
            rng,
        }
    }

    /// Process round-`i` message `[s_i(0), s_i(1)]`.
    ///
    /// Steps:
    ///   1. Check `s_i(0) + s_i(1) == self.current_claim`. If not, return Err.
    ///   2. Sample a random challenge `r in F`.
    ///   3. Update `self.current_claim = s_i(r)`. Recall: for degree-1 `s_i`,
    ///      `s_i(r) = s_i(0) + r * (s_i(1) - s_i(0))`.
    ///   4. Push `r` onto `self.challenges`.
    ///   5. Return `Ok(r)` so the protocol orchestrator can pass it to the prover.
    pub fn process_round_message(&mut self, msg: [Fp; 2]) -> Result<Fp, &'static str> {
        // TODO
        if msg[0] + msg[1] != self.current_claim {
            return Err("round message does not sum to current claim");
        }

        let r = Fp::random(&mut self.rng);
        self.current_claim = msg[0] + r * (msg[1] - msg[0]);
        self.challenges.push(r);
        Ok(r)
    }

    /// Final consistency check: `g(r_1, ..., r_v) == self.current_claim`.
    ///
    /// This is the only place the verifier evaluates `g` directly.
    pub fn final_check(&self, g: &MultilinearPoly) -> Result<(), &'static str> {
        // TODO:
        //   1. Verify self.challenges.len() == self.num_vars.
        //   2. Compute g.evaluate(&self.challenges).
        //   3. Compare with self.current_claim. Return Ok(()) if equal, Err otherwise.
        assert_eq!(self.challenges.len(), self.num_vars, "SumcheckVerifier::final_check: self.challenges.len() != self.num_vars");

        let g_eval = g.evaluate(&self.challenges);
        if g_eval == self.current_claim {
            Ok(())
        } else {
            Err("SumcheckVerifier::final_check: g_eval != self.current_claim")
        }
    }

    /// The challenges sampled so far. Useful for tests and the demo.
    pub fn challenges(&self) -> &[Fp] {
        &self.challenges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn first_round_check_passes_on_consistent_message() {
        // TODO:
        //   1. Build a verifier with claimed_sum = 10, num_vars = 2.
        //   2. Send msg = [s_0(0), s_0(1)] where s_0(0) + s_0(1) = 10 (e.g., [3, 7]).
        //   3. Assert process_round_message returns Ok(_).
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut verifier = SumcheckVerifier::new(Fp::new(10), 2, rng);
        let msg = [Fp::new(3), Fp::new(7)];
        assert!(verifier.process_round_message(msg).is_ok());
    }

    #[test]
    fn first_round_check_fails_on_inconsistent_message() {
        // TODO:
        //   1. Build a verifier with claimed_sum = 10.
        //   2. Send msg = [3, 5]. Sum is 8 != 10.
        //   3. Assert process_round_message returns Err.
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut verifier = SumcheckVerifier::new(Fp::new(10), 2, rng);
        let msg = [Fp::new(3), Fp::new(5)];
        assert!(verifier.process_round_message(msg).is_err());
    }
}
