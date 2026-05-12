//! End-to-end sumcheck protocol orchestration.
//!
//! Honest version: the prover's claim is computed honestly from the polynomial.
//! See [`run_sumcheck_with_claim`] for the version where the claim is a parameter
//! (used by tests to inject a wrong claim and verify the protocol catches it).

use crate::field::Fp;
use crate::polynomial::MultilinearPoly;
use crate::prover::SumcheckProver;
use crate::verifier::SumcheckVerifier;
use rand::Rng;

/// Run sumcheck on the given polynomial with an honest prover.
///
/// Returns `Ok(())` if the verifier accepts. The verifier samples randomness from `rng`.
pub fn run_sumcheck<R: Rng>(polynomial: &MultilinearPoly, rng: R) -> Result<(), &'static str> {
    // TODO: this is just `run_sumcheck_with_claim` with the honest claim.
    let claimed_sum = polynomial.sum_over_hypercube();
    run_sumcheck_with_claim(polynomial, claimed_sum, rng)
}

/// Run sumcheck where the *claimed* sum is provided externally.
/// Used to test the verifier rejects incorrect claims.
pub fn run_sumcheck_with_claim<R: Rng>(
    polynomial: &MultilinearPoly,
    claimed_sum: Fp,
    rng: R,
) -> Result<(), &'static str> {
    // TODO:
    //   1. Construct prover from a clone of the polynomial.
    //   2. Construct verifier from claimed_sum, polynomial.num_vars, rng.
    //   3. Loop until prover is done:
    //        a. msg = prover.compute_round_message()
    //        b. r = verifier.process_round_message(msg)?
    //        c. prover.receive_challenge(r)
    //   4. verifier.final_check(polynomial)
    //
    // Note: we pass `polynomial` to `final_check` because in the protocol, both
    // parties have access to the public polynomial g. (In a SNARK setting, g would
    // be replaced by an oracle; we don't model that here.)
    let mut prover = SumcheckProver::new(polynomial.clone());
    let mut verifier = SumcheckVerifier::new(claimed_sum, polynomial.num_vars, rng);
    while !prover.is_done() {
        let msg = prover.compute_round_message();
        let r = verifier.process_round_message(msg)?;
        prover.receive_challenge(r);
    }
    verifier.final_check(polynomial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn protocol_accepts_honest_run() {
        // TODO: build a small poly, run, assert Ok.
        let polynomial = MultilinearPoly::new(2, vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)]);
        let claimed_sum = polynomial.sum_over_hypercube();
        let rng = &mut rand::rngs::StdRng::seed_from_u64(42);
        assert!(run_sumcheck_with_claim(&polynomial, claimed_sum, rng).is_ok());
    }

    #[test]
    fn protocol_rejects_wrong_claim() {
        // TODO: build a poly, lie about the sum (claimed_sum = honest_sum + Fp::one()),
        // assert Err. (The error might come from any round, depending on randomness.)
        let polynomial = MultilinearPoly::new(2, vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)]);
        let claimed_sum = polynomial.sum_over_hypercube() + Fp::one();
        let rng = &mut rand::rngs::StdRng::seed_from_u64(42);
        assert!(run_sumcheck_with_claim(&polynomial, claimed_sum, rng).is_err());
    }
}
