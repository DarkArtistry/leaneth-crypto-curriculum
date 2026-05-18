//! End-to-end sumcheck orchestration, generic over a summation domain `D`.
//!
//! The orchestrator is the shortest layer in the crate: it wires
//! [`SumcheckProver<D>`] to [`SumcheckVerifier<D, R>`] and runs the round loop.
//! The per-variable domain `D` is **implicit in the polynomial parameter** —
//! both [`run_sumcheck`] and [`run_sumcheck_with_claim`] take a
//! [`MultivariatePoly<D>`] and propagate that `D` to both parties. The caller
//! doesn't name `D` separately.
//!
//! ## One round, concretely
//!
//! Let `k = |D|`, so the prover's message is a length-`k` vector of field
//! elements.
//!
//! 1. Prover computes a `Vec<Fp>` of length `k` (one entry per element of
//!    `D`). See [`prover`] module docs for the partition.
//! 2. Verifier checks `sum_j msg[j] == current_claim`. See [`verifier`] module
//!    docs for the sum-over-`S` check and the polynomial-root soundness bound.
//! 3. Verifier samples `r ∈ F`, sets `current_claim = sum_j L_j(r) * msg[j]`
//!    (k-point Lagrange interpolation), and returns `r`.
//! 4. Prover applies `fix_first_variable(r)`, shrinking by one variable.
//!
//! Repeat `n` times, then evaluate `g` at the accumulated challenges and
//! compare against `current_claim`.
//!
//! **Question for the reader.** Why does the round loop terminate after exactly
//! `n` iterations, and what invariant holds after round `i`?
//! Try to answer before reading on.
//!
//! After the protocol has completed exactly `i` rounds (so we've processed
//! Rounds 0 through `i - 1` and accumulated challenges `r_1, ..., r_i`), the
//! prover's polynomial has `n - i` variables and represents
//! `g(r_1, ..., r_i, X_{i+1}, ..., X_n)` — variables `x_1, ..., x_i` have been
//! folded in via `fix_first_variable` against the challenges sampled so far,
//! while `X_{i+1}, ..., X_n` remain free. The verifier's `current_claim` has
//! correspondingly been updated to
//! `sum over (x_{i+1}, ..., x_n) ∈ S^{n - i} of g(r_1, ..., r_i, x_{i+1}, ..., x_n)`.
//! So once `i = n` rounds have completed there are no free variables left, the
//! polynomial has a single stored value `g(r_1, ..., r_n)`, and `current_claim`
//! is the verifier's expectation for that single value — the orchestrator
//! drops out of the loop and runs `final_check` to compare.
//!
//! ## Boolean walk-through, n = 2
//!
//! Take `D = BooleanHypercube`, `n = 2`, `g` stored as
//! `evals = [1, 2, 3, 4]` (so the Boolean-cube sum `H = 1 + 2 + 3 + 4 = 10`).
//!
//! - Round 0. Prover sends `[1 + 3, 2 + 4] = [4, 6]`. Verifier checks
//!   `4 + 6 == 10` ✓. Samples `r_1`, sets `current_claim = (1 - r_1) * 4 +
//!   r_1 * 6 = 4 + 2 * r_1`. Sends `r_1` to the prover.
//! - Round 1. Prover applies `fix_first_variable(r_1)`, gets a 1-var poly,
//!   sends `[s_1(0), s_1(1)]`. Verifier checks the sum, samples `r_2`,
//!   updates the claim.
//! - Final check. Verifier evaluates `g(r_1, r_2)` and confirms it equals the
//!   running `current_claim`.

use crate::domain::SumDomain;
use crate::field::Fp;
use crate::polynomial::MultivariatePoly;
use crate::prover::SumcheckProver;
use crate::verifier::SumcheckVerifier;
use rand::Rng;

/// Run sumcheck on the given polynomial with an honest prover.
///
/// The claimed sum is derived honestly from the polynomial via
/// [`MultivariatePoly::sum_over_domain`]; see [`run_sumcheck_with_claim`] for
/// the version that takes the claim as a parameter (used by tests to inject a
/// wrong claim and check that the verifier catches it).
///
/// Returns `Ok(())` if the verifier accepts.
pub fn run_sumcheck<D: SumDomain, R: Rng>(
    polynomial: &MultivariatePoly<D>,
    rng: R,
) -> Result<(), &'static str> {
    let claimed_sum = polynomial.sum_over_domain();
    run_sumcheck_with_claim(polynomial, claimed_sum, rng)
}

/// Run sumcheck where the *claimed* sum is provided externally.
///
/// In the protocol, both parties know the public polynomial `g`. The
/// `polynomial` here plays both roles — the prover state and the polynomial
/// the verifier evaluates at the final `(r_1, ..., r_n)`. The two parties
/// share the same `D` automatically because both are constructed from the
/// same `polynomial`.
pub fn run_sumcheck_with_claim<D: SumDomain, R: Rng>(
    polynomial: &MultivariatePoly<D>,
    claimed_sum: Fp,
    rng: R,
) -> Result<(), &'static str> {
    let mut prover = SumcheckProver::new(polynomial.clone());
    let mut verifier = SumcheckVerifier::new(
        polynomial.domain.clone(),
        claimed_sum,
        polynomial.n_vars,
        rng,
    );
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
    use crate::domain::BooleanHypercube;
    use rand::SeedableRng;

    #[test]
    fn protocol_accepts_honest_run() {
        // Honest run on the Boolean cube: the verifier accepts when the claim
        // matches the actual sum over D^n.
        let polynomial = MultivariatePoly::new(
            BooleanHypercube,
            2,
            vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)],
        );
        let claimed_sum = polynomial.sum_over_domain();
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        assert!(run_sumcheck_with_claim(&polynomial, claimed_sum, rng).is_ok());
    }

    #[test]
    fn protocol_rejects_wrong_claim() {
        // Same poly, but lie about the sum by one. The verifier rejects;
        // depending on randomness the failure surfaces in some round (or in
        // the final check), but it always surfaces.
        let polynomial = MultivariatePoly::new(
            BooleanHypercube,
            2,
            vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)],
        );
        let claimed_sum = polynomial.sum_over_domain() + Fp::one();
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        assert!(run_sumcheck_with_claim(&polynomial, claimed_sum, rng).is_err());
    }
}
