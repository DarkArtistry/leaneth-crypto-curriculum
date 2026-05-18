//! Integration tests for the full sumcheck protocol.
//!
//! Most tests instantiate the generic protocol on the Boolean hypercube
//! (`S = {0, 1}`) — the case Lean Ethereum and every modern multilinear
//! SNARK use. The final two tests instantiate the same generic
//! `run_sumcheck` on `Interval3` (`S = {0, 1, 2}`) to exercise the
//! `|S| > 2` Lagrange path end-to-end.
//!
//! [`MultilinearPoly`] is the type alias for `MultivariatePoly<BooleanHypercube>`;
//! both names refer to the same generic storage.

use rand::{rngs::StdRng, SeedableRng};
use sumcheck::domain::{BooleanHypercube, Interval3};
use sumcheck::field::Fp;
use sumcheck::polynomial::{MultilinearPoly, MultivariatePoly};
use sumcheck::protocol::{run_sumcheck, run_sumcheck_with_claim};

/// Honest sumcheck on a 3-var Boolean polynomial: verifier accepts.
#[test]
fn sumcheck_accepts_honest_prover_v3() {
    // Build a multilinear poly in 3 variables, run honestly, assert Ok.
    let evals: Vec<Fp> = (1..=8).map(Fp::new).collect();
    let poly = MultilinearPoly::new(BooleanHypercube, 3, evals);
    let rng = StdRng::seed_from_u64(0);
    assert!(run_sumcheck(&poly, rng).is_ok());
}

/// Honest sumcheck scales to 5 variables (32 random evals) on the Boolean cube.
#[test]
fn sumcheck_accepts_honest_prover_v5() {
    // Same, with 5 variables (32 evals).
    let mut rng_setup = StdRng::seed_from_u64(1);
    let evals: Vec<Fp> = (0..32).map(|_| Fp::random(&mut rng_setup)).collect();
    let poly = MultilinearPoly::new(BooleanHypercube, 5, evals);
    let rng = StdRng::seed_from_u64(2);
    assert!(run_sumcheck(&poly, rng).is_ok());
}

/// Off-by-one cheating claim: verifier rejects (soundness on a deterministic lie).
#[test]
fn sumcheck_rejects_wrong_claim_off_by_one() {
    // Build a poly. Compute honest_sum. Inject claimed_sum = honest_sum + Fp::one().
    // Assert run_sumcheck_with_claim returns Err.
    let evals: Vec<Fp> = (1..=8).map(Fp::new).collect();
    let poly = MultilinearPoly::new(BooleanHypercube, 3, evals);
    let wrong_claim = poly.sum_over_hypercube() + Fp::one();
    let rng = StdRng::seed_from_u64(3);
    assert!(run_sumcheck_with_claim(&poly, wrong_claim, rng).is_err());
}

/// Random wrong claims: verifier rejects with overwhelming probability across many seeds.
#[test]
fn sumcheck_rejects_random_wrong_claim() {
    // Same as above, but use a random Fp as the wrong claim. Repeat 10 times.
    let evals: Vec<Fp> = (1..=8).map(Fp::new).collect();
    let poly = MultilinearPoly::new(BooleanHypercube, 3, evals);
    let honest = poly.sum_over_hypercube();
    let mut rng_seed = StdRng::seed_from_u64(4);
    for seed in 0..10 {
        let wrong = Fp::random(&mut rng_seed);
        if wrong == honest { continue; }  // skip the unlikely match
        let rng = StdRng::seed_from_u64(100 + seed);
        assert!(run_sumcheck_with_claim(&poly, wrong, rng).is_err());
    }
}

/// fix_first_variable agrees with full evaluate: `p_fixed(r2, r3) == p(r, r2, r3)`.
#[test]
fn fix_variable_consistency() {
    // Pick a poly p with v=3 variables.
    //   Pick random r.
    //   Let p_fixed = p.fix_first_variable(r).
    //   For random (r2, r3): assert p_fixed.evaluate(&[r2, r3]) == p.evaluate(&[r, r2, r3]).
    let mut rng = StdRng::seed_from_u64(5);
    let evals: Vec<Fp> = (0..8).map(|_| Fp::random(&mut rng)).collect();
    let poly = MultilinearPoly::new(BooleanHypercube, 3, evals);

    let r = Fp::random(&mut rng);
    let r2 = Fp::random(&mut rng);
    let r3 = Fp::random(&mut rng);

    let fixed = poly.fix_first_variable(r);
    assert_eq!(fixed.evaluate(&[r2, r3]), poly.evaluate(&[r, r2, r3]));
}

/// Generic-over-S path: honest sumcheck on Interval3 (k = 3) instead of Boolean.
#[test]
fn sumcheck_accepts_honest_prover_interval3_n2() {
    // g(x_1, x_2) = x_1 + x_2 on S = {0, 1, 2}. Evaluations in mixed-radix LSB-first:
    //   evals[0] = g(0,0) = 0   evals[1] = g(1,0) = 1   evals[2] = g(2,0) = 2
    //   evals[3] = g(0,1) = 1   evals[4] = g(1,1) = 2   evals[5] = g(2,1) = 3
    //   evals[6] = g(0,2) = 2   evals[7] = g(1,2) = 3   evals[8] = g(2,2) = 4
    // Honest sum: 0+1+2+1+2+3+2+3+4 = 18.
    let evals: Vec<Fp> = vec![
        Fp::new(0), Fp::new(1), Fp::new(2),
        Fp::new(1), Fp::new(2), Fp::new(3),
        Fp::new(2), Fp::new(3), Fp::new(4),
    ];
    let poly = MultivariatePoly::new(Interval3, 2, evals);
    let rng = StdRng::seed_from_u64(7);
    assert!(run_sumcheck(&poly, rng).is_ok());
}

/// Off-by-one cheating claim on Interval3: generic verifier still rejects.
#[test]
fn sumcheck_rejects_wrong_claim_interval3() {
    // Same poly as above; inject the honest sum + 1 as a wrong claim.
    let evals: Vec<Fp> = vec![
        Fp::new(0), Fp::new(1), Fp::new(2),
        Fp::new(1), Fp::new(2), Fp::new(3),
        Fp::new(2), Fp::new(3), Fp::new(4),
    ];
    let poly = MultivariatePoly::new(Interval3, 2, evals);
    let honest_sum = poly.sum_over_domain();
    let wrong = honest_sum + Fp::one();
    let rng = StdRng::seed_from_u64(8);
    assert!(run_sumcheck_with_claim(&poly, wrong, rng).is_err());
}
