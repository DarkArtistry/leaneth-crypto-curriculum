//! Integration tests for the full sumcheck protocol.

use rand::{rngs::StdRng, SeedableRng};
use sumcheck::field::Fp;
use sumcheck::polynomial::MultilinearPoly;
use sumcheck::protocol::{run_sumcheck, run_sumcheck_with_claim};

#[test]
fn sumcheck_accepts_honest_prover_v3() {
    // TODO: build a multilinear poly in 3 variables, run honestly, assert Ok.
    let evals: Vec<Fp> = (1..=8).map(Fp::new).collect();
    let poly = MultilinearPoly::new(3, evals);
    let rng = StdRng::seed_from_u64(0);
    assert!(run_sumcheck(&poly, rng).is_ok());
}

#[test]
fn sumcheck_accepts_honest_prover_v5() {
    // TODO: same, with 5 variables (32 evals).
    let mut rng_setup = StdRng::seed_from_u64(1);
    let evals: Vec<Fp> = (0..32).map(|_| Fp::random(&mut rng_setup)).collect();
    let poly = MultilinearPoly::new(5, evals);
    let rng = StdRng::seed_from_u64(2);
    assert!(run_sumcheck(&poly, rng).is_ok());
}

#[test]
fn sumcheck_rejects_wrong_claim_off_by_one() {
    // TODO: build a poly. Compute honest_sum. Inject claimed_sum = honest_sum + Fp::one().
    // Assert run_sumcheck_with_claim returns Err.
    let evals: Vec<Fp> = (1..=8).map(Fp::new).collect();
    let poly = MultilinearPoly::new(3, evals);
    let wrong_claim = poly.sum_over_hypercube() + Fp::one();
    let rng = StdRng::seed_from_u64(3);
    assert!(run_sumcheck_with_claim(&poly, wrong_claim, rng).is_err());
}

#[test]
fn sumcheck_rejects_random_wrong_claim() {
    // TODO: same as above, but use a random Fp as the wrong claim. Repeat 10 times.
    let evals: Vec<Fp> = (1..=8).map(Fp::new).collect();
    let poly = MultilinearPoly::new(3, evals);
    let honest = poly.sum_over_hypercube();
    let mut rng_seed = StdRng::seed_from_u64(4);
    for seed in 0..10 {
        let wrong = Fp::random(&mut rng_seed);
        if wrong == honest { continue; }  // skip the unlikely match
        let rng = StdRng::seed_from_u64(100 + seed);
        assert!(run_sumcheck_with_claim(&poly, wrong, rng).is_err());
    }
}

#[test]
fn fix_variable_consistency() {
    // TODO: pick a poly p with v=3 variables.
    //   Pick random r.
    //   Let p_fixed = p.fix_first_variable(r).
    //   For random (r2, r3): assert p_fixed.evaluate(&[r2, r3]) == p.evaluate(&[r, r2, r3]).
    let mut rng = StdRng::seed_from_u64(5);
    let evals: Vec<Fp> = (0..8).map(|_| Fp::random(&mut rng)).collect();
    let poly = MultilinearPoly::new(3, evals);

    let r = Fp::random(&mut rng);
    let r2 = Fp::random(&mut rng);
    let r3 = Fp::random(&mut rng);

    let fixed = poly.fix_first_variable(r);
    assert_eq!(fixed.evaluate(&[r2, r3]), poly.evaluate(&[r, r2, r3]));
}
