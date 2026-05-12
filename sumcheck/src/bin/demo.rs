//! A narrative demo of the sumcheck protocol.
//!
//! Run with: `cargo run --bin demo`

use sumcheck::field::Fp;
use sumcheck::polynomial::MultilinearPoly;
use sumcheck::prover::SumcheckProver;
use sumcheck::verifier::SumcheckVerifier;

fn main() {
    // TODO:
    //   1. Build a small multilinear polynomial: v=3 variables, 8 random evals
    //      (use a fixed RNG seed so the output is reproducible).
    //   2. Print the polynomial's evaluations and the honest sum.
    //   3. Set up the prover and verifier.
    //   4. Loop:
    //        - Print "Round i:".
    //        - prover.compute_round_message → print the message.
    //        - verifier.process_round_message → print the challenge.
    //        - prover.receive_challenge.
    //   5. final_check → print accept/reject.
    //
    // This is for human eyes — narrate what's happening at each step.

    println!("=== Sumcheck protocol demo ===");
    println!("(Implement me!)");

    let v = 3;
    let mut rng = rand::thread_rng();

    let evals: Vec<Fp> = (0..(1 << v))
        .map(|_| Fp::random(&mut rng))
        .collect();

    let poly = MultilinearPoly::new(v, evals);
    println!("Polynomial: {:?}", poly);

    let claim_sum = poly.sum_over_hypercube();
    println!("Claimed Sum: {:?}", claim_sum);

    let mut prover = SumcheckProver::new(poly.clone());
    // NOTE: num_vars in the verifier is the number of rounds, variable
    let mut verifier = SumcheckVerifier::new(claim_sum, v, rng);

    println!("Starting sumcheck protocol...");

    for i in 0..v {
        println!("Round {}:", i);
        let msg = prover.compute_round_message();
        println!("  Message: {:?}", msg);
        let r = verifier.process_round_message(msg).unwrap();
        println!("  Challenge: {:?}", r);
        prover.receive_challenge(r);
    }

    let status = if verifier.final_check(&poly).is_ok() {
        "accept"
    } else {
        "reject"
    };

    println!("Final check result: {status}");
}
