//! End-to-end integration tests for the STIR protocol.
//!
//! These tests cover the *whole pipeline* — from input polynomial to
//! verifier decision — by calling [`stir::protocol::run_stir_with_verification`].
//! Each test fixes either an honest input (and asserts the verifier
//! accepts) or a tampered/over-budget input (and asserts the verifier
//! rejects). They are the contract sealants on the protocol's
//! completeness + soundness properties at the integration boundary.
//!
//! `// CAUTION:` these tests use intentionally small parameters
//! (`log_initial_domain_size = 6`, `security_bits = 32`) for run-time. The
//! soundness bounds at these settings are well below cryptographic comfort;
//! the tests still detect *honest* completeness and *catastrophic* cheating
//! (wrong final polynomial, tampered Merkle path), which is what they
//! verify. They are not statistical-power tests of soundness margins.

use reed_solomon::{Fp, UnivariatePoly};
use stir::params::StirParams;
use stir::protocol::run_stir_with_verification;

/// Honest prover on small params: verifier accepts.
#[test]
fn stir_honest_prover_accepts_small_params() {
    // TODO: build StirParams with log_initial_domain_size = 6,
    //       folding_factor = 4, num_rounds = 2.
    //   1. Build a UnivariatePoly of degree < 16 (within bound) — e.g.,
    //      coeffs = [Fp::new(i) for i in 1..=16] gives a degree-15 polynomial.
    //   2. Call run_stir_with_verification(params, poly).
    //   3. Assert matches!(result, Ok((_, true))).
    // WHY: completeness — the headline contract. If this fails, the
    // protocol is broken end-to-end.
    todo!()
}

/// Cheater claiming a high-degree polynomial is low-degree: verifier rejects.
#[test]
fn stir_rejects_high_degree_polynomial() {
    // TODO:
    //   1. Build StirParams with initial_degree_bound = 4.
    //   2. Build a UnivariatePoly of degree 15 (way over the bound).
    //   3. Call run_stir_with_verification(params, poly).
    //   4. Assert the result is Err (the protocol layer rejects on the
    //      degree-overflow input check), OR Ok((_, false)) if the layer
    //      doesn't fast-fail and the verifier catches it.
    // WHY: this is the central soundness contract — the verifier (or the
    // protocol-level guard) must reject inputs outside the declared code.
    todo!()
}

/// Tampered Merkle path in the proof: verifier rejects.
#[test]
fn stir_rejects_tampered_merkle_path() {
    // TODO:
    //   1. Run an honest prove to get a valid (proof, true) pair.
    //   2. Flip a single byte of proof.merkle_paths[0][0].siblings[0].
    //   3. Re-run the verifier (build a fresh transcript with the same
    //      domain separator, then StirVerifier::new(params).verify(&proof, ...)).
    //   4. Assert verifier returns Err.
    // WHY: SHA3-256 collision resistance + correct verify logic should
    // catch any one-bit Merkle-path tamper with probability ≈ 1 - 2^{-256}.
    // If this fails, either the Merkle verify is bypassed or the proof
    // wiring doesn't actually use the prover's paths.
    todo!()
}

/// Tampered OOD reply in the proof: verifier rejects.
#[test]
fn stir_rejects_tampered_ood_reply() {
    // TODO:
    //   1. Run an honest prove to get a valid (proof, true) pair.
    //   2. Mutate proof.ood_replies[0][0] = proof.ood_replies[0][0] + Fp::one().
    //   3. Re-run the verifier and assert Err.
    // WHY: the OOD-reply check is the algebraic cousin of the Merkle
    // path check — it pins down the prover's claimed evaluation at random
    // out-of-domain points, which collapses the list-decoding candidate
    // list (Lemma 4.5). A wrong OOD reply must reject.
    todo!()
}

/// Final polynomial of wrong degree in the proof: verifier rejects.
#[test]
fn stir_rejects_wrong_final_polynomial() {
    // TODO:
    //   1. Run an honest prove to get a valid (proof, true) pair.
    //   2. Replace proof.final_polynomial with a polynomial of degree
    //      params.stopping_degree (i.e., AT the bound — should be < the bound).
    //   3. Re-run the verifier and assert Err.
    // WHY: the final-polynomial degree check is the cheapest possible
    // "is the input low-degree?" check. The verifier explicitly rejects
    // `final_polynomial.degree() >= stopping_degree`. This test pins down
    // that check is wired correctly.
    todo!()
}

/// Final polynomial replaced with a different low-degree polynomial: verifier rejects.
#[test]
fn stir_rejects_wrong_final_polynomial_consistency() {
    // TODO:
    //   1. Run an honest prove to get a valid (proof, true) pair.
    //   2. Replace proof.final_polynomial with a DIFFERENT poly of the
    //      SAME degree (so the degree check doesn't fire — we test the
    //      consistency-with-last-round check instead).
    //   3. Re-run the verifier and assert Err.
    // WHY: the final-polynomial consistency check is what pins the
    // protocol terminator to the last-round shift answers. A different
    // (but still low-degree) polynomial would slip past a naive degree-
    // only check; the verifier must cross-check evaluation values too.
    todo!()
}
