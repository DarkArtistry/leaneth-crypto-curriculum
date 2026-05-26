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
use stir::prover::StirProver;
use stir::transcript::Transcript;
use stir::verifier::StirVerifier;

/// Domain separator used by the protocol layer. Keep in lock-step with
/// `stir::protocol::STIR_DOMAIN_SEPARATOR`.
const STIR_DOMAIN_SEPARATOR: &[u8] = b"stir-protocol-v0";

/// Demo parameters used across the integration tests.
fn demo_params() -> StirParams {
    StirParams::new(6, 16, 4)
        .with_security_bits(32)
        .with_ood_samples(1)
        .with_stopping_degree(4)
        .with_repetition_schedule(vec![8, 4, 2])
}

/// Honest demo polynomial: coefficients `[1, 2, ..., 16]` (degree 15).
fn demo_polynomial() -> UnivariatePoly {
    let coeffs: Vec<Fp> = (1u64..=16).map(Fp::new).collect();
    UnivariatePoly::new(coeffs)
}

/// Re-run the verifier on a (possibly tampered) proof with a fresh
/// transcript. Returns `true` iff the verifier accepts.
fn reverify(params: &StirParams, proof: &stir::StirProof) -> bool {
    let verifier = StirVerifier::new(params.clone());
    let mut transcript = Transcript::new(STIR_DOMAIN_SEPARATOR);
    verifier.verify(proof, &mut transcript).is_ok()
}

/// Honest prover on small params: verifier accepts.
#[test]
fn honest_prover_accepts() {
    let params = demo_params();
    let poly = demo_polynomial();

    let result = run_stir_with_verification(params, poly);
    assert!(
        matches!(result, Ok((_, true))),
        "honest run must produce Ok((_, true)); got {:?}",
        result.as_ref().map(|(_, ok)| *ok).map_err(|e| *e),
    );
}

/// Cheater claiming a high-degree polynomial is low-degree: verifier rejects.
#[test]
fn rejects_high_degree_polynomial() {
    // Default `StirParams::new(6, 16, 4)` has initial_degree_bound = 16.
    // Build a polynomial of degree 17 (over the bound).
    let params = StirParams::new(6, 16, 4);
    let coeffs: Vec<Fp> = (1u64..=18).map(Fp::new).collect(); // degree 17
    let poly = UnivariatePoly::new(coeffs);

    // Either the prover panics (release builds would skip the
    // debug_assert; we catch the panic via std::panic::catch_unwind to
    // make the test robust across build profiles), or the protocol
    // layer fast-fails with Err, or the verifier rejects. The
    // assertion is "the result is not Ok((_, true))".
    let result = std::panic::catch_unwind(|| {
        run_stir_with_verification(params, poly)
    });
    let not_accepted = match result {
        Ok(Ok((_, accepted))) => !accepted,
        Ok(Err(_)) => true,   // protocol-layer Err
        Err(_) => true,       // prover panicked
    };
    assert!(
        not_accepted,
        "high-degree polynomial must not be accepted by the verifier",
    );
}

/// Tampered Merkle path in the proof: verifier rejects.
#[test]
fn rejects_tampered_merkle_path() {
    let params = demo_params();
    let poly = demo_polynomial();

    // Build the honest proof via the lower-level prover so we can
    // re-verify the tampered proof afterwards. (run_stir_with_verification
    // would also work but it would consume the proof in the verifier;
    // we want the proof object to mutate.)
    let prover = StirProver::new(params.clone(), poly);
    let mut prover_transcript = Transcript::new(STIR_DOMAIN_SEPARATOR);
    let mut proof = prover.prove(&mut prover_transcript);

    // Sanity: honest proof verifies.
    assert!(reverify(&params, &proof), "honest proof must verify");

    // Tamper: flip a byte in the first sibling of the first path of
    // round 0. |L_0| = 64 > 1, so the path has at least one sibling.
    assert!(
        !proof.merkle_paths[0][0][0].siblings.is_empty(),
        "expected non-empty sibling vector",
    );
    proof.merkle_paths[0][0][0].siblings[0][0] ^= 0xFF;

    assert!(
        !reverify(&params, &proof),
        "verifier must reject after a Merkle-sibling byte flip",
    );
}

/// Tampered OOD reply in the proof: verifier rejects.
#[test]
fn rejects_tampered_ood_reply() {
    let params = demo_params();
    let poly = demo_polynomial();

    let prover = StirProver::new(params.clone(), poly);
    let mut prover_transcript = Transcript::new(STIR_DOMAIN_SEPARATOR);
    let mut proof = prover.prove(&mut prover_transcript);

    assert!(reverify(&params, &proof), "honest proof must verify");

    // Mutate the first OOD reply of round 0.
    proof.ood_replies[0][0] = proof.ood_replies[0][0] + Fp::one();

    assert!(
        !reverify(&params, &proof),
        "verifier must reject after an OOD-reply mutation",
    );
}

/// Final polynomial of wrong (too-high) degree in the proof: verifier rejects.
#[test]
fn rejects_wrong_final_polynomial_degree() {
    let params = demo_params();
    let poly = demo_polynomial();

    let prover = StirProver::new(params.clone(), poly);
    let mut prover_transcript = Transcript::new(STIR_DOMAIN_SEPARATOR);
    let mut proof = prover.prove(&mut prover_transcript);

    assert!(reverify(&params, &proof), "honest proof must verify");

    // Replace final_polynomial with one whose degree equals
    // stopping_degree (= 4 by demo_params). The check is
    // `degree >= stopping_degree → reject`, so degree 4 hits exactly
    // the rejection boundary. We pad with zeros up to length 5 (so
    // degree() returns Some(4)).
    let stopping_degree = params.stopping_degree;
    let mut high_coeffs = vec![Fp::zero(); stopping_degree + 1];
    high_coeffs[stopping_degree] = Fp::new(7); // non-zero leading coeff
    proof.final_polynomial = UnivariatePoly::new(high_coeffs);

    assert!(
        !reverify(&params, &proof),
        "verifier must reject when final_polynomial.degree() >= stopping_degree",
    );
}

/// Final polynomial replaced with a different low-degree polynomial: verifier rejects.
#[test]
fn rejects_wrong_final_polynomial_consistency() {
    let params = demo_params();
    let poly = demo_polynomial();

    let prover = StirProver::new(params.clone(), poly);
    let mut prover_transcript = Transcript::new(STIR_DOMAIN_SEPARATOR);
    let mut proof = prover.prove(&mut prover_transcript);

    assert!(reverify(&params, &proof), "honest proof must verify");

    // Substitute a *different* polynomial of degree < stopping_degree
    // (= 4). We pick a degree-1 polynomial whose evaluation almost
    // certainly disagrees with the honest one at fresh L_M points.
    // Guard against the astronomically-unlikely case that the honest
    // final_polynomial happens to coincide on the sampled points.
    let candidate = UnivariatePoly::new(vec![Fp::new(42), Fp::new(13)]);
    // If by extreme coincidence the honest poly equals `candidate` as a
    // polynomial, perturb again. (Almost never fires.)
    let new_final = if proof.final_polynomial.coeffs() == candidate.coeffs() {
        UnivariatePoly::new(vec![Fp::new(99), Fp::new(7)])
    } else {
        candidate
    };
    proof.final_polynomial = new_final;

    assert!(
        !reverify(&params, &proof),
        "verifier must reject when final_polynomial disagrees with the \
         folded last-round interpolant",
    );
}
