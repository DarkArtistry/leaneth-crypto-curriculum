//! End-to-end STIR orchestrator.
//!
//! ## Anchor: STIR's top-level orchestrator
//!
//! This module is the user-facing entry point to the STIR proof system.
//! Every other module in this crate ([`crate::prover`], [`crate::verifier`],
//! [`crate::transcript`], [`crate::merkle`], …) provides one building block;
//! [`run_stir`] and [`run_stir_with_verification`] are the two functions that
//! compose those blocks into a complete honest-prover-and-verifier flow.
//! If you're using STIR as a library — to prove proximity of a function to a
//! Reed-Solomon code, or as the proximity-test sub-protocol inside a
//! transparent SNARK — these two functions are what you call. Internally
//! they each construct a fresh [`crate::transcript::Transcript`] seeded with
//! the same domain separator on both sides, then invoke
//! [`crate::prover::StirProver::prove`] followed by
//! [`crate::verifier::StirVerifier::verify`].
//!
//! ## What this module does
//!
//! Two top-level entry points, both wrapping the round-by-round dance
//! over a polynomial `f_0` over the Goldilocks field `F = F_p`,
//! evaluated on the initial domain `L_0` and folded across
//! `M = num_rounds` rounds:
//!
//! - [`run_stir`] — "prove this polynomial is low-degree". Constructs a
//!   [`crate::prover::StirProver`], builds a Fiat-Shamir transcript, drives
//!   the prover to produce a [`crate::proof::StirProof`], and returns it.
//!   Use this when you want a proof object to ship to a separate verifier
//!   (e.g., when STIR is the proximity layer inside a larger SNARK pipeline).
//! - [`run_stir_with_verification`] — "prove + verify in one shot". Calls
//!   [`run_stir`] to get a proof, then constructs a
//!   [`crate::verifier::StirVerifier`], builds a *fresh* transcript with the
//!   same domain separator, and runs `verify`. Returns `(proof, accepted)`.
//!   Use this in tests and demos where you control both ends and want a
//!   one-call sanity check.
//!
//! Both calls fully encapsulate the transcript instantiation. Callers do not
//! need to know about [`crate::transcript::Transcript`] at all; the protocol
//! layer creates and consumes it internally.
//!
//! ## Walk-through, n_rounds = 2
//!
//! Pick a small parameter set: `log_initial_domain_size = 6` (|L_0| = 64),
//! `folding_factor = 4`, `num_rounds = 2`, `initial_degree_bound = 16`,
//! `stopping_degree = 4`, `ood_samples = 1`, `repetition_schedule = [4, 2, 1]`.
//!
//! 1. **Setup.** Caller passes `(params, polynomial)`. `run_stir` validates
//!    that `polynomial.degree() < params.initial_degree_bound` (defensive
//!    debug check; release builds rely on prover's debug-assert).
//! 2. **Prover transcript.** Build a fresh `Transcript` with domain
//!    separator `b"stir-protocol-v0"`.
//! 3. **Prover.** Build `StirProver::new(params, polynomial)` and call
//!    `.prove(&mut transcript)`. The prover absorbs commitments, OOD
//!    replies, shift answers, and PoW into the transcript as it goes;
//!    every challenge (`α_i`, OOD points, shift indices) is read back from
//!    the same transcript.
//! 4. **Round 0** (inside prover):
//!      a. Commit to the function `f_0: L_0 → F` (the evaluation table of
//!         `f_0` on `L_0`, |L_0| = 64).
//!      b. Sample 1 OOD point, reply with `f_0(z)`.
//!      c. Sample 4 shift indices from `[0, 64)`, reply with the 4 leaves +
//!         their Merkle paths.
//!      d. Derive `α_0`, fold/quotient/degcor → `f_1` (degree < 4).
//! 5. **Round 1** (inside prover): same shape, `|L_1| = 32`, 2 shift queries.
//!      After folding by 4, degree drops to < 1 — a constant; this becomes
//!      `final_polynomial`.
//! 6. **Final.** `StirProof` returned with `round_commitments.len() == 2`,
//!    `ood_replies.len() == 2`, etc., plus `final_polynomial`.
//!
//! For `run_stir_with_verification`, a second freshly-initialised transcript
//! (with the same domain separator) is built for the verifier, which then
//! re-derives the same challenges and runs the round-by-round checks.
//!
//! `// CAUTION:` the transcript's domain separator must be unique per
//! protocol invocation. Reusing the same separator across different proofs
//! (say, of different polynomials with different params) is **safe** —
//! Fiat-Shamir folds the absorbed messages in, and different inputs give
//! different transcripts. But reusing the same separator across protocols
//! that share absorbed-message prefixes (e.g., STIR and a custom variant
//! that also commits with SHA3-256 first) opens a replay attack: a proof
//! valid for one protocol might be accepted by the other. Pick a unique
//! per-protocol literal (`b"stir-protocol-v0"`) and bump it on any protocol
//! change.

use reed_solomon::UnivariatePoly;

use crate::params::StirParams;
use crate::proof::StirProof;
use crate::prover::StirProver;
use crate::transcript::Transcript;
use crate::verifier::StirVerifier;

/// The transcript domain separator used by this crate's protocol entry
/// points. Distinct per-protocol literal protects against cross-protocol
/// replay (see module-level `// CAUTION:`).
const STIR_DOMAIN_SEPARATOR: &[u8] = b"stir-protocol-v0";

/// Run the STIR prover end-to-end on an input polynomial.
///
/// # Inputs
/// - `params`: the fully-formed STIR parameters.
/// - `polynomial`: the input `f_0` (the low-degree witness).
///
/// # Outputs
/// A [`StirProof`] suitable to be shipped to an independent verifier.
///
/// # Errors
/// Returns `Err(&'static str)` if the input is structurally invalid (e.g.,
/// `polynomial.degree() >= params.initial_degree_bound`). The prover itself
/// can also report structured errors; those bubble up here.
///
/// # Paper reference
/// STIR (eprint 2024/390), §3 "Protocol" — the prover's end-to-end role.
pub fn run_stir(
    params: StirParams,
    polynomial: UnivariatePoly,
) -> Result<StirProof, &'static str> {
    // (1) Defensive input check. The prover's constructor has a
    //     debug-assert for this; we mirror it as a structured `Err` at
    //     the protocol boundary so release builds report a clean error
    //     instead of a soundness hole.
    if let Some(d) = polynomial.degree() {
        if d >= params.initial_degree_bound {
            return Err("run_stir: polynomial degree exceeds initial_degree_bound");
        }
    }

    // (2) Build a fresh transcript with the protocol's domain separator.
    let mut transcript = Transcript::new(STIR_DOMAIN_SEPARATOR);

    // (3) Construct the prover and produce the proof.
    let prover = StirProver::new(params, polynomial);
    let proof = prover.prove(&mut transcript);

    Ok(proof)
}

/// Run the STIR prover then the verifier on the resulting proof.
///
/// Convenience for tests and demos where both ends are local. Production
/// uses ship the proof from `run_stir` over a wire to a separate verifier.
///
/// # Inputs
/// Same as [`run_stir`].
///
/// # Outputs
/// `(proof, accepted)` where `accepted` is `true` iff the verifier returned
/// `Ok(())`. The proof is returned regardless of the verifier's decision —
/// callers may want to inspect a rejected proof for debugging.
///
/// # Errors
/// Returns `Err(&'static str)` if `run_stir` itself failed (e.g.,
/// wrong-degree input). A verifier-rejection is **not** an `Err`; it's
/// `Ok((proof, false))`.
///
/// # Paper reference
/// STIR (eprint 2024/390), §3 "Protocol" — the end-to-end honest run.
pub fn run_stir_with_verification(
    params: StirParams,
    polynomial: UnivariatePoly,
) -> Result<(StirProof, bool), &'static str> {
    // (1) Run the honest prover. Propagate input-validation errors.
    //     We clone the params first because `run_stir` moves them; the
    //     verifier needs an independent copy.
    let params_for_verifier = params.clone();
    let proof = run_stir(params, polynomial)?;

    // (2) Build a *fresh* verifier transcript with the same domain
    //     separator. Verifier and prover transcripts are independent
    //     objects that converge by absorbing the same prover messages
    //     in the same order — see the module-level CAUTION.
    let mut transcript = Transcript::new(STIR_DOMAIN_SEPARATOR);

    // (3) Construct the verifier and run it. A verifier-side rejection
    //     is NOT an `Err`; it is `Ok((proof, false))`.
    let verifier = StirVerifier::new(params_for_verifier);
    let accepted = verifier.verify(&proof, &mut transcript).is_ok();

    Ok((proof, accepted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reed_solomon::Fp;

    /// Demo params used across the protocol-layer tests.
    fn demo_params() -> StirParams {
        StirParams::new(6, 16, 4)
            .with_security_bits(32)
            .with_ood_samples(1)
            .with_stopping_degree(4)
            .with_repetition_schedule(vec![8, 4, 2])
    }

    fn demo_polynomial() -> UnivariatePoly {
        let coeffs: Vec<Fp> = (1u64..=15).map(Fp::new).collect();
        UnivariatePoly::new(coeffs)
    }

    /// run_stir on a valid input produces a proof (Ok).
    #[test]
    fn run_stir_produces_proof() {
        let params = demo_params();
        let polynomial = demo_polynomial();
        let m = params.num_rounds as usize;

        let result = run_stir(params, polynomial);
        assert!(result.is_ok(), "honest run_stir must succeed");

        let proof = result.unwrap();
        assert_eq!(proof.round_commitments.len(), m);
    }

    /// run_stir_with_verification accepts an honest prover.
    #[test]
    fn run_stir_with_verification_accepts_honest_prover() {
        let params = demo_params();
        let polynomial = demo_polynomial();

        let (proof, accepted) =
            run_stir_with_verification(params, polynomial).unwrap();
        assert!(accepted, "honest prover must convince the verifier");
        assert!(!proof.round_commitments.is_empty());
    }

    /// run_stir_with_verification rejects an input that exceeds the degree bound.
    #[test]
    fn run_stir_with_verification_rejects_low_degree_violator() {
        // Default `StirParams::new(6, 16, 4)` has initial_degree_bound = 16.
        // Build a polynomial whose degree exceeds the bound.
        let params = StirParams::new(6, 16, 4);
        let too_many_coeffs: Vec<Fp> =
            (1u64..=20).map(Fp::new).collect(); // degree 19 ≥ 16
        let polynomial = UnivariatePoly::new(too_many_coeffs);

        let result = run_stir_with_verification(params, polynomial);
        assert!(result.is_err(), "wrong-degree witness must Err");
    }
}
