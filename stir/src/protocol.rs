//! End-to-end STIR orchestrator.
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
// NOTE: `StirProver`, `StirVerifier`, and `Transcript` are referenced by name
// in the TODO blocks below and in the rustdoc. The actual `use` statements
// will be added once the function bodies are implemented; for now they would
// be flagged as unused-imports.
// use crate::prover::StirProver;
// use crate::transcript::Transcript;
// use crate::verifier::StirVerifier;

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
    // TODO:
    //   1. Defensive input check: `if let Some(d) = polynomial.degree() {
    //         if d >= params.initial_degree_bound {
    //             return Err("run_stir: polynomial degree exceeds initial_degree_bound");
    //         }
    //      }`
    //      WHY: catch the most common caller mistake (wrong-degree witness)
    //      at the protocol boundary instead of deep inside the prover.
    //   2. Build a fresh transcript:
    //         let mut transcript = Transcript::new(STIR_DOMAIN_SEPARATOR);
    //      WHY: domain separation prevents cross-protocol replay (see
    //      module-level CAUTION).
    //   3. Construct the prover and run it:
    //         let prover = StirProver::new(params, polynomial);
    //         let proof = prover.prove(&mut transcript);
    //      WHY: ownership cleanly transfers; the prover's internal state
    //      is consumed inside `prove`.
    //   4. Return Ok(proof).
    let _ = (params, polynomial);
    todo!()
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
    // TODO:
    //   1. Call `run_stir` to produce the proof. Propagate Err.
    //      WHY: the prover-side error is the same as `run_stir`'s.
    //   2. Build a FRESH verifier transcript with the same domain separator:
    //         let mut transcript = Transcript::new(STIR_DOMAIN_SEPARATOR);
    //      WHY: verifier and prover transcripts are independent objects;
    //      they must converge by ABSORBING the same prover messages, not
    //      by sharing state.
    //   3. Construct the verifier:
    //         let verifier = StirVerifier::new(params.clone());
    //      (`params` was moved into `run_stir`; clone before the move, or
    //      restructure to keep `params` accessible — implementer's choice.
    //      In a clean implementation, pass `params` by reference into a
    //      shared helper. For the stub we keep the high-level shape clear.)
    //   4. Run `let accepted = verifier.verify(&proof, &mut transcript).is_ok();`
    //   5. Return `Ok((proof, accepted))`.
    let _ = (params, polynomial);
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// run_stir on a valid input produces a proof (Ok).
    #[test]
    fn run_stir_produces_proof() {
        // TODO:
        //   1. Build params (small). Build a low-degree polynomial.
        //   2. let result = run_stir(params, polynomial);
        //   3. assert!(result.is_ok());
        //   4. let proof = result.unwrap();
        //   5. assert_eq!(proof.round_commitments.len(), params.num_rounds).
        // WHY: smoke test the top-level entry point on the happy path.
        todo!()
    }

    /// run_stir_with_verification accepts an honest prover.
    #[test]
    fn run_stir_with_verification_accepts_honest_prover() {
        // TODO:
        //   1. Build params (small). Build a low-degree polynomial.
        //   2. let (proof, accepted) = run_stir_with_verification(params, polynomial).unwrap();
        //   3. assert!(accepted, "honest prover must convince the verifier");
        //   4. assert!(!proof.round_commitments.is_empty()).
        // WHY: completeness of the protocol — the cornerstone happy-path
        // contract.
        todo!()
    }

    /// run_stir_with_verification rejects an input that exceeds the degree bound.
    #[test]
    fn run_stir_with_verification_rejects_low_degree_violator() {
        // TODO:
        //   1. Build params with initial_degree_bound = 4.
        //   2. Build a polynomial of degree 8 (twice the bound).
        //   3. let result = run_stir_with_verification(params, polynomial);
        //   4. assert!(result.is_err()).
        // WHY: the protocol must reject inputs outside its declared bound.
        // This catches mis-specified callers before the verifier even
        // gets a chance to run.
        todo!()
    }
}
