//! # Sumcheck Protocol
//!
//! Educational implementation of the multilinear sumcheck protocol of
//! Lund-Fortnow-Karloff-Nisan (1992), as presented in Thaler §4.1.
//!
//! ## Architecture
//!
//! - [`field`]: a toy prime field `F_p` over the Mersenne prime `2^61 - 1`.
//! - [`polynomial`]: multilinear polynomials in evaluation form over the boolean hypercube.
//! - [`prover`]: the [`prover::SumcheckProver`] type — keeps state across rounds.
//! - [`verifier`]: the [`verifier::SumcheckVerifier`] type — randomized challenger.
//! - [`protocol`]: the [`protocol::run_sumcheck`] orchestrator.
//!
//! ## What sumcheck proves
//!
//! Given a multilinear polynomial `g: F^v -> F` (public, known to both parties)
//! and a claim `H = sum over b in {0,1}^v of g(b)`,
//! sumcheck reduces the hypercube-sum claim to a single-point evaluation claim,
//! with soundness error at most `v · d / |F|` where `d` is the per-variable degree
//! (1 for multilinear).
//!
//! ## Usage
//!
//! ```ignore
//! use sumcheck::{field::Fp, polynomial::MultilinearPoly, protocol::run_sumcheck};
//! use rand::SeedableRng;
//!
//! let evals = vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)];
//! let poly = MultilinearPoly::new(2, evals);
//! let rng = rand::rngs::StdRng::seed_from_u64(42);
//! let result = run_sumcheck(&poly, rng);
//! assert!(result.is_ok());
//! ```

#![warn(missing_docs)]

pub mod field;
pub mod polynomial;
pub mod prover;
pub mod verifier;
pub mod protocol;
