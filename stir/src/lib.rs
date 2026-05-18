//! # STIR — low-degree IOP of proximity
//!
//! Educational implementation of the STIR protocol from
//! Arnon-Chiesa-Fenzi-Yogev, "STIR: Reed-Solomon Proximity Testing with
//! Fewer Queries" (eprint 2024/390). Built over the Goldilocks field
//! `F_p` where `p = 2^64 - 2^32 + 1`, using the Reed-Solomon machinery
//! from the [`reed_solomon`] crate.
//!
//! This is the third objective of the leanEthereum coding curriculum.
//! See `../coding_plan.md §6` and `../README.md` for context.
//!
//! ## What STIR proves, in one paragraph
//!
//! Fix a finite field `F` (here Goldilocks `F_p`, `p = 2^64 − 2^32 + 1`)
//! and an **initial evaluation domain** `L_0`, a finite subset of `F` of
//! size `|L_0| = 2^{log_initial_domain_size}`. Fix also an **initial
//! degree bound** `d_0`. The **Reed-Solomon code** `RS[F, L_0, d_0]` is
//! then the set of evaluation tables on `L_0` of polynomials `p` of
//! degree `< d_0`. A prover commits to a function `f_0: L_0 → F` — that
//! is, a function mapping each point of `L_0` to a field element — by
//! Merkle-hashing its evaluation table. The prover wants to convince a
//! verifier that `f_0` is **δ-close in relative Hamming distance** to
//! some codeword in `RS[F, L_0, d_0]`, where the **proximity parameter**
//! `δ ∈ [0, 1)` means agreement on at least `(1 − δ)·|L_0|` points. STIR
//! is an **interactive oracle proof of proximity (IOPP)**: it lets the
//! verifier accept or reject after only `O(log² d_0)` Merkle queries,
//! exponentially fewer than naively reading the whole evaluation table.
//! Combined with Fiat-Shamir (using BLAKE3 as the random oracle), STIR
//! becomes a non-interactive argument suitable as the proximity-testing
//! sub-protocol inside a transparent SNARK.
//!
//! ## Where it sits in the stack
//!
//! ```text
//!    STARK / SNARK constraint system          ← objective 5 of the curriculum
//!              ↓ "low-degree extension"
//!    STIR (this crate) / WHIR / FRI           ← objective 3 (STIR), 4 (WHIR)
//!              ↓ "RS encoding / commitment"
//!    Reed-Solomon code (objective 2)          ← `reed_solomon` crate
//!              ↓
//!    Goldilocks F_p, FFT, Merkle, transcript
//! ```
//!
//! Everything above the proximity layer reduces "computation is correct"
//! to "the committed function `f: L_0 → F` is close (in Hamming
//! distance) to a Reed-Solomon codeword — i.e., to the evaluation table
//! of some low-degree polynomial". STIR is the proof for the proximity
//! claim.
//!
//! ## STIR vs FRI in one table
//!
//! The protocol runs `M` rounds at a **folding factor** `k` (a power of
//! 2, at least 4). Each round commits to `f_i: L_i → F`, the round-`i`
//! committed function on the round-`i` evaluation domain `L_i`. Per
//! round the verifier asks `t_i` **shift queries** (Merkle openings)
//! and `s` **out-of-domain (OOD) samples**. The whole protocol targets
//! `λ` bits of security.
//!
//! | Quantity                | FRI                       | STIR                                |
//! |-------------------------|---------------------------|-------------------------------------|
//! | Domain shrink per round | `|L| / k`                 | `|L| / (k/2)` (slower!)             |
//! | Rate per round          | constant `ρ_0`            | drops geometrically                 |
//! | Degree shrink per round | `d / k`                   | `d / k` (same)                       |
//! | Per-round repetitions   | `t = λ / log(1/ρ_0)`      | `t_i` — harmonic decrease           |
//! | Total queries           | `O(log d · λ / log(1/ρ))` | `O(log² d)` at fixed `λ`            |
//! | Argument size           | linear in queries         | smaller constant                    |
//! | OOD sampling            | absent                    | `s` per round (collapses list)      |
//! | Prover time             | `O(n log n)`              | `O(n log n)` (slight constant up)   |
//!
//! The STIR row buys its query advantage with two structural ideas:
//! (a) the **domain shrinks slower than the degree**, so the rate
//! `ρ_i = d_i / |L_i|` (where `d_i = d_0 / k^i` is the round-`i` degree
//! bound) drops geometrically — smaller rate gives a larger
//! list-decoding radius and thus a smaller per-query soundness error;
//! (b) **out-of-domain (OOD) sampling** — the verifier asks the prover
//! for evaluations of the folded function at random points *outside*
//! `L_i`, which collapses the list-decoding candidate list down to a
//! single codeword. Together, STIR achieves `O(log² d)` total queries
//! where FRI needs `O(log d · poly(λ))`.
//!
//! ## Architecture
//!
//! Thirteen modules. Phase 1 is the primitives layer; phase 2 is the
//! per-round operations; phase 3 is the protocol itself.
//!
//! Phase 1 — primitives:
//! - [`params`]: [`StirParams`], soundness reporting, validation.
//! - [`transcript`]: Fiat-Shamir BLAKE3 transcript with rejection
//!   sampling.
//! - [`merkle`]: binary Merkle tree on evaluation vectors.
//! - [`domain`]: round-by-round domain shifts `L_0 ⊃ L_1 ⊃ ... ⊃ L_M`
//!   (smooth multiplicative cosets with disjoint offsets).
//!
//! Phase 2 — operations:
//! - [`fold`]: `Fold(f, α, k)` — collapses `k` evaluations of `f` into
//!   one evaluation of the folded function.
//! - [`quotient`]: the OOD quotient polynomial (Lemma 4.4 of the paper).
//! - [`degree_correction`]: post-fold multiplier that pushes the folded
//!   function into exactly `RS[F, L_{i+1}, d_{i+1}]`.
//! - [`ood`]: out-of-domain point sampling and OOD evaluation.
//!
//! Phase 3 — protocol:
//! - [`commitment`]: round-specific Merkle commitment to folded evals.
//! - [`proof`]: [`StirProof`] — the full transcript object.
//! - [`prover`]: [`StirProver::prove`](crate::prover::StirProver::prove)
//!   — builds the proof from the input polynomial.
//! - [`verifier`]: [`StirVerifier::verify`](crate::verifier::StirVerifier::verify)
//!   — checks the proof.
//! - [`protocol`]: [`run_stir`] — convenience wrapper for end-to-end
//!   honest runs.
//!
//! ## Reading map
//!
//! - `README.md` in this directory — narrative tour, worked example,
//!   pitfall list.
//! - `../../stir_study.md` — Kenneth's paper-read notes from 2026-05-05
//!   (phases 0-4, AI-executed; revisit due ~2026-05-12).
//! - Arnon-Chiesa-Fenzi-Yogev, "STIR" (eprint 2024/390), §2-5. The
//!   formal protocol description is §3; soundness analysis is §4;
//!   the parameter-tuning guidance is §5.
//! - For background on the FRI predecessor: Ben-Sasson et al., "Fast
//!   Reed-Solomon Interactive Oracle Proofs of Proximity" (2018).
//!
//! ## Usage
//!
//! ```ignore
//! use stir::{StirParams, run_stir};
//! use reed_solomon::{Fp, UnivariatePoly};
//! use rand::SeedableRng;
//!
//! // Set up: |L_0| = 2^10, d_0 = 256 (rate 1/4), fold by 4 per round.
//! let params = StirParams::new(
//!     /* log_initial_domain_size = */ 10,
//!     /* initial_degree_bound    = */ 256,
//!     /* folding_factor          = */ 4,
//! )
//! .with_security_bits(128);
//!
//! // The function we want to prove is low-degree.
//! let coeffs: Vec<Fp> = (0..256).map(|i| Fp::new(i as u64)).collect();
//! let poly = UnivariatePoly::new(coeffs);
//!
//! let mut rng = rand::rngs::StdRng::seed_from_u64(0xc0ffee);
//! let (proof, accepted) = run_stir(&params, &poly, &mut rng);
//! assert!(accepted, "honest prover must convince the verifier");
//! ```

#![warn(missing_docs)]

pub mod commitment;
pub mod degree_correction;
pub mod domain;
pub mod fold;
pub mod merkle;
pub mod ood;
pub mod params;
pub mod proof;
pub mod protocol;
pub mod prover;
pub mod quotient;
pub mod transcript;
pub mod verifier;

pub use params::{RbrSoundnessReport, StirParams};
pub use proof::StirProof;
pub use protocol::run_stir;
pub use transcript::Transcript;
