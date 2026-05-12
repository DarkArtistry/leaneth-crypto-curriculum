//! # Reed-Solomon Codes
//!
//! Educational implementation of Reed-Solomon error-correcting codes, with
//! FFT-based encoding over the Goldilocks field `F_p` where `p = 2^64 - 2^32 + 1`.
//!
//! This crate is the second objective in the leanEthereum coding curriculum.
//! See `../coding_plan.md §5` for context. The previous objective was
//! [`sumcheck`](../../sumcheck) — sumcheck is a standalone protocol that did
//! not need a structured code; here we build the code that everything from
//! STIR / WHIR / FRI lives inside.
//!
//! ## Architecture
//!
//! - [`field`]: Goldilocks `F_p` with `p = 2^64 - 2^32 + 1`. Two-adicity 32, so
//!   we have primitive `2^k`-th roots of unity for every `k <= 32` — exactly
//!   what FFT needs.
//! - [`polynomial`]: univariate polynomials in **coefficient form** (compare
//!   with sumcheck, which used evaluation form on the Boolean cube).
//! - [`domain`]: smooth multiplicative cosets `L = c · <ω>` where `ω` is a
//!   primitive `|L|`-th root of unity and `|L|` is a power of 2. This is the
//!   evaluation domain on which the Reed-Solomon codeword lives.
//! - [`fft`]: Cooley-Tukey radix-2 FFT and inverse FFT.
//! - [`interpolate`]: Lagrange interpolation from arbitrary points; FFT-based
//!   recovery on smooth domains.
//! - [`encode`]: [`encode::ReedSolomonCode`] — the encoder.
//! - [`decode`]: decoder. Naive interpolation-based decoding is required;
//!   Berlekamp-Welch (which corrects errors) is a stretch goal.
//!
//! ## What Reed-Solomon does, in one paragraph
//!
//! Pick a prime field `F`, an evaluation domain `L ⊆ F` of size `n`, and a
//! degree bound `d < n`. The Reed-Solomon code `RS[F, L, d]` is the set of
//! vectors of the form `(p(x))_{x ∈ L}` for some polynomial `p` of degree
//! `< d`. Every codeword **is** the evaluation table of a low-degree
//! polynomial. The "rate" `ρ = d / n` measures how much redundancy we add:
//! `n - d` extra evaluations let the decoder catch up to `(n - d) / 2` errors
//! (unique-decoding radius).
//!
//! In SNARKs we don't usually decode — we just commit to a *purported*
//! codeword and use a low-degree test (FRI / STIR / WHIR) to check that the
//! committed function is close to a real codeword. But the encoder, the
//! domain machinery, and the FFT are universal — every RS-based proof
//! system needs them.
//!
//! ## Reading map
//!
//! - `../../rs_foundations.md` — your existing notes on the evaluation
//!   domain, smooth cosets, why power-of-2 sizes matter.
//! - Thaler's textbook, RS chapter (or any algebraic-coding-theory
//!   textbook) — for the encoder/decoder. Berlekamp-Welch is in most
//!   coding-theory texts under "the key equation".
//! - Cooley & Tukey (1965), "An algorithm for the machine calculation of
//!   complex Fourier series" — the original FFT paper. The Wikipedia
//!   "Cooley-Tukey FFT algorithm" article is a good first read.
//!
//! ## Crate-level usage sketch (after you implement everything)
//!
//! ```ignore
//! use reed_solomon::{
//!     field::Fp,
//!     polynomial::UnivariatePoly,
//!     domain::EvaluationDomain,
//!     encode::ReedSolomonCode,
//! };
//!
//! // Set up: degree-bound 4, domain of size 16 (rate 1/4).
//! let domain = EvaluationDomain::new_subgroup(4); // 2^4 = 16
//! let code = ReedSolomonCode::new(domain, 4);
//!
//! // A polynomial of degree < 4 — its 4 coefficients are the message.
//! let message = UnivariatePoly::new(vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)]);
//!
//! // Encode: evaluate the polynomial on all 16 points of the domain.
//! let codeword = code.encode(&message);
//! assert_eq!(codeword.len(), 16);
//! ```

#![warn(missing_docs)]

pub mod decode;
pub mod domain;
pub mod encode;
pub mod fft;
pub mod field;
pub mod interpolate;
pub mod polynomial;
