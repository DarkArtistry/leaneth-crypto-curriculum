//! Fiat-Shamir transcript over BLAKE3.
//!
//! ## What this module does, in one paragraph
//!
//! STIR is an **interactive** proof in its native form: prover and
//! verifier exchange messages, with the verifier sampling fresh
//! randomness at each round. To make it **non-interactive**, we apply
//! the **Fiat-Shamir transform**: instead of asking the verifier for
//! randomness, the prover hashes the transcript-so-far and uses the
//! hash output as the "verifier's challenge". The verifier later
//! recomputes the same hashes and checks that the prover's claimed
//! challenges match. This module implements the two operations that
//! make this work: `absorb`, which extends the running hash with a
//! prover message, and `squeeze_*`, which derives uniform random
//! elements of the protocol's finite field `F_p` — Goldilocks,
//! `p = 2^64 − 2^32 + 1` — or array indices from the running hash. The
//! hash function is BLAKE3 (fast, no extension-field arithmetic,
//! well-suited to short-input PRG usage).
//!
//! ## Worked numeric example
//!
//! A two-round STIR transcript could be invoked as:
//!
//! ```text
//! let mut t = Transcript::new(b"stir-demo-v1");
//! t.absorb(&commitment_root_round_0);     // 32-byte Merkle root
//! let alpha_0 = t.squeeze_field();        // Fold randomness for round 0
//! t.absorb_field(beta_0_1);               // OOD answer #1, round 0
//! t.absorb_field(beta_0_2);               // OOD answer #2, round 0
//! let z_0 = t.squeeze_field();            // OOD point for round 1
//! let queries_0 = t.squeeze_indices(8, 64); // 8 Merkle indices in [0, 64)
//! t.absorb(&commitment_root_round_1);
//! // ... and so on.
//! ```
//!
//! The interleaving of absorbs and squeezes is what makes Fiat-Shamir
//! **transcript-binding**: a cheating prover cannot pick the round-0
//! commitment after seeing `alpha_0`, because BLAKE3 is collision-
//! resistant and pre-image-resistant.
//!
//! ## Named theorems & derivations
//!
//! ### Random-oracle soundness of Fiat-Shamir
//!
//! > **Fiat-Shamir theorem (informal, BR93).** If a public-coin
//! > interactive proof `Π` is sound with error `ε(λ)`, and `H` is
//! > modelled as a random oracle, then the non-interactive proof
//! > obtained by replacing the verifier's coin tosses with `H` of the
//! > transcript-so-far is also sound, with error
//! >
//! > ```text
//! > ε_FS(λ) = ε(λ) + Q · 2^{-out_bits(H)}.
//! > ```
//! >
//! > Here `Q` is the number of oracle queries the adversary makes; the
//! > additive term is the probability of *colliding* with a target
//! > challenge. For BLAKE3 (256-bit output) and any realistic `Q ≤ 2^80`,
//! > the additive loss is `≤ 2^{-176}` — negligible.
//!
//! For STIR specifically we need a slightly stronger property:
//! **round-by-round (RBR) soundness**. The interactive STIR is shown
//! RBR-sound in §4 of the paper, which is what licenses our use of
//! Fiat-Shamir here. A merely "soundness in the static sense" proof
//! would NOT suffice — see Canetti-Chen-Holmgren-Lombardi-Ma-Vatandoost
//! (2019) for examples of soundness-but-not-RBR-soundness protocols
//! that Fiat-Shamir breaks.
//!
//! ### Why rejection sampling for `squeeze_field`
//!
//! The field is `F_p` with `p = 2^64 − 2^32 + 1` (Goldilocks). BLAKE3
//! outputs uniform 256-bit strings. Mapping bytes to `F_p` naively via
//! `u64::from_le_bytes(...) % p` introduces **modulo bias**: there are
//! `2^64 / p ≈ 1 + ε` complete cycles of `[0, p)` in the `u64` range,
//! plus a leftover `2^64 mod p = 2^32 - 1` values that map to
//! `[0, 2^32 - 1)`. So the values `[0, 2^32 - 1)` would be sampled
//! with probability `(2^64 / p + 1) / 2^64`, slightly higher than
//! values in `[2^32 - 1, p)`. The bias is small (`~2^{-32}` relative)
//! but cryptographically detectable — and the RBR-soundness reduction
//! assumes uniform sampling. So we **rejection-sample**: draw a fresh
//! `u64`, accept if `< p`, else re-draw. Expected number of attempts
//! is `2^64 / p ≈ 1 + 2^{-32}`, so the cost is negligible.
//!
//! ### Why rejection sampling for `squeeze_indices`
//!
//! Same issue, different range. `squeeze_indices(count, range_exclusive)`
//! returns `count` integers uniformly in `[0, range_exclusive)`. A
//! naive `bytes_as_u32 % range` biases when `range` doesn't divide
//! `2^32`. For Merkle query indices on a domain of size `|L_i|`
//! (always a power of 2 in this codebase), the bias would be zero —
//! BUT we *also* call this function for OOD rejection (when the OOD
//! sampler hits an `F_p` element that happens to live in `L_{i+1}`,
//! we reject and re-sample). To keep the API safe-by-default, we
//! always rejection-sample.
//!
//! ## Caveats for implementers
//!
//! ```text
//! // CAUTION: every prover-to-verifier message MUST be absorbed before
//! //          the next challenge is squeezed. Forgetting even one
//! //          absorb makes the resulting non-interactive proof
//! //          **forgeable** — the prover could choose any
//! //          un-absorbed message to favor its desired challenge.
//! //          This is the #1 Fiat-Shamir implementation bug.
//!
//! // CAUTION: `squeeze_indices` rejection-samples to give uniform
//! //          [0, range_exclusive). Do NOT optimize this to `bytes %
//! //          range`: even when `range` is a power of 2 (and there's
//! //          no bias), the codebase calls this with non-power-of-2
//! //          ranges from `ood::sample_ood_points` rejection logic.
//! //          Keep the rejection sampler universal.
//!
//! // CAUTION: BLAKE3's `update` is order-sensitive (it's a hash). If
//! //          you absorb the same bytes in two different orders, you
//! //          get different transcript states. Helpful — it's what
//! //          makes transcript-binding work — but easy to mess up
//! //          across prover/verifier implementations. Cross-check by
//! //          dumping the absorbed-bytes sequence on both sides.
//! ```
//!
//! ## See also
//!
//! - [`crate::params::StirParams`] for the parameters that determine
//!   how many challenges are squeezed per round.
//! - [`crate::ood`] for the out-of-domain sampler, which calls
//!   [`Transcript::squeeze_field`] then rejects if the result lies in
//!   the upcoming evaluation domain.
//! - [`crate::prover::StirProver::prove`] and
//!   [`crate::verifier::StirVerifier::verify`] for the canonical
//!   absorb/squeeze interleaving.

use reed_solomon::Fp;

/// A Fiat-Shamir transcript backed by BLAKE3.
///
/// The transcript holds a running `blake3::Hasher` state. `absorb`
/// updates it with a prover message; `squeeze_*` finalizes the current
/// state into a fresh PRG stream, takes the requested amount, then
/// re-seeds the state with the consumed PRG output (so the next
/// squeeze produces independent bytes).
///
/// This struct is deliberately not `Clone` — duplicating a transcript
/// is almost always a bug. If you genuinely need a fork (e.g., for a
/// speculative simulation), wrap the field manually.
pub struct Transcript {
    /// Running BLAKE3 state. Updated by `absorb`; finalized by
    /// `squeeze_*` to produce the PRG output, then re-seeded with the
    /// consumed output so subsequent squeezes are independent.
    hasher: blake3::Hasher,
}

impl Transcript {
    /// Initialize a fresh transcript with a **domain separator**.
    ///
    /// The domain separator is a short string identifying the protocol
    /// (e.g. `b"stir-v1"`). It prevents transcripts from one protocol
    /// from being confused with another's — a different domain
    /// separator yields a disjoint set of possible transcript states.
    ///
    /// # Paper reference
    ///
    /// Standard domain-separation practice; see RFC 9380 (hash-to-curve)
    /// for the general principle and BR93 for the original Fiat-Shamir
    /// formulation.
    pub fn new(domain_separator: &[u8]) -> Self {
        // TODO:
        //   1. Create a fresh `blake3::Hasher`.
        //   2. Absorb the domain separator's length (8 bytes,
        //      little-endian) followed by the bytes themselves. The
        //      length prefix is the standard fix to ensure
        //      `absorb(a) ++ absorb(b)` ≠ `absorb(a ++ b)`.
        //   3. Return `Self { hasher }`.
        let _ = domain_separator;
        todo!()
    }

    /// Absorb a raw byte string into the transcript.
    ///
    /// Use for Merkle roots, serialized OOD answers, or any other
    /// prover-to-verifier message that isn't naturally a single field
    /// element. For field elements specifically, prefer
    /// [`Transcript::absorb_field`] for clarity.
    pub fn absorb(&mut self, bytes: &[u8]) {
        // TODO:
        //   1. Absorb the byte string's length (8 bytes, little-endian)
        //      first — same length-prefix discipline as `new`.
        //   2. Absorb the bytes themselves via `hasher.update(bytes)`.
        //   3. No return; mutates `self`.
        //   4. Cross-reference module-doc CAUTION about always absorbing
        //      before squeezing.
        let _ = bytes;
        todo!()
    }

    /// Absorb a single Goldilocks field element.
    ///
    /// Serializes `x` as its 8-byte little-endian canonical form (from
    /// [`reed_solomon::Fp::as_u64`]), then absorbs.
    pub fn absorb_field(&mut self, x: Fp) {
        // TODO:
        //   1. Call `x.as_u64().to_le_bytes()` to canonicalize as 8 bytes.
        //   2. Call `self.absorb(&bytes)` for length-prefixed absorption.
        //   3. No return.
        let _ = x;
        todo!()
    }

    /// Squeeze a single uniformly-random Goldilocks field element.
    ///
    /// Uses rejection sampling (see module docs §"Why rejection
    /// sampling for `squeeze_field`") to avoid modulo bias. Expected
    /// number of inner draws: `~1.0` (acceptance probability
    /// `p / 2^64 > 0.999999999`).
    ///
    /// After squeezing, the transcript's hasher is re-seeded with the
    /// consumed PRG output so that subsequent squeezes are independent.
    pub fn squeeze_field(&mut self) -> Fp {
        // TODO:
        //   1. Domain-separate the squeeze by absorbing a tag like
        //      `b"squeeze_field"` (length-prefixed). This keeps
        //      `squeeze_field` and `squeeze_indices` from producing the
        //      same bytes from the same state.
        //   2. Finalize the hasher with `xof()` to get a PRG stream.
        //   3. Read 8 bytes at a time, interpret as little-endian u64.
        //   4. If `< MODULUS`, accept and break; else re-read.
        //   5. Re-seed the hasher with the *accepted* bytes (to make the
        //      next squeeze independent).
        //   6. Return `Fp::new(accepted_u64)`.
        //   7. Cross-reference §"Why rejection sampling for `squeeze_field`".
        todo!()
    }

    /// Squeeze `count` uniformly-random integers in `[0, range_exclusive)`.
    ///
    /// Returns a `Vec<u32>` of length `count`. Used for Merkle query
    /// indices and as a sub-routine in OOD rejection. Rejection-samples
    /// to avoid modulo bias on non-power-of-2 ranges (see module docs).
    ///
    /// # Panics
    ///
    /// Panics if `range_exclusive == 0` (no valid indices to draw).
    pub fn squeeze_indices(&mut self, count: u32, range_exclusive: u32) -> Vec<u32> {
        // TODO:
        //   1. Panic if `range_exclusive == 0` with a clear message.
        //   2. Domain-separate the squeeze with a tag
        //      `b"squeeze_indices"` (length-prefixed).
        //   3. Compute the largest multiple of `range_exclusive` that
        //      fits in `u32::MAX + 1 = 2^32`, call it `cutoff`:
        //        `cutoff = (2^32 / range_exclusive) * range_exclusive`.
        //      Any draw `< cutoff` can be reduced mod `range_exclusive`
        //      with zero bias; draws `≥ cutoff` are rejected.
        //   4. Finalize the hasher into an XOF stream.
        //   5. Loop until `count` accepted values are collected: read 4
        //      bytes, interpret as little-endian u32, check if `< cutoff`,
        //      if so push `(value % range_exclusive) as u32`.
        //   6. Re-seed the hasher with the consumed XOF bytes (concatenate
        //      all reads so the next squeeze is independent).
        //   7. Return the vector.
        //   8. Cross-reference module-doc CAUTION about uniformity.
        let _ = (count, range_exclusive);
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two transcripts initialized the same way with the same absorbs
    /// must produce the same squeezes — Fiat-Shamir is deterministic.
    #[test]
    fn transcript_is_deterministic() {
        // TODO:
        //   1. Build two transcripts with the same domain separator
        //      `b"test"`.
        //   2. Absorb the same byte string `b"hello"` into both.
        //   3. Squeeze a field element from each and assert equality.
        //   4. Squeeze 5 indices in `[0, 100)` from each and assert
        //      vector equality.
        todo!()
    }

    /// Different absorbs must produce different squeezes — the
    /// transcript must be **non-degenerate**.
    #[test]
    fn different_absorbs_give_different_squeezes() {
        // TODO:
        //   1. Build two transcripts with the same domain separator.
        //   2. Absorb `b"hello"` into the first, `b"world"` into the
        //      second.
        //   3. Squeeze a field element from each and assert inequality
        //      (with cryptographic probability — collision odds ≈ 2^{-64}).
        todo!()
    }

    /// `squeeze_field` must return a value `< MODULUS` (no bias from
    /// failed rejection rounds leaking through).
    #[test]
    fn squeeze_field_is_in_range() {
        // TODO:
        //   1. Build a transcript and squeeze 1000 field elements.
        //   2. For each, assert `result.as_u64() < MODULUS`.
        //   3. Sanity-check the empirical distribution roughly covers
        //      the range (e.g., min < 2^32 with high probability).
        todo!()
    }

    /// `squeeze_indices(count, range)` must return `count` indices, all
    /// in `[0, range)`.
    #[test]
    fn squeeze_indices_are_in_range() {
        // TODO:
        //   1. Build a transcript.
        //   2. Call `squeeze_indices(50, 17)` (17 chosen as a non-power-
        //      of-2 prime to stress the rejection logic).
        //   3. Assert the result has length 50.
        //   4. Assert every entry is `< 17`.
        //   5. Repeat with `range_exclusive = 1` and assert all 50
        //      entries are `0`.
        todo!()
    }
}
