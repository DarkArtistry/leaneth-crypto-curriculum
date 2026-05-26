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
//! ## Anchor: what Fiat-Shamir is and why this transcript exists
//!
//! Fix a **public-coin interactive proof** `Π` between a prover `P` and
//! a verifier `V`. "Public-coin" means every verifier message is a
//! uniformly-random bitstring sent in the clear (no secrets, no
//! conditioning on prover state). The conversation
//!
//! ```text
//!   P → V :  m_1
//!   V → P :  r_1   (uniform random)
//!   P → V :  m_2
//!   V → P :  r_2
//!   ...
//!   P → V :  m_n
//!   V       accepts or rejects based on (m_1, r_1, ..., m_n).
//! ```
//!
//! is the entire interaction. The **Fiat-Shamir transform** replaces
//! every random `r_i` with `H(domain_sep ‖ m_1 ‖ r_1 ‖ ... ‖ m_i)` for
//! some hash `H`. The prover, the verifier, and an external auditor can
//! all compute the same `r_i` deterministically from the transcript-
//! so-far; the verifier no longer needs to be interactive, and the
//! protocol becomes a single **non-interactive argument**: the prover
//! ships `(m_1, ..., m_n)` and the verifier checks acceptance after
//! recomputing the `r_i`.
//!
//! The transcript in this module is exactly the running state
//! `domain_sep ‖ m_1 ‖ r_1 ‖ ... ‖ m_i` — implemented incrementally,
//! one BLAKE3 update at a time. `absorb` extends it with an `m_i`;
//! `squeeze_field` / `squeeze_indices` finalises the current state and
//! reads off the corresponding `r_i`.
//!
//! **Why a transcript, and not "sample r_i = H(m_i)"?** Because that
//! short version is forgeable. A prover who can choose `m_2` after
//! seeing `r_1` can iterate over candidate `m_2` values, recompute the
//! would-be `r_2 = H(m_2)`, and keep searching until it lands on a
//! favourable challenge — defeating the soundness of `Π`. Hashing the
//! **whole transcript** rather than just the last message removes that
//! degree of freedom: `r_i` is locked in by everything that has been
//! said so far, and a prover that wants to change any earlier message
//! has to redo all subsequent challenges from scratch — no incremental
//! advantage.
//!
//! ## Worked numeric example: `new(b"stir-test") → absorb_field(Fp(5)) → squeeze_field()`
//!
//! Walk through the exact bytes the BLAKE3 hasher consumes, in order.
//!
//! 1. **`Transcript::new(b"stir-test")`**. Fresh `blake3::Hasher`,
//!    then absorb the domain separator with a length prefix:
//!
//!    ```text
//!    update(0x09 00 00 00 00 00 00 00)   // (9_u64).to_le_bytes()  — length
//!    update(0x73 74 69 72 2d 74 65 73 74) // b"stir-test"          — payload (9 bytes)
//!    ```
//!
//!    Total bytes absorbed: 17. The hasher's internal state is now
//!    `H_1 := BLAKE3-update(BLAKE3-init(), 0x09…00 ‖ "stir-test")`.
//!
//! 2. **`absorb_field(Fp::new(5))`**. Canonicalise `5` to
//!    `0x05 00 00 00 00 00 00 00` (8 little-endian bytes), then
//!    length-prefix-absorb:
//!
//!    ```text
//!    update(0x08 00 00 00 00 00 00 00)   // (8_u64).to_le_bytes()  — length
//!    update(0x05 00 00 00 00 00 00 00)   // Fp(5).as_u64().to_le_bytes() — payload
//!    ```
//!
//!    Hasher state advances to `H_2 := BLAKE3-update(H_1, 0x08…00 ‖ 0x05…00)`.
//!
//! 3. **`squeeze_field()`**. Domain-separate the squeeze itself so
//!    `squeeze_field()` and `squeeze_indices(...)` can never collide on
//!    the same prior state — absorb the tag `b"squeeze_field"`
//!    (13 bytes), length-prefixed:
//!
//!    ```text
//!    update(0x0d 00 00 00 00 00 00 00)             // (13_u64).to_le_bytes()
//!    update(0x73 71 75 65 65 7a 65 5f 66 69 65 6c 64) // b"squeeze_field"
//!    ```
//!
//!    Call the resulting state `H_3`. Now `finalize_xof()` is invoked
//!    on a *clone* of `H_3` (we do not consume `H_3`; we need to
//!    re-seed from it). Pull 8 bytes from the XOF, interpret as a
//!    little-endian `u64`, call it `v`. If `v < MODULUS`, return
//!    `Fp::new(v)`. Otherwise re-mix with `update(b"\x01")` and pull
//!    8 more bytes; repeat. The acceptance probability per attempt is
//!    `p / 2^64 = 1 − 2^{-32} + 2^{-64}`, so the expected number of
//!    attempts is `≈ 1.000000000232…`.
//!
//! 4. **Re-seed for independence.** Whatever bytes were finally
//!    accepted are absorbed back into the live hasher
//!    (`self.hasher.update(&v.to_le_bytes())`), so the next call to
//!    `absorb`/`squeeze_*` sees a state that depends on the accepted
//!    value. Without this step, two consecutive `squeeze_field()`
//!    calls with no intervening `absorb` would return the same value.
//!
//! ## Named theorems & derivations
//!
//! ### Length-prefixing theorem (collision resistance under variable-length absorbs)
//!
//! > **Theorem (length-prefixing).** Let `H` be a collision-resistant
//! > hash. Define `absorb(s) := update((|s|)_{64-bit LE}) ; update(s)`.
//! > Then the map
//! >
//! > ```text
//! > (s_1, s_2, ..., s_n) ↦ absorb(s_1) ∘ absorb(s_2) ∘ ... ∘ absorb(s_n)
//! > ```
//! >
//! > from *sequences of byte-strings* to hasher states is **injective**
//! > (modulo `H`-collisions): two distinct sequences produce
//! > colliding states with probability `≤ 2^{-out_bits(H)}`.
//! >
//! > **Proof sketch.** Suppose `(s_1, ..., s_n) ≠ (t_1, ..., t_m)`
//! > produce equal final states. Walk the concatenated byte streams
//! >
//! > ```text
//! >   A := |s_1|_LE ‖ s_1 ‖ |s_2|_LE ‖ s_2 ‖ ... ‖ |s_n|_LE ‖ s_n
//! >   B := |t_1|_LE ‖ t_1 ‖ |t_2|_LE ‖ t_2 ‖ ... ‖ |t_m|_LE ‖ t_m
//! > ```
//! >
//! > from left to right. At each step the length prefix tells the
//! > parser exactly how many subsequent bytes belong to the current
//! > chunk — i.e. the boundary between chunks is **encoded in the
//! > byte-stream itself**, not in an external delimiter. Inductively,
//! > equal byte-streams imply equal chunkings imply equal sequences,
//! > contradiction. So `A ≠ B` as byte-strings, and equal hasher
//! > states is an `H`-collision on `(A, B)`. ∎
//! >
//! > **Why this matters.** Without the length prefix,
//! > `absorb("ab") ∘ absorb("c")` and `absorb("a") ∘ absorb("bc")` and
//! > `absorb("abc")` all hash the same three bytes "abc" in the same
//! > order, giving identical states. A cheating prover could split
//! > one logical message into two and reassemble it as one (or vice
//! > versa) to manufacture transcript collisions. The length prefix
//! > makes the *boundaries* part of what gets hashed, killing that
//! > attack.
//!
//! Numerical check on the worked example above: any other absorb
//! sequence producing the same bytes would have to start with a `9`
//! followed by exactly 9 bytes, then an `8` followed by 8 bytes — and
//! BLAKE3's preimage resistance forbids constructing an alternative
//! `(s_1', s_2')` that decodes the same way without colliding the
//! 32-byte output.
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

use reed_solomon::field::MODULUS;
use reed_solomon::Fp;

/// Domain-separation tag used inside [`Transcript::squeeze_field`] so
/// that field squeezes and index squeezes from the same prior state
/// never collide.
const SQUEEZE_FIELD_TAG: &[u8] = b"squeeze_field";

/// Domain-separation tag used inside [`Transcript::squeeze_indices`].
/// Mirror of [`SQUEEZE_FIELD_TAG`] for the index-sampling path.
const SQUEEZE_INDICES_TAG: &[u8] = b"squeeze_indices";

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
        // Length-prefix the domain separator so that
        //     new(b"abc") ∘ absorb(b"def")  ≠  new(b"abcdef") ∘ <nothing>
        // — see the length-prefixing theorem in the module docs.
        let mut transcript = Self {
            hasher: blake3::Hasher::new(),
        };
        transcript.absorb(domain_separator);
        transcript
    }

    /// Absorb a raw byte string into the transcript.
    ///
    /// Use for Merkle roots, serialized OOD answers, or any other
    /// prover-to-verifier message that isn't naturally a single field
    /// element. For field elements specifically, prefer
    /// [`Transcript::absorb_field`] for clarity.
    ///
    /// Always length-prefixes — see the length-prefixing theorem in the
    /// module docs and the caveat about forgetting absorbs.
    pub fn absorb(&mut self, bytes: &[u8]) {
        // Length-prefix discipline: write `(|bytes|)_{u64 LE}` first, then
        // the payload. Cf. module-doc §"Length-prefixing theorem".
        let len = bytes.len() as u64;
        self.hasher.update(&len.to_le_bytes());
        self.hasher.update(bytes);
    }

    /// Absorb a single Goldilocks field element.
    ///
    /// Serializes `x` as its 8-byte little-endian canonical form (from
    /// [`reed_solomon::Fp::as_u64`]), then absorbs.
    pub fn absorb_field(&mut self, x: Fp) {
        // Canonicalise via `as_u64` — the field stores values in
        // `[0, MODULUS)` already, so this is unique per Fp value and
        // matches the verifier's serialisation byte-for-byte.
        let bytes = x.as_u64().to_le_bytes();
        self.absorb(&bytes);
    }

    /// Absorb a Merkle root by its 32 bytes.
    ///
    /// Thin convenience wrapper around [`Transcript::absorb`] that
    /// destructures the root. The prover and verifier both call this on
    /// every round commitment — see
    /// [`crate::prover::StirProver::prove`] /
    /// [`crate::verifier::StirVerifier::verify`].
    pub fn absorb_root(&mut self, root: &crate::merkle::MerkleRoot) {
        // `MerkleRoot` is a newtype around `[u8; 32]`; absorb the raw
        // bytes through the length-prefixed `absorb` path so a future
        // change to root size doesn't silently change the transcript.
        self.absorb(&root.0);
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
        // 1. Domain-separate the squeeze so this call cannot share an
        //    output with `squeeze_indices` on the same prior state.
        self.absorb(SQUEEZE_FIELD_TAG);

        // 2. Snapshot the hasher and pull a PRG stream from the snapshot.
        //    We do NOT consume `self.hasher` — we need to re-seed it
        //    after we know which `u64` was accepted. `finalize_xof`
        //    takes `&self`, so a clone is cheap and explicit about
        //    "this branch of the hasher is for PRG use".
        let mut xof = self.hasher.finalize_xof();

        // 3. Rejection-sample a u64 < MODULUS. Cf. module-doc §"Why
        //    rejection sampling for `squeeze_field`".
        let mut buf = [0u8; 8];
        let accepted: u64 = loop {
            xof.fill(&mut buf);
            let v = u64::from_le_bytes(buf);
            if v < MODULUS {
                break v;
            }
            // else: keep pulling from the same XOF stream — BLAKE3's
            // XOF is an unbounded uniform-byte source, so consecutive
            // reads remain independent. No need to re-mix mid-loop.
        };

        // 4. Re-seed the live hasher with the accepted bytes so the
        //    next absorb/squeeze depends on what we returned. Skip the
        //    length prefix here — we don't care about ambiguity inside
        //    the re-seed; we just need *some* state advance bound to
        //    `accepted`.
        self.hasher.update(&accepted.to_le_bytes());

        Fp::new(accepted)
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
        assert!(
            range_exclusive > 0,
            "squeeze_indices: range_exclusive must be > 0",
        );

        // 1. Domain-separate this squeeze flavour. Cf. module-doc
        //    §"Why rejection sampling for `squeeze_indices`".
        self.absorb(SQUEEZE_INDICES_TAG);

        // 2. Compute the rejection cutoff: the largest multiple of
        //    `range_exclusive` that fits in `[0, 2^32)`.
        //        cutoff = (2^32 / range_exclusive) * range_exclusive
        //    Any draw `< cutoff` reduces mod `range_exclusive` with
        //    zero bias; draws `>= cutoff` are rejected. We compute the
        //    cutoff in u64 to avoid 2^32 overflow when range_exclusive
        //    is 1 (in which case cutoff = 2^32 and *every* draw is
        //    accepted — `u32 as u64 < 2^32` is always true).
        let range = range_exclusive as u64;
        let cutoff: u64 = ((1u64 << 32) / range) * range;

        let mut xof = self.hasher.finalize_xof();
        let mut buf = [0u8; 4];
        let mut out: Vec<u32> = Vec::with_capacity(count as usize);

        // Track the bytes we accepted so we can re-seed reproducibly.
        // (We re-seed with the accepted u32 values — same discipline as
        // `squeeze_field`. Rejected draws need not influence the next
        // state because they did not influence the returned values.)
        while (out.len() as u32) < count {
            xof.fill(&mut buf);
            let v = u32::from_le_bytes(buf) as u64;
            if v < cutoff {
                let idx = (v % range) as u32;
                out.push(idx);
            }
        }

        // 3. Re-seed: absorb every accepted u32, little-endian, into the
        //    live hasher so subsequent squeezes are independent.
        for &idx in &out {
            self.hasher.update(&idx.to_le_bytes());
        }

        out
    }

    /// Thin alias for [`Transcript::squeeze_indices`].
    ///
    /// Provided so prover/verifier call-sites can use the protocol-
    /// vocabulary name "shift indices" (the Merkle query positions in
    /// the round's evaluation domain) without losing the generic
    /// "squeeze indices" implementation underneath. The two are
    /// behaviourally identical — same domain separator, same rejection
    /// cutoff. Adopt the alias at the call site for readability; the
    /// implementation lives in [`Transcript::squeeze_indices`].
    pub fn sample_shift_indices(&mut self, count: u32, range_exclusive: u32) -> Vec<u32> {
        self.squeeze_indices(count, range_exclusive)
    }

    /// **STUB.** Proof-of-work grinding (descoped this iteration).
    ///
    /// In a production STIR pipeline this would force the prover to
    /// search for a `nonce` such that `H(transcript ‖ nonce)` has
    /// `pow_bits` leading zero bits — burning roughly `2^pow_bits`
    /// hash invocations of prover time per round and tightening the
    /// soundness bound by the same factor (each cheating attempt at a
    /// transcript fork costs the same `2^pow_bits` work). The standard
    /// reference is §6 of eprint 2024/390 ("PoW grinding") and the
    /// FRI-with-PoW analysis in ethSTARK.
    ///
    /// In *this* educational iteration we do not implement grinding,
    /// so the method exists only to keep the prover/verifier call sites
    /// type-checking. It always returns `0` and absorbs nothing — the
    /// transcript state is unchanged across a `grind` call. When PoW
    /// is added later, this method should:
    ///
    /// 1. Search for `nonce: u64` with `H(self.hasher ‖ nonce)` having
    ///    ≥ `pow_bits` leading zero bits.
    /// 2. Absorb the found `nonce` so the transcript is bound to it.
    /// 3. Return `nonce` to the caller so the prover can include it in
    ///    the proof and the verifier can recompute the same check.
    ///
    /// The current stub is `cargo`-warning-clean (`pow_bits` consumed
    /// by the `let _ = ...` pattern).
    pub fn grind(&mut self, pow_bits: u32) -> u64 {
        let _ = pow_bits;
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two transcripts initialized the same way with the same absorbs
    /// must produce the same squeezes — Fiat-Shamir is deterministic.
    #[test]
    fn transcript_is_deterministic() {
        let mut a = Transcript::new(b"test");
        let mut b = Transcript::new(b"test");
        a.absorb(b"hello");
        b.absorb(b"hello");

        // Field squeeze must match exactly.
        let fa = a.squeeze_field();
        let fb = b.squeeze_field();
        assert_eq!(fa.as_u64(), fb.as_u64(), "field squeezes must match");

        // Subsequent index squeeze must also match — confirms the
        // re-seed discipline is identical on both sides.
        let ia = a.squeeze_indices(5, 100);
        let ib = b.squeeze_indices(5, 100);
        assert_eq!(ia, ib, "index squeezes must match");
    }

    /// Different absorbs must produce different squeezes — the
    /// transcript must be **non-degenerate**.
    #[test]
    fn different_absorbs_give_different_squeezes() {
        let mut a = Transcript::new(b"test");
        let mut b = Transcript::new(b"test");
        a.absorb(b"hello");
        b.absorb(b"world");

        let fa = a.squeeze_field();
        let fb = b.squeeze_field();
        assert_ne!(
            fa.as_u64(),
            fb.as_u64(),
            "distinct absorbs must produce distinct field squeezes \
             (collision probability ≈ 2^-64)",
        );
    }

    /// Different domain separators must produce different squeezes —
    /// the domain-separation invariant.
    #[test]
    fn different_domain_separators_give_different_squeezes() {
        let mut a = Transcript::new(b"protocol-A");
        let mut b = Transcript::new(b"protocol-B");
        // No absorbs in between — the domain separator alone must be
        // enough to make the streams disjoint.
        let fa = a.squeeze_field();
        let fb = b.squeeze_field();
        assert_ne!(
            fa.as_u64(),
            fb.as_u64(),
            "distinct domain separators must yield distinct field squeezes",
        );
    }

    /// `squeeze_field` must return a value `< MODULUS` (no bias from
    /// failed rejection rounds leaking through).
    #[test]
    fn squeeze_field_is_in_range() {
        let mut t = Transcript::new(b"range-test");
        let mut min_seen = u64::MAX;
        for _ in 0..1000 {
            let v = t.squeeze_field().as_u64();
            assert!(
                v < MODULUS,
                "squeeze_field returned {} which is ≥ MODULUS {}",
                v,
                MODULUS,
            );
            if v < min_seen {
                min_seen = v;
            }
        }
        // With 1000 uniform samples in [0, MODULUS ≈ 2^64), the
        // probability that all of them exceed 2^54 is
        // (1 − 2^{-10})^1000 ≈ e^{-1} ≈ 0.37; so min_seen < 2^54 with
        // probability ≈ 0.63. We use a much looser bound (min < 2^62)
        // so the test is essentially never flaky — failure here would
        // signal a real distribution bug, not a statistical fluke.
        assert!(
            min_seen < (1u64 << 62),
            "empirical distribution looks pathological: min sample = {}",
            min_seen,
        );
    }

    /// `squeeze_indices(count, range)` must return `count` indices, all
    /// in `[0, range)`.
    #[test]
    fn squeeze_indices_are_in_range() {
        let mut t = Transcript::new(b"index-range-test");
        // 17 is a non-power-of-2 prime: exercises the rejection logic
        // for real (cutoff < 2^32 strict).
        let xs = t.squeeze_indices(50, 17);
        assert_eq!(xs.len(), 50);
        for (i, &x) in xs.iter().enumerate() {
            assert!(x < 17, "index #{i} = {x} ≥ 17");
        }

        // Range == 1 is the degenerate case: cutoff = 2^32, every draw
        // accepted, every result `% 1 == 0`.
        let zeros = t.squeeze_indices(50, 1);
        assert_eq!(zeros.len(), 50);
        for (i, &x) in zeros.iter().enumerate() {
            assert_eq!(x, 0, "index #{i} with range 1 should be 0, got {x}");
        }
    }

    /// `squeeze_field` followed by `squeeze_field` must produce
    /// independent values — the re-seed discipline must work.
    #[test]
    fn consecutive_squeezes_are_independent() {
        let mut t = Transcript::new(b"indep-test");
        let v1 = t.squeeze_field().as_u64();
        let v2 = t.squeeze_field().as_u64();
        assert_ne!(
            v1, v2,
            "consecutive squeezes must differ (collision odds ≈ 2^-64); \
             if they're equal, the re-seed step is missing or the squeeze \
             tag is colliding the state on itself",
        );
    }

    /// `absorb_root` and `absorb(&root.0)` must produce the same state
    /// (cross-check the shim).
    #[test]
    fn absorb_root_matches_raw_absorb() {
        let root = crate::merkle::MerkleRoot([0xAB; 32]);

        let mut a = Transcript::new(b"root-test");
        a.absorb_root(&root);
        let fa = a.squeeze_field().as_u64();

        let mut b = Transcript::new(b"root-test");
        b.absorb(&root.0);
        let fb = b.squeeze_field().as_u64();

        assert_eq!(fa, fb, "absorb_root must equal absorb(&root.0)");
    }

    /// `grind` is currently a stub: it must return 0 and leave the
    /// transcript state unchanged. Document the contract so a future
    /// implementation breaks this test loudly (forcing an audit).
    #[test]
    fn grind_is_a_stub() {
        let mut t = Transcript::new(b"grind-test");
        let pre = t.squeeze_field().as_u64();

        let mut t2 = Transcript::new(b"grind-test");
        let nonce = t2.grind(20);
        let post = t2.squeeze_field().as_u64();

        assert_eq!(nonce, 0, "grind stub must return 0");
        assert_eq!(
            pre, post,
            "grind stub must not advance the transcript state",
        );
    }
}
