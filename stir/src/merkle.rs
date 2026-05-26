//! Binary Merkle tree commitments over SHA3-256.
//!
//! This module is a generic Merkle primitive over `Vec<Fp>` (the
//! Goldilocks field `F = F_p`); STIR uses it to commit to each round's
//! evaluation table.
//!
//! ## Anchor: where this sits in the STIR pipeline
//!
//! STIR is a low-degree IOP of proximity: in each round `i = 0, 1, ..., M`
//! the prover hands the verifier a function `f_i: L_i → F` and claims it
//! is Hamming-close to `RS[F, L_i, d_i]`. The verifier cannot afford to
//! download `|L_i|` field elements per round (the whole point of an IOP
//! is sublinear verifier work), so the prover instead sends a 32-byte
//! **Merkle root** committing to the evaluation table of `f_i`, and
//! later opens individual leaves on demand (via random spot-checks
//! drawn by Fiat-Shamir from [`crate::transcript`]).
//!
//! This module is the cryptographic primitive that makes "send the root,
//! open leaves later" possible. It is consumed in two places downstream:
//!
//! 1. [`crate::commitment::StirCommitment`] wraps `MerkleTree` with the
//!    STIR-protocol type-level intent (binds `(root, tree_size)` together
//!    so the verifier cannot be tricked into checking paths against a
//!    wrong-size tree — see §"Binding theorem" below).
//! 2. [`crate::prover`] and [`crate::verifier`] call `open`/`verify` on
//!    indices drawn from the transcript at the spot-check phase of every
//!    round.
//!
//! Without this module, every STIR round would have to transmit `|L_i|`
//! field elements, blowing up proof size from `O(λ · log² d)` (the
//! headline) to `O(d)` (a non-starter).
//!
//! ## What this module does
//!
//! Every STIR round commits to a function `f_i: L_i → F` — the
//! round-`i` committed function on the round-`i` evaluation domain
//! `L_i`, with `|L_i|` field-element entries — by Merkle-hashing its
//! evaluation table. The table is a purported codeword, whose closeness
//! to the Reed-Solomon code `RS[F, L_i, d_i]` (polynomials of degree
//! `< d_i` evaluated on `L_i`, with `d_i = d_0 / k^i` the round-`i`
//! degree bound) is exactly what STIR tests. Given a vector of field
//! elements `leaves = [Fp(a_0), Fp(a_1), ..., Fp(a_{n-1})]`, this
//! module gives you three operations:
//!
//! 1. **Commit.** [`MerkleTree::commit`] builds a binary tree whose root
//!    `r ∈ {0,1}^{256}` cryptographically summarises every leaf. The verifier
//!    receives `r` and nothing else from the prover at commit time.
//! 2. **Open.** [`MerkleTree::open`] produces, for any index `i`, an
//!    authentication path: the chain of sibling hashes from the leaf up to
//!    the root. Path length is `log_2(n)` after power-of-2 padding.
//! 3. **Verify.** [`MerkleTree::verify`] recomputes the root from a claimed
//!    `(index, leaf, path)` triple and checks bitwise equality with the
//!    expected root.
//!
//! The whole point is **succinctness with adversarial trust**: the verifier
//! receives `O(log n)` hashes per query, *cannot* see the rest of the
//! committed vector, and yet is convinced of leaf authenticity by the
//! collision resistance of SHA3-256.
//!
//! ## Hash function choice: SHA3-256
//!
//! We use SHA3-256 (Keccak family, NIST FIPS-202 standardised) for both
//! leaf hashing and internal-node hashing. Reasons:
//!
//! - **Cryptographic separation from BLAKE3.** The Fiat-Shamir transcript
//!    in [`crate::transcript`] uses BLAKE3. Using SHA3-256 here keeps the
//!    two "domains" of randomness independent: a transcript collision tells
//!    us nothing useful about Merkle collisions and vice versa. Mixing
//!    hash functions across protocol roles is a standard defensive practice.
//! - **Educational reach.** SHA3 is the most-deployed hash in the EVM
//!    (`keccak256` is its non-FIPS cousin), so this is the hash STIR-on-Ethereum
//!    audiences will recognise. We'll meet `keccak256` again when we build
//!    smart-contract verifiers; using SHA3-256 here primes that.
//! - **256-bit output.** Matches the contract `MerkleRoot([u8; 32])`. A
//!    32-byte digest puts us comfortably above the 128-bit collision-resistance
//!    target most STARK-family protocols aim for.
//!
//! `// CAUTION:` cross-protocol-reuse of hash functions is itself a subtle
//! security topic. The conservative principle: different *roles* (transcript
//! randomness vs. commitment) should use different hash *primitives* where
//! cheap. We're cheap.
//!
//! ## Tree shape — binary, power-of-2-padded, domain-separated
//!
//! Given `n` leaves, we pad to `m = 2^{ceil(log_2 n)}` zero leaves
//! (`Fp::zero()`, hashed through the same leaf rule — *not* a raw
//! `[0u8; 32]`) and then build a complete binary tree. Conventions:
//!
//! - **Leaf hashes (domain tag `0x00`).** Each `leaf_i = Fp` is first
//!    canonicalised via `leaf_i.as_u64().to_le_bytes()` (eight bytes,
//!    little-endian) and then hashed as
//!    `H_leaf(x) := SHA3-256( 0x00 || x.as_u64().to_le_bytes() )`,
//!    giving a 32-byte digest. The single prefix byte `0x00` is a
//!    **domain separator**: see §"Why domain separation" below.
//! - **Internal hashes (domain tag `0x01`).**
//!    `H_node(l, r) := SHA3-256( 0x01 || l || r )` — one tag byte
//!    followed by the 32-byte left child and the 32-byte right child,
//!    65 bytes total.
//! - **Left subtree first.** Leaf at index `2i` is the left child of
//!    `parent_i`, leaf at index `2i+1` is the right child. The leaf vector
//!    is laid out left-to-right exactly as given.
//! - **Layer storage.** `layers[0] = leaves_hashed`, `layers[k+1][j] =`
//!    `H_node(layers[k][2j], layers[k][2j+1])`, and `layers.last()`
//!    contains the single root hash.
//!
//! ### Why domain separation (the `0x00` / `0x01` tags)
//!
//! The tags prevent two well-known shape-confusion attacks against
//! "untagged" Merkle trees:
//!
//! 1. **Leaf-vs-node confusion.** Without tags, an internal-node hash
//!    `H(l || r)` is a 64-byte preimage; a leaf hash `H(x)` is an
//!    8-byte preimage. A second-preimage attacker who finds a leaf
//!    value `x'` whose 8-byte encoding *happens to equal* the 64-byte
//!    concatenation `l || r` of two known internal hashes could pass
//!    `x'` off as a leaf whose hash matches a known internal node —
//!    forging a "leaf" that opens to a position which was actually an
//!    internal subtree. With our tags, leaf preimages start with `0x00`
//!    and node preimages start with `0x01`, so the two preimage
//!    spaces are disjoint and the attack vanishes by collision-
//!    resistance of SHA3-256 alone (no extra assumption on input
//!    lengths).
//! 2. **Length-extension / cross-protocol replay.** SHA3 (Keccak) is
//!    not vulnerable to length-extension the way SHA-256 is, so this
//!    one is belt-and-suspenders here — but tagging keeps the
//!    construction safe even if the hash is swapped out later for a
//!    Merkle-Damgård variant.
//!
//! `// CAUTION:` the tags are part of the *commit contract*. Verifiers
//! that omit them — or use different byte values — will reject every
//! honestly-built proof. Both [`MerkleTree::commit`] and
//! [`MerkleTree::verify`] route through the same [`sha3_leaf`] and
//! [`sha3_node`] helpers (see §"Helpers" near the bottom of the file)
//! for exactly this reason.
//!
//! ## Worked example: commit to `[Fp(1), Fp(2), Fp(3), Fp(4)]`, open index 2
//!
//! With `n = 4` no padding is needed (`4` is already `2^2`). The tree has
//! three layers — leaves, one intermediate, root.
//!
//! ```text
//! Layer 0 (leaves_hashed), each = SHA3-256(0x00 || 8-byte LE):
//!     h0 = SHA3-256(0x00 || 0x0100000000000000)   // Fp(1).as_u64() = 1
//!     h1 = SHA3-256(0x00 || 0x0200000000000000)
//!     h2 = SHA3-256(0x00 || 0x0300000000000000)
//!     h3 = SHA3-256(0x00 || 0x0400000000000000)
//!
//! Layer 1, each = SHA3-256(0x01 || left || right):
//!     h01 = SHA3-256(0x01 || h0 || h1)
//!     h23 = SHA3-256(0x01 || h2 || h3)
//!
//! Layer 2 (root):
//!     root = SHA3-256(0x01 || h01 || h23)
//! ```
//!
//! **Opening leaf index `2`** (i.e., the leaf `Fp(3)`). The verifier
//! receives `(leaf = Fp(3), siblings = [h3, h01], leaf_index = 2)` plus
//! `(root, tree_size = 4)` from public state. It walks:
//!
//! ```text
//! current = H_leaf(Fp(3))                                = h2
//! level 0: bit 0 of index 2 = 0 → current is LEFT child
//!     current = H_node(current, siblings[0])
//!             = H_node(h2, h3)                           = h23
//! level 1: bit 1 of index 2 = 1 → current is RIGHT child
//!     current = H_node(siblings[1], current)
//!             = H_node(h01, h23)                         = root
//! return current == root.0                               → true
//! ```
//!
//! Note the asymmetric concatenation order — `(current, sibling)` when
//! the index bit is `0`, `(sibling, current)` when it's `1` — is what
//! ties the path to the specific `leaf_index`. Flip a single index bit
//! and the recomputed root almost certainly diverges.
//!
//! ## What a path proves
//!
//! A path `(siblings, leaf_index)` together with a claimed `(leaf, root)`
//! convinces the verifier exactly when **the recomputed hash chain equals
//! `root`**. That happens iff:
//!
//! 1. **The leaf is at the claimed index.** The bit pattern of `leaf_index`
//!    tells the verifier, at each level, whether the current hash is the
//!    left or right child — and therefore the order in which to concatenate
//!    `current || sibling` vs. `sibling || current` before hashing.
//! 2. **The siblings are consistent with `root`.** A second-preimage attack
//!    against SHA3-256 would let the prover forge a path; we assume no such
//!    attack exists (256-bit security target).
//!
//! Soundness reduces to SHA3-256 second-preimage resistance plus
//! collision-resistance — *any* mismatch in `(leaf, path, root)` produces a
//! different recomputed root with probability `≈ 1 - 2^{-256}`.
//!
//! ## Named theorem: Binding of the Merkle root
//!
//! > **Binding theorem (collision resistance of the Merkle root).**
//! > Fix `tree_size = n` and let `m = 2^{⌈log₂ n⌉}` be the padded
//! > size. For any 32-byte root `r`, any leaf index `i ∈ [0, n)`, and
//! > any two `(leaf, path)` pairs `(a, π_a)`, `(b, π_b)` such that
//! > `MerkleTree::verify(r, i, a, π_a, n) = true` and
//! > `MerkleTree::verify(r, i, b, π_b, n) = true`, either `a = b`
//! > (and `π_a = π_b` modulo padding) or one can extract a SHA3-256
//! > collision from `(π_a, π_b)`.
//!
//! **Proof (constructive).** Suppose both verifications accept and
//! `a ≠ b`. Walk both verifications level by level from leaf to root,
//! comparing the current digest at each level.
//!
//! - At level `0`: the leaf hashes are `H_leaf(a)` and `H_leaf(b)`.
//!   Because `a ≠ b`, their 8-byte LE encodings differ, and so do the
//!   1-byte-tagged preimages `0x00 || a_le` and `0x00 || b_le`. Either
//!   `H_leaf(a) ≠ H_leaf(b)` (call this the "diverged" case), OR the
//!   two SHA3-256 calls produced the same output on different inputs —
//!   a collision, extracted.
//! - Inductive step at level `k`. Assume the level-`k` digests differ
//!   between the two verifications. Both verifications then feed their
//!   own digest into `H_node` with `π_a[k]` resp. `π_b[k]` (in the
//!   same left/right order, since `leaf_index = i` is shared).
//!   Both must produce the *same* level-`(k+1)` digest for the final
//!   roots to coincide at level `log₂ m`. But the level-`(k+1)`
//!   preimages — `0x01 || L_a || R_a` vs. `0x01 || L_b || R_b` — differ
//!   in at least 32 bytes (because the inputs at level `k` differ on
//!   one side and the index bit is fixed), so equal outputs are again
//!   a SHA3-256 collision.
//!
//! Tracking the diverged-vs-collision dichotomy up to the root, we
//! conclude: either the verifications produce two different recomputed
//! roots — contradicting both being equal to `r` — or somewhere along
//! the path we extracted an explicit SHA3-256 collision. ∎
//!
//! **Why this matters for STIR.** Every soundness statement about
//! STIR's per-round proximity test implicitly *quantifies over a
//! fixed function `f_i: L_i → F`*. The Binding theorem is what lets
//! us treat `commitment.root` (a 32-byte string) as a stand-in for
//! that fixed function: a malicious prover cannot, after committing,
//! later "open" the same index to two inconsistent values without
//! breaking SHA3-256. The Fiat-Shamir transcript then absorbs `root`
//! *before* the verifier issues challenges, so the prover is forced
//! to commit to `f_i` first and is interrogated against fresh
//! randomness derived from that commitment.
//!
//! **Question for the reader.** We tag leaves and nodes with `0x00`/`0x01`
//! (see §"Why domain separation" above). Why is **index-tagging** —
//! `H_leaf(x, i) := SHA3-256(0x00 || i_le || x_le)` — an even stronger
//! variant, and why do we not use it here?
//!
//! Answer: with plain value hashing, two trees that happen to share a
//! leaf-value have an identical `leaf_hash` at that position. An attacker
//! who learns a path in tree A could try to "shift" it into tree B by
//! finding a smaller / re-padded tree B' whose authentication path
//! coincidentally reuses the same intermediate hashes. Index-tagging the
//! leaf ties every leaf hash to its tree position, blocking such cross-
//! tree replays. For *this* educational implementation we use only the
//! `0x00`/`0x01` shape-confusion tags — flag index-tagging as a
//! production hardening. STIR's binding argument is fine without it
//! because each round's transcript absorbs `tree_size` explicitly.
//!
//! ## Verifier responsibilities
//!
//! The verifier is given `(root, index, leaf, path)`. It must:
//!
//! - Independently know `tree_size` (the number of leaves the prover
//!   *claims* to have committed to). Without that, the verifier can't
//!   tell which paths are valid: an adversary could open the *same*
//!   `(leaf, path)` pair against trees of two different sizes and the
//!   verifier would have no way to detect mismatched levels.
//! - Recompute `m = 2^{ceil(log_2 tree_size)}` and check that
//!   `path.siblings.len() == log_2(m)`.
//! - Walk up: starting from `SHA3-256(leaf.as_u64().to_le_bytes())`, fold
//!   in each sibling according to the index bits, and compare the final
//!   hash to `root`.
//!
//! `// CAUTION:` the verifier MUST derive `tree_size` from public protocol
//! state (the STIR parameters, or a binding from a previous round), NOT
//! from anything the prover sends inside the proof. A prover-supplied
//! size is unauthenticated — accepting it would let the prover commit
//! to one size and "verify against" another, breaking the binding.

use reed_solomon::field::Fp;
use sha3::{Digest, Sha3_256};

/// Domain-separation tag prefixed to **leaf** preimages before SHA3-256.
/// See §"Why domain separation" in the module docs.
const LEAF_TAG: u8 = 0x00;

/// Domain-separation tag prefixed to **internal-node** preimages before
/// SHA3-256. See §"Why domain separation" in the module docs.
const NODE_TAG: u8 = 0x01;

/// A 32-byte Merkle root.
///
/// Wrapper around `[u8; 32]` with `PartialEq`/`Eq`/`Hash` so it can be
/// stored in collections and compared. Equality is byte-exact.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MerkleRoot(pub [u8; 32]);

/// An authentication path opening a single leaf.
///
/// Layout: `siblings[0]` is the sibling at the leaf level, `siblings[1]`
/// is the sibling one level up, ..., `siblings.last()` is the sibling of
/// the layer just below the root. Length equals `log_2(padded_tree_size)`.
///
/// `leaf_index` is the position of the opened leaf in the *unpadded* leaf
/// vector. The verifier reads its bits from least significant up to walk
/// the path: bit 0 says "leaf is left child? right child?" at level 0,
/// bit 1 says the same at level 1, and so on.
#[derive(Clone, Debug)]
pub struct MerklePath {
    /// Sibling hashes from leaf level up to one level below the root.
    pub siblings: Vec<[u8; 32]>,
    /// The position of the opened leaf in the leaves slice passed to
    /// [`MerkleTree::commit`].
    pub leaf_index: usize,
}

/// A fully materialised Merkle tree.
///
/// Stores every layer so [`MerkleTree::open`] can run in `O(log n)` by
/// indexing rather than re-hashing. Memory cost is `2 · padded_n · 32`
/// bytes — fine for STIR's per-round commitments (≤ a few million leaves).
pub struct MerkleTree {
    /// The original (padded) leaves as `Fp` values. Stored so that
    /// [`MerkleTree::open`] can return the field-element value alongside
    /// the authentication path without forcing the caller to keep its
    /// own side table. Length equals `leaves_hashed.len()`.
    pub leaves: Vec<Fp>,
    /// The leaf-level hashes after padding to the next power of 2. Equal
    /// to `layers[0]`; duplicated for ergonomic access.
    pub leaves_hashed: Vec<[u8; 32]>,
    /// All layers of the tree, bottom up.
    ///
    /// - `layers[0]` is the (padded) leaf hashes.
    /// - `layers[k+1][j] = H_node(layers[k][2j], layers[k][2j+1])`
    ///   (tagged SHA3-256, see §"Tree shape" in the module docs).
    /// - `layers.last()` is a single-element vector containing the root.
    pub layers: Vec<Vec<[u8; 32]>>,
}

impl MerkleTree {
    /// Commit to a vector of field elements, returning the materialised tree.
    ///
    /// Pads `leaves` up to the next power of 2 with `Fp::zero()` leaves
    /// (hashed through the same tagged [`sha3_leaf`] rule — *not* a raw
    /// `[0u8; 32]`, see §"Why domain separation" in the module docs), then
    /// builds layers bottom-up via tagged SHA3-256.
    ///
    /// We also stash the original (unpadded) leaves so [`MerkleTree::open`]
    /// can return the field-element value alongside the path. This matches
    /// the `(Fp, MerklePath)` interface promised by the module.
    ///
    /// Cost: `O(n)` hash invocations (one per leaf, one per internal node).
    pub fn commit(leaves: &[Fp]) -> Self {
        // See §"Tree shape" and §"Why domain separation" in the module docs.
        // The rule is:
        //     layers[0][i]     = H_leaf(leaves_padded[i])
        //     layers[k+1][j]   = H_node(layers[k][2j], layers[k][2j+1])
        // with H_leaf and H_node domain-separated by LEAF_TAG / NODE_TAG.

        let n = leaves.len();
        let m = next_power_of_two(n.max(1));

        // Build the padded leaves vector. Padding is `Fp::zero()` hashed
        // through the *same* leaf rule, so the verifier can reproduce
        // padding from `tree_size` alone — no raw zero-hash sentinel.
        let mut padded_leaves: Vec<Fp> = Vec::with_capacity(m);
        padded_leaves.extend_from_slice(leaves);
        padded_leaves.resize(m, Fp::zero());

        // Layer 0: leaf hashes.
        let leaves_hashed: Vec<[u8; 32]> =
            padded_leaves.iter().map(|x| sha3_leaf(*x)).collect();

        // Build layers bottom-up. We stop when the current layer has length 1.
        let mut layers: Vec<Vec<[u8; 32]>> = Vec::new();
        layers.push(leaves_hashed.clone());

        while layers.last().expect("at least one layer").len() > 1 {
            let prev = layers.last().expect("at least one layer");
            let parent: Vec<[u8; 32]> = prev
                .chunks_exact(2)
                .map(|pair| sha3_node(&pair[0], &pair[1]))
                .collect();
            layers.push(parent);
        }

        Self {
            leaves: padded_leaves,
            leaves_hashed,
            layers,
        }
    }

    /// Return the Merkle root of this tree.
    ///
    /// The root is the single hash in `self.layers.last()`. Wrapped in
    /// [`MerkleRoot`] for type safety against raw `[u8; 32]` confusion.
    pub fn root(&self) -> MerkleRoot {
        let top = self.layers.last().expect("tree has at least one layer");
        debug_assert_eq!(
            top.len(),
            1,
            "top layer must be a single root hash; got {} entries",
            top.len()
        );
        MerkleRoot(top[0])
    }

    /// Produce an authentication path for the leaf at `index`.
    ///
    /// Returns `(leaf_value, path)` where `leaf_value` is the original
    /// `Fp` (not its hash) and `path.siblings` is the chain of sibling
    /// hashes from leaf level upward.
    ///
    /// Panics if `index >= self.leaves_hashed.len()` (the padded size).
    /// In normal use the caller passes an index in the *unpadded* range,
    /// which is also within the padded range.
    pub fn open(&self, index: usize) -> (Fp, MerklePath) {
        assert!(
            index < self.leaves_hashed.len(),
            "open index {} out of range (padded tree size {})",
            index,
            self.leaves_hashed.len()
        );

        // The hash chain has `layers.len() - 1` siblings (everything below
        // the root). For a single-leaf tree this is 0 — no siblings.
        let n_levels = self.layers.len().saturating_sub(1);
        let mut siblings: Vec<[u8; 32]> = Vec::with_capacity(n_levels);

        // XOR-by-1 flips the low bit, swapping left/right siblings.
        // See §"Worked example" in the module docs for why this matches
        // the verifier's left/right concatenation rule.
        let mut current_index = index;
        for k in 0..n_levels {
            let sibling_index = current_index ^ 1;
            siblings.push(self.layers[k][sibling_index]);
            current_index >>= 1;
        }

        let leaf_value = self.leaves[index];
        (
            leaf_value,
            MerklePath {
                siblings,
                leaf_index: index,
            },
        )
    }

    /// Verify that `leaf` at `index` is consistent with `root`, given `path`.
    ///
    /// Reconstructs the root by:
    /// 1. Hashing `leaf.as_u64().to_le_bytes()` → `current` 32-byte digest.
    /// 2. For each sibling in `path.siblings`, using the corresponding bit
    ///    of `index` to decide concatenation order, then SHA3-256'ing.
    /// 3. Comparing the final `current` to `root.0`.
    ///
    /// `tree_size` is the prover's *claimed* number of leaves; the verifier
    /// must source this from authenticated public state (see module-level
    /// `// CAUTION:`). The path must have length `ceil(log_2(tree_size))`.
    ///
    /// Returns `false` (not panic) on any mismatch — paths come from
    /// adversaries.
    pub fn verify(
        root: MerkleRoot,
        index: usize,
        leaf: Fp,
        path: &MerklePath,
        tree_size: usize,
    ) -> bool {
        // Cheap structural rejections first — they cost nothing and turn a
        // class of malformed proofs into a fast `false`.
        let padded = next_power_of_two(tree_size.max(1));
        let expected_path_len = if padded <= 1 {
            0
        } else {
            // padded is a power of two ≥ 2, so trailing_zeros is its log2.
            padded.trailing_zeros() as usize
        };

        if path.siblings.len() != expected_path_len {
            return false;
        }
        if index != path.leaf_index {
            return false;
        }
        if index >= tree_size {
            // Indices in the padding-only region are not real leaves —
            // accepting them would let a prover "open" positions it never
            // committed to.
            return false;
        }

        // Walk the path bottom-up, using bit k of `index` to decide the
        // concatenation order. See §"Worked example" in the module docs.
        let mut current = sha3_leaf(leaf);
        for k in 0..expected_path_len {
            let bit = (index >> k) & 1;
            current = if bit == 0 {
                // `current` is the left child.
                sha3_node(&current, &path.siblings[k])
            } else {
                // `current` is the right child.
                sha3_node(&path.siblings[k], &current)
            };
        }

        current == root.0
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Raw SHA3-256 of a byte slice. Internal helper, kept for completeness
/// and for tests that want to recompute leaf hashes by hand.
#[allow(dead_code)]
fn sha3_256(input: &[u8]) -> [u8; 32] {
    Sha3_256::new().chain_update(input).finalize().into()
}

/// Tagged leaf hash: `H_leaf(x) = SHA3-256(LEAF_TAG || x.as_u64().to_le_bytes())`.
///
/// Centralised so [`MerkleTree::commit`] and [`MerkleTree::verify`] cannot
/// drift apart. See §"Why domain separation" in the module docs for why
/// the `LEAF_TAG = 0x00` byte is present.
fn sha3_leaf(x: Fp) -> [u8; 32] {
    let bytes = x.as_u64().to_le_bytes();
    Sha3_256::new()
        .chain_update([LEAF_TAG])
        .chain_update(bytes)
        .finalize()
        .into()
}

/// Tagged internal-node hash: `H_node(l, r) = SHA3-256(NODE_TAG || l || r)`.
///
/// The most common operation in this module. The chained `update` calls
/// avoid allocating a 65-byte temporary buffer per node hash.
fn sha3_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    Sha3_256::new()
        .chain_update([NODE_TAG])
        .chain_update(left)
        .chain_update(right)
        .finalize()
        .into()
}

/// Round up to the next power of 2 (`>= n`). Returns `1` for `n == 0`.
///
/// Single source of truth for the pad-to-power-of-2 contract used by
/// [`MerkleTree::commit`] and [`MerkleTree::verify`].
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        // Stdlib `next_power_of_two` returns the smallest power of 2 ≥ n.
        n.next_power_of_two()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use reed_solomon::field::Fp;

    /// `commit` on a single leaf produces a tree whose root equals the
    /// (tagged) leaf hash — there are no internal nodes to apply `NODE_TAG`.
    #[test]
    fn commit_of_single_leaf_matches_hash() {
        let tree = MerkleTree::commit(&[Fp::new(42)]);
        // Padded size is 1 → only one layer, no internal-node hashing.
        assert_eq!(tree.leaves_hashed.len(), 1);
        assert_eq!(tree.layers.len(), 1);
        // Manually recompute the tagged leaf hash and compare.
        let mut expected_preimage = vec![LEAF_TAG];
        expected_preimage.extend_from_slice(&42u64.to_le_bytes());
        assert_eq!(tree.root().0, sha3_256(&expected_preimage));
    }

    /// Non-power-of-2 leaf counts get padded up to the next power of 2,
    /// and the pad slots are filled with `Fp::zero()` hashed through the
    /// *same* tagged leaf rule (not a raw `[0u8; 32]` sentinel).
    #[test]
    fn commit_pads_non_power_of_2_to_next_power_of_2() {
        let tree = MerkleTree::commit(&[Fp::new(1), Fp::new(2), Fp::new(3)]);
        assert_eq!(tree.leaves_hashed.len(), 4);
        // Pad leaf is `Fp::zero()`, hashed with LEAF_TAG.
        let mut zero_preimage = vec![LEAF_TAG];
        zero_preimage.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(tree.leaves_hashed[3], sha3_256(&zero_preimage));
        // And the stored unpadded-then-padded leaves match.
        assert_eq!(tree.leaves[3], Fp::zero());
    }

    /// Open every leaf, verify the path, expect success — the central
    /// commit → open → verify roundtrip property.
    #[test]
    fn open_then_verify_succeeds() {
        let leaves = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)];
        let tree = MerkleTree::commit(&leaves);
        for i in 0..4 {
            let (leaf, path) = tree.open(i);
            assert_eq!(leaf, leaves[i], "open returned wrong leaf value");
            assert!(
                MerkleTree::verify(tree.root(), i, leaves[i], &path, 4),
                "honest path failed to verify at index {i}",
            );
        }
    }

    /// Verifying with the wrong leaf must fail. Probability of accidental
    /// acceptance is ≈ 2^{-256}.
    #[test]
    fn verify_fails_for_wrong_leaf() {
        let leaves = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)];
        let tree = MerkleTree::commit(&leaves);
        let (_, path) = tree.open(0);
        assert!(!MerkleTree::verify(tree.root(), 0, Fp::new(999), &path, 4));
    }

    /// Verifying after flipping a byte in a sibling hash must fail —
    /// covers the "modify one bit anywhere along the path" adversary.
    #[test]
    fn verify_fails_for_tampered_path() {
        let leaves = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)];
        let tree = MerkleTree::commit(&leaves);
        let (leaf, mut path) = tree.open(0);
        // Flip a byte of the leaf-level sibling.
        path.siblings[0][0] ^= 0xFF;
        assert!(!MerkleTree::verify(tree.root(), 0, leaf, &path, 4));
    }

    /// Verifying with a wrong leaf-index must fail: even if the prover
    /// supplies an otherwise-valid path for index `i`, swapping in a
    /// different `index` argument (or a different `path.leaf_index`)
    /// should be rejected — the index bits drive the left/right
    /// concatenation order, so a wrong index reroutes the recomputed
    /// hash chain off the committed tree.
    #[test]
    fn verify_fails_for_wrong_leaf_index() {
        let leaves = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)];
        let tree = MerkleTree::commit(&leaves);
        let (leaf, path) = tree.open(2);

        // (a) Caller passes wrong `index` while path still reports 2.
        // The structural check `index != path.leaf_index` rejects this.
        assert!(!MerkleTree::verify(tree.root(), 1, leaf, &path, 4));

        // (b) Adversary updates `path.leaf_index` to match the wrong
        // `index` they pass — now both say 0, but the siblings are
        // still those of position 2. The hash-chain walk must reject.
        let mut tampered = path.clone();
        tampered.leaf_index = 0;
        assert!(!MerkleTree::verify(tree.root(), 0, leaf, &tampered, 4));
    }

    /// Different inputs give different roots — distinguishing different
    /// commitments is the whole point. Collision probability is ≈ 2^{-128}
    /// (birthday); never seen in tests.
    #[test]
    fn roots_differ_for_different_inputs() {
        let tree_a = MerkleTree::commit(&[Fp::new(1), Fp::new(2)]);
        let tree_b = MerkleTree::commit(&[Fp::new(1), Fp::new(3)]);
        assert_ne!(tree_a.root(), tree_b.root());
    }
}
