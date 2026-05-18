//! Binary Merkle tree commitments over SHA3-256.
//!
//! This module is a generic Merkle primitive over `Vec<Fp>` (the
//! Goldilocks field `F = F_p`); STIR uses it to commit to each round's
//! evaluation table.
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
//! ## Tree shape — binary, power-of-2-padded
//!
//! Given `n` leaves, we pad to `m = 2^{ceil(log_2 n)}` zero-hashed leaves
//! and then build a complete binary tree. Conventions:
//!
//! - **Leaf hashes.** Each `leaf_i = Fp` is first canonicalised via
//!    `leaf_i.as_u64().to_le_bytes()` (eight bytes, little-endian) and then
//!    SHA3-256'd to give a 32-byte digest. Padding leaves are zero `[u8; 32]`.
//! - **Internal hashes.** `parent = SHA3-256(left || right)` — left child
//!    bytes followed by right child bytes, 64 bytes total.
//! - **Left subtree first.** Leaf at index `2i` is the left child of
//!    `parent_i`, leaf at index `2i+1` is the right child. The leaf vector
//!    is laid out left-to-right exactly as given.
//! - **Layer storage.** `layers[0] = leaves_hashed`, `layers[k+1][j] =`
//!    `SHA3-256(layers[k][2j] || layers[k][2j+1])`, and `layers.last()`
//!    contains the single root hash.
//!
//! ## Worked example: commit to `[Fp(1), Fp(2), Fp(3), Fp(4)]`
//!
//! With `n = 4` no padding is needed (`4` is already `2^2`). The tree has
//! three layers — leaves, one intermediate, root.
//!
//! ```text
//! Layer 0 (leaves_hashed):
//!     h0 = SHA3-256(0x0100000000000000)   // Fp(1).as_u64() = 1, little-endian
//!     h1 = SHA3-256(0x0200000000000000)
//!     h2 = SHA3-256(0x0300000000000000)
//!     h3 = SHA3-256(0x0400000000000000)
//!
//! Layer 1:
//!     h01 = SHA3-256(h0 || h1)
//!     h23 = SHA3-256(h2 || h3)
//!
//! Layer 2 (root):
//!     root = SHA3-256(h01 || h23)
//! ```
//!
//! The path that opens index `2` (i.e., the leaf `Fp(3)`) is
//! `[h3, h01]` — the leaf's right sibling, then the cousin on the next
//! level. Plus the leaf-index `2`, so the verifier knows which side to
//! place each sibling on while rehashing.
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
//! **Question for the reader.** A naive Merkle tree hashes the *value* at
//! each leaf. Why might hashing `value || index` (a tagged hash) be more
//! robust against second-preimage attacks across trees of different sizes?
//!
//! Answer: with plain value hashing, two trees that happen to share a
//! leaf-value have an identical `leaf_hash` at that position. An attacker
//! who learns a path in tree A could try to "shift" it into tree B by
//! finding a smaller / re-padded tree B' whose authentication path
//! coincidentally reuses the same intermediate hashes. Tagging the leaf
//! with its index (`SHA3-256(index_le || value_le)`) ties every leaf hash
//! to its tree position, blocking such cross-tree replays. For *this*
//! educational implementation we stick to plain value hashing — flag
//! tagging as a production improvement (alongside domain-separating
//! internal-node hashing).
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
#[allow(unused_imports)]
use sha3::{Digest, Sha3_256};

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
    /// The leaf-level hashes after padding to the next power of 2. Equal
    /// to `layers[0]`; duplicated for ergonomic access.
    pub leaves_hashed: Vec<[u8; 32]>,
    /// All layers of the tree, bottom up.
    ///
    /// - `layers[0]` is the (padded) leaf hashes.
    /// - `layers[k+1][j] = SHA3-256(layers[k][2j] || layers[k][2j+1])`.
    /// - `layers.last()` is a single-element vector containing the root.
    pub layers: Vec<Vec<[u8; 32]>>,
}

impl MerkleTree {
    /// Commit to a vector of field elements, returning the materialised tree.
    ///
    /// Pads `leaves` up to the next power of 2 with zero-leaf hashes
    /// (`[0u8; 32]`), then builds layers bottom-up via SHA3-256.
    ///
    /// Cost: `O(n)` hash invocations (one per leaf, one per internal node).
    pub fn commit(leaves: &[Fp]) -> Self {
        // TODO:
        //   1. Compute `m = next_power_of_two(leaves.len().max(1))` — the padded size.
        //      WHY: STIR rounds always commit to power-of-2-sized codewords in
        //      practice, but we keep `commit` robust for stub tests.
        //   2. Hash every real leaf: `h_i = SHA3-256(leaves[i].as_u64().to_le_bytes())`.
        //      WHY: an 8-byte canonical encoding ensures byte-for-byte determinism
        //      across machines (vs. e.g. host-endianness or Display formatting).
        //   3. Append `m - leaves.len()` zero hashes (`[0u8; 32]`) to reach length `m`.
        //      WHY: a fixed, public padding rule is what lets the verifier
        //      reconstruct the same tree shape from `tree_size` alone.
        //   4. Build `layers`: start with the padded leaf hashes as `layers[0]`.
        //      For each subsequent level, pair `(left, right)` and hash
        //      `SHA3-256(left || right)` into the parent layer.
        //      WHY: the layer-by-layer build is what makes opening `O(log n)` —
        //      we read off precomputed siblings rather than recomputing them.
        //   5. Stop when the current layer has length 1 — that's the root.
        //   6. Return `Self { leaves_hashed: layers[0].clone(), layers }`.
        let _ = leaves;
        todo!()
    }

    /// Return the Merkle root of this tree.
    ///
    /// The root is the single hash in `self.layers.last()`. Wrapped in
    /// [`MerkleRoot`] for type safety against raw `[u8; 32]` confusion.
    pub fn root(&self) -> MerkleRoot {
        // TODO:
        //   1. Read the last layer of `self.layers`.
        //      WHY: the root lives there by construction in `commit`.
        //   2. Assert it has length 1 (debug-only is fine).
        //      WHY: a mid-build tree would be malformed; defensive check.
        //   3. Return `MerkleRoot(layers.last()[0])`.
        todo!()
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
        // TODO:
        //   1. Sanity check `index < self.leaves_hashed.len()`.
        //      WHY: opening past the end is a programmer bug; better to panic
        //      than to return junk.
        //   2. Recover `leaf_value` by inverting the leaf encoding. (Tricky:
        //      we stored hashes, not values. Pragma: take `leaves` in commit
        //      and keep a Vec<Fp> alongside `leaves_hashed`. OR — simpler for
        //      this stub — assume the caller will retrieve the leaf
        //      separately. Document this and return `Fp::zero()` here as a
        //      placeholder.)
        //      WHY: the interface contract is fixed at `(Fp, MerklePath)`, so
        //      the production implementation stores the original leaves.
        //   3. Walk levels 0..(layers.len() - 1):
        //        - At level k, compute `sibling_index = current_index XOR 1`.
        //          XOR by 1 flips the low bit, which swaps "left child"
        //          and "right child".
        //        - Push `layers[k][sibling_index]` onto the path's siblings vec.
        //        - Update `current_index >>= 1` to move up a level.
        //      WHY: classic Merkle-tree opening — `O(log n)` siblings, one
        //      per level, terminating just below the root.
        //   4. Return `(leaf_value, MerklePath { siblings, leaf_index: index })`.
        let _ = index;
        todo!()
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
        // TODO:
        //   1. Compute `padded = next_power_of_two(tree_size.max(1))`.
        //      WHY: same padding rule as `commit`; verifier must reconstruct it.
        //   2. Compute `expected_path_len = log_2(padded)` (special-case
        //      padded == 1 → length 0).
        //      WHY: a wrong-length path is an immediate sign of tampering.
        //   3. If `path.siblings.len() != expected_path_len` → return false.
        //      Also reject `index != path.leaf_index` and `index >= tree_size`
        //      (a leaf in the padded-only region is not a real leaf).
        //      WHY: cheap structural rejections before doing real hashing.
        //   4. Initialise `current = SHA3-256(leaf.as_u64().to_le_bytes())`.
        //      WHY: same leaf encoding as `commit`; mismatch here would mean
        //      the verifier rejects honest paths.
        //   5. For `k in 0..expected_path_len`:
        //        - bit_k = (index >> k) & 1
        //        - if bit_k == 0 (`current` is left child):
        //              current = SHA3-256(current || path.siblings[k])
        //          else (`current` is right child):
        //              current = SHA3-256(path.siblings[k] || current)
        //      WHY: the concatenation order has to match `commit`'s — getting
        //      this backwards is the classic "verify always returns false" bug.
        //   6. Return `current == root.0` (constant-time eq if you care; for
        //      our educational implementation, plain `==` is fine because
        //      timing leaks on the root don't help an attacker).
        let _ = (root, index, leaf, path, tree_size);
        todo!()
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// SHA3-256 of a single byte slice.
///
/// Thin wrapper so callers don't have to import `sha3::Digest` themselves.
#[allow(dead_code)]
fn sha3_256(input: &[u8]) -> [u8; 32] {
    // TODO:
    //   1. Construct a `Sha3_256` hasher.
    //   2. `update(input)`.
    //   3. `finalize()` into a `[u8; 32]` via `Into::into`.
    // WHY: every leaf and internal-node hash routes through this; centralising
    // it keeps the hash function pluggable for future variants.
    let _ = input;
    todo!()
}

/// SHA3-256 of the concatenation of two 32-byte hashes.
#[allow(dead_code)]
fn sha3_256_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    // TODO:
    //   1. Construct `Sha3_256`.
    //   2. Update with `left`, then `right`.
    //   3. Finalize into `[u8; 32]`.
    // WHY: the most common operation in this module — internal node hashing.
    // Inlining `update` calls avoids allocating a temporary 64-byte buffer.
    let _ = (left, right);
    todo!()
}

/// Round up to the next power of 2 (`>= n`). Returns `1` for `n == 0`.
#[allow(dead_code)]
fn next_power_of_two(n: usize) -> usize {
    // TODO:
    //   - If `n <= 1`, return 1 (we never want an empty tree).
    //   - Otherwise return `(n - 1).next_power_of_two()` — Rust's stdlib
    //     `usize::next_power_of_two` returns the *smallest* power of 2 that
    //     is >= n, which is exactly what we want.
    // WHY: pad-to-power-of-2 is the contract for `commit`; this is the
    // single place to express it so `commit` and `verify` agree.
    let _ = n;
    todo!()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use reed_solomon::field::Fp;

    /// `commit` on a single leaf produces a tree whose root equals the leaf hash.
    #[test]
    fn commit_of_single_leaf_matches_hash() {
        // TODO:
        //   1. tree = MerkleTree::commit(&[Fp::new(42)]).
        //   2. Padded size is 1 → there is only one layer (the leaf itself).
        //   3. Assert `tree.root().0 == sha3_256(&42u64.to_le_bytes())`.
        // WHY: smallest non-trivial case; pins down leaf-encoding determinism.
        todo!()
    }

    /// Non-power-of-2 leaf counts get padded up.
    #[test]
    fn commit_pads_non_power_of_2_to_next_power_of_2() {
        // TODO:
        //   1. tree = MerkleTree::commit(&[Fp::new(1), Fp::new(2), Fp::new(3)]).
        //   2. Assert tree.leaves_hashed.len() == 4 (next power of 2 ≥ 3).
        //   3. Assert tree.leaves_hashed[3] == [0u8; 32] (the pad leaf).
        // WHY: padding contract is what makes verify reproducible from
        // `tree_size` alone.
        todo!()
    }

    /// Open a leaf, verify the path, expect success.
    #[test]
    fn open_then_verify_succeeds() {
        // TODO:
        //   1. leaves = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)].
        //   2. tree = MerkleTree::commit(&leaves).
        //   3. For each i in 0..4:
        //        let (leaf, path) = tree.open(i);
        //        assert!(MerkleTree::verify(tree.root(), i, leaves[i], &path, 4));
        // WHY: round-trip sanity — the central correctness property.
        todo!()
    }

    /// Verifying with the wrong leaf must fail.
    #[test]
    fn verify_fails_for_wrong_leaf() {
        // TODO:
        //   1. leaves = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)].
        //   2. tree = MerkleTree::commit(&leaves).
        //   3. let (_, path) = tree.open(0).
        //   4. assert!(!MerkleTree::verify(tree.root(), 0, Fp::new(999), &path, 4)).
        // WHY: soundness check — pretending the leaf is something it isn't
        // must be rejected. Probability of accidental hit is ≈ 2^-256.
        todo!()
    }

    /// Verifying with a tampered sibling must fail.
    #[test]
    fn verify_fails_for_tampered_path() {
        // TODO:
        //   1. leaves = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)].
        //   2. tree = MerkleTree::commit(&leaves).
        //   3. let (leaf, mut path) = tree.open(0).
        //   4. path.siblings[0][0] ^= 0xFF;  // flip a byte
        //   5. assert!(!MerkleTree::verify(tree.root(), 0, leaf, &path, 4)).
        // WHY: covers the "modify one bit anywhere" adversary.
        todo!()
    }

    /// Different inputs give different roots.
    #[test]
    fn roots_differ_for_different_inputs() {
        // TODO:
        //   1. tree_a = MerkleTree::commit(&[Fp::new(1), Fp::new(2)]);
        //   2. tree_b = MerkleTree::commit(&[Fp::new(1), Fp::new(3)]);
        //   3. assert_ne!(tree_a.root(), tree_b.root());
        // WHY: distinguishing different commitments is the whole point.
        // Collision probability is ≈ 2^-128 (birthday); never seen in tests.
        todo!()
    }
}
