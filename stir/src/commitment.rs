//! STIR-level commitment to a Reed-Solomon evaluation table.
//!
//! ## What this module does
//!
//! At the top of every STIR round the prover must hand the verifier a
//! cryptographic summary of the function it just produced. This module
//! is the thin wrapper that performs that summary: it takes the
//! evaluation table of the round-`i` committed function `f_i: L_i → F`
//! — a function from the round-`i` evaluation domain `L_i` (a finite
//! subset of the Goldilocks field `F = F_p`) into `F`, given as an
//! `|L_i|`-element vector of field elements — hashes it into a binary
//! Merkle tree (see [`crate::merkle::MerkleTree`]), and bundles the
//! resulting root together with the table size into a [`StirCommitment`]
//! value. The commitment is what the prover absorbs into the
//! Fiat-Shamir transcript, and what the verifier later checks Merkle
//! opening paths against. (Honest provers send the evaluation table of
//! a polynomial of degree `< d_i`, where `d_i = d_0 / k^i` is the
//! round-`i` degree bound; STIR's soundness argument only requires
//! that `f_i` is close in Hamming distance to such a codeword.)
//!
//! ## When to use vs [`crate::merkle::MerkleTree::commit`] directly
//!
//! `MerkleTree::commit` is general-purpose: it takes any `&[Fp]` and gives you
//! a tree. This module sits *one layer up* — it is the STIR-protocol-level
//! commitment. The wrapper exists for three reasons:
//!
//! 1. **Type-level intent.** A `StirCommitment` is a commitment to a
//!    **function `f_i: L_i → F`** — equivalently, its evaluation table on
//!    `L_i`, an `|L_i|`-element vector of field elements — not to arbitrary
//!    data. The proximity target (what STIR proves `f_i` is close to) is a
//!    Reed-Solomon codeword in `RS[F, L_i, d_i]`, but the commitment itself
//!    binds the function. Code that consumes one knows the leaves are field
//!    elements indexed by `domain.element(j)`.
//! 2. **Bundling `(root, tree_size)`.** The verifier needs both to reconstruct
//!    the Merkle path layout; [`crate::merkle::MerkleTree::verify`] requires
//!    `tree_size` as a separate argument. Bundling the size at commit time
//!    keeps the verifier's hand-off ergonomic and turns one type-of-bug class
//!    ("forgot to pass the right size") into a compile-time check.
//! 3. **Future extension.** Production STIR variants commit to **chunks** of
//!    the evaluation table (groups of `k` consecutive evaluations as a single
//!    leaf, so the prover opens `k`-vectors per query instead of `k`
//!    individual leaves). The chunked variant changes the leaf encoding but
//!    not the rest of the interface. Wrapping `MerkleTree` here is the
//!    natural place to add that later without churning the prover/verifier
//!    files.
//!
//! ## Worked example
//!
//! Suppose `L_0` has size 8 (so `|L_0| = 8`) and the prover's current-round
//! function `f_0: L_0 → F` has evaluation table `[Fp(1), Fp(2), Fp(3),
//! Fp(4), Fp(5), Fp(6), Fp(7), Fp(8)]` (in honest runs, this would be the
//! evaluations of some degree-`< d_0` polynomial on `L_0`). Then
//!
//! ```text
//! let (commitment, tree) = StirCommitment::commit(&evals);
//! commitment.tree_size == 8
//! commitment.root      == tree.root()
//! ```
//!
//! The prover absorbs `commitment.root.0` (32 bytes) into the transcript, then
//! later — when the verifier asks for the leaf at index `3` — calls
//! `tree.open(3)` to produce the Merkle path. The verifier checks
//!
//! ```text
//! MerkleTree::verify(commitment.root, 3, leaf, &path, commitment.tree_size)
//! ```
//!
//! against the **committed** root, not against the prover's freshly-recomputed
//! tree.
//!
//! ## Theorem invoked (binding)
//!
//! **Binding of Merkle commitments under collision-resistance.** If SHA3-256
//! is collision-resistant, then for any fixed root `r` the set of `(index,
//! leaf, path, tree_size)` quadruples that `MerkleTree::verify` accepts
//! against `r` is uniquely determined except with probability `2^{-256}` in
//! the random-oracle sense. In particular, the prover cannot commit to one
//! evaluation table and later open the same index to a different value. This
//! is the reason every STIR security argument can treat `commitment.root` as
//! a pointer to a *fixed* function `f_i: L_i → F`, even though the verifier
//! never sees the function's full evaluation table.
//!
//! **Question for the reader.** Why does the verifier need `tree_size` if it
//! can recompute it from `params.round_log_domain_size(round)`?
//!
//! Two reasons, both about defense-in-depth. (a) The verifier *does*
//! independently know what `tree_size` should be in any given round — it's
//! derived from the public `StirParams` exactly as you note. The
//! [`StirCommitment::tree_size`] field is what the **prover claims** the size
//! to be. Cross-checking the prover-supplied value against the params-derived
//! expected value catches misalignment bugs (off-by-one in round indexing,
//! wrong-round commitment serialized, etc.) at the boundary of the proof
//! object instead of failing deep inside Merkle verification with a confusing
//! "path length mismatch" error. (b) It makes the proof object *self-
//! describing*: a future verifier with a partial view of the params can still
//! sanity-check internal consistency. Belt and suspenders is cheap and helps
//! the reader of failing tests narrow the failure quickly.
//!
//! `// CAUTION:` STIR commits to a **function `f_i: L_i → F`** via its
//! **evaluation table**, never to coefficients. The Merkle leaf at index `j`
//! is the field element `f_i(domain.element(j))`. If you accidentally pass
//! coefficients to [`StirCommitment::commit`] the protocol silently produces
//! a soundness hole — the verifier's queries will succeed against a
//! *committed-to coefficient table* but the algebraic checks (fold,
//! quotient, OOD) all assume an *evaluation table*, so the soundness
//! argument no longer applies. The signature takes `evals: &[reed_solomon::Fp]`
//! to make the intended payload explicit.

use reed_solomon::Fp;

use crate::merkle::{MerkleRoot, MerkleTree};

/// A STIR per-round commitment to a function `f_i: L_i → F` (via its
/// evaluation table on `L_i`).
///
/// Layout:
/// - `root` is the 32-byte SHA3-256 Merkle root of the (padded) evaluation
///   table. It is the only object the prover sends to the verifier at commit
///   time, and the only object the verifier later verifies Merkle paths
///   against.
/// - `tree_size` is the *unpadded* number of evaluations in the committed
///   table, equal to `evals.len()` from the call to [`Self::commit`]. The
///   verifier reads this and cross-checks against the expected round-`i`
///   domain size derived from public [`crate::params::StirParams`] (see the
///   "Question for the reader" in the module docs).
///
/// `tree_size` is intentionally **not** authenticated by the Merkle structure
/// (a malicious prover could lie about it); see module docs for why the
/// verifier still benefits from carrying it along.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StirCommitment {
    /// SHA3-256 Merkle root of the committed evaluation table.
    pub root: MerkleRoot,
    /// Number of unpadded leaves (i.e., number of evaluations committed to).
    /// Equal to `evals.len()` at the call site.
    pub tree_size: usize,
}

impl StirCommitment {
    /// Commit to the evaluation table `evals` of a STIR-round function
    /// `f_i: L_i → F` (in honest runs, the evaluations of a degree-`< d_i`
    /// polynomial on `L_i`).
    ///
    /// Returns `(commitment, tree)`: the lightweight [`StirCommitment`]
    /// value (root + tree size) for the transcript, and the materialised
    /// [`MerkleTree`] for the prover to keep around — the prover will later
    /// call [`MerkleTree::open`] on it to answer shift-query openings.
    ///
    /// # Inputs
    /// - `evals`: the function's evaluation table on `L_i`, length `|L_i|`.
    ///   The `j`-th entry is `f_i(domain.element(j))` (the `domain` itself
    ///   is not needed at this layer — the indexing convention is the
    ///   caller's responsibility).
    ///
    /// # Outputs
    /// - `commitment.root`: the SHA3-256 Merkle root. Absorb into the
    ///   transcript.
    /// - `commitment.tree_size`: equal to `evals.len()`.
    /// - `tree`: the full Merkle tree. Keep on the prover side only — it
    ///   is not part of the proof and the verifier never sees it.
    ///
    /// # Panics
    /// Panics if `evals.is_empty()` (we can't commit to an empty evaluation
    /// table; a zero-length round would mean a malformed
    /// [`crate::params::StirParams`]).
    ///
    /// # Paper reference
    /// STIR (eprint 2024/390), §3 "Protocol" — every round begins with the
    /// prover sending `root(f_i)`. This function produces `root(f_i)`.
    pub fn commit(evals: &[Fp]) -> (Self, MerkleTree) {
        // TODO: build the STIR-level commitment.
        //   1. Assert `!evals.is_empty()`.
        //      WHY: a zero-length tree is meaningless; protect against
        //      params bugs upstream.
        //   2. Call `MerkleTree::commit(evals)` to materialise the tree.
        //      WHY: this is the central commit operation — SHA3-256 hash
        //      of each leaf, pair-and-hash up to a single root.
        //   3. Read `root = tree.root()` for the lightweight commitment.
        //      WHY: `root` is the only thing the verifier sees at commit
        //      time; `tree_size` rides along as protocol metadata.
        //   4. Capture `tree_size = evals.len()`.
        //      WHY: stored unpadded so the verifier can cross-check it
        //      against the params-derived expected round-`i` size.
        //   5. Return `(Self { root, tree_size }, tree)`.
        let _ = evals;
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Commit returns a root and a tree_size matching the input length.
    #[test]
    fn commit_returns_consistent_root_and_tree_size() {
        // TODO:
        //   1. evals = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)].
        //   2. let (commitment, tree) = StirCommitment::commit(&evals).
        //   3. assert_eq!(commitment.tree_size, 4).
        //   4. assert_eq!(commitment.root, tree.root()).
        // WHY: the commitment's two fields are exactly the "tag" the
        // verifier will later use; pin them down.
        todo!()
    }

    /// Same evals → same root (determinism).
    #[test]
    fn commit_of_same_evals_gives_same_root() {
        // TODO:
        //   1. evals = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)].
        //   2. let (c1, _) = StirCommitment::commit(&evals).
        //   3. let (c2, _) = StirCommitment::commit(&evals).
        //   4. assert_eq!(c1.root, c2.root); assert_eq!(c1.tree_size, c2.tree_size).
        // WHY: determinism is what makes the verifier able to recompute
        // anything; flake here would break the entire protocol.
        todo!()
    }

    /// Different evals → different roots (binding sanity).
    #[test]
    fn different_evals_give_different_roots() {
        // TODO:
        //   1. let (c_a, _) = StirCommitment::commit(&[Fp::new(1), Fp::new(2)]).
        //   2. let (c_b, _) = StirCommitment::commit(&[Fp::new(1), Fp::new(3)]).
        //   3. assert_ne!(c_a.root, c_b.root).
        // WHY: SHA3-256 collision probability is ≈ 2^{-128} (birthday).
        // We never see one in tests; this asserts the "commitment binds
        // to the input" property at the integration boundary.
        todo!()
    }
}
