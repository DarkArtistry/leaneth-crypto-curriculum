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
//! ## Anchor: what the verifier remembers between rounds, and why
//!
//! STIR is an interactive protocol with `M + 1` rounds, but the
//! verifier is sublinear: it must *not* store the prover's `|L_i|`
//! evaluation tables. The role of a [`StirCommitment`] is to be the
//! **constant-size memory token** the verifier carries forward from
//! round `i` into round `i + 1`. Concretely, after the prover sends
//! round-`i`'s function `f_i`:
//!
//! 1. The verifier receives `commitment_i = (root_i, tree_size_i)` —
//!    **64 bytes total** (32 for the SHA3-256 root, 8 for the size on
//!    a 64-bit target).
//! 2. The Fiat-Shamir transcript absorbs `root_i`. From this absorption
//!    the verifier later derives the round's random challenges
//!    (folding randomness, OOD sample points, spot-check indices).
//! 3. When the verifier issues spot-check index `j ∈ [0, tree_size_i)`,
//!    the prover replies with `(leaf, path)` and the verifier calls
//!    [`crate::merkle::MerkleTree::verify`]`(root_i, j, leaf, &path,
//!    tree_size_i)`. Both fields of `commitment_i` are arguments to
//!    that call — neither is optional.
//!
//! So `StirCommitment` is the *minimum* the verifier must remember per
//! round to (a) re-derive the round's randomness and (b) authenticate
//! later openings. Everything else — the full evaluation table, the
//! Merkle layers — stays on the prover side, in the [`MerkleTree`]
//! returned alongside the commitment from [`StirCommitment::commit`].
//! This is what makes the verifier's persistent state `O(M · 64)` bytes
//! across all rounds, independent of `|L_i|`.
//!
//! ## Leaf layout: one `Fp` per leaf (this educational variant)
//!
//! [`crate::merkle::MerkleTree`] hashes **one field element per
//! Merkle leaf**: leaf `j` is `H_leaf(f_i(domain.element(j)))`. Path
//! length is `⌈log₂ |L_i|⌉` and each opening costs that many sibling
//! hashes. We chose the one-`Fp`-per-leaf layout for pedagogical
//! simplicity — it makes the leaf-encoding rule a single line and lets
//! the worked example below fit on screen.
//!
//! `// FORWARD-POINTER:` production STIR (and most STARK deployments)
//! typically **chunk** the evaluation table into groups of `k`
//! consecutive evaluations per leaf, so that each round's `t_i` spot
//! checks open `t_i` chunks instead of `t_i · k` individual leaves —
//! shaving roughly `log₂ k` levels off every Merkle path. The chunked
//! variant changes the leaf encoding (`H_leaf(f_i(x_0), …, f_i(x_{k-1}))`)
//! but neither the [`StirCommitment`] interface nor the rest of the
//! protocol. Adding chunking later is a localised edit to
//! [`crate::merkle`] plus a thin re-shape of `evals` here; the rest of
//! the prover/verifier wiring is unchanged.
//!
//! ## When to use vs [`crate::merkle::MerkleTree::commit`] directly
//!
//! `MerkleTree::commit` is general-purpose: it takes any `&[Fp]` and gives you
//! a tree. This module sits *one layer up* — it is the STIR-protocol-level
//! commitment. The wrapper exists for two reasons:
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
//!    keeps the verifier's hand-off ergonomic and turns a class of bugs
//!    ("forgot to pass the right size") into a localised mismatch detectable
//!    at the proof-object boundary. The named theorem below makes this
//!    rigorous.
//!
//! ## Worked example: commit to `[Fp(1), Fp(2), Fp(3), Fp(4)]`
//!
//! Pick the minimal non-trivial domain: `|L| = 4`, evaluation table
//! `[Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)]`. Since `4 = 2²`
//! is already a power of 2, no zero-padding fires. The build:
//!
//! ```text
//!     let evals = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)];
//!     let (commitment, tree) = StirCommitment::commit(&evals);
//!     // commitment.tree_size == 4   (unpadded length of `evals`)
//!     // commitment.root      == tree.root()
//! ```
//!
//! The Merkle layout (using the tagged hashes from
//! [`crate::merkle`], `LEAF_TAG = 0x00`, `NODE_TAG = 0x01`):
//!
//! ```text
//! Layer 0 (leaf hashes, one Fp per leaf — see "Leaf layout" above):
//!     h0 = SHA3-256(0x00 || 0x0100000000000000)        // Fp(1).as_u64() = 1
//!     h1 = SHA3-256(0x00 || 0x0200000000000000)
//!     h2 = SHA3-256(0x00 || 0x0300000000000000)
//!     h3 = SHA3-256(0x00 || 0x0400000000000000)
//! Layer 1: h01 = SHA3-256(0x01 || h0 || h1)
//!          h23 = SHA3-256(0x01 || h2 || h3)
//! Layer 2: root = SHA3-256(0x01 || h01 || h23)
//! ```
//!
//! Concrete value (verified by `print_worked_example_root` below):
//!
//! ```text
//!     commitment.root.0 = 0x93a945db5d50607d385e6ab99e8587b2_
//!                         73a12fd5577d4fe38736a8f8c9da63d0
//!     commitment.tree_size = 4
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
//! against the **committed** root, not against any prover-recomputed tree.
//!
//! ## Named theorem invoked (binding of the Merkle root)
//!
//! > **Binding theorem (collision resistance of the Merkle root).** Fix
//! > a root `r`. Under SHA3-256 collision resistance, for every leaf
//! > index `i` and every `tree_size n`, there is at most one leaf value
//! > `a ∈ Fp` such that some path `π` makes
//! > [`crate::merkle::MerkleTree::verify`]`(r, i, a, π, n)` accept;
//! > extracting two distinct accepting `(a, π_a)`, `(b, π_b)` pairs
//! > exhibits an explicit SHA3-256 collision. (See
//! > [`crate::merkle`] §"Named theorem: Binding of the Merkle root"
//! > for the constructive proof.)
//!
//! This is the property that lets every STIR soundness statement
//! quantify over a *fixed* function `f_i: L_i → F` after the verifier
//! has seen only the 32-byte `root_i`.
//!
//! ## Named theorem: Tree-Size Binding
//!
//! > **Tree-Size Binding theorem.** Let `commitment_i = (root_i,
//! > tree_size_i)` be the STIR-round-`i` commitment, and let the
//! > verifier carry both fields forward. Under SHA3-256 collision
//! > resistance, accepting a Merkle opening at index `j` against
//! > `commitment_i` is equivalent to accepting it against a function
//! > `f_i: L_i → F` of **exactly `tree_size_i` evaluations** —
//! > equivalently, a Merkle tree of padded height
//! > `⌈log₂ tree_size_i⌉`. A prover cannot, having committed
//! > `(root_i, tree_size_i)`, later answer openings as though the
//! > committed table had a different length.
//!
//! **Proof (constructive).** Suppose the verifier accepts an opening
//! `(j, leaf, path)` against `commitment_i = (root_i, n)` with `n =
//! tree_size_i`. From [`crate::merkle::MerkleTree::verify`], the
//! verifier rejects unless `path.siblings.len() == ⌈log₂(max(n, 1))⌉`
//! (the structural length check) **and** the recomputed hash chain at
//! that height equals `root_i`. So acceptance forces a *specific*
//! Merkle height, which is the height of a tree with exactly `n`
//! padded leaves. If a malicious prover claims the same `root_i` came
//! from a different table length `n' ≠ n` with `⌈log₂ n'⌉ ≠ ⌈log₂ n⌉`,
//! then the path length for any honest opening against the `n'`-leaf
//! tree differs from that for the `n`-leaf tree — the structural check
//! rejects automatically. If `⌈log₂ n'⌉ = ⌈log₂ n⌉` (same padded
//! height but different unpadded `n`), the structural check passes,
//! **but** the verifier additionally rejects `j ≥ n` (indices in the
//! padding-only region), so any opening at `j ∈ [n, n')` is rejected
//! against the smaller `n` — and any honest leaf-vs-pad confusion at
//! `j ∈ [min(n,n'), max(n,n'))` reduces to a SHA3-256 collision by the
//! Binding theorem above (the pad leaf is `H_leaf(Fp::zero())`, which
//! must collide with the prover's claimed leaf to be accepted). ∎
//!
//! **Why this matters.** Without binding `tree_size` together with
//! `root`, a malicious prover could carry out a **length-extension
//! attack**: commit to a 4-leaf table, then later open "leaves" at
//! indices 4, 5, 6, 7 of a fictitious 8-leaf table that happens to
//! pad-extend the original. The padding rule `[real_0, …, real_3, 0,
//! 0, 0, 0]` plus a single extra `H_node` layer would produce a *new*
//! 8-leaf root the prover never committed to — but if the verifier
//! sloppily accepted the prover's claim that the original commitment
//! is "actually" an 8-leaf tree, it would walk an 8-deep hash chain
//! against the original 4-leaf root and the structural check would
//! catch nothing. The Tree-Size Binding theorem says: as long as
//! `tree_size_i` is fixed *at commit time* and the verifier passes it
//! into every `verify`, length-extension is structurally impossible.
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
        // Cross-ref: the *pair* `(root, tree_size)` is the verifier's
        // round-`i` memory token (see module doc §"Anchor"). Capturing
        // both at commit time — and binding them together in the
        // returned struct — is exactly what the Tree-Size Binding
        // theorem (module doc) needs: storing `tree_size` separately
        // and threading it through every later [`MerkleTree::verify`]
        // call is what makes length-extension attacks structurally
        // impossible (the structural path-length check fires before
        // any hash work).
        assert!(
            !evals.is_empty(),
            "StirCommitment::commit: empty evaluation table — \
             a zero-length round signals a malformed StirParams upstream",
        );

        // One `Fp` per Merkle leaf — see module doc §"Leaf layout".
        // [`MerkleTree::commit`] pads up to the next power of 2 with
        // `Fp::zero()` leaves internally; we record the *unpadded*
        // length here so the verifier reproduces padding the same way
        // (cf. [`MerkleTree::verify`]'s `next_power_of_two(tree_size)`
        // computation).
        let tree = MerkleTree::commit(evals);
        let root = tree.root();
        let tree_size = evals.len();

        (Self { root, tree_size }, tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle::MerkleTree;

    /// Commit returns a root and a tree_size matching the input length,
    /// and the returned tree's root matches the commitment's root.
    #[test]
    fn commit_returns_consistent_root_and_tree_size() {
        let evals = [Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)];
        let (commitment, tree) = StirCommitment::commit(&evals);
        // `tree_size` is the *unpadded* count — see field-level doc on
        // `StirCommitment::tree_size`. evals.len() == 4 is already a
        // power of 2, so the padded size also equals 4.
        assert_eq!(commitment.tree_size, 4);
        assert_eq!(commitment.root, tree.root());
    }

    /// Different evals → different roots (binding sanity).
    ///
    /// SHA3-256 collision probability is `≈ 2^{-128}` (birthday); we
    /// never observe one in tests. This pins down the *binding* half of
    /// the Merkle commitment property at the STIR-level boundary, i.e.,
    /// asserts the commitment actually depends on the input.
    #[test]
    fn different_evals_give_different_roots() {
        let (c_a, _) = StirCommitment::commit(&[Fp::new(1), Fp::new(2)]);
        let (c_b, _) = StirCommitment::commit(&[Fp::new(1), Fp::new(3)]);
        assert_ne!(c_a.root, c_b.root);
    }

    /// The tree returned by `commit` is the prover's "stash" — it must
    /// be openable, and the resulting `(leaf, path)` triple must verify
    /// against the commitment's root via `MerkleTree::verify`.
    ///
    /// This is the end-to-end roundtrip the prover/verifier pair will
    /// rely on every round: commit → absorb root → (later) open at a
    /// challenged index → verify against the committed root.
    #[test]
    fn opened_leaf_verifies_against_commitment_root() {
        let evals = [Fp::new(10), Fp::new(20), Fp::new(30), Fp::new(40)];
        let (commitment, tree) = StirCommitment::commit(&evals);
        for i in 0..evals.len() {
            let (leaf, path) = tree.open(i);
            assert_eq!(leaf, evals[i]);
            assert!(
                MerkleTree::verify(
                    commitment.root.clone(),
                    i,
                    leaf,
                    &path,
                    commitment.tree_size,
                ),
                "honest path at index {i} failed to verify against StirCommitment",
            );
        }
    }

    /// Print the worked-example root to stdout. Run with
    /// `cargo test --lib commitment::tests::print_worked_example_root -- --nocapture`.
    /// Used to bake the actual hex prefix into the module docs; left
    /// behind as a one-shot diagnostic.
    #[test]
    #[ignore]
    fn print_worked_example_root() {
        let (c, _) = StirCommitment::commit(&[
            Fp::new(1),
            Fp::new(2),
            Fp::new(3),
            Fp::new(4),
        ]);
        print!("root_hex = ");
        for b in c.root.0.iter() {
            print!("{b:02x}");
        }
        println!();
        println!("tree_size = {}", c.tree_size);
    }
}
