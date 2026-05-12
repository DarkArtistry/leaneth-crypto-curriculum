# leanEthereum Crypto Curriculum

A Rust curriculum that rebuilds [leanEthereum](https://ethereum.org/)'s
post-quantum cryptographic stack from first principles. Every primitive is
implemented from scratch — no external crypto libraries — with extensive
inline pedagogy aimed at making the underlying math click before the code is
written.

## Layout

This is a Cargo **workspace**. Each top-level directory under the workspace
root is one objective, implemented as its own crate. Later objectives depend
on the primitives implemented in earlier ones.

| #  | Crate                                  | Topic                                                          | Status        |
|----|----------------------------------------|----------------------------------------------------------------|---------------|
| 1  | [`sumcheck/`](./sumcheck/)             | Multilinear sumcheck (Lund-Fortnow-Karloff-Nisan 1992)         | complete      |
| 2  | [`reed_solomon/`](./reed_solomon/)     | Goldilocks field, NTT, RS encoder/decoder                      | in progress   |
| 3  | `stir/`                                | STIR low-degree test (eprint 2024/390)                         | pending       |
| 4  | `whir/`                                | Multilinear extension of STIR (eprint 2024/1586)               | pending       |
| 5  | `stark/`                               | AIR + pluggable low-degree-test STARK                          | pending       |
| 6  | `stark_whir/`                          | STARK + WHIR integration                                       | pending       |
| 7  | `xmss/`                                | Generalized XMSS (Drake et al., hash-based signatures)         | pending       |
| 8  | `lean_eth/`                            | N-validator XMSS aggregation under a STARK                     | pending       |

## Working in this repo

### Build / test a single objective

```bash
cargo build -p sumcheck                       # build one crate
cargo test  -p reed_solomon                   # run that crate's tests
cargo run   -p sumcheck --bin demo            # run its narrative demo
```

### Build / test all objectives

```bash
cargo build                                   # all workspace members
cargo test                                    # all unit + integration tests
```

### Formatting + lints

```bash
cargo fmt --all                               # apply rustfmt
cargo fmt --all --check                       # CI-style check
cargo clippy --all-targets -- -D warnings     # treat clippy warnings as errors
```

## How to read each sub-crate

Each objective ships with the same shape:

- **`README.md`** at its root — the goal, the order to work through the
  files, and the definition of done.
- **Module-level rustdoc** (the `//!` block at the top of each
  `src/<module>.rs`) — walks through the math with worked numeric examples
  before showing the algorithm. This is the load-bearing pedagogy; the
  code is the punchline.
- **`///` doc comments** on every public method, plus retrospective
  inline comments inside method bodies explaining *why* the implementation
  is what it is. Where a load-bearing theorem is invoked (Fermat's little
  theorem, Lagrange's theorem, the Fundamental Theorem of Cyclic Groups,
  the zero-product property of fields, etc.) the theorem is named and a
  short inline proof is given.

**If you're reading to learn the math:** start with the module-level
docstrings and the worked examples. The code is meant to be the result of
internalising the math, not a substitute for it.

**If you're reading to evaluate the implementation:** start with each
crate's `tests/integration.rs` and `src/<module>.rs#tests`. The tests
exercise the end-to-end happy path plus the edge cases the math says
should matter.

## Style guidelines

- Every public method has a `///` doc comment.
- Named theorems are cited explicitly with short inline proofs.
- No `unwrap()` outside `#[cfg(test)]` code; use `?` or pattern-matching.
- `#[derive(Clone, Debug, PartialEq, Eq)]` liberally on data types.
- Each crate's tests must pass with `cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` clean.

## Philosophy

The curriculum is dependency-ordered: sumcheck is the simplest IOP-style
protocol and introduces the round-by-round soundness pattern; Reed-Solomon
underpins every low-degree test (FRI, STIR, WHIR); STIR/WHIR are concrete
low-degree tests; STARK is the encoding-plus-LDT combination; XMSS is the
signature scheme; and the final integration aggregates `N` XMSS signatures
under a STARK proof — the leanEthereum vision in miniature.

Each objective is implemented from scratch (no `ark-*`, no `p3-*`) so the
algorithms are forced into the surface of the code rather than hidden
behind dependencies. Libraries enter only where they don't pay for
themselves pedagogically — currently just `rand`, `sha3` (for STIR), and
`blake3` (for STIR).
