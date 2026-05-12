# Sumcheck — Objective 1

This is the first objective in the leanEthereum coding curriculum. See
`../coding_plan.md` for the full curriculum and `../../sumcheck_study.md`
for the conceptual notes.

## Your job

Fill in every `todo!()` body in `src/`. The structure is:

```
src/
├── lib.rs           # crate root, module declarations, top-level docs
├── field.rs         # toy F_p over the Mersenne prime 2^61 - 1
├── polynomial.rs    # multilinear polynomials in evaluation form
├── prover.rs        # SumcheckProver — keeps state across rounds
├── verifier.rs      # SumcheckVerifier — randomized challenger
├── protocol.rs      # run_sumcheck — orchestrates one full session
└── bin/
    └── demo.rs      # narrative demo printing each round
tests/
└── integration.rs   # end-to-end tests
```

## Recommended order

1. **`field.rs`** — get arithmetic working before anything else. Run
   the unit tests in `field.rs` (you'll need to write them too — they're
   currently empty).
2. **`polynomial.rs`** — multilinear polynomials. Test
   `fix_first_variable` thoroughly: it's the operation sumcheck depends on.
3. **`prover.rs`** — round-by-round state machine. Test
   `compute_round_message` against the hand-computed example in the test.
4. **`verifier.rs`** — the per-round check. Test rejection of
   inconsistent messages.
5. **`protocol.rs`** — glue it together. Run integration tests.
6. **`bin/demo.rs`** — narrative output.

## Build / test commands

```bash
# Build (from the workspace root, /code-practice)
cargo build -p sumcheck

# Test
cargo test -p sumcheck

# Run the demo
cargo run -p sumcheck --bin demo

# Format check
cargo fmt -p sumcheck --check

# Lint
cargo clippy -p sumcheck --all-targets -- -D warnings
```

## Definition of done

- All tests pass with `cargo test -p sumcheck`.
- `cargo clippy -p sumcheck --all-targets -- -D warnings` is clean.
- `cargo fmt -p sumcheck --check` passes.
- The `demo` binary prints a clean narrative.
- A reviewing agent (Claude will spawn one) approves the code.

## Hints

### Rust-specific

- Use `derive(Clone, Copy, Debug, PartialEq, Eq)` on `Fp`. The deriving
  works because the underlying `u64` has these traits.
- For `MultilinearPoly`, derive `Clone, Debug, PartialEq, Eq` but **not**
  `Copy` (it owns a `Vec<Fp>`).
- `Result<_, &'static str>` is fine for now. We'll switch to a proper
  error enum in objective 3.
- Use `?` in `protocol.rs` to propagate errors from `process_round_message`.
- For RNG, `SeedableRng::seed_from_u64(42)` gives reproducible randomness.

### Math-specific

- The Mersenne prime `2^61 - 1` is small enough to fit in `u64`, big
  enough to give meaningful soundness (~2^-61 per round).
- For multiplication: `let prod = (a.0 as u128) * (b.0 as u128); Fp((prod % MODULUS as u128) as u64)`.
- For inverse: Fermat's little theorem says `a^(p-1) = 1 mod p` for `a != 0`,
  so `a^(p-2) = a^(-1)`.
- The `s_i(0) + s_i(1) == previous_claim` check is the heart of the protocol.
- `s_i(r) = s_i(0) + r * (s_i(1) - s_i(0))` for degree-1 `s_i`.

### Common pitfalls

- Forgetting to reduce mod `MODULUS` somewhere → tests randomly fail.
- Off-by-one in `fix_first_variable` indexing → first round looks fine,
  later rounds explode.
- Using `polynomial.num_vars` after `fix_first_variable` mutates it → use
  the prover's stored `initial_num_vars` to know when to stop.
- Forgetting that `MultilinearPoly::evaluate` must work for *any* point in
  `F^v`, not just `{0,1}^v`. Test with a non-boolean point explicitly.

## When you're done

Tell Claude "sumcheck done" in chat. Claude will spawn review agents and
consolidate feedback. Iterate until clean, then move to objective 2 (Reed–Solomon).
