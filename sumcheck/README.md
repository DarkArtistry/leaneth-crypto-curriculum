# Sumcheck — Objective 1

This is the first objective in the leanEthereum coding curriculum. See
`../coding_plan.md` for the full curriculum and `../../sumcheck_study.md`
for the conceptual notes.

## Background: The Sumcheck Protocol

### The problem

Let `g : F^n -> F` be a multilinear polynomial known to both prover and
verifier — `n` is the number of variables of `g`. The prover claims

```text
H = sum over b in {0,1}^n of g(b)
```

The protocol implementation in this crate is generic over the
per-variable summation domain `S` (see `src/domain.rs` for the
`SumDomain` trait and the `BooleanHypercube` / `Interval3` instances).
The worked example below uses `S = {0, 1}` — the Boolean hypercube,
which is what Lean Ethereum and every modern multilinear SNARK use —
but the same `run_sumcheck` works unchanged for any `S: SumDomain`, as
the `Interval3` integration tests demonstrate.

A naive verifier evaluates `g` at every Boolean corner and adds the
`2^n` results — exponential in `n`. For `n = 20` that's a million
evaluations; for `n = 40` it's a trillion. The whole point of
delegation is to **avoid** doing this. Sumcheck is the reduction that
turns the hypercube-sum claim into a *single-point* evaluation claim
`g(r_1, ..., r_n) = y` in `O(n)` rounds. The verifier evaluates `g`
exactly once, at a random point of the verifier's choosing.

(A "multilinear" polynomial has per-variable degree at most 1 — no
`x_i^2`, no `x_i^7`. See `src/polynomial.rs` for the full primer.)

### Why it matters

Sumcheck is the foundational IOP primitive underneath GKR, Spartan,
HyperPlonk, Jolt, Lasso. The "send a degree-bounded message, check it
sums correctly, fold with a random challenge" pattern recurs in STIR,
WHIR, and FRI — though those operate on Reed-Solomon codewords, not
hypercube sums. Learn sumcheck first; everything else is a variation.

### The protocol, round by round

Both parties know `g` and `v`. The prover initially declares `H`.

**Round i** (for `i = 1, ..., n`):

1. Prover sends the univariate polynomial
   ```text
   s_i(X) = sum over (b_{i+1},...,b_n) in {0,1}^{v-i} of
              g(r_1, ..., r_{i-1}, X, b_{i+1}, ..., b_v)
   ```
   Since `g` is multilinear, `s_i` has degree 1, so two values suffice
   — the prover sends `[s_i(0), s_i(1)]`.
2. Verifier checks `s_i(0) + s_i(1) == previous_claim`. Reject on
   mismatch.
3. Verifier samples `r_i` uniformly at random from `F`, sends it back.
4. Both parties update `previous_claim := s_i(r_i)`.

**Final check.** After round `n` the verifier evaluates `g(r_1, ..., r_n)`
once and compares with the final claim. Accept iff equal.

#### Worked example: n = 2

Take `g(x_1, x_2)` with evaluations on `{0,1}^2`:

```text
g(0, 0) = 1     g(1, 0) = 2     g(0, 1) = 3     g(1, 1) = 4
```

Honest sum: `H = 1 + 2 + 3 + 4 = 10`.

**Round 1.** Prover computes
- `s_1(0) = g(0,0) + g(0,1) = 1 + 3 = 4`
- `s_1(1) = g(1,0) + g(1,1) = 2 + 4 = 6`

Check: `s_1(0) + s_1(1) = 4 + 6 = 10 = H`. Verifier samples `r_1 = 5`
(say). New claim: `s_1(5) = 4 + 5·(6 - 4) = 14`.

**Round 2.** Prover internally fixes `x_1 = 5` (linear interpolation
between corners; see `polynomial::fix_first_variable` in
`src/polynomial.rs`). The reduced polynomial `g'(x_2) = g(5, x_2)` has
- `g'(0) = g(5, 0) = (1-5)·1 + 5·2 = 6`
- `g'(1) = g(5, 1) = (1-5)·3 + 5·4 = 8`

So `s_2(0) = 6`, `s_2(1) = 8`. Check: `6 + 8 = 14`. Verifier samples
`r_2 = 7`. Final claim: `s_2(7) = 6 + 7·(8 - 6) = 20`.

**Final.** Verifier evaluates `g(5, 7)` itself using the MLE formula
(`evaluate` in `src/polynomial.rs`) and gets `20`. Accept.

### Why the per-round check works

The crucial identity: `s_i(0) + s_i(1)` is, by definition,

```text
sum over b_i in {0,1} of sum over (b_{i+1},...,b_n) in {0,1}^{v-i} of
    g(r_1, ..., r_{i-1}, b_i, b_{i+1}, ..., b_v)
```

which is `g`'s sum over the remaining `n - i + 1` Boolean variables.
That's *exactly* what the previous round's claim asserted (round `i-1`
left the verifier expecting that this sum equals `s_{i-1}(r_{i-1})`;
round 0 was the original `H`). So an honest `s_i` always passes the
check.

The random `r_i` injection is what stops a cheating prover from
choosing `s_i` to satisfy step 2 while lying everywhere else.

### Soundness analysis

Suppose the prover lies in round `i`: sends `hat_s_i ≠ s_i` (where
`s_i` is the honest univariate slice). For the cheat to survive, the
verifier's random sample `r_i` must satisfy `hat_s_i(r_i) = s_i(r_i)`,
i.e., the two polynomials agree at `r_i`.

**Schwartz-Zippel** (or just the polynomial root bound — a degree-`d`
nonzero univariate polynomial over a field `F` has at most `d` roots,
by the factor theorem combined with the zero-product property in
integral domains). Apply it to `hat_s_i - s_i`, which is a nonzero
polynomial of degree at most `d = 1` (multilinear case): it has at
most 1 root in `F`. Probability that a uniform `r_i` hits that root:

```text
Pr[cheat survives round i]  ≤  d / |F|  =  1 / |F|
```

**Union bound** across all `n` rounds (failure modes don't need to be
independent):

```text
Pr[verifier fooled]  ≤  n · d / |F|  =  n / |F|     (multilinear)
```

For this crate's field `F_p` with `p = 2^61 - 1` and `n = 20`,
multilinear (`|S| = 2`, so `d = 1`):

```text
total error  ≤  20 / 2^61  ≈  2^-56.5
```

Cryptographically sound. For general `|S| = k` (with the storage
convention that per-variable degree is `< k`), the per-round error is
`(k - 1) / |F|` — the `hat_s_i - s_i` polynomial has degree `< k`,
hence at most `k - 1` roots. For `|S| = 3` (`Interval3`) with the same
`n = 20`, total error doubles to `≤ 40 / 2^61 ≈ 2^-55.5`. Still fine.

For larger `n` or smaller fields, see the field-size table in
`../../sumcheck_study.md` §3 — short version: Goldilocks (`2^64`) at
`n = 25` gives `2^-59` which is fine for soundness amplification but
not standalone-`2^-100`-secure, hence the move to extension fields in
production systems like Plonky3.

### Where it appears in this crate

- `SumDomain`, `BooleanHypercube`, `Interval3` in `src/domain.rs` — the
  per-variable summation domain `S` and its two concrete instances.
- `MultivariatePoly<D>` (alias `MultilinearPoly` for `D = BooleanHypercube`)
  in `src/polynomial.rs` — evaluation-form storage on `S^n`,
  `evaluate`, `sum_over_domain` / `sum_over_hypercube`, `fix_first_variable`
  (generic `|S|`-point Lagrange).
- `SumcheckProver` in `src/prover.rs` — per-round state machine.
- `SumcheckVerifier` in `src/verifier.rs` — the sum-over-`S` check and
  random challenge.
- `run_sumcheck` in `src/protocol.rs` — the orchestrator that wires
  them together, generic over any `D: SumDomain`.

The crate-level `src/lib.rs` has the framing for "what's generic now vs
what's still future work" (higher-per-variable-degree, univariate
sumcheck over a coset, batched sumcheck) — read it once.

### References

- Lund, Fortnow, Karloff, Nisan, "Algebraic methods for interactive
  proof systems," JACM 39(4), 1992. The original.
- Thaler, *Proofs, Arguments, and Zero-Knowledge*, §4.1. The modern
  textbook treatment this crate follows.
- `../../sumcheck_study.md` — Kenneth's own study notes (Thaler §4.1
  five-pass walkthrough, soundness drill, modern systems landscape).

## Your job

Fill in every `todo!()` body in `src/`. The structure is:

```
src/
├── lib.rs           # crate root, module declarations, top-level docs
├── field.rs         # toy F_p over the Mersenne prime 2^61 - 1
├── domain.rs        # SumDomain trait + BooleanHypercube / Interval3
├── polynomial.rs    # MultivariatePoly<D> in evaluation form on S^n
├── prover.rs        # SumcheckProver — keeps state across rounds
├── verifier.rs      # SumcheckVerifier — randomized challenger
├── protocol.rs      # run_sumcheck — orchestrates one full session
└── bin/
    └── demo.rs      # narrative demo printing each round
tests/
└── integration.rs   # end-to-end tests (Boolean + Interval3)
```

## Recommended order

1. **`field.rs`** — get arithmetic working before anything else. Run
   the unit tests in `field.rs` (you'll need to write them too — they're
   currently empty).
2. **`domain.rs`** — the per-variable summation domain `S`. Short file;
   defines the `SumDomain` trait and the `BooleanHypercube` / `Interval3`
   instances that downstream modules are generic over.
3. **`polynomial.rs`** — `MultivariatePoly<D>` on `S^n`. Test
   `fix_first_variable` thoroughly: it's the operation sumcheck depends on,
   and the generic `|S|`-point Lagrange path is the easiest place to
   silently introduce bugs.
4. **`prover.rs`** — round-by-round state machine. Test
   `compute_round_message` against the hand-computed example in the test.
5. **`verifier.rs`** — the sum-over-`S` check + Lagrange interpolation
   to update the running claim. Test rejection of inconsistent messages.
6. **`protocol.rs`** — glue it together. Run integration tests
   (Boolean + Interval3).
7. **`bin/demo.rs`** — narrative output for the Boolean-hypercube case.

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
  `F^v`, not just `{0,1}^n`. Test with a non-boolean point explicitly.

## When you're done

Tell Claude "sumcheck done" in chat. Claude will spawn review agents and
consolidate feedback. Iterate until clean, then move to objective 2 (Reed–Solomon).
