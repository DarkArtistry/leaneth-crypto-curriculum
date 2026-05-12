# Reed-Solomon — Objective 2

The second objective of the leanEthereum coding curriculum. See
`../coding_plan.md §5` for the full curriculum context, and
`../../rs_foundations.md` for the conceptual notes.

This is materially bigger than sumcheck: 8 source modules instead of 5,
because we now have a structured code (the RS code) plus its supporting
machinery (a field that supports FFT, an evaluation domain, the FFT
itself, Lagrange interpolation, and the encoder/decoder layered on top).

## Your job

Fill in every `todo!()` body in `src/`. The structure is:

```
src/
├── lib.rs           # crate root — already written; module declarations + crate docs
├── field.rs         # Goldilocks field F_p with p = 2^64 - 2^32 + 1
├── polynomial.rs    # UnivariatePoly in coefficient form
├── domain.rs        # EvaluationDomain (smooth multiplicative coset)
├── fft.rs           # Cooley-Tukey radix-2 FFT
├── interpolate.rs   # Lagrange interpolation
├── encode.rs        # ReedSolomonCode — encode by evaluating poly on the domain
├── decode.rs        # interpolation-based decoder + Berlekamp-Welch (stretch)
└── bin/
    └── demo.rs      # narrative demo
tests/
└── integration.rs   # end-to-end tests
```

## Recommended order

Two phases. Get all of phase 1 green before starting phase 2.

**Working pattern for every file:**

1. Re-read the file's module-level docstring (the `//!` block at the top) —
   it contains a worked numeric example you'll cross-check against.
2. Implement methods in the sub-order listed below.
3. Run `cargo test -p reed_solomon --lib <module>::` after each chunk.
4. If a test fails, walk the worked example by hand and find the first
   place your code disagrees with the math.
5. Don't start the next file until the current one's tests are clean. RS
   layers each module on the previous — a quiet `field.rs` bug becomes a
   noisy `fft.rs` bug, and you'll waste hours.

**Cross-reference the sumcheck crate** for `field.rs` and the operator
impls. Most code transfers directly; the deltas are `MODULUS`,
`TWO_ADICITY`, `Product`, and `primitive_root_of_unity`.

### Phase 1 — the math layer

**1. `field.rs`** — Goldilocks arithmetic + `primitive_root_of_unity`.
   1. Constructors: `new`, `zero`, `one`, `random`, `as_u64`.
   2. Operators: `Add`, `Sub`, `Neg`, `Mul` plus the `*Assign` siblings
      (use a `u128` intermediate for `Mul`).
   3. `pow` (square-and-multiply) and `inverse` (Fermat: `self.pow(MODULUS - 2)`).
   4. `Sum` and `Product` iterator impls.
   5. `primitive_root_of_unity(log_n)` — see module docs for why the
      formula works (Fermat + `g` is a generator).

   **Sanity check:** `primitive_root_of_unity(1)` must equal `MODULUS - 1`
   (i.e., `-1` mod `p`). Cleanest one-liner verification of the construction.

**2. `polynomial.rs`** — `UnivariatePoly` in coefficient form.
   1. Constructors / accessors: `new` (with trailing-zero stripping),
      `from_coeffs_unstripped`, `zero`, `one`, `is_zero`, `degree`, `coeffs`.
   2. `evaluate` via Horner.
   3. `Add`, `Sub` (pointwise on the longer length, then re-canonicalize via `new`).
   4. `Mul` — schoolbook is fine, `O(d²)`.

   **Sanity check:** `p(X) = 1 + 2X + 3X²` evaluated at `x = 5` is `86`
   (module docs walk through Horner's reduction).

   **Strongest correctness test:** `mul_evaluates_pointwise`. If
   `(p · q).evaluate(x) ≠ p.evaluate(x) · q.evaluate(x)`, your `Mul` is wrong.

**3. `domain.rs`** — `EvaluationDomain` (smooth multiplicative coset).
   1. Constructors: `new_subgroup`, `new_coset`.
   2. Trivial accessors: `size`, `log_size`, `generator`, `offset`.
   3. `element(i)` — `offset · generator^i`. Wraps around.
   4. `iter()` and `DomainIter::next` — use a **running product**, not a
      `pow` per element. The iterator should be `O(n)` field ops total;
      a `pow` per call would be `O(n log n)`.

   **Sanity check:** for a subgroup, `domain.element(0) == Fp::one()` and
   `domain.element(domain.size()) == domain.element(0)`.

**4. `fft.rs`** — Cooley-Tukey radix-2. The hardest file in the crate;
budget extra time and re-read the module docs before starting.
   1. `fft_subgroup` — base case (`n == 1`) first, then the butterfly combine.
   2. `ifft_subgroup` — forward FFT at `omega^-1`, divide every entry by `n`.
   3. `fft_on_domain` — pre-scale coefficients by powers of `c`, then `fft_subgroup`.
   4. `ifft_on_domain` — `ifft_subgroup`, then de-scale by powers of `c^-1`.

   **Critical test:** `fft_matches_naive_evaluation_subgroup`. **If this
   fails for `n = 4`, your butterfly is wrong** — walk through the `n = 4`
   worked example in the module docs by hand and find the first divergence.

**5. `interpolate.rs`** — Lagrange interpolation.
   1. `scalar_mul` (used by Lagrange — implement first).
   2. `lagrange_interpolate`.

   **Sanity check:** `(1, 1), (2, 4), (3, 9)` interpolates to `X²`
   (module docs walk through the algebra in `F_p`).

   **Strongest correctness test:** `interpolate_recovers_random_polynomial` —
   build a random poly, evaluate at `n` distinct points, interpolate, recover.

### Phase 2 — the protocol

**6. `encode.rs`** — `ReedSolomonCode`.
   1. `new`, accessors, `rate`.
   2. `encode_naive` — `domain.iter().map(|x| message.evaluate(x)).collect()`.
   3. `encode` — zero-pad `message.coeffs()` to `domain.size()`, then `fft_on_domain`.

   **Critical test:** `encode_matches_encode_naive`. Both methods must
   agree on every random input you throw at them.

**7. `decode.rs`** — interpolation-based decoder.
   1. `decode_via_interpolation` — `ifft_on_domain` + canonicalize via
      `UnivariatePoly::new` (which strips trailing zeros).
   2. *Stretch goal:* `decode_berlekamp_welch`. Leave as `unimplemented!()`
      for the first pass; come back before objective 3 (STIR).

   **Sanity check:** `tests/integration.rs::encode_decode_round_trip_subgroup`.

**8. `bin/demo.rs`** — narrative demo. Encode a small message, simulate
one corruption, decode, print before and after. Run with
`cargo run -p reed_solomon --bin demo`.

## Build / test commands

```bash
# Build (from the workspace root, /code-practice)
cargo build -p reed_solomon

# Test
cargo test -p reed_solomon

# Run the demo
cargo run -p reed_solomon --bin demo

# Format check
cargo fmt -p reed_solomon --check

# Lint
cargo clippy -p reed_solomon --all-targets -- -D warnings
```

## Definition of done

- All tests pass with `cargo test -p reed_solomon`.
- `cargo clippy -p reed_solomon --all-targets -- -D warnings` is clean.
- `cargo fmt -p reed_solomon --check` passes.
- The `demo` binary prints a clean narrative.
- A reviewing agent (Claude will spawn one) approves the code.

Berlekamp-Welch can be left as `unimplemented!()` for the first
"definition of done" pass — we'll come back to it before objective 3
(STIR), since STIR's analysis assumes you can reason about the
proximity radius.

## Hints — Goldilocks-specific

- `MODULUS = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001`. **It fits in `u64`**,
  but only just — `MODULUS - 1 = 0xFFFFFFFF00000000` is the largest
  representable element minus one. Always use a `u128` intermediate for
  any multiply.
- `MODULUS - 1 = 2^32 · (2^32 - 1)`, so 2-adicity is 32. You can build
  primitive `2^k`-th roots of unity for `k = 0, 1, ..., 32`. For `k > 32`,
  no such root exists in the prime field.
- `7` is a multiplicative generator of `F_p^*` for Goldilocks (standard
  fact — take on faith). Combined with **Fermat's little theorem**
  (`a^(p-1) ≡ 1 mod p` for any non-zero `a`), this gives:

      omega = 7^((p-1)/2^k)   ⟹   omega^(2^k) = 7^(p-1) = 1,

  so `omega` is a `2^k`-th root of unity. It's **primitive** (order
  exactly `2^k`, not smaller) because `7` has order *exactly* `p - 1` —
  see `field.rs` module docs for the precise argument.
- Adding two field elements in `[0, p)` can overflow `u64` (sum can be up
  to `2p - 2 ≈ 2^65`). Use a `u128` intermediate, or use
  `overflowing_add` + conditional subtract.
- The "fast" Goldilocks reduction (using `2^64 ≡ 2^32 - 1 mod p`) is a
  performance optimization — skip it for now; `value % MODULUS` is fine.

## Hints — FFT-specific

- Cooley-Tukey radix-2: split a length-`n` evaluation into even-indexed
  and odd-indexed sub-DFTs of length `n/2`, recurse on each at the
  squared root of unity `ω^2`, then combine with butterflies. The
  module-level docstring in `fft.rs` walks through the math.
- A recursive implementation is much clearer than the iterative
  bit-reversal version. Start there. The iterative version is a
  performance optimization you can do later.
- For a coset `L = c · <ω>`, the FFT factors as: scale coefficients
  `a_i ← c^i · a_i`, then run the subgroup FFT.
- The inverse FFT is the forward FFT at `ω^-1`, divided by `n`.

## Common pitfalls

- Forgetting to canonicalize after subtraction or negation → tests fail
  intermittently.
- Off-by-one in the FFT recursion base case (length 1 vs length 2).
- Confusing "primitive `n`-th root of unity" (order exactly `n`) with
  "any root of `X^n - 1 = 0`" (order divides `n`).
- Treating `UnivariatePoly` as having a fixed length: it doesn't.
  Trailing-zero coefficients are stripped to keep `degree()` honest.
- Forgetting that the encoder treats the message as a polynomial of
  degree `< d`, not `<= d`. A "degree-bound 4" code accepts messages of
  length 4: `a_0, a_1, a_2, a_3`.

## When you're done

Tell Claude "reed-solomon done" in chat. Claude will spawn review
agents and consolidate feedback. Iterate until clean, then move to
objective 3 (STIR).
