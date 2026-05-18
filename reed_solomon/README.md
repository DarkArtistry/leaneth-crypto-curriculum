# Reed-Solomon — Objective 2

The second objective of the leanEthereum coding curriculum. See
`../coding_plan.md §5` for the full curriculum context, and
`../../rs_foundations.md` for the conceptual notes.

This is materially bigger than sumcheck: 8 source modules instead of 5,
because we now have a structured code (the RS code) plus its supporting
machinery (a field that supports FFT, an evaluation domain, the FFT
itself, Lagrange interpolation, and the encoder/decoder layered on top).

## Background: Reed-Solomon Codes

### 1. Definition — by worked example first

Fix a field `F`, an **evaluation domain** `L ⊆ F` of size `n`, and a
**degree bound** `d < n`. The Reed-Solomon code is the set

```text
RS[F, L, d] = { (p(x))_{x ∈ L} : p ∈ F[X], deg p < d }
```

In words: **a codeword IS the evaluation table of a low-degree
polynomial on `L`.** Nothing more. There's no separate "encoded" form
and no extra parity bits bolted on — the redundancy is baked in by
making `L` strictly bigger than the polynomial needs.

**Concrete in `F_17`.** Take `L = {1, 2, 3, 4}` and `d = 2`, so each
codeword is the evaluation of some `p(X) = a + b·X` (one of 17² = 289
polynomials) at those four points. The polynomial `p(X) = 1 + 2X` gives:

```text
p(1) = 3,  p(2) = 5,  p(3) = 7,  p(4) = 9
   → codeword = (3, 5, 7, 9).
```

Four-symbol vector; encodes a 2-symbol message `(a, b) = (1, 2)`. The
extra two symbols are the redundancy that lets the decoder catch errors.

### 2. Why every modern post-quantum SNARK is built on RS

FRI, STIR, WHIR, ethSTARK, Plonky2/3 — every transparent SNARK in
production today reduces "I have a valid computation" to "this committed
function is close to an RS codeword". The prover Merkle-commits to a
purported codeword on `L`; the verifier runs a **low-degree test** (FRI
or its successors) that queries a handful of positions and accepts iff
the function passes a proximity check against `RS[F, L, d]`.

The reason post-quantum SNARK designers reach for RS rather than (say)
algebraic-geometric codes or LDPC codes: RS codes are **MDS** (maximum
distance separable — see §5), they have an `O(n log n)` encoder via
FFT, and we have decades of mature soundness analysis for low-degree
tests over them. The encoder, domain machinery, and FFT in this crate
are the universal substrate underneath the next three objectives (STIR,
WHIR, FRI).

### 3. The encoder

Message = polynomial `p` with `d` coefficients (`a_0, ..., a_{d-1}`),
i.e., `deg p < d`. Encode by evaluating `p` at every `x ∈ L`. Two paths:

- **Naive (`O(n · d)`):** Horner's rule at each of the `n` points.
- **Fast (`O(n log n)`):** zero-pad the coefficients to length `n` and
  run the FFT, when `L` is a smooth multiplicative coset. This is the
  whole reason we bothered to build the FFT machinery.

See `src/encode.rs` for the wrapper and `src/fft.rs` for the FFT itself.

### 4. The rate `ρ`

```text
ρ = d / n     (message length / codeword length)
```

Smaller `ρ` = more redundancy = larger decoding radius = longer
codeword for the same message. Typical STARK family choices are
`ρ ∈ {1/2, 1/4, 1/8, 1/16}`. ethSTARK and Plonky2 use `1/8` by default.

**Concrete.** `d = 1024`, `n = 8192` → `ρ = 1/8`. The codeword is 8× as
long as the message. The 7168 "extra" evaluations are what let the
verifier sample sparsely and still catch a cheating prover.

### 5. Minimum distance, MDS, and error correction

**Definitions first.** The **Hamming distance** between two vectors is
the number of positions where they differ. The **minimum distance**
`Δ` of a code is the smallest Hamming distance over all distinct pairs
of codewords.

**Singleton bound.** Any `[n, d]`-code over any alphabet has minimum
distance `Δ ≤ n - d + 1`. (Proof sketch: there are at most
`|F|^d` codewords; if `Δ > n - d + 1`, distinct codewords would have
to differ in too many positions for that many to fit.)

**RS codes meet the bound with equality** — they are **MDS** (Maximum
Distance Separable). The proof is short and worth knowing:

> **Polynomial root-bound theorem.** A nonzero polynomial of degree
> `< d` over a field has at most `d - 1` roots.
>
> Proof: each root `r` contributes a factor `(X - r)`; the integral-
> domain structure of the field (no zero divisors) prevents extra
> factors from sneaking in. So `# roots ≤ deg`.

Two distinct degree-`< d` polynomials `p, q` differ by `p - q`, a
nonzero polynomial of degree `< d`, which has `≤ d - 1` roots. So their
evaluation tables on `L` agree at **at most `d - 1` positions**, hence
differ at **at least `n - (d - 1) = n - d + 1` positions**. That's the
Singleton bound with equality. ✓

**Consequences for decoding:**

- **Unique-decoding radius** = `⌊(n - d) / 2⌋` errors. Any received
  word within this many errors of a unique codeword can be recovered
  (Berlekamp-Welch — `src/decode.rs`).
- **List-decoding (Johnson bound).** Up to roughly `n - √(d · n)`
  errors, with the output a *list* of candidate codewords of bounded
  size (Guruswami-Sudan). FRI / STIR / WHIR soundness analyses lean on
  list-decoding bounds for their proximity-parameter `δ`.

### 6. The decoder

Two algorithms live in `src/decode.rs`:

- **`decode_via_interpolation` (required).** No errors assumed. Given
  an intact codeword, inverse-FFT to recover the coefficients in
  `O(n log n)`. Wraps `fft::ifft_on_domain`. If you only had `d`
  points (not all of `L`), Lagrange interpolation on those `d` points
  recovers `p` in `O(d²)` — see `src/interpolate.rs`.
- **`decode_berlekamp_welch` (stretch).** Up to
  `⌊(n - d) / 2⌋` errors. Sets up a linear system in the coefficients
  of an error-locator `E(X)` (roots at the corrupted positions) and a
  product polynomial `Q(X) = p(X) · E(X)`, solves it over `F_p`, then
  polynomial-divides `Q / E` to recover `p`. `O(n³)` naive, `O(n² log n)`
  with care. Leave as `unimplemented!()` for the first pass; revisit
  before objective 3 (STIR's soundness analysis assumes you can reason
  about the proximity radius).

### 7. Why Goldilocks (one sentence)

RS codes need fast polynomial evaluation on `n` structured points; the
FFT achieves `O(n log n)` *iff* the field contains a smooth subgroup of
size `n` — which Goldilocks does (2-adicity 32, so FFTs up to size
`2^32 ≈ 4 billion`). Full derivation in `src/field.rs`.

### 8. Where each piece lives

| Concept                        | Type / function                            | Module               |
|--------------------------------|--------------------------------------------|----------------------|
| Field `F_p` (Goldilocks)       | `Fp`                                       | `src/field.rs`       |
| Polynomial in coefficient form | `UnivariatePoly`                           | `src/polynomial.rs`  |
| Evaluation domain `L`          | `EvaluationDomain`                         | `src/domain.rs`      |
| FFT / inverse FFT              | `fft_subgroup`, `fft_on_domain` (and `i*`) | `src/fft.rs`         |
| Lagrange interpolation         | `lagrange_interpolate`                     | `src/interpolate.rs` |
| Encoder                        | `ReedSolomonCode::encode{,_naive}`         | `src/encode.rs`      |
| Decoder                        | `decode_via_interpolation`, `decode_berlekamp_welch` | `src/decode.rs` |

### 9. References

- **Original:** Reed & Solomon, "Polynomial codes over certain finite
  fields" (1960). The four-page paper that started the field.
- **Application angle:** Ben-Sasson et al., "Fast Reed-Solomon
  Interactive Oracle Proofs of Proximity" (FRI, 2018);
  Arnon-Chiesa-Fenzi-Yogev, "STIR" (2024/390);
  Arnon-Chiesa-Spooner et al., "WHIR" (2024/1586). Each is a proximity
  test against an RS code.
- **Kenneth's own foundations notes:** `../../rs_foundations.md` —
  evaluation domains, smooth multiplicative cosets, the architect's
  decision flow.
- **Source-level docs in this crate:** every module starts with a
  `//!`-block that walks through a worked numeric example. Read those
  before implementing.

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

**4. `fft.rs`** — Cooley-Tukey radix-2 FFT, a.k.a. the **Number Theoretic
Transform (NTT)**. This is the hardest and most important file in the
crate, so this section is much longer than the others. The module docs
at the top of `src/fft.rs` are the canonical reference for the proofs;
the walkthrough here is a markdown-friendly pre-implementation tour
with worked numerics.

#### 4a. What the FFT computes

Given a polynomial in **coefficient form**

```text
p(X) = a_0 + a_1·X + a_2·X² + ... + a_{n-1}·X^{n-1}
```

return its **values at the n powers of a primitive n-th root of unity `ω`**:

```text
[ p(ω⁰), p(ω¹), p(ω²), ..., p(ω^{n-1}) ]
```

For Reed-Solomon this is literally the encoding step — the n outputs
*are* the codeword. So "fast polynomial evaluation on n points" is the
same problem as "fast RS encoding".

- **Naive cost: O(n²).** Horner's rule at each of n points: O(n) per
  evaluation × n evaluations.
- **FFT cost: O(n log n).** For n = 10⁶, that's roughly a 50,000× speedup
  over naive. The speedup is *the* reason production STARKs are feasible.

#### 4b. The trick: evaluation points pair up as ± opposites

The points `1, ω, ω², ..., ω^{n-1}` aren't arbitrary — they have a
crucial symmetry. The key identity is

```text
ω^{n/2} = -1
```

so the second half of the domain is the first half negated:

```text
ω^{i + n/2} = ω^i · ω^{n/2} = ω^i · (-1) = -ω^i
```

That ± pairing lets us compute `p(x_i)` and `p(-x_i)` from **one shared
sub-computation** — see §4c for how. Recursing on this halving turns
O(n²) into O(n log n).

**Why `ω^{n/2} = -1`** (short version): `ω^{n/2}` has order exactly 2 in
`F_p^*` (since `(ω^{n/2})² = ω^n = 1` and `ω^{n/2} ≠ 1` because `ω` is
*primitive*), and the unique element of order 2 in any field is `-1`
(solve `x² = 1` by factoring, then use the zero-product property). Full
proof with the integral-domain step is in `src/fft.rs` module docs.

#### 4c. The Cooley-Tukey recursion

Split `p`'s coefficients into even-indexed and odd-indexed bundles, using
a **fresh variable `Y`** for the sub-polynomials (distinct from `X` in
the original `p(X)`):

```text
p_even(Y) = a_0 + a_2·Y + a_4·Y² + ... + a_{n-2}·Y^{n/2 - 1}
p_odd(Y)  = a_1 + a_3·Y + a_5·Y² + ... + a_{n-1}·Y^{n/2 - 1}
```

Then reassemble via `Y = X²`:

```text
p(X) = p_even(X²) + X · p_odd(X²)
```

The asymmetric `X ·` is because `Y = X²` only produces *even* powers of
`X`; multiplying by `X` bumps the odd-indexed coefficients to their
correct odd powers. (Module docs walk through the derivation step by
step.)

Now evaluate at a ± pair `(x, -x)` where `x = ω^i`:

```text
p( x) = p_even(x²) + x · p_odd(x²)
p(-x) = p_even((-x)²) + (-x) · p_odd((-x)²)
      = p_even(x²) − x · p_odd(x²)        ← (-x)² = x², so the sub-evals match
```

That gives the **butterfly**: given the shared values `e = p_even(x²)`
and `o = p_odd(x²)`, both outputs come from one extra multiplication
plus an add/subtract:

```text
e = p_even(x²)
o = p_odd(x²)
p( x) = e + x · o      ← "+" output
p(-x) = e − x · o      ← "−" output
```

The squared values `x_0², x_1², ..., x_{n/2 - 1}²` are the powers of `ω²`,
which is a primitive `(n/2)`-th root of unity. So `p_even` and `p_odd`
need their length-`(n/2)` FFTs at `ω²`. Recurse. Base case: `n = 1`
returns the single coefficient unchanged.

**Twiddle factors.** In the butterfly above, `x = ω^k` is the **twiddle
factor** for index `k`. As `k` ranges over `0, 1, ..., n/2 - 1`, the
twiddle walks through `ω⁰, ω¹, ..., ω^{n/2 - 1}`. The implementation
maintains this as a running product, advancing by one factor of `ω` each
iteration.

#### 4d. Worked example: n = 4

Take `p(X) = 1 + 2X + 3X² + 4X³`, so `coeffs = [1, 2, 3, 4]`. Pick `ω`
to be a primitive 4th root of unity, so `ω⁴ = 1` and `ω² = -1`.

**Split:**

```text
p_even(Y) = 1 + 3·Y    (from [1, 3])
p_odd(Y)  = 2 + 4·Y    (from [2, 4])
```

**Recurse**: each sub-FFT is length 2 at root `ω² = -1`.

```text
p_even on <-1> = [p_even(1),  p_even(-1)] = [1 + 3,  1 − 3] = [ 4, -2]
p_odd  on <-1> = [p_odd(1),   p_odd(-1) ] = [2 + 4,  2 − 4] = [ 6, -2]
```

**Combine** with the butterfly. For `k = 0` the twiddle is `ω⁰ = 1`;
for `k = 1` it's `ω¹ = ω`:

```text
p(ω⁰) =  4 + 1·6      = 10                  p(ω²) =  4 − 1·6      = -2
p(ω¹) = -2 + ω·(-2)   = -2 − 2ω             p(ω³) = -2 − ω·(-2)   = -2 + 2ω
```

**Cross-check** by Horner at the two values of `ω` we already know:

```text
p( 1) = 1 + 2 + 3 + 4 = 10    ✓
p(-1) = 1 − 2 + 3 − 4 = -2    ✓
```

The other two depend on the concrete numerical value of `ω`, but the
mechanism is identical.

#### 4e. Inverse FFT — recovering coefficients

Forward FFT: coefficients → evaluations. Inverse FFT: the reverse —
given the evaluation table, recover the coefficient vector.

**Formula:**

```text
ifft(evals, ω) = (1/n) · fft(evals, ω⁻¹)
```

In words: **run the forward FFT at the inverse root `ω⁻¹`, then scale
every entry by `1/n`**. Both pieces are necessary; missing either gives
garbage.

**Why this works** (one-paragraph version): the **Discrete Fourier
Transform (DFT)** — the linear map "take coefficients, return evaluations
on `ω⁰, ω¹, ..., ω^{n-1}`" — is matrix multiplication. The FFT and NTT
are just *fast algorithms for computing the DFT*; the DFT is the underlying
operation. Its matrix is `V[i, j] = ω^{ij}` (size `n × n`). Forward FFT
is `V · coeffs`; inverse is `V⁻¹ · evals`. The identity `V · V* = n · I`
— where `V*[i, j] = ω^{-ij}` is the DFT matrix at `ω⁻¹` — gives
`V⁻¹ = (1/n) · V*`. So undoing the FFT is "forward at `ω⁻¹`, then divide
by `n`". The proof of `V · V* = n · I` reduces to a geometric-series sum
of n-th roots of unity, which equals `n` on the diagonal and `0`
off-diagonal. Full proof in `src/fft.rs` module docs.

**Round-trip example (n = 2).** Take `coeffs = [3, 5]`, `n = 2`,
`ω = -1` (the primitive 2nd root of unity).

```text
Forward:  p(X) = 3 + 5X
          fft = [p(1), p(-1)] = [3 + 5, 3 − 5] = [8, -2]

Inverse:  ω⁻¹ = (-1)⁻¹ = -1, so re-use the same root.
          Treat evals = [8, -2] as coefficients of q(X) = 8 − 2X.
          fft([8, -2], -1) = [q(1), q(-1)] = [8 − 2, 8 + 2] = [6, 10]
          Scale by 1/n = 1/2:  [6/2, 10/2] = [3, 5]   ✓ back to original.
```

The "two passes of forward FFT" structure is peculiar to `n = 2` where
`ω⁻¹ = ω`; for larger `n` the inverse goes through a different recursion
path. But the *formula* is the same shape: forward at the inverse root,
then divide by `n`.

#### 4f. Coset FFT — when the offset c ≠ 1

For a coset `L = c · ⟨ω⟩` we want the codeword on `c, c·ω, c·ω², ...`
instead of `1, ω, ω², ...`. The trick: evaluating `p` on `c · ⟨ω⟩` is
the same as evaluating `p_c(X) := p(c·X)` on `⟨ω⟩`, because
`p_c(ω^i) = p(c · ω^i)` — exactly the values we want.

Expanding `p_c`:

```text
p_c(X) = a_0 + (a_1·c)·X + (a_2·c²)·X² + ... + (a_{n-1}·c^{n-1})·X^{n-1}
```

So `p_c` has the same shape as `p`, with **each coefficient `a_i`
multiplied by `c^i`** — a *running product* of `c`. Then a plain
subgroup FFT at `ω` finishes the job.

**Coset inverse FFT** mirrors this: subgroup-iFFT first, then de-scale
the *output* by powers of `c⁻¹`. Two common bugs to avoid:

- **Using the same `c` instead of `c^i`** in the pre-scale (forward) or
  de-scale (inverse). The fix is a running product: `pow_c` starts at
  `c⁰ = 1`, gets used for `a_0`, advances to `c¹` for `a_1`, etc.
- **Applying the de-scale to the wrong vector** in `ifft_on_domain`.
  The de-scale acts on the **output** of `ifft_subgroup` (the recovered
  scaled coefficients), not on the input `evals`.

Also: compute `c⁻¹` (and `n⁻¹`) **once outside the loop**. Calling
`.inverse()` per iteration is O(n · log p); hoisting it out is O(log p + n).

#### 4g. Implementation order

Implement in this order, with `cargo test -p reed_solomon --lib fft::`
after each:

1. **`fft_subgroup`** — base case `n == 1` returns `vec![coeffs[0]]`;
   recursive case splits even/odd, recurses at `ω²`, combines with
   butterfly.
2. **`ifft_subgroup`** — `forward FFT at omega.inverse()` then scale by
   `n⁻¹` (computed once via `Fp::new(n as u64).inverse()`).
3. **`fft_on_domain`** — pre-scale by running product of `c`, then
   `fft_subgroup`. Shortcut if `offset == Fp::one()` (just call
   `fft_subgroup` directly).
4. **`ifft_on_domain`** — `ifft_subgroup` first, then de-scale by
   running product of `c⁻¹`. Same `offset == 1` shortcut.

**Critical test:** `fft_matches_naive_evaluation_subgroup`. The FFT
output at index `i` must equal `poly.evaluate(omega.pow(i))`. **If this
fails for `n = 4`, your butterfly is wrong** — walk through §4d above
and find the first divergence.

**Round-trip test:** `fft_round_trip_subgroup` and
`fft_round_trip_coset` — apply forward then inverse, must recover the
original coefficients exactly. If forward and inverse don't compose to
identity, one of the two is buggy.

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
  and odd-indexed sub-FFTs (each a smaller Discrete Fourier Transform)
  of length `n/2`, recurse on each at the squared root of unity `ω²`,
  then combine with butterflies. The module-level docstring in
  `fft.rs` walks through the math.
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
