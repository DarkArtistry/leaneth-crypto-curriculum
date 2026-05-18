# STIR — Objective 3

The third objective of the leanEthereum coding curriculum. See
`../coding_plan.md §6` for the full curriculum context, and
`../../stir_study.md` for the conceptual notes pre-loaded from the
2026-05-05 paper read (eprint 2024/390).

## Overview

STIR is a proof system. We pick a finite field `F` — in this crate,
**Goldilocks** `F_p` with `p = 2^64 − 2^32 + 1`, inherited from
`reed_solomon::Fp` — and a finite set of points `L_0 ⊆ F` called the
**initial evaluation domain**, fixed when the protocol is set up. The
prover holds a function that maps each point in `L_0` to a field
element; call this function `f: L_0 → F`. The prover claims that `f`
"looks like" the evaluations of a low-degree polynomial. Concretely:
there exists some polynomial `p` whose degree is below a fixed bound
called the **initial degree bound**, written `d_0`, such that
`f(x) = p(x)` for every `x ∈ L_0`.

The set of all functions on `L_0` that arise this way — every
evaluation table `(p(x))_{x ∈ L_0}` for some polynomial `p` of degree
`< d_0` — is the **Reed-Solomon code** `RS[F, L_0, d_0]`. So the
prover's claim is "my `f` is a codeword of `RS[F, L_0, d_0]`", and
STIR is the proof system that lets the prover convince a verifier of
this. But a literal proof would force the verifier to read all of
`f` — that's `|L_0|` field elements, defeating the point. So STIR
proves a slightly weaker, practically sufficient claim: that `f` is
*close* to a codeword. To make "close" precise, introduce the
**proximity parameter** `δ ∈ [0, 1)`. We say `f` is **δ-close** to a
codeword `c ∈ RS[F, L_0, d_0]` iff `f(x) = c(x)` for at least
`(1 − δ) · |L_0|` values of `x ∈ L_0` — equivalently, they differ on
at most a `δ`-fraction of points. This is **Hamming distance**.
STIR is an **interactive oracle proof of proximity (IOPP)**: an
interactive protocol where the verifier only reads `f` at a few
sampled positions — **sublinear** in `|L_0|`.

How does it work? The protocol runs `M` rounds. Each round divides
the polynomial-degree-bound by a **folding factor** `k` — a power of
2, at least 4 — so after `i` rounds the prover is working with a
degree-bound of `d_i = d_0 / k^i` on a smaller evaluation domain
called `L_i` (the round-`i` evaluation domain, with `L_0` being the
initial one). The prover commits to one function per round:
`f_i: L_i → F`, the round-`i` analogue of `f`, with `f_0 = f`. Each
round the verifier makes `t_i` random **shift queries** (opening
Merkle paths into `f_i`'s evaluation table) plus `s`
**out-of-domain (OOD) samples** (asking for `f_i`'s value at a point
*outside* `L_i`). After all `M` rounds the prover sends a final
low-degree polynomial in the clear. The whole protocol targets `λ`
bits of security (typically 80 or 128), and a key headline is that
total query count scales as `O(log² d_0)` — substantially fewer than
FRI's `O(log d_0)` queries per round across `O(log d_0)` rounds.

The rest of this README walks through *why* this works: the two
structural innovations vs FRI (domain shifting and OOD sampling),
the per-module roles, and the implementation order.

## Sumcheck vs Reed-Solomon vs STIR — what's the same, what's new

STIR builds directly on top of the previous two objectives — Reed-Solomon
supplies the code and FFT machinery, and the polynomial-root-bound proof
technique you learned in sumcheck reappears at the heart of STIR's
soundness analysis. The table below makes the differences explicit.

| Dimension | Sumcheck (obj 1) | Reed-Solomon (obj 2) | STIR (obj 3, this crate) |
|---|---|---|---|
| **Category** | Interactive proof (IP) | Error-correcting code | Interactive oracle proof of proximity (IOPP) |
| **Interactive?** | **Yes** — `n` rounds of prover-verifier message exchange | **No** — pure algebraic encode/decode, no prover or verifier or rounds | **Yes** — `M ≈ log_k(d_0)` rounds |
| **Does it use sumcheck / Reed-Solomon internally?** | n/a — sumcheck is its own thing | **No** — Reed-Solomon does not use sumcheck. RS is non-interactive; sumcheck is unrelated. | **Uses Reed-Solomon** (this crate's `reed_solomon` is a path dependency). **Does NOT use sumcheck.** |
| **What it proves / does** | The claimed sum `H = Σ_{x ∈ S^n} g(x)` is correct, for a *public* multivariate polynomial `g` | Encodes a polynomial `p` of degree `< d` as its evaluation table on a finite domain `L`; decoder recovers `p` or corrects up to `⌊(n−d)/2⌋` errors | The *committed* function `f: L_0 → F` is δ-close to a codeword in `RS[F, L_0, d_0]` — i.e., to the evaluations of some polynomial of degree `< d_0` |
| **Field** | Mersenne `F_p`, `p = 2^61 − 1` (fits in `u64`, simple modular reduction) | Goldilocks `F_p`, `p = 2^64 − 2^32 + 1` (2-adicity 32 supports FFT up to `2^32`) | Goldilocks (inherited from `reed_solomon::Fp`) |
| **Polynomial type** | **Multivariate**, `n` variables; evaluation form on `S^n` in mixed-radix LSB-first order | **Univariate**, degree `< d`; coefficient form | **Univariate**, degree `< d_0`; the prover holds an evaluation table on `L_0`, polynomial is implicit |
| **Object cryptographically committed to** | None — `g` is public | None — encoding is purely algebraic | Per-round Merkle root over `f_i`'s evaluation table |
| **Crypto primitives** | None | None | Merkle tree (SHA3-256) + Fiat-Shamir transcript (BLAKE3) |
| **Number of rounds** | `n` (one per variable of `g`) | None — encode/decode are non-interactive | `M ≈ log_k(d_0)` (logarithmic in degree) |
| **Verifier's reading pattern** | Reads `k = |S|` field elements per round (prover-sent) + 1 final evaluation of `g` at `(r_1, ..., r_n)` | No verifier role; decoder reads all `n` codeword positions | **Sublinear**: a few hundred Merkle openings total, independent of `|L_0|` |
| **Soundness bound** | `n · (k − 1) / |F|` — polynomial root bound × union bound | Minimum distance `n − d + 1`; unique-decoding radius `⌊(n − d) / 2⌋` | `2^{−λ}` for target security `λ`, achieved via per-round repetition × Johnson-bound list collapse |
| **Pedagogical foundations** | Polynomial root bound (Schwartz-Zippel univariate); multivariate Lagrange interpolation | Polynomial root bound (gives min distance); Lagrange interpolation; Cooley-Tukey FFT lemmas | Both of the above + **Johnson list-decoding bound** + **STIR Lemma 4.4** (quotient distance preservation) |

### What's shared, and what isn't

The three protocols are **independent of each other**, not nested:

- **Reed-Solomon has no interactive part.** Objective 2 contains no prover,
  no verifier, and no protocol rounds. Encoding is one FFT; decoding is one
  inverse FFT (or, for Berlekamp-Welch, a linear system solve). If you went
  looking for "the interactive part of Reed-Solomon" you'd find nothing,
  because there isn't one.
- **Sumcheck is not part of Reed-Solomon.** Sumcheck is a generic
  interactive technique for proving claims of the form `H = Σ g(x)` over a
  finite domain. The polynomial `g` can be anything multivariate — it
  never needed Reed-Solomon, and Reed-Solomon never needed it.
- **STIR uses Reed-Solomon as a foundation, but does not use sumcheck.**
  STIR's rounds are *STIR's* invention — folding + quotienting + degree
  correction + Merkle commitments + Fiat-Shamir. None of this is "RS's
  interactive part" (RS has none); none of this is sumcheck.

The one thing all three share is the **polynomial-root-bound lemma**: a
nonzero univariate polynomial of degree `< d` has at most `d − 1` roots
in `F`. Sumcheck uses it for per-round soundness. Reed-Solomon uses it to
derive minimum distance `n − d + 1`. STIR uses it twice — once via RS's
distance bound, once via Schwartz-Zippel-style arguments in its own
soundness analysis. Same lemma, three different applications.

> **Composition vs subsumption.** Higher-level zkSNARKs (ethSTARK,
> Aurora, Plonky-style systems) often *compose* sumcheck with RS-based
> proximity tests — sumcheck for AIR constraint enforcement, FRI or STIR
> for the low-degree proximity check. That's two protocols running in
> series, not one being part of the other. In this curriculum the
> composition shows up in **objective 5 (STARK)** and **objective 6
> (STARK + WHIR)**, not here.

### Headline structural changes at STIR

The headline structural changes at STIR: **(a)** the verifier no longer
reads everything — it makes `O(log² d_0)` Merkle openings instead of
touching the whole evaluation table; **(b)** the protocol introduces
**cryptographic commitments** (Merkle + Fiat-Shamir hash) where the
prior objectives needed none; **(c)** STIR proves *proximity* to a
codeword, not equality with a known function — this weaker claim is
what enables the sublinear verifier. Everything else (the field, the
polynomial-root bound underpinning soundness, the FFT used to evaluate
polynomials) is recycled from objectives 1 and 2.

## Background: STIR

### 1. What STIR proves — by worked example first

Fix a Reed-Solomon code `RS[F, L_0, d_0]` (see `../reed_solomon`). A
prover holds a function `f: L_0 → F` and wants to convince a verifier
that `f` is **δ-close in Hamming distance** to some codeword
`c ∈ RS[F, L_0, d_0]` — i.e., to the evaluation table of some
polynomial `p` of degree `< d_0`.

**Concrete in Goldilocks.** Take `|L_0| = 2^20 ≈ 10^6`, rate `ρ_0 = 1/4`
so `d_0 = 2^18`, folding factor `k = 4`. The prover commits to `f` by
Merkle-hashing its evaluation table `(f(x))_{x ∈ L_0}`. STIR then runs ~`log_k(d_0) = 9`
rounds. In each round the prover folds the function by `k`, shifts to a
smaller-but-rate-shrinking domain, and answers the verifier's queries.
The verifier accepts after ~`λ / log(1/ρ)` Merkle openings — sublinear
in `|L_0|`. The whole protocol is `O(log² d_0)` query complexity, vs
FRI's `O(log d_0 · λ / log(1/ρ))`.

### 2. Why STIR beats FRI — the two innovations

> **STIR.** Arnon, Chiesa, Fenzi, Yogev (eprint 2024/390). "Reed-Solomon
> Proximity Testing with Fewer Queries." The headline: same security
> level as FRI, **fewer prover-to-verifier rounds and dramatically
> fewer queries**, because of two structural changes.

**Innovation 1: domain shift between rounds.** FRI keeps the rate `ρ`
constant across rounds (every round halves both the domain and the
degree). STIR keeps the **degree** halving by `k` per round but
**shrinks the domain by only `k/2`** — so the rate per round drops
geometrically. Smaller rate → larger list-decoding radius → smaller
soundness error per query → **fewer queries needed at the same `λ`**.

**Innovation 2: out-of-domain (OOD) sampling.** A round of STIR
includes the verifier sampling a few field elements `z_1, ..., z_s` from
`F \ L_i` and asking the prover for the folded function's value there.
This is **not** a Merkle query against the committed evaluation table
— it's a separate algebraic check that collapses the list of candidate
codewords down to a single one (the **list-decoding radius reduction
lemma** in §4 of the paper). Without OOD, soundness would degrade
catastrophically when `δ` is close to `1 - √ρ`.

The combination of (a) and (b) is what buys STIR its `O(log² d)`
query complexity — and what makes WHIR (objective 4) possible, since
WHIR extends STIR's machinery to multilinear codes.

### 3. Reed-Solomon below, STARK / WHIR above

```text
   ┌─────────────────────────────────────────────────────────┐
   │           STARK / SNARK constraint system                │
   │     (arithmetization, constraints, AIR, lookups, ...)    │
   └─────────────────────────────────────────────────────────┘
                          ↓ "this is a low-degree extension"
   ┌─────────────────────────────────────────────────────────┐
   │       STIR / WHIR / FRI — IOP of proximity              │
   │       ← THIS CRATE implements STIR ←                     │
   └─────────────────────────────────────────────────────────┘
                          ↓ "polynomial commitment / RS encoding"
   ┌─────────────────────────────────────────────────────────┐
   │           Reed-Solomon code (objective 2)                │
   └─────────────────────────────────────────────────────────┘
                          ↓
   ┌─────────────────────────────────────────────────────────┐
   │     Goldilocks `F_p`, FFT, Merkle tree, Fiat-Shamir      │
   └─────────────────────────────────────────────────────────┘
```

STIR is the **central piece** of every modern post-quantum SNARK. The
constraint system on top reduces to "prove the committed function `f`
is close to a Reed-Solomon codeword (the evaluation table of some
low-degree polynomial)", and STIR is the proof. WHIR generalizes to
multilinear; the WHIR objective in this curriculum will reuse most of
this crate's machinery with one module swapped.

### 4. STIR vs FRI in one table

| Quantity                     | FRI                        | STIR                                |
|------------------------------|----------------------------|-------------------------------------|
| Domain shrink per round      | `|L| / k` (factor `k`)     | `|L| / (k/2)` (factor `k/2`)         |
| Rate per round               | constant `ρ`               | drops geometrically                 |
| Degree shrink per round      | `d / k`                    | `d / k` (same)                       |
| Per-round repetitions        | `t = λ / log(1/ρ)`         | `t_i` — harmonic, decreasing         |
| Total queries                | `O(log d · λ / log(1/ρ))`  | `O(log² d)` at fixed `λ`             |
| Argument size                | linear in queries          | linear in queries (smaller constant) |
| Out-of-domain sampling       | absent                     | `s` samples per round (collapses list) |

The harmonic decline of `t_i` is the source of the `log²` (rather than
`log`) — see §4.3 of the paper for the exact formula.

### 5. STIR in pseudocode — one round

```text
Round i, with current function f_i: L_i → F, current degree bound d_i:

  1. Verifier samples folding randomness α_i ∈ F (Fiat-Shamir).
  2. Prover folds:                f_{i+1}(x) := Fold(f_i, α_i, k)(x)
                                  for x ∈ L_{i+1}.
                                  This collapses k evaluations of f_i
                                  into 1 evaluation of f_{i+1} via
                                  Lagrange interpolation.
  3. Prover Merkle-commits to evals of f_{i+1} on L_{i+1}.
  4. Verifier samples OOD points  z_1, ..., z_s ∈ F \ L_{i+1}.
  5. Prover replies              β_1, ..., β_s where β_j = f_{i+1}(z_j).
  6. Verifier samples queries     idx_1, ..., idx_{t_i} ∈ [|L_i|].
  7. Prover opens                 f_i at those idx (k of them per query,
                                  to recover the fold).
  8. Verifier checks            (a) Merkle paths,
                                (b) Fold(f_i restricted to {idx_*})(α_i)
                                    matches f_{i+1} at the right point,
                                (c) the OOD answers are consistent
                                    with a quotient polynomial of degree
                                    `< d_{i+1}` — this is the
                                    `quotient.rs` Lemma 4.4 check.

  ... domain shift: L_{i+1} := next smaller smooth coset,
                    d_{i+1} := d_i / k,
                    ρ_{i+1} := d_{i+1} / |L_{i+1}|   ← DROPS, not constant.
```

The final round (`i = M`) doesn't fold further — the prover just sends
the remaining `f_M` in plain (it's tiny) and the verifier checks it
directly. Stopping degree is configured by `StirParams::stopping_degree`.

### 6. Worked end-to-end example

Tiny instance, chosen so the numbers fit in a hand calculation:

- **Initial domain:** `log₂|L_0| = 6`, so `|L_0| = 64`.
- **Initial degree bound:** `d_0 = 16` (rate `ρ_0 = 16/64 = 1/4`).
- **Folding factor:** `k = 4`.
- **Number of rounds:** `M = 2` (`d_0 → d_1 = 4 → d_2 = 1`).
- **OOD samples per round:** `s = 2`.
- **Repetitions:** `t_0 = 8, t_1 = 4, t_final = 4`.

**Round 0.** `|L_0| = 64`, `d_0 = 16`. Verifier picks `α_0`. Prover
folds `f_0` (64 evals) into `f_1` (32 evals on `L_1`, since
`|L_1| = |L_0| / (k/2) = 64 / 2 = 32`). Verifier OOD-queries at
`z_1, z_2 ∈ F \ L_1`; prover replies with `β_1, β_2`. Verifier picks 8
Merkle indices on `L_0` and checks the fold + quotient.

**Round 1.** `|L_1| = 32`, `d_1 = 4`, `ρ_1 = 4/32 = 1/8` (dropped from
`1/4`). Same dance, smaller. `|L_2| = 16`, `d_2 = 1`.

**Final.** `d_2 = 1` means `f_2` is a constant. Prover sends the
constant in plain. Verifier checks Merkle openings on `f_2` against 4
random indices on `L_2`, plus the final fold consistency.

Total queries: `t_0 · k + t_1 · k + t_final = 32 + 16 + 4 = 52` Merkle
openings. Plus `s · M = 4` OOD points. For a security target of
`λ = 128`, the harmonic schedule would in reality be tuned higher; this
is just to show the shapes.

## Your job

Fill in every `todo!()` body in `src/`. The structure (13 modules):

```
src/
├── lib.rs                # crate root — module declarations + crate docs
├── params.rs             # StirParams, RbrSoundnessReport
├── transcript.rs         # Fiat-Shamir BLAKE3 transcript
├── merkle.rs             # binary Merkle tree (commits, openings, verify)
├── domain.rs             # round-by-round domain shifts L_0 ⊃ L_1 ⊃ ... ⊃ L_M
├── fold.rs               # Fold(f, α, k) — collapses k evaluations into 1
├── quotient.rs           # quotient polynomial for OOD consistency (Lemma 4.4)
├── degree_correction.rs  # degree-correction polynomial after each fold
├── ood.rs                # out-of-domain sampling: pick z, evaluate, send β
├── commitment.rs         # round-specific commit (Merkle on folded evals)
├── proof.rs              # StirProof — the full transcript object
├── prover.rs             # run_prover: builds the proof
├── verifier.rs           # run_verifier: checks the proof
├── protocol.rs           # run_stir wrapper, end-to-end honest run
└── bin/
    └── demo.rs           # narrative demo, prints round-by-round
tests/
└── integration.rs        # honest-prover-accepts, cheater-rejects
```

## Recommended order

Three phases. Get each phase green before starting the next.

### Phase 1 — primitives

These have no STIR-specific algebra; they're the substrate. Should be
done in a single session.

1. **`params.rs`** — `StirParams`, `RbrSoundnessReport`. Validate that
   `k >= 4`, `k` is a power of 2, `repetition_schedule.len() == num_rounds + 1`.
   Implement `round_degree_bound` (returns `d_0 / k^round`) and
   `round_log_domain_size` (returns `log|L_0| - round · log_2(k/2)`).
2. **`transcript.rs`** — `Transcript` wrapping `blake3::Hasher`. `absorb`,
   `absorb_field`, `squeeze_field` (rejection-sample over Goldilocks),
   `squeeze_indices` (rejection-sample to avoid modulo bias).
3. **`merkle.rs`** — binary Merkle on `Vec<Fp>`. Commit returns root + tree.
   Open returns sibling path. Verify recomputes root from leaf + path +
   index. Use BLAKE3.
4. **`domain.rs`** — `RoundDomain { coset: EvaluationDomain, round: u32 }`.
   `next_domain(prev, k)` returns the round-`i+1` coset: half the size
   of the previous, but with a different offset so the cosets are
   disjoint (see paper §3.2 for the construction).

### Phase 2 — operations

These are the STIR-specific algebraic primitives. Each is a pure
function on polynomials/evaluations.

5. **`fold.rs`** — `fold(evals, α, k) -> Vec<Fp>`. Given `f` on `L_i` and
   folding randomness `α`, produce `f_folded` on `L_{i+1}`. The fold
   takes each block of `k` consecutive evaluations of `f` (a
   "coset slice" of `L_i` over `L_{i+1}`), Lagrange-interpolates them,
   evaluates at `α`. The result is one element of `f_folded`.
6. **`quotient.rs`** — `quotient_poly(g, points: &[(Fp, Fp)]) -> UnivariatePoly`.
   Given a polynomial `g` and a list of `(z_j, β_j)` such that
   `g(z_j) = β_j` for all `j`, return the quotient `q` of degree
   `< deg g - s` such that `g(X) - I(X) = q(X) · ∏(X - z_j)`, where `I`
   is the interpolant through `(z_j, β_j)`. This is the **list-decoding
   collapse**: the verifier checks `q` is itself low-degree.
7. **`degree_correction.rs`** — multiplies the folded function by a
   carefully chosen polynomial to push it back to degree exactly `d_{i+1}`
   (the fold itself produces something of degree `< d_{i+1}` but
   "non-canonical" degree, see §3.3 of the paper).
8. **`ood.rs`** — `sample_ood_points(transcript, s, exclude: &EvaluationDomain) -> Vec<Fp>`.
   Squeeze `s` field elements from the transcript, reject any that lie
   in `L_{i+1}`. This is the OOD that collapses the list.

### Phase 3 — protocol

Tie the operations together into the IOP.

9. **`commitment.rs`** — `Commitment { root: [u8; 32], tree: MerkleTree }`.
   `commit(evals)` runs the Merkle commit; `open(idx)` returns the leaf +
   sibling path; verifier-side `verify(root, leaf, path, idx)` checks.
10. **`proof.rs`** — `StirProof` struct holding every round's
    `(commitment_root, fold_alpha, ood_pairs, query_openings)` plus the
    final low-degree polynomial. Implement `serialize` / `deserialize`
    helpers (raw byte concatenation is fine for educational purposes).
11. **`prover.rs`** — `run_prover(params, f0_evals, &mut rng) -> StirProof`.
    Implements the M-round loop: for each round, fold, commit, OOD, open
    Merkle paths at squeezed indices. Final round sends `f_M` in plain.
12. **`verifier.rs`** — `run_verifier(params, commitment_root, proof) -> bool`.
    Walks the same Fiat-Shamir transcript as the prover, checks each
    round's Merkle paths, fold consistency, OOD quotient consistency.
13. **`protocol.rs`** — `run_stir(params, poly, rng) -> (StirProof, bool)`.
    Encodes `poly` on `L_0`, runs the prover, runs the verifier, returns
    both for inspection.

## Build / test commands

```bash
# Build (from the workspace root, /code-practice)
cargo build -p stir

# Test
cargo test -p stir

# Run the demo
cargo run -p stir --bin demo

# Format check
cargo fmt -p stir --check

# Lint
cargo clippy -p stir --all-targets -- -D warnings
```

## Definition of done

- All 13 modules implemented (no `todo!()` remaining).
- All tests pass with `cargo test -p stir`.
- Integration test `honest_prover_accepts` succeeds end-to-end on a
  non-trivial instance (`|L_0| >= 2^10, k = 4, num_rounds >= 2`).
- Integration test `cheater_rejects_on_corrupted_codeword` rejects a
  prover who feeds in `f0` with `δ > 1 - √ρ` from any codeword.
- The `demo` binary prints round-by-round numbers and ends with
  `verifier accepted: true`.
- `cargo clippy -p stir --all-targets -- -D warnings` is clean.
- `cargo fmt -p stir --check` passes.

## Hints — STIR-specific gotchas

These are the **flagged pitfalls** from Kenneth's paper read; don't
discover them again the hard way.

- **`folding_factor >= 4` is a hard boundary.** STIR's soundness
  analysis (§4.2 of the paper) assumes `k >= 4`. For `k = 2` the OOD
  collapse and quotient bound both degrade, and the per-round error
  inflates by `~ρ^{-1/2}` instead of `~ρ`. Code should panic — `K_MIN`
  in `params.rs` exists exactly to enforce this.
- **OOD points MUST come from `F \ L_{i+1}`, never from `L_{i+1}`.** If
  even one OOD sample collides with the next round's evaluation domain,
  the quotient polynomial is ill-defined (division by zero at the
  collision point) AND the list-decoding collapse no longer holds. The
  `ood.rs` sampler rejection-samples specifically for this reason. The
  probability of collision per sample is `|L_{i+1}| / |F|`, which is
  cryptographically negligible — but you must enforce it.
- **Quotient distance fatality (Lemma 4.4).** If the prover claims
  `g(z_j) = β_j` for some β_j with `g(z_j) ≠ β_j` for the true `g`, the
  quotient polynomial picks up a pole at `z_j`, and the resulting
  function on `L_{i+1}` is **`1 - 1/|L_{i+1}|` far** from any low-degree
  codeword. That's nearly the whole code — making the verifier's
  proximity check trivially reject. This fatality is what gives STIR
  its strong RBR (round-by-round) soundness, and what fails silently if
  you implement the quotient wrong. **Cross-check the quotient against
  an honest run with a polynomial division before trusting your
  implementation.**
- **`t_i` is a harmonic decline, not geometric.** The repetitions for
  round `i` decrease roughly as `t_i ≈ t_0 / (i + 1)`. Sum is
  `O(t_0 · log M) = O(log² d)`. A geometric schedule (e.g. `t_i = t_0 / 2^i`)
  would converge to a constant total and break the security analysis.
- **RBR soundness is what licenses Fiat-Shamir.** STIR is round-by-round
  sound, not just sound — meaning each round's transcript-state has a
  bounded "cheating probability", independent of future rounds. This is
  what makes the Fiat-Shamir transform with BLAKE3 valid. If you switch
  the analysis to "interactive only" sound, BLAKE3-based Fiat-Shamir is
  no longer justified, and you'd need a stronger random-oracle argument.
- **Domain shift is to a `disjoint` smaller coset, not a subgroup.**
  `L_{i+1}` is constructed by taking `L_i`'s offset, squaring (or
  k/2-th-powering), and using a different smooth generator. The
  cosets `L_0, L_1, ..., L_M` are pairwise disjoint smooth multiplicative
  cosets — this is what makes the OOD points "outside" `L_{i+1}` a
  well-defined notion. See paper §3.2 for the explicit construction.
- **Degree correction matters.** After folding by `k`, the resulting
  function on `L_{i+1}` has degree `< d_i / k = d_{i+1}` in principle,
  but the *literal* polynomial recovered by interpolation has degree
  `< |L_{i+1}|`. The degree-correction step picks a specific
  low-degree multiplier to push the function back into
  `RS[F, L_{i+1}, d_{i+1}]` cleanly. Without it, the inner-round
  proximity tests don't compose.

## Common pitfalls (general)

- Forgetting to absorb every prover message into the transcript before
  the verifier squeezes the next challenge. **Fiat-Shamir is brittle:
  one missed absorb and the proof is forgeable.**
- Confusing `round` with `round_index + 1` in `round_degree_bound` and
  `round_log_domain_size`. The convention here: `round = 0` is the
  *input* domain (`L_0`, `d_0`), and after `M` rounds you arrive at
  `(L_M, d_M)`.
- Treating the folded polynomial's degree as `d_i / k` when it's
  actually `< d_i / k` (strict). Off-by-one in the degree bound is the
  source of most "almost-but-not-quite" verifier failures.
- Sampling Merkle query indices with `bytes % range` when `range` is not
  a power of 2 — introduces modulo bias. `Transcript::squeeze_indices`
  rejection-samples for exactly this reason.

## When you're done

Push to GitHub. Tell Claude "stir done"; review agents will spawn for
soundness / cleanliness / correctness passes. Once clean, move to
objective 4 — WHIR, which extends STIR to the multilinear setting.
