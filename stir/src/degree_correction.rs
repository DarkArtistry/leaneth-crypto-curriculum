//! **DegCor** (degree correction) and **Combine** — STIR's degree-alignment
//! primitives.
//!
//! ## Anchor: why STIR needs degree correction
//!
//! STIR's per-round arithmetic produces a folded/quotiented polynomial whose
//! *actual* degree bound `d_high` is **smaller** than the degree bound `d*`
//! that the **next round's Reed-Solomon code** is built around. Concretely,
//! after Fold by factor `k` the degree drops by `k` but after Quotient the
//! degree drops by a further `|S|` (the size of the OOD + shifted-query set);
//! meanwhile the next round's RS code on the shrunken domain `L_{i+1}` was
//! sized for a specific `d*_{i+1}` set by the rate-halving schedule (see
//! [`crate::params`]'s rate-drop formula). The two numbers don't line up: the
//! function in hand has degree `< d_high`, but to participate in the next
//! round's proximity test it must be a codeword of `RS[F, L_{i+1}, d*]`.
//!
//! Re-evaluating the polynomial on the new domain would defeat STIR's whole
//! reason to exist (no FFT per round). Instead, DegCor multiplies pointwise
//! by a scaling polynomial `g_r(X)` of degree exactly `e = d* − d_high` —
//! cheap, evaluation-form, **and** binds the new function to the verifier's
//! randomness `r`. This last property is load-bearing: if the multiplier were
//! the deterministic `X^e`, an adversarial prover would have a free hand to
//! engineer the high coefficients; randomising via `g_r` forces them to
//! commit to a single polynomial-in-`r` whose zero set Schwartz-Zippel says
//! is a `(d*−d_high)/|F|` fraction of the field (the named theorem below).
//!
//! ## What this module does
//!
//! Given a polynomial `f` of degree `< d_high` over the Goldilocks field
//! `F = F_p` and a **target degree bump** `e ≥ 0`, DegCor produces a "scaled"
//! polynomial `f · g_r` of degree `< d_high + e = d*` whose evaluation table
//! is bound to `f`'s by a verifier-chosen random scalar `r ∈ F_p` (the
//! round's degree-correction randomness). The scaling polynomial is the
//! **geometric-sum scaling polynomial**
//! `g_r(X) := sum_{i=0..e} (rX)^i`, defined formally below. A malicious
//! prover who cheats on `f` cannot hide the cheat by claiming the wrong
//! degree, because `r` enters the scaling factor.
//!
//! Combine is the simpler sibling: given two codewords `a` and `b` (already
//! on the same domain at the same target degree), `Combine(a, b, r) := a + r
//! · b` returns their random linear combination. STIR uses Combine to merge
//! two corrected codewords into one before passing to the next-round
//! Fold/Quotient step.
//!
//! ## What DegCor does
//!
//! Given evaluations of `f` of degree `< d` and a target degree bound `d* ≥
//! d`, compute the evaluations of
//!
//! ```text
//! f(X) · g_r(X),    where  g_r(X) := sum_{i=0..e} (rX)^i,   e := d* - d.
//! ```
//!
//! The product has degree `deg f + e ≤ (d - 1) + (d* - d) = d* - 1`, i.e.,
//! degree `< d*` — the new target. `g_r` is a "geometric-sum scaling
//! polynomial" whose role is to bump the degree by exactly `e` while binding
//! the result to the verifier's randomness `r`.
//!
//! ## Why we need it
//!
//! Across STIR rounds, the actual polynomial degree `d_i = d_0 / k^i` is set
//! by the folding schedule, but the round's RS code requires a specific
//! degree `d*_i` that may differ from `d_i` (because the *rate* drops between
//! rounds via the domain-shrinkage schedule). The two would-be-compatible
//! degree bounds don't line up. DegCor multiplies by `g_r` to raise `d_i` to
//! `d*_i`, after which Quotient subtracts `|S|` constraints to land on the
//! next round's actual degree.
//!
//! ## Computing `g_r(X)` efficiently
//!
//! A geometric sum has a closed form:
//!
//! ```text
//! g_r(X) = 1 + (rX) + (rX)^2 + ... + (rX)^e
//!        = (1 - (rX)^{e+1}) / (1 - rX),     for rX != 1.
//! ```
//!
//! Both numerator and denominator are linear-in-(`rX`) algebra. On
//! **evaluation form**, evaluating `g_r` at a point `x` is:
//!
//! ```text
//! g_r(x)  =  (1 - (r·x)^{e+1}) · (1 - r·x)^{-1}      if r·x != 1
//!         =  e + 1                                   if r·x == 1.
//! ```
//!
//! The `r·x == 1` case avoids the 0/0 — both numerator and denominator vanish,
//! and L'Hôpital (or just direct summation) gives `g_r(x) = e + 1`.
//!
//! ## Worked example (`F_17`, `d = 1`, `d* = 3`, `r = 5`)
//!
//! Take `f(X) = 2 + 3X` (degree 1). Target `d* = 3` so `e = d* - d = 2` and
//! `g_r(X) = 1 + 5X + 25X²`. Reduce `25 ≡ 8 (mod 17)`, so `g_r(X) = 1 + 5X + 8X²`.
//!
//! Pick the size-4 subgroup `L = {1, 4, 16, 13}` of `F_17^*` (the same one as
//! in the fft sanity check). Walking through:
//!
//! ```text
//! L[0] = 1:   g_r(1)  = 1 + 5 + 8 = 14;       f(1)  = 5;     product = 70 ≡ 2  (mod 17)
//! L[1] = 4:   g_r(4)  = 1 + 20 + 128 = 1 + 3 + 128 mod 17.
//!             128 = 7·17 + 9, so g_r(4) = 1 + 3 + 9 = 13.    f(4)  = 14;    product = 182 ≡ 12 (mod 17)
//! L[2] = 16:  g_r(16) = 1 + 80 + 2048.   80 ≡ 80-4·17 = 12;  2048 = 120·17 + 8, ≡ 8.
//!             g_r(16) = 1 + 12 + 8 = 21 ≡ 4.                 f(16) = 2 + 48 = 50 ≡ 16; product = 64 ≡ 13.
//! L[3] = 13:  g_r(13) = 1 + 65 + 1352.   65 ≡ 65-3·17 = 14;  1352 = 79·17+9, ≡ 9.
//!             g_r(13) = 1 + 14 + 9 = 24 ≡ 7.                 f(13) = 2 + 39 = 41 ≡ 7;  product = 49 ≡ 15.
//! ```
//!
//! So `deg_cor(evals_of_f, L, 3, 5) = [2, 12, 13, 15]` (mod 17), with each
//! entry being `f(L[j]) · g_r(L[j])` as computed above. None of the four
//! `r·x` values hit `1`, so the closed-form branch is exercised throughout.
//!
//! ## Worked example for the Degree-Correction theorem (`d_high = 2`, `d_target = 4`, `α = 3`)
//!
//! This second example focuses on the structural theorem above. Take
//! `d_high = 2` (polynomial degrees `< 2`, i.e. constants and linears),
//! `d_target = 4`, so `e := d_target − d_high = 2`. In this codebase the
//! geometric sum `g_r(X) := sum_{i=0..e} (rX)^i` runs `i = 0, 1, ..., e` —
//! one term per "slot of degree bump" plus the constant — so for our
//! parameters and `α = 3`,
//!
//! ```text
//! g_r(X)  =  1 + (3X) + (3X)^2 + (3X)^3
//!         =  1 + 3X + 9X^2 + 27X^3.
//! ```
//!
//! The leading term is `27 X^3`, so `deg(g_r) = 3 = e + 1`. The theorem
//! gives `deg(f · g_r) ≤ deg(f) + deg(g_r) ≤ (d_high − 1) + (e + 1) =
//! d_target − 1` — i.e. `f · g_r ∈ RS[F, L, d_target]` as promised. Pick a
//! concrete `f(X) = 1 + X ∈ RS[F, L, d_high = 2]`:
//!
//! ```text
//! f(X) · g_r(X)  =  (1 + X) · (1 + 3X + 9X^2 + 27X^3)
//!                =  (1 + 3X + 9X^2 + 27X^3)
//!                 + (X + 3X^2 + 9X^3 + 27X^4)
//!                =  1 + 4X + 12X^2 + 36X^3 + 27X^4.
//! ```
//!
//! Top degree `4 = d_target`, so the product is in `RS[F, L, d_target + 1]`
//! when `g_r` has degree exactly `e + 1` — which is the slack the protocol
//! actually wants (it leaves "headroom" for the next round's Quotient to
//! shave off the OOD/shifted-query constraints without dropping below the
//! next round's RS code dimension). For a tight `RS[F, L, d_target]`
//! membership use the `(e)`-degree variant `g_r(X) = 1 + 3X + 9X^2` instead.
//!
//! Pointwise on a hypothetical domain `L = {0, 1, 2}`:
//!
//! ```text
//! L[0] = 0: f(0) = 1,   g_r(0) = 1,                    product = 1
//! L[1] = 1: f(1) = 2,   g_r(1) = 1 + 3 + 9 + 27 = 40,  product = 80
//! L[2] = 2: f(2) = 3,   g_r(2) = 1 + 6 + 36 + 216 = 259, product = 777.
//! ```
//!
//! The convolution insight is that the high-coefficient `27 = α^{e+1}` of
//! `f · g_r` is a polynomial of degree `e + 1` in `α`, so any adversary's
//! claim about it is Schwartz-Zippel-bounded by `(e+1)/|F| = (d* − d)/|F|`
//! up to a constant — the soundness theorem's bound.
//!
//! ## Named theorems
//!
//! ### Degree-Correction theorem (the structural statement)
//!
//! > **Degree-Correction.** Let `f ∈ RS[F, L, d_high]` (i.e. `f`'s evaluation
//! > table on `L` agrees with some polynomial of degree `< d_high`). Let
//! > `e := d_target − d_high ≥ 0` and let `g_r(X) := sum_{i=0..e} (r X)^i`
//! > be the geometric-sum scaling polynomial for any `r ∈ F`. Then
//! >
//! > ```text
//! > f · g_r  ∈  RS[F, L, d_target].
//! > ```
//! >
//! > **Proof.** `f` is the evaluation table of some `p ∈ F[X]` with
//! > `deg(p) < d_high`, i.e. `deg(p) ≤ d_high − 1`. `g_r` is a polynomial of
//! > degree exactly `e` (its leading term `r^e X^e` is non-zero whenever
//! > `r ≠ 0`, and is the zero polynomial collapsed to the constant `1` when
//! > `r = 0`, in which case `deg(g_r) = 0 ≤ e`). In either case
//! >
//! > ```text
//! > deg(p · g_r)  ≤  deg(p) + deg(g_r)
//! >               ≤  (d_high − 1) + e
//! >               =  d_high + e − 1
//! >               =  d_target − 1.
//! > ```
//! >
//! > So `p · g_r` has degree `< d_target`, and its evaluation table on `L`
//! > is exactly the pointwise product `f[j] · g_r(L[j])` — which is the
//! > vector this module's `deg_cor` returns. Hence `f · g_r ∈ RS[F, L,
//! > d_target]`. ∎
//!
//! ### DegCor Soundness theorem (the adversarial statement)
//!
//! > **DegCor Soundness.** Let `f` be a function on the evaluation domain `L`
//! > and `RS_d`, `RS_{d*}` the Reed-Solomon codes of degree bounds `< d` and
//! > `< d*` on `L`. If `f` is δ-far (in relative Hamming distance) from
//! > `RS_d`, then for a uniformly random `r ∈ F`,
//! >
//! > ```text
//! > Pr_r[ f · g_r is < δ-far from RS_{d*} ]  ≤  (d* - d) / |F|.
//! > ```
//!
//! Proof sketch: if `f` is δ-far from `RS_d`, then `f - h` is a non-zero
//! length-`|L|` vector for every `h ∈ RS_d`, with Hamming weight `≥ δ · |L|`.
//! For `f · g_r` to land in `RS_{d*}`, an attacker would need a specific
//! relationship between the `e + 1` powers `(r^0, r^1, ..., r^e)` and the
//! columns of `f`'s "error matrix" — a polynomial equation in `r` of degree
//! `≤ e = d* - d`. By Schwartz-Zippel a non-zero such polynomial vanishes on
//! at most `(d* - d) / |F|` of the field, hitting the relation with
//! negligible probability.
//!
//! ## Socratic prompt
//!
//! **Question for the reader.** Why does `g_r(X)` use a geometric-sum form
//! rather than just `X^{d* - d}` to bump the degree?
//!
//! Answer: a simple shift `f · X^{d* - d}` doesn't randomize. The high
//! coefficients of the product would just be `f`'s coefficients in shifted
//! positions — deterministic, no `r` dependence. A cheating prover who picked
//! `f` adversarially can keep cheating after the shift; nothing has been
//! "bound" by verifier randomness. `g_r` mixes the `e + 1` powers
//! `1, rX, (rX)², ..., (rX)^e` non-trivially with `r`, so the product `f ·
//! g_r` is a *polynomial in `r`* of degree `e`, and Schwartz-Zippel says a
//! non-zero polynomial in `r` of small degree vanishes on a tiny fraction of
//! the field. That gives the soundness bound `(d* - d) / |F|` in the theorem
//! above. Without randomness, the bound would be vacuous.
//!
//! ## Cross-module interface
//!
//! - [`deg_cor`] evaluates `f · g_r` on the domain (the operationally
//!   important case for an evaluation-form prover).
//! - [`poly_deg_cor`] multiplies a **coefficient-form** polynomial by
//!   `g_r(X)` directly, returning a new polynomial of degree
//!   `deg(poly) + e`. Used by the refactored prover (see
//!   `stir-full-spec.md` §6) which keeps its round-`i` polynomial in
//!   coefficient form (Fold/Quotient/DegCor all operate naturally on
//!   coefficients in the refactor; no per-round IFFT needed).
//! - [`eval_g_r`] evaluates the geometric-sum scaling polynomial
//!   `g_r(X) = Σ_{i=0..e} (rX)^i` at a single field point using the
//!   closed form `(1 − (rx)^{e+1}) / (1 − rx)` with the `rx == 1`
//!   L'Hôpital branch returning `e + 1`. The verifier uses this to do
//!   the per-query DegCor step pointwise without ever materialising
//!   `g_r` as a polynomial.
//! - [`combine`] does the round's linear combination of two evaluation
//!   vectors.
//!
//! `deg_cor` and `combine` work on raw evaluation tables; `poly_deg_cor`
//! and `eval_g_r` are the coefficient-form / pointwise siblings. The
//! **prover** in this refactor uses `poly_deg_cor` (one polynomial
//! multiply per round, deg-`< d_{i+1}` output), while the **verifier**
//! uses `eval_g_r` (one closed-form geometric-sum eval per shift query).
//! Both compute the same algebraic object — the same `f · g_r` of the
//! Degree-Correction theorem — at different granularities.

use reed_solomon::{EvaluationDomain, Fp, UnivariatePoly};

/// Degree-correct an evaluation table: scale `f` so that its effective
/// degree bound becomes `target_degree`.
///
/// Computes `(f · g_r)(L[j])` for each `L[j] ∈ L`, where `g_r(X) =
/// sum_{i=0..e} (rX)^i` with `e = target_degree - input_degree_bound`.
///
/// # Inputs
///
/// - `evals`: `f(L[0]), ..., f(L[n-1])`. The caller-implicit "input degree
///   bound" is `evals.len()` (one evaluation per coefficient, in the
///   unencoded sense). For STIR's purposes, the input is on the domain `L`
///   so `evals.len() == |L|`, and we infer the polynomial's degree bound from
///   the protocol's per-round schedule rather than from `evals.len()` alone.
/// - `domain`: the evaluation domain `L`.
/// - `target_degree`: `d*`. Must satisfy `target_degree >= input_degree_bound`.
/// - `randomness`: `r ∈ F_p`. The verifier's challenge.
///
/// # Output
///
/// A `Vec<Fp>` of length `|L|`, with entry `j` equal to `(f · g_r)(L[j])`.
///
/// # Panics
///
/// Panics if `evals.len() != domain.size()`, or if `target_degree` is
/// strictly less than the implied input degree bound (which would make `e
/// < 0` and `g_r` ill-defined — degree *shrinking* is Quotient's job, not
/// DegCor's).
///
/// # Implementation note: input degree bound
///
/// This function's signature doesn't take an explicit `input_degree_bound`
/// argument. In STIR the protocol schedule fixes the round's actual degree
/// `d_i`, and `target_degree = d*_i` is set against it. A common convention
/// (followed here) is to treat `input_degree_bound` as `target_degree`'s
/// caller-side companion: the caller invokes `deg_cor` knowing both numbers.
/// If you find yourself wanting both as arguments, that's a sign you want a
/// thin wrapper struct (`DegCor { d, d_star, r }`); for the educational
/// scaffold it's fine to expose only the target and let the caller pass
/// `target_degree == input_degree_bound` for the identity case.
///
/// (STIR §4, DegCor definition; soundness Lemma 4.5.)
pub fn deg_cor(
    evals: &[Fp],
    domain: &EvaluationDomain,
    target_degree: usize,
    randomness: Fp,
) -> Vec<Fp> {
    // Calling convention (cross-ref §"What this module does" and the
    // Degree-Correction theorem in the module docs): `target_degree` is the
    // **degree bump** `e := d_target − d_high` that the caller has already
    // computed from the protocol's round-by-round schedule. We don't take
    // `d_high` separately — the caller tracks it in a STIR round struct and
    // hands us only the gap. So:
    //
    //   - `target_degree == 0` ⇒ e = 0 ⇒ g_r(X) = 1   (identity; see test (a))
    //   - `target_degree  > 0` ⇒ e > 0 ⇒ g_r has e+1 terms.
    //
    // The "ill-defined degree-shrinking" case the scaffold cautioned about
    // is now structurally impossible: `e` is a `usize`, hence `≥ 0`.
    assert_eq!(
        evals.len(),
        domain.size(),
        "deg_cor: evals.len() ({}) must equal domain.size() ({})",
        evals.len(),
        domain.size(),
    );

    let e = target_degree;
    // `g_r` evaluated at a single domain point `x`, using the closed form
    // (1 - (rx)^{e+1}) / (1 - rx) when rx ≠ 1, and the direct sum e + 1
    // when rx = 1 (the 0/0 L'Hôpital branch).
    let gr_at = |x: Fp| -> Fp {
        let rx = randomness * x;
        if rx == Fp::one() {
            // Direct geometric sum: 1 + 1 + ... (e+1 ones) = e + 1.
            Fp::new((e as u64) + 1)
        } else {
            let num = Fp::one() - rx.pow((e as u64) + 1);
            let den = Fp::one() - rx;
            num * den.inverse().expect("1 - rx ≠ 0 in this branch")
        }
    };

    // One pass: pointwise product f(L[j]) · g_r(L[j]).
    domain
        .iter()
        .zip(evals.iter())
        .map(|(x, &fx)| fx * gr_at(x))
        .collect()
}

/// Random linear combination of two codewords.
///
/// Computes `c[j] = a[j] + r · b[j]` for each `j`. Used in STIR's
/// "DegCor + Combine" pattern: when the round merges two intermediate
/// codewords (already on the same domain, at the same degree bound) into a
/// single codeword for the next-round operations.
///
/// # Inputs
///
/// - `evals_a`, `evals_b`: two same-length evaluation tables.
/// - `randomness`: `r ∈ F_p`. The verifier's challenge.
///
/// # Output
///
/// A `Vec<Fp>` of the same length, entry `j` equal to `evals_a[j] + r ·
/// evals_b[j]`.
///
/// # Panics
///
/// Panics if `evals_a.len() != evals_b.len()`.
///
/// (STIR §4, Combine definition.)
pub fn combine(
    evals_a: &[Fp],
    evals_b: &[Fp],
    randomness: Fp,
) -> Vec<Fp> {
    // Simplest of the module's primitives — but load-bearing for STIR
    // soundness: like `deg_cor`, it binds two codewords by a random scalar,
    // so a cheating prover can't fix one codeword's error after seeing the
    // challenge.
    assert_eq!(
        evals_a.len(),
        evals_b.len(),
        "combine: length mismatch ({} vs {})",
        evals_a.len(),
        evals_b.len(),
    );

    evals_a
        .iter()
        .zip(evals_b.iter())
        .map(|(&a, &b)| a + randomness * b)
        .collect()
}

/// Coefficient-form degree correction: multiply `poly` by the geometric-sum
/// scaling polynomial `g_r(X) = Σ_{i=0..e} (r·X)^i`, returning a polynomial
/// whose degree bound bumps up by exactly `e`.
///
/// This is the **prover's** companion to [`deg_cor`]: it computes the same
/// algebraic object (the `f · g_r` of the Degree-Correction theorem above)
/// but lands in coefficient form rather than on an evaluation table. The
/// refactored prover (see `stir-full-spec.md` §2.1 and §6) keeps its
/// round-`i` polynomial in coefficient form across Quotient → Fold →
/// DegCor; this helper closes the loop so the round update stays
/// coefficient-form end to end.
///
/// # Inputs
/// - `poly`: input polynomial in coefficient form.
/// - `e`: degree bump (= `d_target − d_high`). If `e == 0`,
///   `g_r(X) = 1` and this returns `poly.clone()`.
/// - `r`: verifier randomness (the `r_comb_i` squeeze in the per-round
///   transcript schedule; see `stir-full-spec.md` §1.1).
///
/// # Output
/// `UnivariatePoly` representing `poly(X) · g_r(X)`. Its degree bound is
/// `deg(poly) + e`; trailing-zero stripping by `UnivariatePoly::new`
/// gives the tight degree.
///
/// # Algorithm
/// 1. Build the coefficient vector of `g_r`: `[1, r, r^2, ..., r^e]`,
///    length `e + 1`. One running multiply per coefficient.
/// 2. Convolve `poly.coeffs()` with `g_r_coeffs` naively in
///    `O((d + e) · e)` field operations. For demo parameters `e ≤ 2`
///    so this is effectively `O(d)`; no FFT needed.
/// 3. Wrap via `UnivariatePoly::new` (strips trailing zeros — important
///    for the Prover-Round Invariant's `deg(current_poly) < d_{i+1}`
///    bookkeeping).
///
/// # Cross-reference
/// See [`eval_g_r`] for the single-point evaluation used by the verifier
/// to check `f · g_r` at a query point without building the polynomial.
/// See the module-level **Degree-Correction theorem** for the algebraic
/// guarantee `deg(poly · g_r) ≤ (d_high − 1) + e = d_target − 1`.
pub fn poly_deg_cor(poly: &UnivariatePoly, e: usize, r: Fp) -> UnivariatePoly {
    if e == 0 {
        return poly.clone();
    }

    // (1) g_r coeffs: [1, r, r^2, ..., r^e].
    let mut g_r_coeffs: Vec<Fp> = Vec::with_capacity(e + 1);
    let mut acc = Fp::one();
    for _ in 0..=e {
        g_r_coeffs.push(acc);
        acc = acc * r;
    }

    let in_coeffs = poly.coeffs();
    if in_coeffs.is_empty() {
        // poly is the zero polynomial — `0 · g_r = 0`.
        return UnivariatePoly::zero();
    }

    // (2) Naive convolution. out[i+j] += a_i · b_j.
    let out_len = in_coeffs.len() + g_r_coeffs.len() - 1;
    let mut out = vec![Fp::zero(); out_len];
    for (i, &a) in in_coeffs.iter().enumerate() {
        for (j, &b) in g_r_coeffs.iter().enumerate() {
            out[i + j] = out[i + j] + a * b;
        }
    }

    // (3) Wrap, stripping trailing zeros.
    UnivariatePoly::new(out)
}

/// Single-point evaluation of the geometric-sum scaling polynomial
/// `g_r(X) = Σ_{i=0..e} (r·X)^i` at `x ∈ F_p`.
///
/// Uses the closed form
///
/// ```text
/// g_r(x) = (1 − (r·x)^{e+1}) / (1 − r·x),     r·x ≠ 1
///        = e + 1,                              r·x  = 1   (L'Hôpital branch)
/// ```
///
/// This is the **verifier's** companion to [`poly_deg_cor`]: at a shift
/// query the verifier has a single point `z_k` and needs `g_r(z_k)` to
/// apply the DegCor multiplier locally; building `g_r` as a polynomial
/// would be silly when one closed-form evaluation suffices.
///
/// # Inputs
/// - `r`: verifier randomness (`r_comb_i`).
/// - `x`: evaluation point.
/// - `e`: degree of `g_r` minus one — i.e. `g_r` has `e + 1` terms.
///   When `e == 0`, `g_r ≡ 1`, so the return value is `Fp::one()`
///   regardless of `r` and `x` (the closed-form numerator `1 − rx`
///   matches the denominator `1 − rx`, giving `1`; the special case is
///   not strictly required but is documented here for cross-reference
///   with [`poly_deg_cor`]'s `e == 0` early-return).
///
/// # Output
/// `g_r(x) ∈ F_p`.
///
/// # Cross-reference
/// See [`poly_deg_cor`] for the coefficient-form sibling. See the
/// module-level "Computing `g_r(X)` efficiently" section for the
/// derivation of the closed form and the `rx == 1` L'Hôpital branch.
pub fn eval_g_r(r: Fp, x: Fp, e: usize) -> Fp {
    let rx = r * x;
    if rx == Fp::one() {
        // Direct geometric sum: 1 + 1 + ... + 1 (e + 1 ones).
        Fp::new((e as u64) + 1)
    } else {
        let num = Fp::one() - rx.pow((e as u64) + 1);
        let den = Fp::one() - rx;
        num * den.inverse().expect("1 - rx ≠ 0 in this branch")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// With `r = 0`, `g_r(X) = 1 + 0 + 0 + ... = 1` (a constant), so
    /// `f · g_r = f` and DegCor is the identity at the evaluation level —
    /// even with a non-trivial degree bump `e > 0`.
    ///
    /// Note: the domain `L` is the size-8 multiplicative subgroup of `F_p`,
    /// and `0 ∈ F_p` is **not** in `L` (subgroups don't contain 0). So no
    /// domain element `x` satisfies `r·x = 0·x = 0 = 1`, and the closed-form
    /// branch (not the L'Hôpital branch) is what's exercised here.
    #[test]
    fn deg_cor_with_r_zero_is_identity() {
        let domain = EvaluationDomain::new_subgroup(3); // |L| = 8
        let evals: Vec<Fp> = (1..=8).map(Fp::new).collect();
        // e = 5 (any positive bump); with r = 0 the result should still be `evals`.
        let result = deg_cor(&evals, &domain, 5, Fp::zero());
        assert_eq!(result, evals, "r = 0 ⇒ g_r ≡ 1 ⇒ deg_cor is identity");
    }

    /// When `target_degree == 0` (i.e. `e = 0`), `g_r(X) = (rX)^0 = 1`
    /// (a single term). DegCor is the identity in that case regardless of `r`.
    #[test]
    fn deg_cor_with_e_zero_returns_input() {
        let domain = EvaluationDomain::new_subgroup(3); // |L| = 8
        let evals: Vec<Fp> = (1..=8).map(Fp::new).collect();
        // Non-trivial randomness shouldn't matter — e = 0 collapses g_r to 1.
        let result = deg_cor(&evals, &domain, 0, Fp::new(13));
        assert_eq!(result, evals, "e = 0 ⇒ g_r ≡ 1 ⇒ deg_cor is identity");
    }

    /// `combine(a, b, 0) = a`: the `b` term vanishes.
    #[test]
    fn combine_with_r_zero_returns_first() {
        let a = vec![Fp::new(10), Fp::new(20), Fp::new(30), Fp::new(40)];
        let b = vec![Fp::new(7), Fp::new(11), Fp::new(13), Fp::new(17)];
        let out = combine(&a, &b, Fp::zero());
        assert_eq!(out, a);
    }

    /// `combine(a, b, 1) = a + b` pointwise. Also checks one entry against a
    /// hand-computed value with `r = 5` to exercise the non-trivial path.
    #[test]
    fn combine_with_r_one_is_pointwise_sum() {
        let a = vec![Fp::new(10), Fp::new(20), Fp::new(30), Fp::new(40)];
        let b = vec![Fp::new(7), Fp::new(11), Fp::new(13), Fp::new(17)];

        let out_one = combine(&a, &b, Fp::one());
        for j in 0..a.len() {
            assert_eq!(out_one[j], a[j] + b[j], "mismatch at j = {}", j);
        }

        // Bonus: hand-checked entry with r = 5 → a[0] + 5·b[0] = 10 + 35 = 45.
        let out_five = combine(&a, &b, Fp::new(5));
        assert_eq!(out_five[0], Fp::new(45));
        assert_eq!(out_five[1], Fp::new(20) + Fp::new(5) * Fp::new(11));
    }

    /// `poly_deg_cor` with `e = 0` is the identity (returns input unchanged),
    /// regardless of `r`. Mirror of the eval-form `e = 0` test above.
    #[test]
    fn poly_deg_cor_with_e_zero_is_identity() {
        let p = UnivariatePoly::new(vec![Fp::new(7), Fp::new(11), Fp::new(13)]);
        // Even with non-trivial r, e=0 ⇒ g_r ≡ 1 ⇒ result == input.
        let out = poly_deg_cor(&p, 0, Fp::new(99));
        assert_eq!(out, p);
    }

    /// `poly_deg_cor` matches the coefficient-by-coefficient convolution
    /// against a hand-built `g_r` polynomial. Worked example from the
    /// module docs: `f(X) = 1 + X`, `e = 2`, `r = 3` ⇒
    /// `g_r(X) = 1 + 3X + 9X²`, and
    /// `(1 + X) · (1 + 3X + 9X²) = 1 + 4X + 12X² + 9X³`.
    #[test]
    fn poly_deg_cor_matches_hand_computed_convolution() {
        let f = UnivariatePoly::new(vec![Fp::one(), Fp::one()]); // 1 + X
        let out = poly_deg_cor(&f, 2, Fp::new(3));
        // Expected coefficients [1, 4, 12, 9].
        let expected = UnivariatePoly::new(vec![
            Fp::new(1),
            Fp::new(4),
            Fp::new(12),
            Fp::new(9),
        ]);
        assert_eq!(out, expected);
        // Degree bumps up by exactly e = 2.
        assert_eq!(out.degree(), Some(3));
    }

    /// `poly_deg_cor(f, e, r)` evaluated pointwise on a domain agrees with
    /// `deg_cor(evals_of_f, domain, e, r)` — i.e., coefficient-form DegCor
    /// and evaluation-form DegCor produce the same algebraic object (the
    /// `f · g_r` of the Degree-Correction theorem). This is the load-bearing
    /// equivalence between the prover's coefficient-form path and the
    /// (pre-existing) evaluation-form helper.
    #[test]
    fn poly_deg_cor_matches_eval_form_on_domain() {
        let domain = EvaluationDomain::new_subgroup(3); // |L| = 8
        let f = UnivariatePoly::new(vec![Fp::new(2), Fp::new(5), Fp::new(11), Fp::new(7)]);
        let r = Fp::new(13);
        let e = 3usize;

        // Evaluate f on the domain pointwise.
        let evals: Vec<Fp> = (0..domain.size())
            .map(|j| f.evaluate(domain.element(j)))
            .collect();

        // Eval-form DegCor.
        let evals_corrected = deg_cor(&evals, &domain, e, r);

        // Coef-form DegCor, then evaluate on the domain.
        let f_corrected = poly_deg_cor(&f, e, r);
        let poly_corrected_on_domain: Vec<Fp> = (0..domain.size())
            .map(|j| f_corrected.evaluate(domain.element(j)))
            .collect();

        assert_eq!(evals_corrected, poly_corrected_on_domain);
    }

    /// `eval_g_r(r, x, e)` matches the direct geometric sum
    /// `Σ_{i=0..e} (r·x)^i` at every domain point. Exercises the closed-form
    /// branch (the `rx == 1` branch is exercised separately below).
    #[test]
    fn eval_g_r_matches_direct_sum_on_domain() {
        let domain = EvaluationDomain::new_subgroup(3); // |L| = 8
        let r = Fp::new(13);
        let e = 4usize;

        for j in 0..domain.size() {
            let x = domain.element(j);
            let got = eval_g_r(r, x, e);

            // Direct sum.
            let rx = r * x;
            let mut expected = Fp::zero();
            let mut pow = Fp::one();
            for _ in 0..=e {
                expected = expected + pow;
                pow = pow * rx;
            }
            assert_eq!(got, expected, "g_r mismatch at L[{j}]");
        }
    }

    /// `eval_g_r` triggers the `r·x == 1` L'Hôpital branch and returns
    /// `e + 1`. Construct it explicitly by picking `x = r^{-1}`.
    #[test]
    fn eval_g_r_lhopital_branch_when_rx_is_one() {
        let r = Fp::new(7);
        let x = r.inverse().expect("7 ≠ 0 in Goldilocks");
        // Sanity: r · x == 1.
        assert_eq!(r * x, Fp::one());

        for &e in &[0usize, 1, 2, 5] {
            let got = eval_g_r(r, x, e);
            assert_eq!(
                got,
                Fp::new((e as u64) + 1),
                "L'Hôpital branch must return e + 1 (e = {e})",
            );
        }
    }

    /// Sanity check the closed-form path: pick a tiny case where we can
    /// compare against a direct sum.
    ///
    /// Take e = 3, r = α, and verify g_r(x) = 1 + αx + (αx)^2 + (αx)^3 at
    /// every domain point. We do this by computing `deg_cor` with `f ≡ 1`
    /// (all-ones evals) and checking the output equals the direct geometric
    /// sum evaluated point-by-point.
    #[test]
    fn deg_cor_closed_form_matches_direct_sum() {
        let domain = EvaluationDomain::new_subgroup(3); // |L| = 8
        let ones = vec![Fp::one(); domain.size()];
        let alpha = Fp::new(3);
        let e = 3usize;

        let got = deg_cor(&ones, &domain, e, alpha);

        // Expected: at each x ∈ L, g_r(x) = sum_{i=0..=e} (αx)^i.
        for (j, x) in domain.iter().enumerate() {
            let rx = alpha * x;
            // Direct geometric sum, no closed form.
            let mut expected = Fp::zero();
            let mut pow = Fp::one();
            for _ in 0..=e {
                expected = expected + pow;
                pow = pow * rx;
            }
            assert_eq!(got[j], expected, "g_r mismatch at L[{}] = {:?}", j, x);
        }
    }
}
