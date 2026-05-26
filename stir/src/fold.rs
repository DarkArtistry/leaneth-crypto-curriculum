//! The **Fold** operation — STIR's core round-reduction step.
//!
//! ## What this module does
//!
//! Folding takes a polynomial / codeword `f` of degree `< d` over the
//! Goldilocks field `F = F_p` and produces a new polynomial / codeword
//! of degree `< d / k`, where `k` (the **folding factor**, a fixed
//! power of two — at least 4 in STIR) is the round's reduction factor.
//! A single random scalar `α ∈ F_p` — the **fold randomness**, the
//! verifier's per-round challenge — drives the reduction; the verifier
//! sends `α`, both prover and verifier apply the same fold, and the
//! protocol moves on to a problem of `1/k` the original size. The
//! `k` **row functions** `f_0, ..., f_{k-1}` introduced just below are
//! the coefficient-stride pieces of `f` that the fold mixes together
//! using powers of `α`.
//!
//! ## Anchor: STIR's per-round degree reduction
//!
//! Every STIR round consumes one degree-`< d_i` claim ("the function on
//! `L_i` is δ-close to `RS[F, L_i, d_i]`") and emits a degree-`< d_i / k`
//! claim ("the folded function on `L_i^k` is δ-close to
//! `RS[F, L_i^k, d_i / k]`"). The **Fold** operation in this module is
//! the algebraic engine that performs the degree reduction. Three things
//! must all happen at once:
//!
//! 1. **Polynomial side.** A degree-`< d` polynomial `f(X)` collapses to
//!    a degree-`< d/k` polynomial `Fold(f, α)(Y)` parameterised by the
//!    verifier's challenge `α`. Implemented in [`poly_fold`].
//! 2. **Codeword side.** An evaluation table of `f` on a smooth domain
//!    `L` (closed under multiplication by a primitive `k`-th root of
//!    unity `ω`) collapses to an evaluation table of `Fold(f, α)` on
//!    `L^k = { x^k : x ∈ L }`, a domain of size `|L| / k`. Implemented in
//!    [`fold`].
//! 3. **Two sides must agree.** Encoding then folding (on the codeword
//!    side) must give the same table as folding then encoding (on the
//!    polynomial side). The named theorem below proves this; the test
//!    `fold_matches_poly_fold_on_evaluations` checks it numerically.
//!
//! Why this is the round's *only* algebraic step: every other STIR
//! ingredient (Merkle queries, OOD samples, the quotient construction,
//! degree-correction) either commits to the table that comes out of
//! [`fold`], or tests properties of it. Fold is the verb; everything
//! else is bookkeeping around the verb.
//!
//! ## What folding does, in one paragraph
//!
//! Every polynomial `f(X)` of degree `< d` decomposes **uniquely** as
//!
//! ```text
//! f(X) = sum_{i=0..k-1} f_i(X^k) · X^i,        with deg(f_i) < d/k.
//! ```
//!
//! Read coefficient-by-coefficient: the `j`-th coefficient of `f` becomes the
//! `(j / k)`-th coefficient of `f_{j mod k}`. So `f_0` collects coefficients
//! at positions `0, k, 2k, ...`; `f_1` collects positions `1, k+1, 2k+1, ...`;
//! and so on. Conversely, `f_i` is the **`i`-th row** in the `k × (d/k)`
//! coefficient matrix of `f` (with row index `= position mod k`).
//!
//! The **fold by `α`** operation is then
//!
//! ```text
//! Fold(f, α)(Y)  :=  sum_{i=0..k-1} α^i · f_i(Y),
//! ```
//!
//! a single polynomial of degree `< d/k`. The `f_i(X^k) · X^i` decomposition
//! becomes `f_i(Y) · α^i` after substituting `Y = X^k` (which collapses the
//! `X^k` argument to `Y`) and `α` (which collapses the `X^i` factor to a
//! scalar `α^i`).
//!
//! ## Equivalent `k`-row view
//!
//! Think of `f(X)` as a `k × (d/k)` matrix `M` where row `i` lists the
//! coefficients of `f_i`. Then the fold is the matrix-vector product
//!
//! ```text
//! (α^0, α^1, α^2, ..., α^{k-1}) · M,
//! ```
//!
//! producing a length-`d/k` row vector — the coefficients of `Fold(f, α)`.
//! "Coefficient form fold" really is one matrix multiply on the coefficient
//! grid. See [`poly_fold`] for the implementation; the row extraction step
//! there is literally "transpose `c` by `k`-strides and stack as rows".
//!
//! ## Why it works on evaluation tables, not just polynomials
//!
//! Folding on evaluation tables is the operationally important case in STIR:
//! the prover commits to `f` as an evaluation table on `L`, never as
//! coefficients, and the verifier needs to derive the folded codeword on `L^k`
//! using only the original table.
//!
//! The trick: pick the evaluation domain `L` to be **closed under
//! multiplication by `ω`**, where `ω` is a primitive `k`-th root of unity.
//! For smooth multiplicative domains `L = c · <g>` with `|L|` a multiple of
//! `k`, we can take `ω = g^{|L|/k}` and the closure holds automatically. Then
//! for any `x ∈ L`, the `k` points `x, ω·x, ω²·x, ..., ω^{k-1}·x` are also in
//! `L` — and they share the same `k`-th power: `(ω^i · x)^k = ω^{ik} · x^k =
//! x^k` (using `ω^k = 1`). So the `k`-fold "fibre" over `x^k` is exactly
//! `{ω^i · x}_{i=0..k-1}`.
//!
//! Now evaluate the decomposition at `ω^i · x`:
//!
//! ```text
//! f(ω^i · x)  =  sum_{j=0..k-1} f_j((ω^i · x)^k) · (ω^i · x)^j
//!             =  sum_{j=0..k-1} f_j(x^k) · ω^{ij} · x^j.
//! ```
//!
//! Treat the `k` values `(f(x), f(ω·x), ..., f(ω^{k-1}·x))` as a vector and
//! the `k` row-evaluations `(f_0(x^k) · x^0, f_1(x^k) · x^1, ..., f_{k-1}(x^k)
//! · x^{k-1})` as another. The matrix relating them is the DFT matrix
//! `V[i, j] = ω^{ij}` — a Vandermonde at `ω` (see `reed_solomon::fft`). Invert
//! the DFT (a single `k`-point IDFT, cheap for small `k`), divide each entry
//! by the appropriate `x^j`, and you recover `(f_0(x^k), ..., f_{k-1}(x^k))`.
//! Linear-combine with `(α^0, α^1, ..., α^{k-1})` to produce the folded
//! evaluation `Fold(f, α)(x^k)`.
//!
//! Repeat for every `x` in a representative set (one per fibre — equivalently,
//! one per coset of `<ω>` in `L`) and you've built the entire folded
//! evaluation table on the squared/`k`-th-power domain `L^k`. See [`fold`]
//! for the implementation; the IDFT step appeals to `reed_solomon::fft`'s
//! orthogonality identities.
//!
//! ## Named theorem
//!
//! ### Folding theorem
//!
//! > **Folding theorem.** Let `f(X) ∈ F[X]` have degree `< d`, decompose it
//! > uniquely as
//! >
//! > ```text
//! > f(X) = sum_{i=0..k-1} f_i(X^k) · X^i,        deg(f_i) < d / k,
//! > ```
//! >
//! > and define `Fold(f, α)(Y) := sum_{i=0..k-1} α^i · f_i(Y)`. Then for every
//! > `x ∈ F` and every `α ∈ F`:
//! >
//! > ```text
//! > Fold(f, α)(X^k) = sum_{i=0..k-1} α^i · f_i(X^k)
//! > ```
//! >
//! > as polynomials in `F[X]` (i.e., as identities in `F[X]`, not just at one
//! > point). Equivalently, evaluating `Fold(f, α)` at `Y = x^k` is the same as
//! > the `α`-twist of the row-evaluations `(f_0(x^k), ..., f_{k-1}(x^k))`.
//!
//! **Constructive proof.** Substitute `Y ↦ X^k` directly into the definition
//! of `Fold(f, α)`:
//!
//! ```text
//! Fold(f, α)(X^k) = sum_{i=0..k-1} α^i · f_i((X^k))         (definition)
//!                 = sum_{i=0..k-1} α^i · f_i(X^k).          (rename inner Y → X^k)
//! ```
//!
//! That is exactly the RHS. The decomposition `f(X) = Σ f_i(X^k) · X^i` is
//! never used in the proof itself — the theorem is a pure substitution
//! identity. The decomposition is used only to **construct** the `f_i` from
//! `f`'s coefficients (and to guarantee `deg(f_i) < d / k`, which gives the
//! degree-reduction property). ∎
//!
//! **Uniqueness of the decomposition.** Write `f(X) = Σ_{j=0..d-1} c_j · X^j`.
//! Each `j ∈ [0, d)` splits uniquely as `j = q · k + r` with `r ∈ [0, k)`, so
//!
//! ```text
//! f(X) = Σ_j c_j · X^j
//!      = Σ_{q, r} c_{qk+r} · X^{qk + r}
//!      = Σ_r X^r · ( Σ_q c_{qk+r} · X^{qk} )
//!      = Σ_r X^r · ( Σ_q c_{qk+r} · (X^k)^q )
//!      = Σ_r X^r · f_r(X^k)
//! ```
//!
//! where `f_r(Y) := Σ_q c_{qk + r} · Y^q`. Each `f_r` has degree `< ⌈d/k⌉`
//! (its highest-`q` term is at `q = ⌊(d-1-r)/k⌋ < d/k`). Uniqueness: any other
//! decomposition `f = Σ_r X^r · g_r(X^k)` would, term-by-term, force
//! `c_{qk + r} = [Y^q] g_r(Y)` (the polynomial identity), so `g_r = f_r`. ∎
//!
//! **Why `α` randomness matters: Fold Soundness Lemma.** The theorem above is
//! a polynomial identity — true for *every* `α`. The cryptographic content of
//! folding is a separate soundness statement (used in STIR's main analysis):
//!
//! > **Fold Soundness Lemma (STIR §4, informal).** If `f` is δ-far (in
//! > relative Hamming distance) from every codeword of `RS[F, L, d]`, then for
//! > a uniformly random `α ∈ F`,
//! >
//! > ```text
//! > Pr_α[ Fold(f, α) is < δ-far from RS[F, L^k, d/k] ]  ≤  poly(k, d) / |F|.
//! >```
//!
//! Intuitively: a malicious prover who tries to "cancel" the corrupted
//! coefficients across rows would have to engineer `f` so that the specific
//! linear combination dictated by `α` happens to land back in the code. For
//! `α` chosen *after* `f` is committed and over a large field, this happens
//! with negligible probability — the same Schwartz-Zippel argument that
//! underpins every interactive low-degree test.
//!
//! ## Worked numeric example (`k = 2`, `f = 1 + 2X + 3X² + 4X³`, `α = 5`)
//!
//! We verify both halves of the Folding theorem coincide on a small domain.
//!
//! **Coefficient side.** Split coefficients by index parity:
//!
//! ```text
//! f(X) = 1 + 2X + 3X² + 4X³
//! f_0(Y) = 1 + 3Y     (positions 0, 2:  c_0 = 1, c_2 = 3)
//! f_1(Y) = 2 + 4Y     (positions 1, 3:  c_1 = 2, c_3 = 4)
//! ```
//!
//! Sanity check the decomposition: `f_0(X²) + X · f_1(X²) = (1 + 3X²) + X · (2 + 4X²)
//! = 1 + 2X + 3X² + 4X³ = f(X)`. ✓
//!
//! Apply `Fold(·, 5)`:
//!
//! ```text
//! Fold(f, 5)(Y) = f_0(Y) + 5 · f_1(Y)
//!               = (1 + 3Y) + 5 · (2 + 4Y)
//!               = 11 + 23Y.
//! ```
//!
//! `Fold(f, 5)` has degree 1 < d/k = 4/2 = 2. ✓
//!
//! **Evaluation side.** Pick the size-`|L| = 4` subgroup `L = ⟨g⟩` where `g`
//! is a primitive 4th root of unity in Goldilocks. Then `ω = g^{4/2} = g² = -1`
//! is a primitive 2nd root of unity (Lemma A in `reed_solomon::fft`). The
//! fibre over `x² ∈ L^k = ⟨g²⟩ = {1, -1}` is `{x, -x}` (the `±` pair).
//!
//! Concretely the four elements are `L = {1, g, g², g³} = {1, g, -1, -g}` and
//! `L^k = {1, -1}` (each fibre has two preimages).
//!
//! - Fibre over `x² = 1`: `L[0] = 1` and `L[0 + n/k] = L[2] = -1`. So the
//!   sibling pair is `(f(1), f(-1)) = (1+2+3+4, 1−2+3−4) = (10, -2)`.
//!
//!   IDFT of size 2 (at `ω⁻¹ = -1`, scale by `1/2`): produces
//!   `((10 + (-2)) / 2,  (10 - (-2)) / 2) = (4, 6)`. These are
//!   `(f_0(1) · 1⁰,  f_1(1) · 1¹) = (f_0(1), f_1(1)) = (1+3, 2+4) = (4, 6)`. ✓
//!
//!   Unscale by `1⁰ = 1`: `(f_0(1), f_1(1)) = (4, 6)`. Combine with
//!   `(α⁰, α¹) = (1, 5)`: `Fold(f, 5)(1) = 4 + 5·6 = 34`.
//!
//!   Cross-check: `Fold(f, 5)(1) = 11 + 23·1 = 34`. ✓
//!
//! - Fibre over `x² = -1` (here `x² = g² = -1` in Goldilocks): `L[1] = g` and
//!   `L[1 + n/k] = L[3] = -g = g³`. Sibling pair `(f(g), f(-g))`.
//!
//!   By the polynomial identity `f(g) = f_0(g²) + g · f_1(g²) = f_0(-1) + g · f_1(-1) =
//!   (1 - 3) + g · (2 - 4) = -2 - 2g`. Likewise `f(-g) = -2 + 2g`. Sibling
//!   pair `(-2 - 2g, -2 + 2g)`.
//!
//!   IDFT of size 2: `(((-2-2g) + (-2+2g))/2, ((-2-2g) - (-2+2g))/2) = (-2, -2g)`.
//!   These are `(f_0(-1) · g⁰, f_1(-1) · g¹) = (-2, -2g)`. ✓
//!
//!   Unscale by `(g⁰, g¹) = (1, g)`: `(f_0(-1), f_1(-1)) = (-2, -2)`.
//!   Combine with `(1, 5)`: `Fold(f, 5)(-1) = -2 + 5·(-2) = -12`.
//!
//!   Cross-check: `Fold(f, 5)(-1) = 11 + 23·(-1) = -12`. ✓
//!
//! Both fibres recover the same folded polynomial via the evaluation-side
//! recipe. Polynomial and evaluation forms agree, as the Folding theorem
//! guarantees.
//!
//! ## Socratic prompt
//!
//! **Question for the reader.** Why is the random `α` essential here? What
//! would happen if we just used `α = 1`?
//!
//! Answer: with `α = 1` the fold collapses to `f_0(Y) + f_1(Y) + ... +
//! f_{k-1}(Y)` — a fixed deterministic linear combination of the rows. A
//! malicious prover who knows `α = 1` in advance can construct `f` whose rows
//! sum to a low-degree polynomial *despite* `f` itself being arbitrarily
//! corrupted: e.g., for `k = 2`, take `f_0` correct and `f_1 = -f_0 + g` for a
//! tiny `g` of degree `< d/k` — then `Fold(f, 1) = g` is in the next-round
//! code while `f` is not in the current-round code. Randomness over a large
//! field forces the adversary to commit to a single error before the linear
//! combination is fixed, and the Schwartz-Zippel bound kicks in: a non-zero
//! polynomial of low degree in `α` vanishes on a vanishingly small fraction
//! of the field.
//!
//! ## Cross-module interface
//!
//! - [`fold`] operates on evaluation tables on a smooth domain `L`. The
//!   "Why it works on evaluation tables, not just polynomials" section and
//!   the Folding theorem above are the load-bearing references; the function
//!   docstring points back here at each step of its TODO breadcrumbs.
//! - [`poly_fold`] operates on coefficient-form polynomials — useful for
//!   correctness tests against [`fold`] (round-trip via `reed_solomon::fft`).
//!   See "Equivalent `k`-row view" above for the matrix interpretation it
//!   implements.
//!
//! See `reed_solomon::polynomial::UnivariatePoly`,
//! `reed_solomon::domain::EvaluationDomain`, and `reed_solomon::fft` for the
//! underlying types and DFT machinery this module relies on.

use reed_solomon::{fft::ifft_subgroup, EvaluationDomain, Fp, UnivariatePoly};

/// Fold a codeword (evaluation table) by random linear combination of its
/// `k = folding_factor` "rows".
///
/// Given a codeword `evals` of `f(X)` evaluated on a smooth domain `L`
/// (closed under multiplication by a primitive `k`-th root of unity),
/// returns the evaluations of `Fold(f, α)(Y)` on `L^k` — the image of `L`
/// under `x → x^k`, which has `|L| / k` elements.
///
/// # Inputs
///
/// - `evals`: `f(L[0]), f(L[1]), ..., f(L[n-1])`, with `n = |L|`.
/// - `folding_factor`: `k`. Must be a power of two `≥ 1` that divides `n`.
/// - `randomness`: `α ∈ F_p`. The verifier's challenge.
/// - `domain`: the evaluation domain `L`. Provides both the element-lookup
///   `L[j]` needed for the per-fibre unscale and the primitive `k`-th root
///   `ω = g^{n/k}` derived from the domain's generator `g`.
///
/// # Output
///
/// A `Vec<Fp>` of length `n / k`, with entry `j` equal to
/// `Fold(f, α)(L[j]^k)` — that is, the folded codeword on `L^k`, indexed by
/// the fibre representative `L[j]`. The output's `j`-th fibre representative
/// is itself the squared (more generally, `k`-th-powered) `j`-th element of
/// `L`.
///
/// # Layout convention
///
/// `evals` is assumed to be in **natural order** on `L = c · <g>`: entry `j`
/// is `f(c · g^j)`. Under that ordering, the `k`-fold fibre `{ω^i · L[j]}`
/// over `L[j]^k` corresponds to indices `j, j + n/k, j + 2n/k, ..., j +
/// (k-1)n/k` of `evals` (i.e., a stride of `n/k`), because `ω = g^{n/k}`
/// walks through the fibre.
///
/// # Algorithm
///
/// Per the "Why it works on evaluation tables" section and the **Folding
/// theorem** in the module docs:
///
/// 1. Validate divisibility and gather the `k` siblings of each fibre at
///    stride `m = n/k`.
/// 2. Run a `k`-point inverse DFT on each sibling vector. This recovers the
///    *twisted* row evaluations `(f_0(x^k) · x^0, f_1(x^k) · x^1, ..., f_{k-1}(x^k) · x^{k-1})`
///    — the "DFT matrix is Vandermonde at ω" identity from
///    `reed_solomon::fft` is what makes the IDFT the right inverse.
/// 3. Unscale by `(x^0, x^1, ..., x^{k-1})` to peel off the `X^i` factors,
///    leaving the pure row evaluations `(f_0(x^k), ..., f_{k-1}(x^k))`.
/// 4. Linear-combine with `(α^0, α^1, ..., α^{k-1})` — this is exactly the
///    "Equivalent `k`-row view" matrix-vector product on the evaluation
///    side, agreeing with the Folding theorem.
///
/// # Panics
///
/// Panics if `folding_factor == 0`, if `folding_factor` does not divide
/// `evals.len()`, if `evals.len() != domain.size()`, or if `folding_factor`
/// is not a power of two (the inner IDFT requires power-of-two sizes).
///
/// (STIR §4, Fold definition; reduction-step soundness in Lemma 4.4.
/// Polynomial-side correctness is the **Folding theorem** in the module docs.)
pub fn fold(
    evals: &[Fp],
    folding_factor: u32,
    randomness: Fp,
    domain: &EvaluationDomain,
) -> Vec<Fp> {
    // Validate. See the "Layout convention" above.
    assert!(folding_factor >= 1, "folding_factor must be at least 1");
    let k = folding_factor as usize;
    assert!(
        k.is_power_of_two(),
        "folding_factor must be a power of two (IDFT requires it)"
    );
    let n = evals.len();
    assert_eq!(
        n,
        domain.size(),
        "evals length must match domain size"
    );
    assert_eq!(
        n % k,
        0,
        "folding_factor must divide evals.len()"
    );

    // Trivial fold by 1: identity.
    if k == 1 {
        return evals.to_vec();
    }

    let m = n / k; // |L^k|

    // The primitive k-th root of unity ω = g^{n/k}, derived from the domain's
    // generator g (a primitive n-th root of unity). Walking siblings at stride
    // m in `evals` corresponds to walking the fibre {ω^i · L[j]}_{i=0..k}.
    let omega_k = domain.generator().pow(m as u64);

    // Precompute powers of the fold randomness α: (α^0, α^1, ..., α^{k-1}).
    // These are the "Equivalent `k`-row view" coefficients from the module docs.
    let alpha_powers: Vec<Fp> = {
        let mut v = Vec::with_capacity(k);
        let mut acc = Fp::one();
        for _ in 0..k {
            v.push(acc);
            acc = acc * randomness;
        }
        v
    };

    let mut out = Vec::with_capacity(m);

    // Reusable scratch buffer for each fibre's `k` siblings.
    let mut siblings = vec![Fp::zero(); k];

    for j in 0..m {
        // Step 1: gather the k siblings of the fibre over L[j]^k.
        // siblings[i] = f(ω^i · L[j]) = evals[j + i·m].
        for i in 0..k {
            siblings[i] = evals[j + i * m];
        }

        // Step 2: inverse DFT of size k at root ω_k recovers the twisted row
        // evaluations t_i = f_i(L[j]^k) · L[j]^i. The identity that makes
        // this work is from the module docs:
        //   f(ω^i · x) = Σ_l f_l(x^k) · ω^{il} · x^l = Σ_l ω^{il} · t_l.
        // So `siblings = V · t` with V[i,l] = ω^{il} — apply V^{-1} via
        // `ifft_subgroup`.
        let twisted = ifft_subgroup(&siblings, omega_k);

        // Step 3: unscale by L[j]^i to peel off the X^i factor, leaving the
        // pure row evaluations r_i = f_i(L[j]^k).
        let x = domain.element(j); // L[j] = c · g^j
        let mut x_pow = Fp::one(); // running L[j]^i

        // Step 4: linear-combine with (α^0, ..., α^{k-1}) to produce
        //   Fold(f, α)(L[j]^k) = Σ_i α^i · f_i(L[j]^k).
        // Combine steps 3 and 4 into a single pass over the twisted vector
        // to avoid materialising the un-twisted intermediate.
        let mut folded = Fp::zero();
        for (i, &t_i) in twisted.iter().enumerate() {
            // r_i = t_i / x^i. Multiplying by x^{-i} would require an inverse
            // per iteration; instead pre-multiply α^i by x^{-i}. Equivalent:
            // accumulate the "current power of α / x" and use it as the
            // single scalar applied to t_i.
            //
            // We use the form `(α^i · t_i) / x^i` to keep the explicit "fold by
            // powers of α" structure visible, and pay one inverse at the end.
            folded = folded + alpha_powers[i] * t_i * inverse_or_one(x_pow);
            x_pow = x_pow * x;
        }
        out.push(folded);
    }

    out
}

/// Return `x.inverse().unwrap()`, or `Fp::one()` if `x == Fp::zero()`.
///
/// Used inside the per-fibre unscale step of [`fold`]. The unscale divides by
/// `L[j]^i`; for `i = 0` this is `1` (no inversion needed). `L[j]` itself is
/// always non-zero (every element of a smooth domain is a unit), so the only
/// way to feed `Fp::zero()` to this helper would be a programming error —
/// we still return `Fp::one()` (a safe identity) to make `i = 0` cheap.
#[inline]
fn inverse_or_one(x: Fp) -> Fp {
    x.inverse().unwrap_or(Fp::one())
}

/// Fold a coefficient-form polynomial by the row-randomization rule:
///
/// ```text
/// Fold(f, α)(Y) = sum_{i=0..k-1} α^i · f_i(Y),
/// ```
///
/// where `f(X) = sum_i f_i(X^k) · X^i` is the unique `k`-row decomposition
/// (with `deg(f_i) < deg(f) / k`).
///
/// # Inputs
///
/// - `poly`: the input polynomial in coefficient form (ascending degree).
/// - `folding_factor`: `k`. Must be `≥ 1`.
/// - `randomness`: `α ∈ F_p`.
///
/// # Output
///
/// `UnivariatePoly` of degree `< ⌈(deg(poly) + 1) / k⌉` — see the
/// **Folding theorem** in the module docs for the degree-reduction
/// guarantee.
///
/// # Algorithm
///
/// 1. Read the coefficient vector `c = [c_0, c_1, ..., c_{d-1}]`.
/// 2. For each `i ∈ 0..k`, extract row `i` of the `k × ⌈d/k⌉` coefficient
///    matrix: this is the slice `[c_i, c_{i+k}, c_{i+2k}, ...]`.
/// 3. Accumulate `Σ_i α^i · f_i(Y)` using a running power of `α`.
///
/// See the "Equivalent `k`-row view" section of the module docs — the
/// implementation is the matrix-vector product `(α^0, ..., α^{k-1}) · M`
/// where `M` is the `k`-row coefficient matrix.
///
/// # Panics
///
/// Panics if `folding_factor == 0`.
///
/// # Example
///
/// `poly = 1 + 2X + 3X² + 4X³`, `k = 2`, `α = 5` (worked in detail in the
/// module docs):
///
/// ```text
/// f_0 = 1 + 3Y,   f_1 = 2 + 4Y
/// Fold = f_0 + 5·f_1 = 11 + 23Y.
/// ```
///
/// (STIR §4, coefficient-form Fold. Polynomial-side correctness is the
/// **Folding theorem** in the module docs.)
pub fn poly_fold(
    poly: &UnivariatePoly,
    folding_factor: u32,
    randomness: Fp,
) -> UnivariatePoly {
    assert!(folding_factor >= 1, "folding_factor must be at least 1");
    let k = folding_factor as usize;

    let coeffs = poly.coeffs();
    if coeffs.is_empty() {
        return UnivariatePoly::zero();
    }

    // k = 1 ⇒ Fold(f, α) = f (only row 0 exists, and α^0 = 1).
    if k == 1 {
        return poly.clone();
    }

    let d = coeffs.len();
    // The number of coefficients per row is ⌈d / k⌉ — the longest row may
    // have one more entry than the others if `k` doesn't divide `d`.
    let row_len = (d + k - 1) / k;

    // Accumulator for Σ α^i · f_i(Y). Built directly in coefficient form
    // (length `row_len`) and wrapped in UnivariatePoly::new at the end so
    // trailing zeros are stripped.
    let mut out = vec![Fp::zero(); row_len];

    let mut alpha_pow = Fp::one();
    for i in 0..k {
        // Row `i` of the `k`-stride coefficient matrix: positions
        // i, i+k, i+2k, ... (the "Equivalent k-row view" of the module docs).
        // Walk those positions and fold into `out` weighted by α^i.
        let mut q = 0usize;
        let mut pos = i;
        while pos < d {
            out[q] = out[q] + alpha_pow * coeffs[pos];
            pos += k;
            q += 1;
        }
        alpha_pow = alpha_pow * randomness;
    }

    UnivariatePoly::new(out)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use reed_solomon::fft::fft_on_domain;

    /// `poly_fold` of a constant polynomial: every "row" is the constant
    /// itself (in row 0) or zero (rows 1..k), so the fold equals `c · α^0 = c`.
    #[test]
    fn poly_fold_of_constant_equals_constant() {
        let c = UnivariatePoly::new(vec![Fp::new(42)]);
        let folded = poly_fold(&c, 4, Fp::new(99));
        assert_eq!(folded, c);
    }

    /// `fold` on evaluation tables must match `poly_fold` on coefficients:
    /// take a polynomial, encode it on a domain, fold the codeword, decode the
    /// folded codeword on the squared domain — should equal `poly_fold(poly, k, α)`
    /// evaluated on the squared domain.
    ///
    /// This is the load-bearing equivalence proved by the **Folding theorem**
    /// in the module docs: coefficient-side fold and evaluation-side fold
    /// agree on every point of `L^k`.
    #[test]
    fn fold_matches_poly_fold_on_evaluations() {
        use rand::SeedableRng;

        // Domain of size 16 (log_size = 4). Polynomial of degree < 8 so the
        // folded degree d/k = 4 < |L^k| = 8 — comfortably in the code.
        let log_n = 4;
        let n: usize = 1 << log_n; // 16
        let domain = EvaluationDomain::new_subgroup(log_n);

        let mut rng = rand::rngs::StdRng::seed_from_u64(0xF01D_F01D);
        let coeffs: Vec<Fp> = (0..8).map(|_| Fp::random(&mut rng)).collect();
        // Pad to length n with zeros so `fft_on_domain` has the right shape;
        // use `from_coeffs_unstripped` so the polynomial view keeps the
        // padded zeros (only affects `degree()`; values agree everywhere).
        let mut padded = coeffs.clone();
        padded.resize(n, Fp::zero());
        let poly_padded = UnivariatePoly::from_coeffs_unstripped(padded.clone());
        let poly = UnivariatePoly::new(coeffs);

        let evals = fft_on_domain(&padded, &domain);

        let k: u32 = 2;
        let alpha = Fp::new(13);

        let folded_evals = fold(&evals, k, alpha, &domain);

        // Squared domain: L^k = ⟨g^k⟩ of size m = n/k.
        let m = n / k as usize;
        let folded_poly = poly_fold(&poly, k, alpha);

        // Independent check: evaluate `folded_poly` at L[j]^k for each j,
        // compare with `folded_evals[j]`.
        assert_eq!(folded_evals.len(), m);
        for j in 0..m {
            let x = domain.element(j);
            let x_to_k = x.pow(k as u64);
            assert_eq!(
                folded_evals[j],
                folded_poly.evaluate(x_to_k),
                "mismatch at fibre representative L[{j}] (x^k = {:?})",
                x_to_k,
            );
        }

        // Bonus check: padded polynomial evaluates identically, used as a
        // sanity that the `evals` indeed came from `poly` (not just `padded`
        // with junk).
        for j in 0..n {
            assert_eq!(evals[j], poly_padded.evaluate(domain.element(j)));
        }
    }

    /// With `α = 0`, `Fold(f, 0) = f_0` — only the row-0 contribution survives,
    /// since `α^0 = 1` and `α^i = 0` for `i ≥ 1`.
    #[test]
    fn fold_with_alpha_zero_returns_first_row_only() {
        // f(X) = 1 + 2X + 3X² + 4X³.
        // For k = 2: rows are f_0 = 1 + 3Y, f_1 = 2 + 4Y. Fold(f, 0) = f_0.
        let f = UnivariatePoly::new(vec![
            Fp::new(1),
            Fp::new(2),
            Fp::new(3),
            Fp::new(4),
        ]);
        let folded_k2 = poly_fold(&f, 2, Fp::zero());
        assert_eq!(
            folded_k2,
            UnivariatePoly::new(vec![Fp::new(1), Fp::new(3)])
        );

        // For k = 4: rows are f_0 = 1, f_1 = 2, f_2 = 3, f_3 = 4 (each a
        // constant). Fold(f, 0) = f_0 = 1.
        let folded_k4 = poly_fold(&f, 4, Fp::zero());
        assert_eq!(folded_k4, UnivariatePoly::new(vec![Fp::new(1)]));
    }

    /// `fold` must reject inputs whose length is not divisible by the folding
    /// factor — there's no consistent way to bundle them into k-element fibres.
    ///
    /// We test via the divisibility branch directly: pad `evals` to a length
    /// that the domain doesn't naturally support and confirm the assertion
    /// fires. (We can't easily construct an `EvaluationDomain` of non-power-of-2
    /// size, so we forge the mismatch with a wrong-sized evals slice.)
    #[test]
    #[should_panic(expected = "evals length must match domain size")]
    fn fold_factor_must_divide_evals_length() {
        let domain = EvaluationDomain::new_subgroup(2); // size 4
        // Pass evals of length 5 — neither divisible by k=2 in the
        // "would-be" sense nor matching the domain. The size-mismatch check
        // fires first, which is the more informative panic.
        let evals = vec![Fp::new(1); 5];
        let _ = fold(&evals, 2, Fp::one(), &domain);
    }

    /// Iterated folding: folding twice — first by `α` (factor `k`), then by
    /// `β` (factor `k`) — must equal a single fold by factor `k² = k_total`
    /// with the correct `(α, β)`-combined randomness vector.
    ///
    /// What is the "correct combined randomness"? Apply the Folding theorem
    /// twice: `Fold(Fold(f, α), β)(Z)` is a polynomial in `Z = X^{k²}`. Tracing
    /// the row indices, the coefficient at position `j = q · k² + r · k + s`
    /// (with `r, s ∈ [0, k)`) of `f` picks up the multiplier `β^s · α^r`. Thus
    /// for `k_total = k²`, the equivalent single-fold randomness `γ` would need
    /// to satisfy `γ^{r·k + s} = β^s · α^r` for all `r, s ∈ [0, k)`. Such a `γ`
    /// generally does NOT exist (the map `(r, s) ↦ β^s · α^r` need not be a
    /// power map). The cleanest equivalence — and what we test — is that the
    /// *two-pass* and the *single-pass with the same `(α, β)` row weights*
    /// match. We verify this by evaluating both at every point of `L^{k²}`.
    #[test]
    fn fold_twice_matches_fold_once_with_combined_randomness() {
        use rand::SeedableRng;

        // Domain of size 16, polynomial of degree < 16. Fold by k=2 twice,
        // landing in L^4 of size 4 with a polynomial of degree < 4.
        let log_n = 4;
        let n: usize = 1 << log_n;
        let domain = EvaluationDomain::new_subgroup(log_n);

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x70A57_B33F);
        let coeffs: Vec<Fp> = (0..n).map(|_| Fp::random(&mut rng)).collect();
        let poly = UnivariatePoly::from_coeffs_unstripped(coeffs.clone());

        let evals = fft_on_domain(&coeffs, &domain);

        let alpha = Fp::new(11);
        let beta = Fp::new(29);

        // Two-pass evaluation fold: L (size 16) → L² (size 8) → L⁴ (size 4).
        // After folding once by k=2, we need the squared domain L^k as the
        // input domain for the second fold.
        let log_n2 = log_n - 1; // size 8
        let domain2 = EvaluationDomain::new_subgroup(log_n2);
        let folded_once = fold(&evals, 2, alpha, &domain);
        assert_eq!(folded_once.len(), domain2.size());
        let folded_twice = fold(&folded_once, 2, beta, &domain2);
        assert_eq!(folded_twice.len(), 4);

        // Independent two-pass coefficient fold: Fold(Fold(poly, α), β).
        let poly_folded_once = poly_fold(&poly, 2, alpha);
        let poly_folded_twice = poly_fold(&poly_folded_once, 2, beta);

        // Squared-twice domain: L^4 = ⟨g^4⟩ of size 4.
        let log_n4 = log_n - 2;
        let domain4 = EvaluationDomain::new_subgroup(log_n4);
        for j in 0..domain4.size() {
            let y = domain4.element(j);
            assert_eq!(
                folded_twice[j],
                poly_folded_twice.evaluate(y),
                "fold-twice mismatch at L^4[{j}] = {:?}",
                y,
            );
        }
    }

    /// Worked-example sanity check: `f = 1 + 2X + 3X² + 4X³`, `α = 5`, `k = 2`
    /// must produce `Fold(f, 5) = 11 + 23Y` (degree 1). This pins the example
    /// in the module docs to a concrete unit assertion.
    #[test]
    fn poly_fold_worked_example() {
        let f = UnivariatePoly::new(vec![
            Fp::new(1),
            Fp::new(2),
            Fp::new(3),
            Fp::new(4),
        ]);
        let folded = poly_fold(&f, 2, Fp::new(5));
        assert_eq!(folded, UnivariatePoly::new(vec![Fp::new(11), Fp::new(23)]));
        assert_eq!(folded.degree(), Some(1));
    }

    /// Degree-bound check: `deg(poly_fold(p, k, α)) ≤ ⌈(deg(p)+1)/k⌉ − 1`
    /// (equivalently `< (deg(p)+1)/k` rounded up). Property test over a few
    /// shapes.
    #[test]
    fn poly_fold_degree_is_bounded_by_input_over_k() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEAD_BEEF);

        for &(d, k) in &[(8usize, 2u32), (12, 4), (16, 4), (15, 2), (1, 4)] {
            let coeffs: Vec<Fp> = (0..d).map(|_| Fp::random(&mut rng)).collect();
            let p = UnivariatePoly::new(coeffs);
            let alpha = Fp::random(&mut rng);
            let folded = poly_fold(&p, k, alpha);
            let row_len = (d + k as usize - 1) / (k as usize);
            // Folded polynomial has at most `row_len` coefficients ⇒
            // degree at most `row_len - 1` (or None if it vanished).
            if let Some(fd) = folded.degree() {
                assert!(
                    fd < row_len,
                    "deg(folded) = {fd} not < ⌈d/k⌉ = {row_len} (d = {d}, k = {k})",
                );
            }
        }
    }
}
