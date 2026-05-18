//! Cooley-Tukey radix-2 FFT — **a.k.a. the Number Theoretic Transform (NTT)**.
//!
//! ## What problem does the FFT solve?
//!
//! Suppose you have a polynomial in **coefficient form**:
//!
//! ```text
//! p(X) = a_0 + a_1·X + a_2·X² + ... + a_{n-1}·X^{n-1}
//! ```
//!
//! and you want its **values at `n` specific points** — its evaluation form:
//!
//! ```text
//! [p(x_0), p(x_1), p(x_2), ..., p(x_{n-1})]
//! ```
//!
//! **Two reasons this matters.** For Reed-Solomon, the `n` outputs *are*
//! the codeword — "fast polynomial evaluation on `n` points" IS "fast RS
//! encoding". Beyond RS, evaluation form is interesting in its own right:
//! **polynomial multiplication is `O(n)` pointwise** in evaluation form
//! (multiply two evaluation vectors entry-by-entry) vs `O(n²)` in
//! coefficient form (convolve the two coefficient vectors). The catch —
//! and the historical motivation for the FFT — is that converting
//! between forms is itself `O(n²)` naively, wiping out the multiplication
//! speedup. The FFT fixes the conversion.
//!
//! There's also a uniqueness guarantee underneath all of this:
//!
//! > **Polynomial Interpolation Theorem (a.k.a. Lagrange's theorem on
//! > polynomials).** Given `n` distinct points `x_0, ..., x_{n-1}` and
//! > any `n` values `y_0, ..., y_{n-1}`, there is **exactly one**
//! > polynomial of degree `< n` such that `p(x_i) = y_i` for all `i`.
//!
//! So the coefficient ↔ evaluation form conversion is a genuine bijection
//! on degree-`< n` polynomials, not a lossy encoding. Forward FFT goes one
//! way, inverse FFT goes the other, and they compose to the identity.
//!
//! **Naive approach.** Use Horner's rule at each point. Each evaluation
//! costs `O(n)` field operations; doing it `n` times costs `O(n²)` total.
//! Encoding a polynomial of degree `10^6` would take `~10^12` field
//! operations — prohibitive.
//!
//! **FFT approach.** The same job in `O(n log n)`. For `n = 10^6` that's
//! `~2 · 10^7` operations — roughly a **50,000× speedup** over naive.
//! That speedup is *the* reason production STARKs are feasible at all.
//!
//! ## Sanity check: roots of unity in `F_17`
//!
//! Before the abstract Goldilocks machinery, let's see roots of unity in
//! a tiny field a reader can verify by hand. Take `p = 17`, so
//! `|F_17^*| = 16 = 2⁴` (plenty of FFT-friendly structure). Pick `ω = 13`
//! and compute its powers mod 17.
//!
//! (We pick `ω = 13` here — a different primitive 4th root of unity than
//! `ω = 4` used in [`domain`](crate::domain). Both generate the same
//! order-4 subgroup `{1, 4, 13, 16}`. Concretely, `13 = 4³ mod 17` (check:
//! `4² = 16`, `4³ = 64 = 3·17 + 13 = 13`), so `ω = 13` traverses the
//! subgroup in reverse cyclic order to `ω = 4`. The choice is convenient
//! for the round-trip walk-through below.)
//!
//! ```text
//! ω¹ = 13
//! ω² = 13 · 13 = 169 = 10·17 - 1   ≡ -1 ≡ 16   (mod 17)
//! ω³ = 13 · 16 = 208 = 12·17 +  4  ≡  4        (mod 17)
//! ω⁴ = 13 ·  4 =  52 =  3·17 +  1  ≡  1        (mod 17)   ← back to start
//! ```
//!
//! `ω = 13` has order exactly 4 — a **primitive 4th root of unity** in
//! `F_17`. It generates the subgroup `⟨ω⟩ = {1, 13, 16, 4}` of order 4
//! inside `F_17^*`.
//!
//! Two facts that will recur throughout this file:
//!
//! - **`ω^{n/2} = ω² = 16 = -1`** (since `-1 ≡ 16 mod 17`). The pairing
//!   identity `ω^{n/2} = -1` holds — proved in general below.
//! - **The four roots split into `±` pairs:** `(1, 16) = (1, -1)` and
//!   `(13, 4) = (ω, -ω)`. Second half of the domain is the first half
//!   negated.
//!
//! Keep `{1, 13, 16, 4} ⊂ F_17` in mind as the running concrete example.
//! We'll come back to it when introducing the Vandermonde matrix and the
//! inverse FFT.
//!
//! ## The DFT matrix is Vandermonde
//!
//! The FFT (and its slower naive cousin) is, at its core, **a single
//! matrix-vector product**. The matrix has a name — it's a **Vandermonde
//! matrix** — and noticing this up front makes everything that follows
//! cleaner.
//!
//! ### Vandermonde matrix (general)
//!
//! For points `x_0, x_1, ..., x_{n-1}` in `F_p`, the **Vandermonde matrix**
//! is the `n × n` matrix whose `i`-th row is the powers `0, 1, ..., n-1`
//! of `x_i`:
//!
//! ```text
//! V[i, j] = x_i^j     for 0 ≤ i, j < n.
//! ```
//!
//! Why this particular matrix? Evaluating a polynomial
//! `p(X) = a_0 + a_1·X + ... + a_{n-1}·X^{n-1}` at the point `x_i` is the
//! **dot product of row `i` of `V` with the coefficient vector**:
//!
//! ```text
//! p(x_i) = a_0·x_i^0 + a_1·x_i^1 + ... + a_{n-1}·x_i^{n-1}
//!        = V[i, *] · [a_0, a_1, ..., a_{n-1}]ᵀ.
//! ```
//!
//! Stacking all `n` evaluations into one matrix-vector product:
//!
//! ```text
//! ⎡  p(x_0)   ⎤   ⎡ x_0^0     x_0^1     ...  x_0^{n-1}     ⎤   ⎡ a_0     ⎤
//! ⎢  p(x_1)   ⎥ = ⎢ x_1^0     x_1^1     ...  x_1^{n-1}     ⎥ · ⎢ a_1     ⎥
//! ⎢    ⋮      ⎥   ⎢   ⋮                                    ⎥   ⎢  ⋮      ⎥
//! ⎣ p(x_{n-1})⎦   ⎣ x_{n-1}^0 x_{n-1}^1 ...  x_{n-1}^{n-1} ⎦   ⎣ a_{n-1} ⎦
//! ```
//!
//! That's `evals = V · coeffs`. **Polynomial evaluation at `n` points is
//! exactly left-multiplication by a Vandermonde matrix.**
//!
//! ### The DFT matrix
//!
//! Specialize to `x_i = ω^i` where `ω` is a primitive `n`-th root of
//! unity (the FFT's "smart choice of evaluation points"). Each entry
//! becomes
//!
//! ```text
//! V[i, j] = (ω^i)^j = ω^{ij}.
//! ```
//!
//! This `n × n` matrix is the **DFT matrix at `ω`**. It's the protagonist
//! of the rest of this file:
//!
//! - Forward FFT = `V · coeffs`.
//! - Inverse FFT = `V⁻¹ · evals` (and `V⁻¹` turns out to be another
//!   Vandermonde matrix — see the Inverse FFT section).
//! - Cooley-Tukey = a particular `O(n log n)` way of computing
//!   `V · coeffs` by exploiting `V`'s structure (specifically, that
//!   `ω^{n/2} = -1`).
//!
//! ### Concrete DFT matrix in `F_17` (n = 4)
//!
//! Using `ω = 13` from the sanity check above:
//!
//! ```text
//!         j=0   j=1   j=2   j=3
//! i=0  [   1     1     1     1  ]    ← powers of ω⁰ = 1
//! i=1  [   1    13    16     4  ]    ← powers of ω¹ = 13
//! i=2  [   1    16     1    16  ]    ← powers of ω² = 16 = -1
//! i=3  [   1     4    16    13  ]    ← powers of ω³ = 4 = -ω
//! ```
//!
//! Verify row `i = 2` by hand: the entries are
//! `(ω²)⁰, (ω²)¹, (ω²)², (ω²)³ = 1, ω², ω⁴, ω⁶ = 1, 16, 1, 16` (using
//! `ω⁴ = 1` and `ω⁶ = ω² = 16`). ✓
//!
//! Notice the matrix is **symmetric** in this small example, because
//! `ω^{ij} = ω^{ji}` — that's a general property of the DFT matrix:
//! `V[i, j] = V[j, i]`. So `V = Vᵀ` always.
//!
//! A forward FFT of `coeffs = [a_0, a_1, a_2, a_3]` over this domain is
//! literally `V · coeffs` where `V` is the matrix above. The naive way
//! takes `O(n²) = 16` multiplies. The Cooley-Tukey algorithm we'll
//! derive below does it in `O(n log n) = 8`. For `n = 4` that's a small
//! win; for `n = 10⁶` it's the speedup that makes STARKs possible.
//!
//! ## Why is the FFT faster? The trick in one sentence
//!
//! The evaluation points aren't arbitrary — they're chosen to be the `n`
//! powers of a primitive `n`-th root of unity `ω`:
//!
//! ```text
//! x_i = ω^i,    so the n points are  [1, ω, ω², ..., ω^{n-1}].
//! ```
//!
//! Because `ω^{n/2} = -1` (proved just below), the points pair up as `±`
//! opposites:
//!
//! ```text
//! x_{i + n/2} = ω^{i + n/2} = ω^i · ω^{n/2} = ω^i · (-1) = -x_i.
//! ```
//!
//! That symmetry lets us compute `p(x_i)` and `p(x_{i + n/2})` **together**,
//! sharing the squared substitution `x_i² = x_{i + n/2}²` between two
//! sub-evaluations. Recursing on this halving gives `O(n log n)` total.
//! The precise math is in "The recursion" below.
//!
//! ### Why `ω^{n/2} = -1`
//!
//! Two facts combine:
//!
//! **Fact 1.** `ω^{n/2}` has order *exactly* 2.
//! - `(ω^{n/2})² = ω^n = 1`, so the order of `ω^{n/2}` divides 2.
//! - `ω^{n/2} ≠ 1`, else `ω` itself would have order `≤ n/2`,
//!   contradicting that `ω` is a *primitive* `n`-th root of unity (order
//!   exactly `n`).
//! - Order divides 2 and is not 1, so it's exactly 2. ✓
//!
//! **Fact 2.** In any field, the unique element of order *exactly* 2 is `-1`.
//!
//! To see this, solve `x² = 1` by factoring:
//!
//! ```text
//! x² - 1 = 0   ⟺   (x - 1)(x + 1) = 0   ⟺   x = 1  or  x = -1.
//! ```
//!
//! The last `⟺` uses the **zero-product property of fields**:
//!
//! > **Theorem.** In a field (more generally, in any **integral domain**),
//! > `a · b = 0` implies `a = 0` or `b = 0`.
//!
//! "Integral domain" is just the name for a commutative ring with this
//! property. Every field is an integral domain (proof: if `a ≠ 0` then
//! `a` has a multiplicative inverse `a⁻¹`; multiplying `a · b = 0` on
//! the left by `a⁻¹` gives `b = 0`). The contrapositive — "no zero
//! divisors" — is what lets us conclude from `(x - 1)(x + 1) = 0` that
//! one of the factors must be zero.
//!
//! (Aside: this is the same property that gives the **polynomial
//! root-bound theorem** — a nonzero degree-`d` polynomial over a field
//! has at most `d` roots, because each root `r` contributes a factor
//! `(X - r)` and the integral-domain structure prevents extra factors
//! from sneaking in. We use the special case `d = 2` here, but the
//! general theorem is what justifies "factor and read off the roots"
//! as a complete solution method over any field.)
//!
//! Back to the original problem: `X² - 1` has exactly two roots, `1`
//! and `-1`. Of those, `1` has order 1 and `-1` has order 2 (assuming
//! `p > 2`, which holds for every useful crypto field including
//! Goldilocks). So `-1` is the unique element with order *exactly* 2.
//!
//! **Combining Fact 1 and Fact 2:** `ω^{n/2}` has order 2 (Fact 1) and
//! the unique element of order 2 is `-1` (Fact 2). Therefore
//! `ω^{n/2} = -1`. ✓
//!
//! ## Three micro-lemmas about roots of unity
//!
//! The recursion below leans on three small named facts. Stating them
//! once here makes the algorithm derivation cleaner.
//!
//! **Lemma A (Squaring lemma).** If `ω` is a primitive `n`-th root of
//! unity and `n` is even, then `ω²` is a primitive `(n/2)`-th root of
//! unity.
//!
//! *Proof.* `(ω²)^{n/2} = ω^n = 1`, so the order of `ω²` divides `n/2`.
//! Suppose `(ω²)^j = 1` for some `0 < j < n/2`. Then `ω^{2j} = 1` with
//! `0 < 2j < n`, contradicting that `ω` has order exactly `n`. Hence
//! `ω²` has order exactly `n/2`. ∎
//!
//! This is what lets the recursion *re-use the FFT machinery* at half
//! the size: the squared root of unity at the next level is itself a
//! primitive root of unity, so the recursive call is a smaller, identical
//! problem.
//!
//! **Lemma B (Sign by parity).** `(ω^s)^{n/2} = (-1)^s`.
//!
//! *Proof.* `(ω^s)^{n/2} = (ω^{n/2})^s = (-1)^s` by the previous section's
//! result `ω^{n/2} = -1`. ∎
//!
//! Equivalent statements of the same identity: `ω^{i + n/2} = -ω^i`
//! (used in the trick-in-one-sentence section above); and "the second
//! half of the evaluation domain is the first half negated".
//!
//! **Lemma C (Square-root pairing).** The squaring map `x ↦ x²` from
//! `⟨ω⟩` to `⟨ω²⟩` is 2-to-1: each `(n/2)`-th root of unity has exactly
//! two `n`-th-root preimages, and those preimages are `±x` (additive
//! inverses).
//!
//! *Proof.* `(ω^k)² = ω^{2k}`, and `(ω^k)² = (ω^{k + n/2})²` because
//! `ω^{k + n/2} = -ω^k` (Lemma B with `s = 1`) and `(-y)² = y²`. So the
//! two preimages of `ω^{2k}` are `ω^k` and `ω^{k + n/2} = -ω^k`. They're
//! distinct (one is the negative of the other and we're not in
//! characteristic 2). ∎
//!
//! This is the structural reason the butterfly gets **two outputs from
//! one shared sub-computation**: `p(ω^k)` and `p(ω^{k + n/2}) = p(-ω^k)`
//! share the squared substitution `x² = (-x)²`, so `p_even(x²)` and
//! `p_odd(x²)` only need to be computed once per preimage pair.
//!
//! ## Why the FFT requires `n` a power of 2
//!
//! The Cooley-Tukey recursion halves `n` at every step. The halving is
//! algebraic, not just bookkeeping: at depth `d` we recurse on `n / 2^d`
//! coefficients at root `ω^{2^d}`, and Lemma A (Squaring lemma) says
//! `ω^{2^d}` is a primitive `(n / 2^d)`-th root of unity — *provided the
//! current size is still even so the squaring lands in an order-2 quotient*.
//!
//! For the recursion to terminate cleanly at the base case `n = 1`, every
//! intermediate size must be even — i.e. `n = 2^k` for some `k`. If `n`
//! were `12 = 4 · 3`, after two halvings we'd be stuck at `n = 3`, and the
//! squaring map `x → x²` is a bijection on the order-3 subgroup (3 is odd,
//! so the kernel `{x : x² = 1}` is trivial in a group of odd order, and a
//! bijection on a finite set has equal image and domain). Therefore
//! `⟨ω²⟩ = ⟨ω⟩` — squaring doesn't produce a smaller subgroup to recurse
//! into. Lemma A breaks; the butterfly identity `ω^{n/2} = -1` requires
//! `n` even and breaks the moment it isn't.
//!
//! Bluestein's algorithm and mixed-radix FFTs handle composite `n` by
//! falling back to a different recursion (or a chirp transform). This
//! crate doesn't implement them — every codeword size and FRI fold here
//! is `2^k` by construction.
//!
//! ## "Wait, where are the complex numbers?"
//!
//! If you've seen the FFT before — EE class, signal processing, numerical
//! libraries — it was probably over `C`, with twiddle factors like
//! `e^(2πi/n)`. **This is the same algorithm, over `F_p` instead of `C`.**
//!
//! Cooley-Tukey doesn't actually need the complex numbers. All it needs is:
//!
//! - A field that contains a primitive `n`-th root of unity `ω` (we built
//!   one in [`crate::field::Fp::primitive_root_of_unity`]). In `F_p`, such
//!   an `ω` exists iff `n | (p - 1)` — a consequence of the **Fundamental
//!   Theorem of Cyclic Groups**, proven in detail in `field.rs` module docs.
//! - `n` to be a power of 2 (so the recursion splits cleanly into two
//!   halves of size `n/2`).
//!
//! The classical FFT plugs in `ω = e^(2πi/n) ∈ C`. The NTT plugs in
//! `ω = g^((p-1)/n) ∈ F_p`. Same butterflies, same `O(n log n)` cost —
//! different host arithmetic. When you read this code and see `Fp`
//! everywhere with no `f64` in sight, that's why: it's an FFT over a
//! finite field.
//!
//! ### Unit-circle mental model
//!
//! Picture a clock with `n` ticks evenly spaced around its face. Over
//! `C`, the `n` primitive `n`-th roots of unity literally sit at those
//! tick marks (positions `e^(2πi · k / n)` on the unit circle), and the
//! FFT visits each one in turn. Tick `i` and tick `i + n/2` are
//! **diametrically opposite** — adding `n/2` is a half-turn — so the
//! complex numbers at those positions are additive inverses
//! (`e^(2πi(i + n/2)/n) = -e^(2πi · i / n)`).
//!
//! Over `F_p`, the picture has no geometry — there's no "circle" to look
//! at — but the **arithmetic structure is identical**. The `n` roots
//! `1, ω, ω², ..., ω^{n-1}` still form a cyclic group of order `n`;
//! index `i` and index `i + n/2` still differ by a factor of `-1`
//! (Lemma B above). So if the clock-face mental model helps you intuit
//! butterflies and `±` pairing in the complex case, it transfers
//! directly to the finite-field case — just stop drawing the picture and
//! think in field arithmetic.
//!
//! ## What the FFT computes
//!
//! Given coefficients `[a_0, a_1, ..., a_{n-1}]` and a primitive `n`-th root
//! of unity `ω`, the FFT computes the evaluation vector
//!
//! ```text
//! [p(ω^0), p(ω^1), p(ω^2), ..., p(ω^{n-1})]
//! ```
//!
//! where `p(X) = a_0 + a_1·X + ... + a_{n-1}·X^{n-1}`. That is: it converts
//! a polynomial from coefficient form to evaluation form on the multiplicative
//! subgroup `<ω>`. **`n` must be a power of 2** for radix-2.
//!
//! ## The recursion (Cooley-Tukey 1965)
//!
//! Split the `n` coefficients of `p` into two `n/2`-long sub-sequences by
//! **index parity** — the even-indexed coefficients and the odd-indexed
//! ones — and bundle each into its own polynomial. The trick is to use a
//! **fresh variable `Y`** for these sub-polynomials, distinct from the
//! `X` in `p(X)`:
//!
//! ```text
//! p_even(Y) = a_0 + a_2·Y + a_4·Y² + a_6·Y³ + ...   (length n/2)
//! p_odd(Y)  = a_1 + a_3·Y + a_5·Y² + a_7·Y³ + ...   (length n/2)
//! ```
//!
//! Read carefully: the **coefficients** of `p_even` are the even-indexed
//! ones from `p` (`a_0, a_2, a_4, ...`), but the **powers of `Y`** are
//! the ordinary `0, 1, 2, 3, ...` — they look "normal". `p_even` is just
//! a polynomial like any other; nothing about its appearance shouts "even".
//! The compression is in the *coefficient indexing*, not in the variable's
//! powers.
//!
//! The even-vs-odd structure of the ORIGINAL polynomial reappears when
//! we substitute `Y = X²`. The substitution **doubles every power**:
//! whatever sits at position `k` in a sub-polynomial gets moved to
//! position `2k` in `X`. Watch the two halves separately.
//!
//! **For `p_even`** the doubling is exactly what we want:
//!
//! ```text
//! p_even(X²) = a_0 + a_2·X² + a_4·X⁴ + a_6·X⁶ + ...
//! ```
//!
//! Coefficient `a_{2k}` lands at power `X^{2k}` — matching the
//! even-degree terms of the original `p(X)`. ✓
//!
//! **For `p_odd`** the same substitution alone gives:
//!
//! ```text
//! p_odd(X²) = a_1 + a_3·X² + a_5·X⁴ + a_7·X⁶ + ...
//! ```
//!
//! Coefficient `a_{2k+1}` lands at power `X^{2k}` — but we *want* it at
//! `X^{2k+1}`. **Off by one power.** The fix: multiply the whole thing
//! by `X`, which shifts every exponent up by 1:
//!
//! ```text
//! X · p_odd(X²) = a_1·X + a_3·X³ + a_5·X⁵ + a_7·X⁷ + ...
//! ```
//!
//! Now `a_{2k+1}` sits at `X^{2k+1}` — matching the odd-degree terms of `p`. ✓
//!
//! ### Why the asymmetric `X ·`
//!
//! That's the whole reason the formula is `p_even(X²) + X · p_odd(X²)`
//! and not just `p_even(X²) + p_odd(X²)`. The substitution `Y = X²` only
//! produces *even* powers of `X`, but the odd-indexed coefficients of `p`
//! need to land on *odd* powers. The extra `X ·` is the off-by-one shift
//! that fixes that. The even-indexed coefficients don't need it — their
//! target powers are already even.
//!
//! (Another way to see the same thing: factor `X` out of the odd-degree
//! part of `p`:
//!
//! ```text
//! a_1·X + a_3·X³ + a_5·X⁵ + ... = X · (a_1 + a_3·X² + a_5·X⁴ + ...).
//! ```
//!
//! The bracketed inside is a polynomial in `X²`, which is precisely
//! `p_odd(X²)`. So the odd-degree part of `p` factors as `X · p_odd(X²)`
//! by construction.)
//!
//! Adding the two halves back recovers the original polynomial:
//!
//! ```text
//! p(X)  =  p_even(X²)  +  X · p_odd(X²)
//!          └─even-deg─┘   └──odd-deg──┘
//! ```
//!
//! Now substitute `X = ω^k`:
//!
//! ```text
//! p(ω^k) = p_even(ω^{2k}) + ω^k · p_odd(ω^{2k})
//! ```
//!
//! and observe that `ω^2` is a primitive `(n/2)`-th root of unity. So
//! `p_even(ω^{2k})` and `p_odd(ω^{2k})` are values that a length-`(n/2)` FFT
//! at root `ω^2` would already give us. The recursion has depth `log n` with
//! `O(n)` work per level → `O(n log n)` overall.
//!
//! For `k ∈ [0, n/2)`:
//!
//! ```text
//! p(ω^k)         = p_even(ω^{2k}) + ω^k · p_odd(ω^{2k})        ("first half")
//! p(ω^{k + n/2}) = p_even(ω^{2k}) - ω^k · p_odd(ω^{2k})        ("second half")
//! ```
//!
//! The minus sign in the second formula follows from `ω^{n/2} = -1` (since
//! `ω` has order exactly `n` and `(ω^{n/2})^2 = 1`, so `ω^{n/2}` is the
//! unique field element of order 2). This pair `(plus, minus)` is the
//! **butterfly** — Cooley-Tukey's signature operation.
//!
//! **Question for the reader.** The butterfly extracts *two* output values
//! `out[k]` and `out[k + n/2]` from a single recursive sub-computation.
//! Which one identity of `ω` (the primitive `n`-th root) makes that pairing
//! possible? What goes wrong if it doesn't hold?
//! Try to answer before reading on.
//!
//! Answer: `ω^{n/2} = -1`. Without it, the second equation
//! `even[k] + ω^{k + n/2}·odd[k]` does not simplify to `even[k] − ω^k·odd[k]`,
//! so you can't recover the `(k + n/2)`-th output from the same `(even, odd)`
//! pair — you'd need a second recursive call, killing the halving and the
//! `O(n log n)` cost. The "load-bearing identity" subsection just below
//! shows the cancellation explicitly.
//!
//! ### The load-bearing identity, isolated
//!
//! The single fact `ω^{n/2} = -1` is what lets the butterfly extract *two*
//! outputs from *one* sub-recursion. Concretely:
//!
//! ```text
//! out[k]        = even[k] + ω^k · odd[k]
//! out[k + n/2]  = even[k] + ω^{k + n/2} · odd[k]
//!               = even[k] − ω^k · odd[k]              ← uses ω^{n/2} = -1.
//! ```
//!
//! Without `ω^{n/2} = -1`, the second line is just "another butterfly with
//! a different twiddle"; it doesn't fall out of the same `(even, odd)` pair
//! and you'd need a *second* recursive call to get it. The halving of work
//! at each level — the whole reason the FFT is `O(n log n)` instead of `O(n²)`
//! — is this one identity, dressed up.
//!
//! ## Coset FFT
//!
//! For a coset `L = c · <ω>` (offset `c ≠ 1`), the FFT of `p` on `L` is
//! exactly the subgroup-FFT of `p_c(X) := p(c·X)` on `<ω>`. The coefficients
//! of `p_c` are `[a_0, c·a_1, c^2·a_2, ..., c^{n-1}·a_{n-1}]`. So the
//! algorithm is:
//!
//! 1. Pre-multiply `a_i ← c^i · a_i` (use a running product).
//! 2. Run the subgroup FFT.
//!
//! See [`fft_on_domain`] below for the wrapper.
//!
//! ## Inverse FFT
//!
//! The inverse FFT recovers coefficients from evaluations — it undoes the
//! forward FFT. The formula is short:
//!
//! ```text
//! ifft(evals, ω) = (1/n) · fft(evals, ω⁻¹).
//! ```
//!
//! In words: **run the forward FFT at the inverse root `ω⁻¹`, then scale
//! every entry by `1/n`.** Both pieces are necessary; missing either gives
//! garbage.
//!
//! **Intuition.** The forward FFT evaluates the polynomial at the `n`-th
//! roots of unity. Inverting that map is equivalent to solving the
//! resulting Vandermonde system — and the special structure of roots of
//! unity collapses the answer to a clean closed form: the *same* algorithm,
//! run at the inverse root, then normalised by the domain size.
//!
//! Stated another way: **the iFFT is Lagrange polynomial interpolation,
//! specialised to the roots-of-unity domain.** Lagrange interpolation on
//! `n` arbitrary points takes `O(n²)` work (see [`crate::interpolate`]).
//! The roots-of-unity structure collapses that to `O(n log n)`, which is
//! exactly the speedup the forward FFT delivers for evaluation. Forward
//! and inverse are dual problems with dual speedups.
//!
//! (`1/n` exists in `F_p` because `n` is non-zero modulo `p`. For practical
//! FFT sizes `n < p`, so `n mod p = n ≠ 0`, making `n` a unit. Note this
//! is a *separate* condition from `n | (p-1)` — the latter is what makes
//! `ω` exist; this one is what makes the scaling step well-defined. Both
//! hold whenever you can run an NTT at all.)
//!
//! ### The structural identity: the inverse of a Vandermonde is another Vandermonde
//!
//! The forward FFT, established at the top of this file, is
//! `evals = V · coeffs` where `V[i, j] = ω^{ij}` is the DFT matrix at `ω`.
//! The inverse FFT is therefore `coeffs = V⁻¹ · evals`. The "click" of the
//! whole inverse-FFT theory is:
//!
//! > **Theorem (Vandermonde inverse).** Let `V` be the DFT matrix at `ω`
//! > (a primitive `n`-th root of unity). Then
//! >
//! > ```text
//! > V⁻¹ = (1/n) · V(ω⁻¹)
//! > ```
//! >
//! > where `V(ω⁻¹)[i, j] := ω^{-ij}` is the DFT matrix at `ω⁻¹`.
//!
//! In words: **the inverse of a Vandermonde matrix built from roots of
//! unity is itself a Vandermonde matrix** — just at `ω⁻¹` with a `1/n`
//! scale. The structural fact is *not* "the inverse is some complicated
//! linear operator"; it's "the inverse has the **same shape** as the
//! forward, with one parameter flipped".
//!
//! This is why "run the forward FFT at `ω⁻¹` then divide by `n`" works:
//! the inverse matrix admits the **same Cooley-Tukey recursion** as the
//! forward, because it's the same kind of matrix. We get the iFFT
//! algorithm for free.
//!
//! Stated as the formula at the top of this section:
//!
//! ```text
//! coeffs = V⁻¹ · evals
//!        = (1/n) · V(ω⁻¹) · evals
//!        = (1/n) · forward_FFT(evals, ω⁻¹).
//! ```
//!
//! ### Proof: orthogonality of roots of unity ⟹ V⁻¹ = (1/n) · V(ω⁻¹)
//!
//! Both pieces of the theorem (the `ω → ω⁻¹` flip and the `1/n` scale)
//! fall out of a single named lemma:
//!
//! > **Orthogonality Lemma.** For a primitive `n`-th root of unity `ω`
//! > and any integer `d`:
//! >
//! > ```text
//! > sum_{j = 0..n} ω^{jd} = { n   if d ≡ 0 (mod n),
//! >                         { 0   otherwise.
//! > ```
//!
//! Why "orthogonality"? Read the sum as the inner product of two row
//! vectors of length `n`: `[1, ω^a, ω^{2a}, ..., ω^{(n-1)a}]` and
//! `[1, ω^{-b}, ω^{-2b}, ..., ω^{-(n-1)b}]` with `d = a - b`. The lemma
//! says **distinct rows of the DFT matrix are orthogonal**, and each
//! row's self-inner-product is `n`. That's exactly the condition we need
//! for `V · V(ω⁻¹) = n · I`.
//!
//! *Proof of the lemma.* Two cases on `d`:
//!
//! - **`d ≡ 0 (mod n)`:** every term is `ω^{jd} = (ω^n)^{j(d/n)} = 1`,
//!   so the sum is `n`.
//! - **`d ≢ 0 (mod n)`:** apply the standard geometric-series identity
//!
//!   ```text
//!   sum_{j = 0..n} r^j = (1 − r^n) / (1 − r),       valid whenever r ≠ 1
//!   ```
//!
//!   with `r = ω^d`. Numerator: `1 − (ω^n)^d = 1 − 1 = 0`. Denominator:
//!   `1 − ω^d ≠ 0` since `ω` has order exactly `n` and `d ≢ 0 (mod n)`
//!   means `ω^d ≠ 1`. So the sum is `0 / non-zero = 0`. ∎
//!
//! ### From orthogonality to the Vandermonde inverse
//!
//! Plug the lemma into the matrix product. Compute the `(i, k)`-th entry
//! of `V · V(ω⁻¹)`:
//!
//! ```text
//! (V · V(ω⁻¹))[i, k] = sum_{j = 0..n} V[i, j] · V(ω⁻¹)[j, k]
//!                    = sum_{j = 0..n} ω^{ij} · ω^{-jk}
//!                    = sum_{j = 0..n} ω^{j(i - k)}.
//! ```
//!
//! Apply the orthogonality lemma with `d = i - k`:
//!
//! - **`i = k`:** `d = 0`, sum is `n`. The diagonal of `V · V(ω⁻¹)` is `n`.
//! - **`i ≠ k`:** `0 < |i - k| < n`, so `d ≢ 0 (mod n)`. Sum is `0`. The
//!   off-diagonal is zero.
//!
//! So `V · V(ω⁻¹) = n · I`, hence `V⁻¹ = (1/n) · V(ω⁻¹)`. ∎
//!
//! ### Coset inverse FFT
//!
//! For a coset `L = c · ⟨ω⟩`, the inverse is: subgroup-iFFT, then
//! de-scale `b_i ← c^{−i} · b_i`. This mirrors the forward direction's
//! "scale by `c^i`, then run subgroup FFT" — every coset operation is
//! its subgroup counterpart sandwiched between a scaling and an
//! inverse scaling.
//!
//! ### Worked inverse round-trip (n = 2)
//!
//! Small enough to verify by hand. Take `coeffs = [3, 5]` and `n = 2`,
//! so `ω` is a primitive 2nd root of unity = `-1`.
//!
//! **Forward FFT** evaluates `p(X) = 3 + 5X` at the powers of `ω`:
//!
//! ```text
//! p(ω^0) = p( 1) = 3 + 5·1 =  8
//! p(ω^1) = p(-1) = 3 − 5   = -2
//! evals = [8, -2].
//! ```
//!
//! **Inverse FFT** applies `(1/n) · fft(evals, ω⁻¹)`. For `n = 2` we have
//! `ω⁻¹ = (-1)⁻¹ = -1` (each non-zero element of order 2 is its own
//! inverse), so we run the forward FFT *again* at the same root, this
//! time treating `evals` as coefficients of `q(X) = 8 − 2X`:
//!
//! ```text
//! q( 1) = 8 − 2 =  6
//! q(-1) = 8 + 2 = 10
//! fft(evals, -1) = [6, 10].
//! ```
//!
//! Scale by `1/n = 1/2`:
//!
//! ```text
//! [6/2, 10/2] = [3, 5] = original coeffs. ✓
//! ```
//!
//! Round-trip closes. The "two passes of the forward FFT" structure is
//! peculiar to `n = 2` (where `ω⁻¹ = ω`); for larger `n` you genuinely
//! need `ω⁻¹ ≠ ω` and the inverse uses a different recursion path. But
//! the *formula* is the same shape: forward at the inverse root, then
//! divide by `n`.
//!
//! ### Worked inverse round-trip (n = 4, symbolic)
//!
//! Now an `ω⁻¹ ≠ ω` case. Re-use the forward example from
//! "Worked example: n = 4" below — `coeffs = [1, 2, 3, 4]`, `ω` a
//! primitive 4th root of unity (so `ω² = -1` and `ω⁴ = 1`). The forward
//! FFT produced
//!
//! ```text
//! evals = [10,  -2 − 2ω,  -2,  -2 + 2ω].
//! ```
//!
//! For `n = 4`, `ω⁻¹ = ω³ = -ω` (since `ω · ω³ = ω⁴ = 1`, and
//! `ω³ = ω² · ω = -ω`). The inverse FFT runs the forward FFT on `evals`
//! at root `ω⁻¹ = -ω`, then scales by `1/4`.
//!
//! Treat `evals` as the coefficient vector of a polynomial
//! `q(Y) = 10 + (-2 - 2ω)·Y + (-2)·Y² + (-2 + 2ω)·Y³` and evaluate at
//! `(ω⁻¹)⁰, (ω⁻¹)¹, (ω⁻¹)², (ω⁻¹)³`:
//!
//! ```text
//! (ω⁻¹)⁰ =  1
//! (ω⁻¹)¹ = -ω
//! (ω⁻¹)² =  ω² = -1   (since (-ω)² = ω²)
//! (ω⁻¹)³ =  ω         (since (-ω)³ = -ω³ = -(-ω) = ω)
//! ```
//!
//! Evaluate `q` at each, using `ω⁴ = 1` and `ω² = -1` to simplify:
//!
//! ```text
//! q(1)  = 10 + (-2 - 2ω)·1 + (-2)·1   + (-2 + 2ω)·1
//!       = 10 - 2 - 2ω - 2 - 2 + 2ω
//!       = 4.
//!
//! q(-ω) = 10 + (-2 - 2ω)·(-ω) + (-2)·ω² + (-2 + 2ω)·(-ω³)
//!       = 10 + (2ω + 2ω²)     + (-2)·(-1)   + (-2 + 2ω)·(ω)
//!       = 10 + 2ω + 2·(-1)    + 2           + (-2ω + 2ω²)
//!       = 10 + 2ω - 2 + 2 - 2ω + 2·(-1)
//!       = 10 - 2
//!       = 8.
//!
//! q(-1) = 10 + (-2 - 2ω)·(-1) + (-2)·1   + (-2 + 2ω)·(-1)
//!       = 10 + 2 + 2ω - 2 + 2 - 2ω
//!       = 12.
//!
//! q(ω)  = 10 + (-2 - 2ω)·ω    + (-2)·ω²  + (-2 + 2ω)·ω³
//!       = 10 + (-2ω - 2ω²)    + 2        + (-2ω³ + 2ω⁴)
//!       = 10 - 2ω - 2·(-1)    + 2        - 2·(-ω) + 2
//!       = 10 - 2ω + 2 + 2 + 2ω + 2
//!       = 16.
//! ```
//!
//! So `fft(evals, ω⁻¹) = [4, 8, 12, 16]`. Scale by `1/n = 1/4`:
//!
//! ```text
//! (1/4) · [4, 8, 12, 16] = [1, 2, 3, 4] = original coeffs.   ✓
//! ```
//!
//! Round-trip closes for `n = 4` with `ω⁻¹ ≠ ω`. Every `ω` term in the
//! intermediate computations cancels exactly — that's the orthogonality
//! lemma at work in concrete numerics. The recipe really is just
//! **forward at the inverse root, then divide by `n`**.
//!
//! ## Worked example: n = 4
//!
//! Take `p(X) = 1 + 2X + 3X^2 + 4X^3`, so `coeffs = [1, 2, 3, 4]`.
//! Pick `ω` = primitive 4th root of unity, so `ω^4 = 1` and `ω^2 = -1`.
//! `p_even(Y) = 1 + 3Y` (from `[1, 3]`), `p_odd(Y) = 2 + 4Y` (from `[2, 4]`).
//!
//! Running length-2 FFTs at root `ω^2 = -1`:
//!
//! ```text
//! p_even at <-1> = [p_even(1), p_even(-1)] = [4, -2]
//! p_odd  at <-1> = [p_odd(1),  p_odd(-1) ] = [6, -2]
//! ```
//!
//! Combining:
//!
//! ```text
//! p(ω^0) = 4 + 1·6  = 10        p(ω^2) = 4 - 1·6  = -2
//! p(ω^1) = -2 + ω·(-2) = -2 - 2ω  p(ω^3) = -2 - ω·(-2) = -2 + 2ω
//! ```
//!
//! Cross-check by Horner: `p(1) = 1 + 2 + 3 + 4 = 10` ✓ and `p(-1) = 1 - 2 + 3 - 4 = -2` ✓.
//! The other two depend on the actual numerical value of `ω`.

use crate::domain::EvaluationDomain;
use crate::field::Fp;

/// Compute the forward FFT of `coeffs` on the subgroup `<ω>` of size `n = coeffs.len()`.
///
/// # Notation: `omega` is `ω`
///
/// The math throughout this module uses `ω` (Greek omega) for the primitive
/// root of unity. Rust identifiers can't be named `ω`, so the parameter
/// below is spelled `omega`. **They are the same thing** — `omega` *is* `ω`.
/// (Same goes for `omega_sq` in the body: that's just `ω²`.)
///
/// # Returns
///
/// The evaluation vector `[p(ω^0), p(ω^1), p(ω^2), ..., p(ω^{n-1})]`, where
/// `p(X) = a_0 + a_1·X + ... + a_{n-1}·X^{n-1}` is the polynomial whose
/// coefficients are `coeffs`.
///
/// # Preconditions
///
/// - `coeffs.len() == n` is a power of 2.
/// - `omega` (i.e., `ω`) is a primitive `n`-th root of unity in `F_p`:
///   `omega.pow(n) == Fp::one()` and `omega.pow(n/2) != Fp::one()`.
///
/// Panics if `n` is not a power of 2. The "primitive root" precondition isn't
/// fully checked at runtime — we trust the caller. Adding
/// `debug_assert!(omega.pow(n as u64) == Fp::one())` would cost an `O(log n)`
/// pow per call; reasonable as a debug-only guardrail.
///
/// # Subgroup vs coset: this is the inner kernel
///
/// The output is `p` evaluated on the **subgroup** `⟨ω⟩` and nothing else.
/// This function has no notion of a coset offset — if you want evaluations
/// on a coset `c · ⟨ω⟩` for some `c ≠ 1`, the coset is handled at the
/// **coefficient** level, never at the `omega` level.
///
/// The identity that makes the trick work:
///
/// ```text
/// p(c · ω^j) = Σ_i a_i · (c · ω^j)^i = Σ_i (a_i · c^i) · ω^(ij).
/// ```
///
/// Coset evaluations of `p(X) = Σ a_i · X^i` thus equal subgroup evaluations
/// of `p_c(X) := p(c·X)`, whose coefficients are the pre-scaled
/// `a_i' = c^i · a_i`. The call pattern is:
///
/// ```text
/// pre_scaled[i] = c^i · coeffs[i]                          (caller-side)
/// fft_subgroup(&pre_scaled, ω)                             (this function)
///   ↳ returns [p(c), p(c·ω), p(c·ω²), ..., p(c·ω^{n-1})]
/// ```
///
/// The wrapper [`fft_on_domain`] does this pre-scaling for you given an
/// [`EvaluationDomain`] (subgroup *or* coset) — prefer it unless you have a
/// specific reason to call the kernel directly. In the subgroup case
/// (`c = 1`) the wrapper degenerates: `c^i = 1` for all `i`, the pre-scale
/// loop is a no-op, and the wrapper just forwards `coeffs` and `ω` to
/// `fft_subgroup` unchanged.
///
/// **Anti-pattern.** Don't pass `c · ω` as `omega` hoping to "shift" the
/// domain. For most `c ∉ ⟨ω⟩`, the product `c · ω` does not have order
/// exactly `n`, so the primitive-root precondition fails and the function
/// silently produces garbage — no panic, no error, just wrong numbers. The
/// distinction is load-bearing: **`omega` always names a subgroup
/// generator; the coset offset (if any) lives in the coefficients.**
///
/// # Twiddle factors
///
/// A **twiddle factor** is jargon for one of the powers of `ω` that gets
/// multiplied into a butterfly — specifically the `ω^k` in:
///
/// ```text
/// u + ω^k · v     ("+" output of the butterfly)
/// u − ω^k · v     ("−" output of the butterfly)
/// ```
///
/// Different butterflies use different powers of `ω` (different `k`);
/// collectively those powers are called "the twiddle factors of the FFT".
/// In the implementation below, the local variable `twiddle` walks through
/// `ω^0, ω^1, ω^2, ...` as `k` advances through the combine loop.
pub fn fft_subgroup(coeffs: &[Fp], omega: Fp) -> Vec<Fp> {
    // TODO: return `[p(ω^0), p(ω^1), ..., p(ω^{n-1})]` in O(n log n).
    //   1. Base case: `n == 1` returns `vec![coeffs[0]]` (constant poly evaluation).
    //   2. Split coeffs by index parity into `evens` (a_0, a_2, ...) and `odds` (a_1, a_3, ...).
    //   3. Recurse at `ω²` — primitive `(n/2)`-th root by Lemma A (Squaring lemma).
    //   4. Butterfly combine: `result[k] = even[k] + ω^k · odd[k]`,
    //      `result[k + n/2] = even[k] − ω^k · odd[k]`. The sign flip is Lemma B
    //      (`(ω^{k + n/2}) = −ω^k`), so both outputs share one sub-computation.
    // See "The recursion (Cooley-Tukey 1965)" and "Three micro-lemmas" above.
    //
    // Reference implementation below.
    let n = coeffs.len();
    assert!(n.is_power_of_two(), "coeffs length must be a power of two");

    if n == 1 {
        vec![coeffs[0]]
    } else {
        let evens: Vec<Fp> = coeffs.iter().step_by(2).copied().collect();
        let odds:  Vec<Fp> = coeffs.iter().skip(1).step_by(2).copied().collect();
        let omega_sq = omega * omega;
        let even_dft = fft_subgroup(&evens, omega_sq);
        let odd_dft  = fft_subgroup(&odds,  omega_sq);
        let mut result = vec![Fp::zero(); n];
        let mut twiddle = Fp::one();
        for k in 0..n/2 {
            let t = twiddle * odd_dft[k];
            result[k]         = even_dft[k] + t;
            result[k + n/2]   = even_dft[k] - t;
            twiddle = twiddle * omega;
        }
        result
    }
}

/// Inverse FFT on the subgroup `⟨ω⟩`. Recovers coefficients from evaluations.
///
/// # What this computes
///
/// Given `evals = [p(ω^0), p(ω^1), ..., p(ω^{n-1})]`, returns the
/// coefficient vector `[a_0, a_1, ..., a_{n-1}]` of the unique
/// polynomial `p(X) = sum_i a_i · X^i` (of degree `< n`) whose
/// evaluations on `⟨ω⟩` are `evals`. In particular,
/// `ifft_subgroup(fft_subgroup(coeffs, ω), ω) == coeffs`.
///
/// # Formula and why it works
///
/// ```text
/// ifft_subgroup(evals, ω) = (1/n) · fft_subgroup(evals, ω⁻¹).
/// ```
///
/// See the **"Inverse FFT"** section in the module docs for the full
/// derivation. The one-paragraph version: viewing the DFT as a matrix
/// `V[i, j] = ω^{ij}`, the inverse satisfies `V⁻¹ = (1/n) · V*` where
/// `V*[i, j] = ω^{-ij}` is the DFT matrix at `ω⁻¹`. So undoing the FFT
/// = running the forward FFT at `ω⁻¹` (computes `V* · evals`) and then
/// scaling by `1/n` (turns `V*` into `V⁻¹`).
///
/// # Preconditions
///
/// Same as [`fft_subgroup`]:
///
/// - `evals.len() == n` is a power of 2.
/// - `omega` (i.e., `ω`) is a primitive `n`-th root of unity in `F_p`.
pub fn ifft_subgroup(evals: &[Fp], omega: Fp) -> Vec<Fp> {
    // TODO: recover coefficients from evaluations on `⟨ω⟩`.
    //   1. Run `fft_subgroup` at `ω⁻¹` (the inverse root). This produces
    //      `V(ω⁻¹) · evals`, which differs from the true inverse by a `1/n`
    //      scalar by the Vandermonde-inverse identity `V⁻¹ = (1/n)·V(ω⁻¹)`.
    //   2. Multiply every output by `Fp::new(n).inverse()` to apply the
    //      `1/n` factor and finish the inversion.
    // See "Inverse FFT" / "structural identity" sections above.
    //
    // Reference implementation below.
    let n = evals.len();
    let n_inv = Fp::new(n as u64)
        .inverse()
        .expect("n is a non-zero power of 2, so 1/n exists in F_p");

    let evals_fft = fft_subgroup(
        evals,
        omega.inverse().expect("primitive roots of unity are non-zero"),
    );

    evals_fft.into_iter().map(|x| x * n_inv).collect()
}

/// Forward FFT of `coeffs` on a (possibly proper) coset domain.
///
/// `coeffs.len()` must equal `domain.size()`. Returns
/// `[p(c), p(c·ω), p(c·ω^2), ..., p(c·ω^{n-1})]`.
///
/// Internally pre-multiplies `coeffs` by powers of `c` (the offset) and
/// then runs the subgroup FFT at `ω` = `domain.generator()`.
///
/// # The coset trick *is* a change of variables
///
/// The identity `p(c·ω^j) = Σ_i (a_i · c^i) · ω^(ij)` says exactly that
/// evaluating `p` on the coset `c·H` is the same as evaluating
/// `p_c(X) := p(c·X)` on `H` — and `p_c` has coefficients `a_i · c^i`.
/// So the "pre-scale by powers of `c`" step is *literally* the act of
/// rewriting `p(X)` as `p_c(X)`; nothing about the FFT kernel itself changes.
pub fn fft_on_domain(coeffs: &[Fp], domain: &EvaluationDomain) -> Vec<Fp> {
    // TODO: evaluate `p` on the (possibly proper) coset `L = c·⟨ω⟩`.
    //   1. If `domain.offset() == 1`, short-circuit to `fft_subgroup(coeffs, ω)` —
    //      the pre-scale would be a no-op anyway (`c^i = 1`).
    //   2. Otherwise pre-scale `a_i ← c^i · a_i` (running product `pow_c`).
    //      This is the change-of-variables trick: evaluating `p` on `c·⟨ω⟩` equals
    //      evaluating `p_c(X) := p(c·X)` on `⟨ω⟩`, and `p_c` has coefficients `c^i · a_i`.
    //   3. Call `fft_subgroup` on the rescaled vector with `ω = domain.generator()`.
    // See "Subgroup vs coset: this is the inner kernel" above for the identity.
    //
    // Reference implementation below.

    assert_eq!(coeffs.len(), domain.size(), "coefficients length must match domain size");
    if domain.offset() == Fp::one() {
        fft_subgroup(coeffs, domain.generator())
    } else {
        let offset = domain.offset();
        let mut pow_c = Fp::one();
        let scaled: Vec<Fp> = coeffs.iter().map(|&a| {
            let scaled_a = a * pow_c;
            pow_c = pow_c * offset;
            scaled_a
        }).collect();
        fft_subgroup(&scaled, domain.generator())
    }
}

/// Inverse FFT on a coset domain.
///
/// Recovers coefficients `[a_0, a_1, ..., a_{n-1}]` from evaluations
/// `[p(c), p(c·ω), ..., p(c·ω^{n-1})]`.
pub fn ifft_on_domain(evals: &[Fp], domain: &EvaluationDomain) -> Vec<Fp> {
    // TODO: invert the coset FFT — recover `a_i` from coset evaluations.
    //   1. If `offset == 1` call `ifft_subgroup` and we're done (no pre-scale was applied).
    //   2. Otherwise first `ifft_subgroup` — this returns `[c^i · a_i]`, the
    //      *scaled* coefficients matching what `fft_on_domain` pre-fed.
    //   3. Unscale by `c^{-i}` (running product `pow_c_inv`) to recover the true `a_i`.
    // See "Coset inverse FFT" in module docs — symmetric to the forward direction.
    //
    // Reference implementation below.

    let mut scaled_coeffs = ifft_subgroup(evals, domain.generator());
    let offset = domain.offset();
    if offset != Fp::one() {
        let c_inv = offset.inverse().expect("coset offset is non-zero by construction");
        let mut pow_c_inv = Fp::one();
        for entry in scaled_coeffs.iter_mut() {
            *entry = *entry * pow_c_inv;
            pow_c_inv = pow_c_inv * c_inv;
        }
    }
    scaled_coeffs
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use crate::polynomial::UnivariatePoly;
    use super::*;

    #[test]
    fn fft_of_constant_polynomial() {
        // TODO: coeffs = [c, 0, 0, 0]. The FFT should be [c, c, c, c]
        // (a constant polynomial evaluates to c at every point).
        // omega is the primitive 4th root of unity.

        // The constant polynomial p(X) = 42 has coefficients [42, 0, 0, 0]
        // in length-4 coefficient form. Evaluating it at any point returns
        // 42, so the FFT output should be [42, 42, 42, 42] regardless of ω.
        let c = Fp::new(42);
        let coeffs = vec![c, Fp::zero(), Fp::zero(), Fp::zero()];
        let omega = Fp::primitive_root_of_unity(2); // n = 2^2 = 4

        let evals = fft_subgroup(&coeffs, omega);

        assert_eq!(evals, vec![c; 4]);
    }

    #[test]
    fn fft_of_x_polynomial() {
        // TODO: coeffs = [0, 1, 0, 0] represents p(X) = X.
        // FFT should be [omega^0, omega^1, omega^2, omega^3]
        //               = [1, omega, omega^2, omega^3].

        let c = Fp::zero();
        let coeffs = vec![c, Fp::one(), Fp::zero(), Fp::zero()];
        let omega = Fp::primitive_root_of_unity(2); // n = 2^2 = 4
        let evals = fft_subgroup(&coeffs, omega);

        assert_eq!(evals, vec![Fp::one(), omega, omega.pow(2), omega.pow(3)]);
    }

    #[test]
    fn fft_round_trip_subgroup() {
        // Round-trip on the subgroup `<ω>`: forward FFT then inverse FFT
        // must recover the original coefficients exactly. If they don't,
        // one of the two functions is buggy.
        use rand::SeedableRng;

        let log_n = 3;
        let n = 1 << log_n; // 8

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_5EED);
        let coeffs: Vec<Fp> = (0..n).map(|_| Fp::random(&mut rng)).collect();

        let omega = Fp::primitive_root_of_unity(log_n);
        let evals = fft_subgroup(&coeffs, omega);
        let recovered = ifft_subgroup(&evals, omega);

        assert_eq!(recovered, coeffs);
    }

    #[test]
    fn fft_round_trip_coset() {
        // TODO: same as `fft_round_trip_subgroup` above, but exercise the
        // coset path: build an `EvaluationDomain::new_coset(log_n, offset)`
        // with offset != 1, then call `fft_on_domain` / `ifft_on_domain`
        // and check the round-trip recovers the original coefficients.
        //
        // Hint: pick a fixed non-zero offset (e.g., `Fp::new(7)`) rather
        // than `Fp::random(...)` — `new_coset` panics on `offset == zero`,
        // and a random offset makes test failures non-reproducible.
        //
        // new: swapped thread_rng() for a seeded StdRng so failures are
        // reproducible.
        use rand::SeedableRng;

        let log_n = 3;
        let n = 1 << log_n; // 8

        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC05E7_5EED);
        let coeffs: Vec<Fp> = (0..n).map(|_| Fp::random(&mut rng)).collect();

        // Fixed non-zero offset. `Fp::new(7)` happens to match Goldilocks's
        // multiplicative generator — any non-zero, non-one element works.
        let offset = Fp::new(7);
        let domain = EvaluationDomain::new_coset(log_n, offset);

        let evals = fft_on_domain(&coeffs, &domain);
        let recovered = ifft_on_domain(&evals, &domain);

        assert_eq!(recovered, coeffs);
    }

    #[test]
    fn fft_matches_naive_evaluation_subgroup() {
        // TODO: build a random poly of degree < 8 (so coeffs has length 8 padded with zeros,
        // OR length d < 8 padded with zeros to length 8 before calling FFT — your choice).
        // For a domain of size 8:
        //   fast = fft_subgroup(&coeffs, omega)
        //   slow = (0..8).map(|i| poly.evaluate(omega.pow(i))).collect::<Vec<_>>()
        // Assert fast == slow.
        //
        // Note: if you pad with zeros to length 8, you must build a `UnivariatePoly`
        // via `from_coeffs_unstripped` or `evaluate` will see only the original (d) coeffs.
        // Easier: use the unstripped constructor.

        // Independent oracle: Horner evaluation via UnivariatePoly. The FFT
        // output at index i must equal poly.evaluate(omega^i). Catches bugs
        // that the round-trip test can't (e.g., compensating sign errors
        // between fft and ifft).
        use rand::SeedableRng;

        let log_n = 3;
        let n = 1 << log_n; // 8
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xFA57_C0DE);

        let coeffs: Vec<Fp> = (0..n).map(|_| Fp::random(&mut rng)).collect();
        let omega = Fp::primitive_root_of_unity(log_n);

        // Use the unstripped constructor so trailing-zero coefficients (which
        // can happen by chance) aren't dropped from the polynomial's view.
        let poly = UnivariatePoly::from_coeffs_unstripped(coeffs.clone());

        let fast = fft_subgroup(&coeffs, omega);
        let slow: Vec<Fp> = (0..n as u64).map(|i| poly.evaluate(omega.pow(i))).collect();

        assert_eq!(fast, slow);
    }

    #[test]
    fn fft_matches_naive_evaluation_on_coset() {
        // TODO: same as `fft_matches_naive_evaluation_subgroup` above,
        // but on a coset. For each i in 0..n:
        //   fast[i] = fft_on_domain(&coeffs, &domain)[i]
        //   slow[i] = poly.evaluate(domain.element(i))
        // should be equal. Build the polynomial via `from_coeffs_unstripped`
        // so trailing-zero coefficients aren't dropped from the polynomial's view.
        //
        // Independent oracle: `poly.evaluate(domain.element(i))` computed
        // via Horner's rule. The fast coset-FFT must produce the same
        // vector. Catches single-side bugs in `fft_on_domain` that the
        // round-trip test can't (e.g., wrong powers in the pre-scale step).
        //
        // new: rewrote entirely — previous body did a subgroup round-trip
        // in a loop, which doesn't test what the name says.
        use rand::SeedableRng;

        let log_n = 3;
        let n: usize = 1 << log_n; // 8
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC05E7_FA57);

        let coeffs: Vec<Fp> = (0..n).map(|_| Fp::random(&mut rng)).collect();
        let domain = EvaluationDomain::new_coset(log_n, Fp::new(7));

        // Unstripped so trailing-zero coefficients (rare but possible from
        // random sampling) aren't dropped from the polynomial's view.
        let poly = UnivariatePoly::from_coeffs_unstripped(coeffs.clone());

        let fast = fft_on_domain(&coeffs, &domain);
        let slow: Vec<Fp> = (0..n).map(|i| poly.evaluate(domain.element(i))).collect();

        assert_eq!(fast, slow);
    }

    #[test]
    fn fft_size_one_is_identity() {
        // TODO: coeffs = [Fp::new(42)]; omega = Fp::one(); fft → [Fp::new(42)].
        // (Edge case — make sure your base case is correct.)
        //
        // Hint: test the FORWARD FFT in isolation, not a fft→ifft
        // round-trip. The n=1 iFFT is also the identity, so a round-trip
        // version would pass even if your base case is broken.
        //
        // new: changed from fft→ifft round-trip to a direct forward call
        // so the base case is actually isolated.
        let coeffs = vec![Fp::new(42)];
        let omega = Fp::one(); // primitive 1st root of unity = 1
        let evals = fft_subgroup(&coeffs, omega);
        assert_eq!(evals, coeffs);
    }
}
