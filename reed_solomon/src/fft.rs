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
//! For Reed-Solomon, this is exactly the encoding step: the `n` outputs
//! *are* the codeword. So "fast polynomial evaluation on `n` points" is
//! the same problem as "fast RS encoding".
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
//! ## "Wait, where are the complex numbers?"
//!
//! If you've seen the FFT before — EE class, signal processing, numerical
//! libraries — it was probably over `C`, with twiddle factors like
//! `e^(2πi/n)`. **This is the same algorithm, over `F_p` instead of `C`.**
//!
//! Cooley-Tukey doesn't actually need the complex numbers. All it needs is:
//!
//! - A field that contains a primitive `n`-th root of unity `ω` (we built
//!   one in [`crate::field::Fp::primitive_root_of_unity`]).
//! - `n` to be a power of 2 (so the recursion splits cleanly into two
//!   halves of size `n/2`).
//!
//! The classical FFT plugs in `ω = e^(2πi/n) ∈ C`. The NTT plugs in
//! `ω = g^((p-1)/n) ∈ F_p`. Same butterflies, same `O(n log n)` cost —
//! different host arithmetic. When you read this code and see `Fp`
//! everywhere with no `f64` in sight, that's why: it's an FFT over a
//! finite field.
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
//! The inverse FFT is **the forward FFT with `ω^-1` substituted for `ω`**,
//! followed by a division of every output by `n`. This is a property of the
//! DFT matrix: it's unitary up to scaling.
//!
//! For a coset, the inverse is: subgroup-iFFT, then de-scale `b_i ← c^{-i} · b_i`.
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
    // TODO:
    //   1. Let n = coeffs.len().
    //   2. Assert n is a power of 2 (n.is_power_of_two()).
    //   3. Base case: if n == 1, return vec![coeffs[0]].
    //   4. Split into evens and odds:
    //        let evens: Vec<Fp> = coeffs.iter().step_by(2).copied().collect();
    //        let odds:  Vec<Fp> = coeffs.iter().skip(1).step_by(2).copied().collect();
    //   5. Recurse at omega^2:
    //        let omega_sq = omega * omega;
    //        let even_dft = fft_subgroup(&evens, omega_sq);
    //        let odd_dft  = fft_subgroup(&odds,  omega_sq);
    //   6. Combine:
    //        let mut result = vec![Fp::zero(); n];
    //        let mut twiddle = Fp::one();
    //        for k in 0..n/2 {
    //            let t = twiddle * odd_dft[k];
    //            result[k]         = even_dft[k] + t;
    //            result[k + n/2]   = even_dft[k] - t;
    //            twiddle = twiddle * omega;
    //        }
    //        result
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

/// Inverse FFT on the subgroup. Recovers coefficients from evaluations.
///
/// Computes `forward_FFT(evals, omega^-1) / n`.
///
/// Preconditions: same as [`fft_subgroup`].
pub fn ifft_subgroup(evals: &[Fp], omega: Fp) -> Vec<Fp> {
    // TODO:
    //   1. Compute omega_inv = omega.inverse().expect("omega must be non-zero").
    //   2. Run forward FFT at omega_inv.
    //   3. Divide every entry by n. Implementation: multiply by Fp::new(n as u64).inverse().
    //
    // Hint: Fp::new(n as u64) is fine for n up to MODULUS, which is more
    // than we'll ever hit. Compute n_inv ONCE outside the loop.
    let _ = (evals, omega);
    todo!()
}

/// Forward FFT of `coeffs` on a (possibly proper) coset domain.
///
/// `coeffs.len()` must equal `domain.size()`. Returns
/// `[p(c), p(c·ω), p(c·ω^2), ..., p(c·ω^{n-1})]`.
///
/// Internally pre-multiplies `coeffs` by powers of `c` (the offset) and
/// then runs the subgroup FFT at `ω` = `domain.generator()`.
pub fn fft_on_domain(coeffs: &[Fp], domain: &EvaluationDomain) -> Vec<Fp> {
    // TODO:
    //   1. Assert coeffs.len() == domain.size().
    //   2. If domain.offset() == Fp::one(), just call fft_subgroup directly.
    //   3. Otherwise: build scaled = [c^0·a_0, c^1·a_1, c^2·a_2, ...] using
    //      a running product `pow_c`. Then call fft_subgroup(&scaled, domain.generator()).
    let _ = (coeffs, domain);
    todo!()
}

/// Inverse FFT on a coset domain.
///
/// Recovers coefficients `[a_0, a_1, ..., a_{n-1}]` from evaluations
/// `[p(c), p(c·ω), ..., p(c·ω^{n-1})]`.
pub fn ifft_on_domain(evals: &[Fp], domain: &EvaluationDomain) -> Vec<Fp> {
    // TODO:
    //   1. Run ifft_subgroup at omega = domain.generator().
    //   2. If domain.offset() != Fp::one(), de-scale: divide entry i by c^i,
    //      i.e., multiply by (c^-1)^i. Use a running product over c.inverse().
    let _ = (evals, domain);
    todo!()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_of_constant_polynomial() {
        // TODO: coeffs = [c, 0, 0, 0]. The FFT should be [c, c, c, c]
        // (a constant polynomial evaluates to c at every point).
        // omega is the primitive 4th root of unity.
        todo!()
    }

    #[test]
    fn fft_of_x_polynomial() {
        // TODO: coeffs = [0, 1, 0, 0] represents p(X) = X.
        // FFT should be [omega^0, omega^1, omega^2, omega^3]
        //               = [1, omega, omega^2, omega^3].
        todo!()
    }

    #[test]
    fn fft_round_trip_subgroup() {
        // TODO: pick log_n = 3 (size 8). Generate random coeffs.
        // Compute fft → ifft → original. Use the field::Fp::random + a seeded RNG.
        todo!()
    }

    #[test]
    fn fft_round_trip_coset() {
        // TODO: same as above but using fft_on_domain / ifft_on_domain on a
        // coset with offset != 1.
        todo!()
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
        todo!()
    }

    #[test]
    fn fft_matches_naive_evaluation_on_coset() {
        // TODO: same as above but on a coset.
        // For each i in 0..n: fast[i] should equal poly.evaluate(domain.element(i)).
        todo!()
    }

    #[test]
    fn fft_size_one_is_identity() {
        // TODO: coeffs = [Fp::new(42)]; omega = Fp::one(); fft → [Fp::new(42)].
        // (Edge case — make sure your base case is correct.)
        todo!()
    }
}
