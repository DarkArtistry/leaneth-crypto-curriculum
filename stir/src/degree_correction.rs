//! **DegCor** (degree correction) and **Combine** — STIR's degree-alignment
//! primitives.
//!
//! ## What this module does
//!
//! STIR's per-round arithmetic involves multiple polynomials of *different
//! actual degrees* that the protocol wants to treat uniformly. DegCor
//! patches the gap: given a polynomial `f` of degree `< d` over the
//! Goldilocks field `F = F_p` and a **target degree bound** `d* ≥ d`,
//! DegCor produces a "scaled" polynomial `f · g_r` of degree `< d*` whose
//! evaluation table is bound to `f`'s by a verifier-chosen random scalar
//! `r ∈ F_p` (the round's degree-correction randomness). The scaling
//! polynomial is the **geometric-sum scaling polynomial**
//! `g_r(X) := sum_{i=0..e} (rX)^i` with `e := d* − d`, defined formally
//! below. A malicious prover who cheats on `f` cannot hide the cheat by
//! claiming the wrong degree, because `r` enters the scaling factor.
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
//! ## Named theorem
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
//!   important case).
//! - [`combine`] does the round's linear combination of two evaluation
//!   vectors.
//!
//! Both work on raw evaluation tables — no coefficient-form polynomial gets
//! built or destroyed.

use reed_solomon::{EvaluationDomain, Fp};

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
    // TODO: pointwise multiply by g_r(L[j]).
    //   1. Validate `evals.len() == domain.size()`.
    //   2. Decide the input degree bound. In the educational scaffold we treat
    //      it as implicit — say, derived from a STIR round struct the caller
    //      maintains. For the test cases below, `target_degree == evals.len() - 1`
    //      corresponds to "input is already at target degree" (e = 0, g_r = 1,
    //      identity). When implementing, pick a calling convention and stick
    //      to it: either take an extra argument, or document that
    //      `target_degree - 0` (caller-tracked `e`) is passed.
    //   3. Compute `e = target_degree - input_degree_bound`. Build a closure
    //      `gr_at = |x: Fp| -> Fp` implementing the closed form:
    //
    //        let rx = randomness * x;
    //        if rx == Fp::one() {
    //            // r·x = 1: the closed form 0/0 → direct sum is e + 1 ones.
    //            Fp::new((e as u64) + 1)
    //        } else {
    //            // (1 - (rx)^{e+1}) / (1 - rx)
    //            let num = Fp::one() - rx.pow((e as u64) + 1);
    //            let den = Fp::one() - rx;
    //            num * den.inverse().unwrap()
    //        }
    //
    //   4. Walk `domain.iter()` and `evals` in lockstep:
    //        output.push(evals[j] * gr_at(L[j]));
    //   5. Return.
    //
    // CAUTION: if `target_degree < evals.len() - 1` (i.e. degree *shrinking*,
    // not growing), this function is ill-defined; assert `target_degree >=
    // input_degree_bound` (equivalently, `e >= 0`). Reduction is done via
    // Quotient, not DegCor — the two operations are designed in tandem and
    // confusing them silently would corrupt the round-by-round degree
    // accounting.
    let _ = (evals, domain, target_degree, randomness);
    todo!()
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
    // TODO: pointwise random linear combination.
    //   1. Validate `evals_a.len() == evals_b.len()`.
    //   2. `output[j] = evals_a[j] + randomness * evals_b[j]`. One pass.
    //   3. Return.
    //
    // This is the simplest of the three modules' primitives — but it's
    // load-bearing for STIR soundness: like DegCor, it binds two codewords
    // by a random scalar, so a cheating prover can't fix one codeword's
    // error after seeing the challenge.
    let _ = (evals_a, evals_b, randomness);
    todo!()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// With `r = 0`, `g_r(X) = 1 + 0 + 0 + ... = 1` (a constant), so
    /// `f · g_r = f` and DegCor is the identity at the evaluation level.
    #[test]
    fn deg_cor_with_r_zero_is_identity() {
        // TODO:
        //   1. Build a domain `L` of size 8 and any evaluation vector
        //      `evals = [Fp::new(1), Fp::new(2), ..., Fp::new(8)]`.
        //   2. Pick `target_degree = some value > implied input bound` so
        //      `e > 0` and `g_r(X)` actually has more than one term.
        //   3. `result = deg_cor(&evals, &L, target_degree, Fp::zero())`.
        //   4. Assert `result == evals`: with `r = 0`, every `rx` term vanishes,
        //      `g_r(x) = 1` for all `x`, and the pointwise product is `f` itself.
        todo!()
    }

    /// When `target_degree` equals the input degree bound, `e = 0` and
    /// `g_r(X) = 1` (a single term). DegCor is the identity in that case
    /// regardless of `r`.
    #[test]
    fn deg_cor_with_target_equals_input_returns_input() {
        // TODO:
        //   1. Build a domain of size 8 and `evals = [Fp::new(1), ..., Fp::new(8)]`.
        //   2. Pick `target_degree = (whatever counts as the input degree bound
        //      in your calling convention)`. With `e = 0`, `g_r(X) = (rX)^0 = 1`.
        //   3. Call `deg_cor` with a non-trivial `randomness = Fp::new(13)`.
        //   4. Assert the result equals `evals` entry-by-entry.
        todo!()
    }

    /// `combine(a, b, 0) = a`: the `b` term vanishes.
    #[test]
    fn combine_with_r_zero_returns_first() {
        // TODO:
        //   1. Build `a = [Fp::new(10), Fp::new(20), Fp::new(30), Fp::new(40)]`.
        //   2. Build `b = [Fp::new(7), Fp::new(11), Fp::new(13), Fp::new(17)]`.
        //   3. Assert `combine(&a, &b, Fp::zero()) == a` entry-by-entry.
        todo!()
    }

    /// `combine(a, b, 1) = a + b` pointwise.
    #[test]
    fn combine_with_r_one_is_pointwise_sum() {
        // TODO:
        //   1. Same `a, b` as the previous test.
        //   2. Assert `combine(&a, &b, Fp::one())[j] == a[j] + b[j]` for all j.
        //   3. (Bonus: pick a non-trivial `r = Fp::new(5)` and check one entry
        //      by hand against `a[j] + 5 · b[j]`.)
        todo!()
    }
}
