//! Out-of-domain (OOD) sampling for STIR's soundness amplification.
//!
//! ## What this module does
//!
//! Once the prover commits to its round-`i` function `g_i: L_i → F`
//! — the round-`i` committed function (a.k.a. `f_i`) on the round-`i`
//! evaluation domain `L_i`, a subset of the Goldilocks field `F = F_p`
//! — the verifier issues `s` **out-of-domain** challenges (`s` being
//! the per-round OOD sample count): random points `r ∈ F_p \ L_i`. For
//! each such `r`, the verifier asks the prover for `g_i(r)`, and bakes
//! those values into the Fiat-Shamir transcript.
//!
//! Why "out of domain"? Because the prover has already committed (via the
//! Merkle root) to the function `g_i: L_i → F` *on `L_i`*, but **not** to
//! any value at points outside `L_i`. If the verifier samples a fresh `r`
//! after the commitment is locked in, the prover cannot pre-pick `g_i(r)`
//! to be consistent with two incompatible "low-degree extensions" of the
//! committed values — assuming `g_i` is close to a unique degree-`< d`
//! polynomial `p` (the proximity target), `p(r)` is determined and the
//! prover is forced to reveal it.
//!
//! ## Johnson bound, briefly
//!
//! STIR's prover-soundness analysis hinges on **list-decoding the
//! committed function**. Roughly:
//!
//! > **Johnson bound (informal).** For a Reed-Solomon code of rate
//! > `ρ = d/n` and distance fraction `δ = 1 - √ρ - η` for some slack
//! > `η > 0`, the number of codewords within Hamming distance `δ·n` of
//! > any received word is at most `1 / (2 η √ρ)`.
//!
//! Call that list size `ℓ`. After committing, the prover is informally
//! "pinned" to a list of at most `ℓ` candidate low-degree polynomials.
//! OOD sampling collapses this list:
//!
//! > **Lemma (informal).** Fix any two distinct degree-`< d` polynomials
//! > `f, g` in the list. Their difference `f - g` has at most `d - 1` roots
//! > in `F_p`, so `Pr_{r ∈ F_p \ L}[f(r) = g(r)] ≤ (d - 1) / (|F_p| - |L|)`.
//! > Union-bounding over `ℓ²/2` pairs and demanding all `s` OOD samples
//! > agree, the probability that **any** two list members survive `s`
//! > points of OOD filtering is
//! >
//! > ```text
//! > ≤ (ℓ² / 2) · ((d - 1) / (|F_p| - |L|))^s.
//! > ```
//!
//! For typical Goldilocks-scale STIR params, **`s = 1` already gives
//! negligible probability** — so the list collapses to a single
//! polynomial after one OOD round.
//!
//! ## Worked example
//!
//! Take Goldilocks with `|F_p| ≈ 2^{64}`, a STIR round with `|L_i| ≈ 2^{20}`
//! evaluation points, degree bound `d ≈ 2^{18}`, and `s = 1` OOD sample.
//! The single-pair collision probability is
//!
//! ```text
//! (d - 1) / (|F_p| - |L_i|)
//!     ≈ 2^{18} / (2^{64} - 2^{20})
//!     ≈ 2^{18} / 2^{64}
//!     =  2^{-46}.
//! ```
//!
//! With list size `ℓ ≤ 32` (well within Johnson for rate `1/8` and modest
//! `η`), the `ℓ²/2 ≈ 512 = 2^9` union bound gives:
//!
//! ```text
//! total collision probability ≤ 2^9 · 2^{-46} = 2^{-37}.
//! ```
//!
//! Multiple OOD samples make this exponentially smaller. STIR typically
//! takes `s ≥ 2` to push below 128-bit soundness comfortably.
//!
//! **Question for the reader.** Why does the OOD point have to be outside
//! `L_i`? What happens if the verifier samples `r ∈ L_i` by accident?
//!
//! Answer: if `r ∈ L_i`, the prover *already committed* to `g_i(r)` inside
//! the Merkle tree at commit time. Sampling at such an `r` retrieves
//! exactly what the Merkle root binds — it provides no new constraint
//! beyond what a normal Merkle query already does. The list-decoding list
//! cannot be filtered using a point that's already determined by the
//! commitment. The whole point of "out of domain" is that the prover is
//! *not* pre-bound at `r`, so the response is forced to be the true
//! polynomial's value (assuming the prover is honest) or detectable
//! random noise (assuming the prover cheats).
//!
//! `// CAUTION:` to sample `r ∈ F_p \ L_i`, **rejection-sample** from
//! `transcript.squeeze_field()`: draw a candidate, check membership in
//! `forbidden_domain`, and retry if it landed inside. **Never** XOR, shift,
//! or otherwise "adjust" a candidate to push it out of `L_i` — that
//! biases the distribution and the soundness analysis (which assumes
//! uniform over `F_p \ L_i`) goes out the window.
//!
//! The rejection probability is `|L_i| / |F_p|`. For Goldilocks
//! `|F_p| ≈ 2^{64}` and `|L_i| ≤ 2^{32}`, that is at most `2^{-32}` per
//! draw — utterly negligible. Expected draws per accepted sample
//! is `1 + 2^{-32}`.

use reed_solomon::domain::EvaluationDomain;
use reed_solomon::field::Fp;
use reed_solomon::polynomial::UnivariatePoly;

/// Sample `count` field elements uniformly from `F_p \ forbidden_domain`.
///
/// Draws from the Fiat-Shamir transcript via
/// [`Transcript::squeeze_field`](crate::transcript::Transcript::squeeze_field)
/// and rejects any candidate that lies in `forbidden_domain`. Output is
/// length-`count`, all elements distinct only with high probability (the
/// caller does not de-duplicate explicitly because the collision
/// probability over `≤ s = 8` samples in a `2^{64}`-element field is
/// `≤ s² / |F_p| ≈ 2^{-58}`).
///
/// Used per round: the verifier calls
/// `sample_ood_points(&mut transcript, params.ood_samples, &domain.round_domains[i])`
/// after absorbing the round-`i` Merkle root.
///
/// Cost: `O(count · |L_i|)` field comparisons in the worst case via
/// linear membership scan. For STIR's regime (`count ≤ 8`, `|L| ≤ 2^{20}`)
/// this is fine; production code uses a `HashSet<Fp>` of `L_i` for `O(1)`
/// membership.
pub fn sample_ood_points(
    transcript: &mut crate::transcript::Transcript,
    count: u32,
    forbidden_domain: &EvaluationDomain,
) -> Vec<Fp> {
    // TODO:
    //   1. Allocate `out: Vec<Fp> = Vec::with_capacity(count as usize)`.
    //      WHY: known capacity → no reallocation.
    //   2. Loop until `out.len() == count as usize`:
    //        a. `candidate = transcript.squeeze_field()`.
    //           WHY: pulls a uniformly random field element from the
    //           Fiat-Shamir state; deterministic given prior absorbs.
    //        b. Check membership: scan `forbidden_domain.iter()` and test
    //           equality. If `candidate` appears, *discard* and loop.
    //           WHY: rejection sampling preserves uniformity over
    //           `F_p \ L_i`. See module-level CAUTION on why we don't
    //           "adjust" candidates instead.
    //        c. Otherwise push `candidate` into `out`.
    //      WHY: rejection probability `|L|/|F|` is `<< 1`, expected
    //      iterations per sample is `1 + O(|L|/|F|)`.
    //   3. Return `out`.
    //
    //   Implementation note: for `|L|` much larger, replace the scan with
    //   a `HashSet<Fp>` constructed once before the loop. The interface
    //   stays the same.
    let _ = (transcript, count, forbidden_domain);
    todo!()
}

/// Evaluate `poly` at `point`. Thin wrapper around
/// [`UnivariatePoly::evaluate`].
///
/// Exposed in this module so prover code calling "OOD reply = poly(r)"
/// reads naturally next to the [`sample_ood_points`] call.
pub fn evaluate_at(poly: &UnivariatePoly, point: Fp) -> Fp {
    // TODO:
    //   1. Return `poly.evaluate(point)`.
    //      WHY: trivial pass-through. The thin wrapper exists so callers
    //      write `ood::evaluate_at(...)` rather than reaching into
    //      `reed_solomon::UnivariatePoly`, which keeps the OOD-related
    //      operations grouped semantically.
    let _ = (poly, point);
    todo!()
}

/// Build the verifier's interpolant through OOD answers and shift answers.
///
/// In STIR's combined-low-degree check, the verifier needs the *unique*
/// polynomial of degree `< (count of input points)` that passes through:
///
/// - `(ood_points[i], ood_replies[i])` for `i in 0..ood_points.len()`,
/// - `(shift_points[j], shift_answers[j])` for `j in 0..shift_points.len()`.
///
/// `ood_points.len() == ood_replies.len()` and `shift_points.len() ==
/// shift_answers.len()` are required (debug-time assert recommended).
///
/// This is the polynomial the verifier expects the prover's claimed
/// folded function to match — disagreement at any point is detected via
/// the next round's Merkle queries.
///
/// Cost: `O((n_ood + n_shift)²)` field operations through
/// [`reed_solomon::interpolate::lagrange_interpolate`]. The combined point
/// count is small (`≤ 16` in practice), so the quadratic is cheap.
///
/// Paper reference: eprint 2024/390 §4, "Combined-check polynomial".
pub fn verifier_interpolant(
    ood_points: &[Fp],
    ood_replies: &[Fp],
    shift_points: &[Fp],
    shift_answers: &[Fp],
) -> UnivariatePoly {
    // TODO:
    //   1. Assert ood_points.len() == ood_replies.len()
    //      and shift_points.len() == shift_answers.len()
    //      and (ood_points.len() + shift_points.len()) >= 1.
    //      WHY: an empty input would have no unique interpolating
    //      polynomial; preconditions of `lagrange_interpolate`.
    //   2. Build `points: Vec<(Fp, Fp)>` of length
    //      `ood_points.len() + shift_points.len()`:
    //        - Push (ood_points[i], ood_replies[i]) for each i.
    //        - Push (shift_points[j], shift_answers[j]) for each j.
    //      WHY: lagrange_interpolate takes a flat slice of (x, y) pairs;
    //      we don't care about the relative order, only that x-coords
    //      are distinct.
    //   3. Return `reed_solomon::interpolate::lagrange_interpolate(&points)`.
    //      WHY: degree `< n` is the Polynomial Interpolation Theorem —
    //      a unique fit. `lagrange_interpolate` panics on duplicate
    //      x-coords, which propagates a useful error if the OOD point
    //      sampler ever accidentally produces a value that coincides
    //      with a shift point.
    let _ = (ood_points, ood_replies, shift_points, shift_answers);
    todo!()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Sampled OOD points are never inside the forbidden domain.
    #[test]
    fn ood_points_are_outside_forbidden_domain() {
        // TODO:
        //   1. domain = EvaluationDomain::new_subgroup(4); // |L| = 16
        //   2. Build a transcript with some fixed seed (e.g.,
        //      Transcript::new(b"test-seed")).
        //   3. let ood = sample_ood_points(&mut transcript, 5, &domain);
        //   4. let forbidden: HashSet<Fp> = domain.iter().collect();
        //   5. For each r in &ood: assert!(!forbidden.contains(r)).
        // WHY: the rejection-sampling contract — never returns a point
        // inside the committed evaluation domain.
        todo!()
    }

    /// With high probability, OOD points are pairwise distinct.
    #[test]
    fn ood_points_are_distinct_with_high_probability() {
        // TODO:
        //   1. Construct domain + transcript as above.
        //   2. Sample, say, 8 OOD points.
        //   3. Insert into a HashSet<Fp>; assert set.len() == 8.
        // WHY: collision probability `≤ 8²/|F_p| ≈ 2^{-58}` for
        // Goldilocks — never occurs in a test run. If this ever flakes,
        // the transcript is broken (degenerate distribution), not the OOD
        // logic.
        todo!()
    }

    /// The interpolant passes through every input point.
    #[test]
    fn interpolant_passes_through_all_input_points() {
        // TODO:
        //   1. ood_points = [Fp::new(101), Fp::new(102)].
        //      ood_replies = [Fp::new(7), Fp::new(9)].
        //   2. shift_points = [Fp::new(50), Fp::new(60)].
        //      shift_answers = [Fp::new(3), Fp::new(5)].
        //   3. let p = verifier_interpolant(&ood_points, &ood_replies,
        //                                   &shift_points, &shift_answers);
        //   4. For each (x, y) pair: assert_eq!(p.evaluate(x), y).
        //   5. Assert p.degree() == Some(3)  // 4 distinct points → degree < 4.
        // WHY: cross-checks the Lagrange-interpolation contract end-to-end
        // in the STIR module surface. Catches "wrong x-axis sign"
        // sign errors and "wrong y-axis vector" mixups in one shot.
        todo!()
    }
}
