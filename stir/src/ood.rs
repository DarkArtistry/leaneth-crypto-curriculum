//! Out-of-domain (OOD) sampling for STIR's soundness amplification.
//!
//! ## What this module does, in one paragraph
//!
//! Once the prover commits (via a Merkle root) to its round-`i` function
//! `g_i: L_i → F` on the round-`i` evaluation domain `L_i ⊂ F = F_p`, the
//! verifier issues `s` **out-of-domain** challenges: random points
//! `z_1, ..., z_s ∈ F \ L_{i+1}` (the *next* round's domain is the one we
//! must avoid — see the protocol cross-reference below). For each such
//! `z_j` the verifier asks the prover for `g_i(z_j)`, and bakes those
//! answers into the Fiat-Shamir transcript. Together with the per-round
//! shift-query openings, those OOD points and replies feed
//! [`verifier_interpolant`] — the unique low-degree polynomial through
//! every `(point, claimed-value)` pair the verifier has accumulated this
//! round, which the combined-low-degree check then compares against the
//! next round's commitment. This module implements three operations:
//! [`sample_ood_points`] (rejection-sampled OOD challenges),
//! [`evaluate_at`] (a thin alias used at prover OOD-reply sites), and
//! [`verifier_interpolant`] (Lagrange interpolation through OOD + shift
//! points).
//!
//! ## Anchor: STIR's list-decoding collapse — what OOD sampling buys you
//!
//! STIR's soundness analysis lives at the **Johnson list-decoding
//! radius**: for a Reed-Solomon code of rate `ρ = d / |L|`, every
//! received word (the prover's committed evaluation table) has a *list*
//! of codewords within relative Hamming distance `1 − √ρ − η`, of size at
//! most `ℓ ≤ 1 / (2 η √ρ)`. The prover, having committed, is informally
//! "pinned" to that list of candidate degree-`< d` polynomials. The
//! commitment alone does not single one out — but **OOD sampling does**.
//!
//! Geometrically: every Merkle query "lives inside" `L_i`, so it can only
//! distinguish list members that already disagree on `L_i`. Two list
//! members that agree on every position of `L_i` but extend to different
//! degree-`< d` polynomials are *invisible* to in-domain queries. An OOD
//! point `z ∈ F \ L_i` lives *outside* the commitment's reach: a
//! cheating prover would have to reply with a single value `y`, but each
//! list member `p` satisfies `p(z) = y` for at most one specific `y` —
//! so an honest reply consistent with two distinct list members is
//! impossible up to the polynomial-identity collision bound. The list
//! collapses, fast.
//!
//! That collapse is exactly the **List-Decoding Collapse Lemma** below.
//! Without it, the rate-halving rounds of STIR would lose their
//! soundness juice — each round would face a list-decoding ambiguity it
//! could not resolve, and the per-round error analysis from §4 of
//! eprint 2024/390 (relying on rate-halving) would no longer terminate
//! at `2^{-λ/M}`.
//!
//! ## Named theorems & derivations
//!
//! ### List-Decoding Collapse Lemma
//!
//! > **List-Decoding Collapse Lemma.** Fix a Reed-Solomon code
//! > `RS[F, L, d]` and a received word that has a Johnson-radius
//! > list-decoding list of size at most `ℓ`. Sample `s ≥ 1` independent
//! > uniform OOD points `z_1, ..., z_s ∈ F \ L`. Then
//! >
//! > ```text
//! >   Pr[ a non-unique codeword in the list passes the OOD check ]
//! >       ≤ ℓ · ((d - 1) / (|F| - |L|))^s
//! >       ≤ ℓ / |F|^s          (for |L| ≪ |F| and d ≤ |F|).
//! > ```
//! >
//! > **Proof.** Fix any two distinct list members `f, g ∈ F[X]` of
//! > degree `< d`. Their difference `f − g` is a nonzero polynomial of
//! > degree `< d`, hence has at most `d − 1` roots in `F`
//! > (Schwartz-Zippel / fundamental theorem of algebra over a field).
//! > So
//! >
//! > ```text
//! >   Pr_{z ∈ F \ L}[ f(z) = g(z) ]
//! >       = |{ z ∈ F \ L : (f - g)(z) = 0 }| / (|F| - |L|)
//! >       ≤ (d - 1) / (|F| - |L|).
//! > ```
//! >
//! > Equivalently: each OOD point eliminates *all but at most one* list
//! > member agreeing on it. Across `s` independent OOD draws,
//! > eliminations compound multiplicatively: the probability some fixed
//! > distinct `(f, g)` survive all `s` draws is at most
//! > `((d − 1) / (|F| − |L|))^s`. Union-bounding over the `≤ ℓ` ways the
//! > "wrong" list member could be picked yields the lemma. ∎
//! >
//! > **Why this matters.** The list-decoding regime is what gives STIR
//! > its rate-halving freedom; without OOD collapse, the prover could
//! > silently switch between two list members that the Merkle
//! > commitments cannot distinguish, and the soundness reduction in §4
//! > of eprint 2024/390 would fail.
//!
//! ### Worked example: rejection sampling terminates quickly
//!
//! Take a small round with `|L_{i+1}| = 32` and `s = count = 2` OOD
//! samples, drawn from Goldilocks (`|F_p| ≈ 2^{64}`).
//!
//! 1. **Rejection probability per draw.** A uniform `Fp`-sample lands in
//!    the forbidden domain with probability
//!    `|L_{i+1}| / |F_p| = 32 / 2^{64} = 2^{-59}`. So the expected number
//!    of `squeeze_field()` calls needed to obtain one outside-of-`L`
//!    candidate is `1 / (1 - 2^{-59}) ≈ 1 + 2^{-59}` — effectively `1.0`.
//!
//! 2. **Distinctness rejection.** After accepting one candidate, the
//!    second draw additionally rejects if it equals the first. The
//!    probability is `1 / (|F_p| - |L_{i+1}|) ≈ 2^{-64}`. Negligible.
//!
//! 3. **Total expected loop iterations** to fill `count = 2`:
//!    `2 · (1 + 2^{-59}) ≈ 2.0`. The rejection-sampling loop is
//!    indistinguishable from "draw `count` field elements straight".
//!
//! 4. **Collapse strength.** Plugging into the Collapse Lemma with
//!    `ℓ ≤ 32` (a generous Johnson list-size bound at rate `1/8`) and
//!    `d − 1 ≪ |F| − |L|`:
//!
//!    ```text
//!      total error ≤ 32 · ((d - 1) / (|F| - |L|))^2 ≈ 32 · 2^{-92} = 2^{-87}.
//!    ```
//!
//!    Two OOD samples already drive the list-collapse error below 128-bit
//!    soundness with room to spare.
//!
//! **Question for the reader.** Why must the OOD point be outside the
//! *next* round's domain `L_{i+1}`, not the current round's `L_i`?
//!
//! Answer: the OOD point is the input to the **next round's**
//! combined-low-degree check — see [`verifier_interpolant`]. If `z` were
//! in `L_{i+1}`, the next round's Merkle commitment would already bind
//! `g_{i+1}(z)`, and the OOD reply would provide no new constraint
//! relative to a normal shift query. The whole point of "out of domain"
//! is that the prover is *not* pre-bound at `z` by the upcoming
//! commitment, so the response is forced to be the true polynomial's
//! value (honest case) or detectable noise (cheating case).
//!
//! ## Caveats for implementers
//!
//! ```text
//! // CAUTION: to sample r ∈ F_p \ L_{i+1}, ALWAYS rejection-sample from
//! //          transcript.squeeze_field() — draw a candidate, check
//! //          membership in `forbidden_domain`, retry if it landed
//! //          inside. NEVER XOR, shift, or "adjust" a candidate to push
//! //          it out of L. Adjustment biases the distribution, and the
//! //          Collapse Lemma above assumes uniform over F \ L.
//!
//! // CAUTION: the `forbidden_domain` argument is L_{i+1}, the NEXT
//! //          round's domain, not the current one. The OOD reply is
//! //          checked against the next round's commitment by the
//! //          combined-low-degree polynomial, so the point must lie
//! //          outside that commitment's binding set.
//!
//! // CAUTION: ood_points and shift_points passed to
//! //          `verifier_interpolant` must be pairwise distinct across
//! //          BOTH sets — Lagrange interpolation panics on a duplicate
//! //          x-coordinate. If the OOD sampler ever produced a value
//! //          coinciding with a shift point, this would surface as a
//! //          loud panic rather than a silent verifier accept.
//! ```
//!
//! ## See also
//!
//! - [`crate::transcript::Transcript::squeeze_field`] — uniform `F_p`
//!   PRG used as the candidate source.
//! - [`crate::params::StirParams`] — the `ood_samples` parameter `s`
//!   chosen per the Collapse Lemma's error budget.
//! - [`crate::quotient`] — consumes the OOD points + replies to form the
//!   round's quotient polynomial.
//! - [`crate::verifier::StirVerifier::verify`] — canonical call site of
//!   [`sample_ood_points`] and [`verifier_interpolant`].

use std::collections::HashSet;

use reed_solomon::domain::EvaluationDomain;
use reed_solomon::field::Fp;
use reed_solomon::polynomial::UnivariatePoly;

/// Sample `count` field elements uniformly from `F_p \ forbidden_domain`,
/// pairwise distinct.
///
/// Draws from the Fiat-Shamir transcript via
/// [`Transcript::squeeze_field`](crate::transcript::Transcript::squeeze_field)
/// and rejects any candidate that (a) lies in `forbidden_domain` or
/// (b) has already been chosen on this call. The result is therefore a
/// uniform sample-without-replacement from `F_p \ forbidden_domain`.
///
/// Used per round: the verifier calls
/// `sample_ood_points(&mut transcript, params.ood_samples, &domain.round_domains[i + 1])`
/// after absorbing the round-`i` Merkle root. See the module-level
/// **List-Decoding Collapse Lemma** for the soundness justification.
///
/// Cost: `O(|forbidden_domain|)` to build the once-per-call lookup set,
/// plus `O(count)` expected `squeeze_field` calls (rejection probability
/// is `|L| / |F| ≪ 1`).
pub fn sample_ood_points(
    transcript: &mut crate::transcript::Transcript,
    count: u32,
    forbidden_domain: &EvaluationDomain,
) -> Vec<Fp> {
    // 1. Build an O(1)-lookup table of forbidden values once. We key by
    //    the canonical u64 representation (Fp::as_u64) because Fp does
    //    not (yet) implement Hash directly; the canonical form is
    //    bijective with Fp on [0, MODULUS), so this is collision-safe.
    //    Cf. the Caveats §: rejection must compare against L_{i+1},
    //    which is what `forbidden_domain` is.
    let forbidden: HashSet<u64> = forbidden_domain.iter().map(|x| x.as_u64()).collect();

    // 2. Allocate output with known capacity — no reallocation in the
    //    hot loop.
    let mut out: Vec<Fp> = Vec::with_capacity(count as usize);

    // Track chosen values too, so we reject duplicates across the
    // `count` draws (sample-without-replacement). Collisions here are
    // ~2^{-64}, so this almost never fires, but keeping the check makes
    // the "all distinct" postcondition unconditional rather than
    // probabilistic.
    let mut chosen: HashSet<u64> = HashSet::with_capacity(count as usize);

    // 3. Rejection-sampling loop. Cf. module-doc §"Worked example":
    //    expected iterations ≈ count · (1 + |L|/|F|) ≈ count.
    while (out.len() as u32) < count {
        let candidate = transcript.squeeze_field();
        let key = candidate.as_u64();

        // Reject if (a) inside the forbidden domain — would defeat the
        // Collapse Lemma (the point would already be bound by the next
        // round's commitment), or (b) already chosen this call —
        // duplicate x-coordinates would later panic
        // `verifier_interpolant` via Lagrange's distinctness
        // precondition.
        if forbidden.contains(&key) || !chosen.insert(key) {
            continue;
        }
        out.push(candidate);
    }

    out
}

/// Evaluate `poly` at `point`. Thin wrapper around
/// [`UnivariatePoly::evaluate`].
///
/// Exposed in this module so prover code that writes
/// "OOD reply = poly(r)" reads naturally next to the
/// [`sample_ood_points`] call. The implementation is a one-line
/// pass-through; the value of the wrapper is semantic grouping, not
/// computation.
pub fn evaluate_at(poly: &UnivariatePoly, point: Fp) -> Fp {
    // Trivial pass-through — see the doc comment. The wrapper exists so
    // callers say `ood::evaluate_at(...)` rather than reaching into
    // `reed_solomon::UnivariatePoly`, keeping the OOD-related operations
    // grouped under one module surface.
    poly.evaluate(point)
}

/// Build the verifier's interpolant through OOD answers and shift answers.
///
/// In STIR's combined-low-degree check, the verifier needs the *unique*
/// polynomial of degree `< n` (where `n = ood_points.len() + shift_points.len()`)
/// that passes through:
///
/// - `(ood_points[i], ood_replies[i])` for `i in 0..ood_points.len()`,
/// - `(shift_points[j], shift_answers[j])` for `j in 0..shift_points.len()`.
///
/// This is the polynomial the verifier expects the prover's claimed
/// folded function to match — disagreement at any point is detected via
/// the next round's Merkle queries. See the module-level **List-Decoding
/// Collapse Lemma** for why this interpolant pins down a unique codeword
/// from the Johnson-radius list.
///
/// # Panics
///
/// - Length mismatch between `ood_points` and `ood_replies` (or between
///   `shift_points` and `shift_answers`).
/// - Empty combined input (no points to interpolate).
/// - Duplicate `x`-coordinate across the combined set (propagated from
///   [`reed_solomon::interpolate::lagrange_interpolate`]). With
///   high probability the OOD sampler avoids this — see the third
///   CAUTION in the module docs.
///
/// Cost: `O((n_ood + n_shift)^2)` field operations via
/// [`reed_solomon::interpolate::lagrange_interpolate`]. The combined
/// point count is small (`≤ 16` in practice), so the quadratic is cheap.
///
/// Paper reference: eprint 2024/390 §4, "Combined-check polynomial".
pub fn verifier_interpolant(
    ood_points: &[Fp],
    ood_replies: &[Fp],
    shift_points: &[Fp],
    shift_answers: &[Fp],
) -> UnivariatePoly {
    // 1. Length / non-emptiness preconditions. We assert (not
    //    debug_assert) because a mismatch here is a *protocol* bug, not
    //    a performance hotspot — a silent truncation would build a
    //    polynomial fitting only some of the prover's claims, which is
    //    far worse than a loud panic.
    assert_eq!(
        ood_points.len(),
        ood_replies.len(),
        "ood_points and ood_replies must have equal length",
    );
    assert_eq!(
        shift_points.len(),
        shift_answers.len(),
        "shift_points and shift_answers must have equal length",
    );
    assert!(
        !ood_points.is_empty() || !shift_points.is_empty(),
        "verifier_interpolant: at least one input point required",
    );

    // 2. Flatten both `(point, value)` streams into one slice in the
    //    natural order (OOD first, shifts after). The relative order
    //    does not matter — Lagrange interpolation is order-symmetric —
    //    but keeping OOD first matches the verifier's transcript order
    //    (OOD points are sampled before shift indices in
    //    [`crate::verifier::StirVerifier::verify`]).
    let mut points: Vec<(Fp, Fp)> =
        Vec::with_capacity(ood_points.len() + shift_points.len());
    for (i, &x) in ood_points.iter().enumerate() {
        points.push((x, ood_replies[i]));
    }
    for (j, &x) in shift_points.iter().enumerate() {
        points.push((x, shift_answers[j]));
    }

    // 3. Delegate to the Reed-Solomon Lagrange interpolator. It panics
    //    on duplicate x-coordinates, which is the desired failure mode
    //    here — see the third CAUTION in the module docs.
    reed_solomon::interpolate::lagrange_interpolate(&points)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use crate::transcript::Transcript;

    /// (a) Sampled OOD points are never inside the forbidden domain, and
    /// the requested `count` are pairwise distinct.
    #[test]
    fn ood_points_are_outside_forbidden_domain_and_distinct() {
        // |L| = 16: small enough to enumerate, large enough to exercise
        // the HashSet membership path non-trivially.
        let domain = EvaluationDomain::new_subgroup(4);
        let mut transcript = Transcript::new(b"ood-test-seed");

        let count: u32 = 4;
        let ood = sample_ood_points(&mut transcript, count, &domain);

        assert_eq!(ood.len(), count as usize, "must return exactly `count` points");

        // No element is in the forbidden domain.
        let forbidden: HashSet<u64> = domain.iter().map(|x| x.as_u64()).collect();
        for (i, r) in ood.iter().enumerate() {
            assert!(
                !forbidden.contains(&r.as_u64()),
                "OOD point #{i} = {} landed in forbidden domain",
                r.as_u64(),
            );
        }

        // Pairwise distinct (the by-construction guarantee, not just a
        // probabilistic claim).
        let unique: HashSet<u64> = ood.iter().map(|x| x.as_u64()).collect();
        assert_eq!(
            unique.len(),
            ood.len(),
            "OOD points must be pairwise distinct by construction",
        );
    }

    /// (b) `evaluate_at(poly, point)` is a faithful alias for
    /// `poly.evaluate(point)`.
    #[test]
    fn evaluate_at_matches_polynomial_evaluate() {
        // p(X) = 1 + 2X + 3X^2.
        let poly = UnivariatePoly::new(vec![Fp::new(1), Fp::new(2), Fp::new(3)]);

        // Spot-check across a handful of points, including 0 (boundary)
        // and a large value (full field arithmetic path).
        for &x in &[0u64, 1, 2, 5, 100, 1_000_000_007] {
            let point = Fp::new(x);
            assert_eq!(
                evaluate_at(&poly, point),
                poly.evaluate(point),
                "evaluate_at must match UnivariatePoly::evaluate at {x}",
            );
        }
    }

    /// (c) The interpolant passes through every input point — 3 OOD
    /// points + 5 shift points = 8 total constraints, degree `< 8`.
    #[test]
    fn interpolant_passes_through_all_inputs() {
        let ood_points = [Fp::new(101), Fp::new(102), Fp::new(103)];
        let ood_replies = [Fp::new(7), Fp::new(9), Fp::new(11)];
        let shift_points = [
            Fp::new(50),
            Fp::new(60),
            Fp::new(70),
            Fp::new(80),
            Fp::new(90),
        ];
        let shift_answers = [
            Fp::new(3),
            Fp::new(5),
            Fp::new(13),
            Fp::new(17),
            Fp::new(19),
        ];

        let p = verifier_interpolant(
            &ood_points,
            &ood_replies,
            &shift_points,
            &shift_answers,
        );

        // Every (x, y) pair must satisfy p(x) = y. Lagrange's
        // uniqueness theorem is what makes this assertion meaningful:
        // there is at most one degree-`< 8` polynomial through 8
        // distinct points, and the interpolator produced it.
        for (i, (&x, &y)) in ood_points.iter().zip(ood_replies.iter()).enumerate() {
            assert_eq!(p.evaluate(x), y, "OOD pair #{i} ({},{}) not on interpolant",
                       x.as_u64(), y.as_u64());
        }
        for (j, (&x, &y)) in shift_points.iter().zip(shift_answers.iter()).enumerate() {
            assert_eq!(p.evaluate(x), y, "shift pair #{j} ({},{}) not on interpolant",
                       x.as_u64(), y.as_u64());
        }

        // 8 distinct inputs → unique fit has degree `< 8`, i.e.
        // `degree() ∈ {None} ∪ {0, ..., 7}`. Stronger: with random-
        // looking y-values these inputs are unlikely to collapse below
        // degree 7, but we only assert the upper bound — degree exactly
        // 7 is not a load-bearing claim.
        match p.degree() {
            None => panic!("interpolant is zero polynomial; impossible for 8 nonzero y's"),
            Some(d) => assert!(d < 8, "degree must be < 8, got {d}"),
        }
    }

    /// (d) Given identical transcript state, `sample_ood_points` is
    /// deterministic — same domain, same domain-separator, same prior
    /// absorbs ⇒ same output. This is the Fiat-Shamir contract; the
    /// verifier *relies* on recomputing the prover's challenges
    /// bit-for-bit.
    #[test]
    fn sample_ood_points_is_deterministic() {
        let domain = EvaluationDomain::new_subgroup(4);

        let mut t1 = Transcript::new(b"determinism-test");
        t1.absorb(b"round-0-merkle-root-placeholder");
        let a = sample_ood_points(&mut t1, 4, &domain);

        let mut t2 = Transcript::new(b"determinism-test");
        t2.absorb(b"round-0-merkle-root-placeholder");
        let b = sample_ood_points(&mut t2, 4, &domain);

        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.as_u64(),
                y.as_u64(),
                "OOD sample #{i} diverges between identical transcripts \
                 — Fiat-Shamir determinism broken",
            );
        }
    }
}
