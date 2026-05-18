//! The evaluation domain `L`.
//!
//! In Reed-Solomon (and FRI / STIR / WHIR), the codeword lives on a
//! distinguished subset `L` of the field. **`L` is fixed by system design**,
//! not picked at runtime — see `../../rs_foundations.md §1` for the full
//! story on who chooses it and why.
//!
//! For RS / FRI / STIR, `L` must be a **smooth multiplicative coset** of
//! `F_p^*`:
//!
//! ```text
//! L = c · <ω> = { c, c·ω, c·ω^2, ..., c·ω^{n-1} }
//! ```
//!
//! where:
//! - `ω` is a primitive `n`-th root of unity in `F_p`.
//! - `c` is a non-zero offset (often `c = 1`, in which case `L = <ω>` is the
//!   subgroup itself rather than a coset).
//! - `n = |L|` is a power of 2 (the "smooth" part — what makes the FFT possible).
//!
//! This module gives you a single struct, [`EvaluationDomain`], that holds
//! `(generator, size, offset)`. It exposes element access by index and an
//! iterator. The FFT module reads `(generator, offset)` to do its work.
//!
//! ## What is a coset, exactly?
//!
//! ### Mental model: a translated copy of a subgroup
//!
//! In an abelian group like `F_p^*` under multiplication, "translation"
//! means multiplication by a fixed element `c`. A **coset** of a
//! subgroup `H` by `c` is the set you get when you multiply every
//! element of `H` by `c` — same shape, shifted location.
//!
//! ### Worked example in `F_17`
//!
//! Take `p = 17`. The group `F_17^*` has 16 elements (`1, 2, ..., 16`).
//! Set `ω = 4`; it's a primitive 4th root of unity, because:
//!
//! ```text
//! ω^0 = 1
//! ω^1 = 4
//! ω^2 = 16     (since 4 · 4 = 16)
//! ω^3 = 13     (since 4 · 16 = 64 = 3·17 + 13)
//! ω^4 = 1      ← back to start, so the order is exactly 4
//! ```
//!
//! So the subgroup `H = ⟨ω⟩ = {1, 4, 16, 13}`. Now pick some `c ∈ F_17^*`
//! that is *not* in `H`, say `c = 3`. The **coset** `cH` is what you get
//! by multiplying every element of `H` by `c`:
//!
//! ```text
//! c · 1  = 3
//! c · 4  = 12
//! c · 16 = 48 mod 17 = 14
//! c · 13 = 39 mod 17 =  5
//!
//! cH = {3, 12, 14, 5}.
//! ```
//!
//! Two things to notice:
//!
//! - **Same size as `H`** (four elements each).
//! - **Same internal structure**: consecutive elements still differ by a
//!   factor of `ω = 4`, just starting from `3` instead of `1`. It's the
//!   same circle of multiplications, rotated.
//!
//! Keep going with other shifts and you partition all 16 nonzero elements:
//!
//! ```text
//!  1·H = { 1,  4, 16, 13}
//!  3·H = { 3, 12, 14,  5}
//!  2·H = { 2,  8, 15,  9}
//!  6·H = { 6,  7, 11, 10}
//! ```
//!
//! Four cosets, four elements each, 16 elements total — every nonzero
//! element appears in exactly one coset. This is a consequence of
//! **Lagrange's theorem**: cosets of a subgroup partition the parent
//! group into equal-sized chunks.
//!
//! **Question for the reader.** In `F_17^*`, the subgroup `⟨ω⟩ = {1, 4, 16, 13}`
//! with `ω = 4` (a primitive 4th root). Multiply every element by `c = 2`.
//! What set do you get? Why is it the same *size* as the subgroup but *no
//! longer closed under multiplication*?
//! Try to answer before reading on.
//!
//! Answer: `c·⟨ω⟩ = {2, 8, 15, 9}` (compute `2·1 = 2, 2·4 = 8, 2·16 = 32 ≡ 15,
//! 2·13 = 26 ≡ 9 mod 17`). Same size four because `c` is non-zero and
//! `F_17^*` is a group, so multiplication by `c` is a bijection. Closure
//! fails because, e.g., `2 · 2 = 4 ∉ {2, 8, 15, 9}` — products of two
//! elements of `c·H` land in `c² · H`, a *different* coset.
//!
//! ### Abstract definition
//!
//! For a group `G` and a subgroup `H ≤ G`, the **(left) coset** of `H`
//! by `c ∈ G` is:
//!
//! ```text
//! cH = { c · h : h ∈ H }
//! ```
//!
//! In abelian groups (`F_p^*` under multiplication is abelian), left and
//! right cosets coincide — we just say "coset". Properties:
//!
//! - `|cH| = |H|` (same size).
//! - If `c ∈ H`, then `cH = H` (the trivial coset — no real shift; the
//!   subgroup viewed as one of its own cosets).
//! - If `c ∉ H`, then `cH ∩ H = ∅` — a **proper coset**, disjoint from `H`.
//! - Any two cosets of the same `H` are either identical or disjoint.
//!
//! ### Two precise facts: not a group, but in set-bijection with `H`
//!
//! The "translated copy of a subgroup" picture is a useful first
//! mental model, but it's slightly imprecise. Two facts about the
//! coset `cH` carry the whole FRI / RS story, and the contrast
//! between them is what most pedagogical accounts gloss over:
//!
//! 1. **A coset is *not* a group.** When `c ∉ H`, `cH` fails the
//!    group axioms in two distinct ways:
//!    - **No identity.** `1 ∈ cH` would require some `h ∈ H` with
//!      `c·h = 1`, i.e. `h = c^{-1}` — which lies in `H` only when
//!      `c ∈ H`. On the `F_17` example, `1 ∉ cH = {3, 12, 14, 5}` ✓.
//!    - **Not closed under multiplication.**
//!      `(c·h_1) · (c·h_2) = c^2·(h_1·h_2)` sits in `c^2·H`, a
//!      *different* coset of `H`. On `F_17`: `3 · 5 = 15`, and
//!      `15 ∉ cH = {3, 12, 14, 5}` ✓.
//!
//! 2. ***But* `cH` *is* in set-bijection with `H`.** The map
//!    `φ: H → cH` defined by `φ(h) = c·h` is a perfect one-to-one
//!    correspondence, with inverse `φ^{-1}(x) = c^{-1}·x`. It is
//!    **not** a group homomorphism — the target has no group
//!    structure to be a homomorphism *into* — but it **is** an
//!    **isomorphism of sets**.
//!
//! The algebraic *axioms* of `H` do not transfer to `cH` — that's (1).
//! The *set-shape* of `H` does transfer, via `φ` — that's (2). A
//! common slip is to call `H` and `cH` "homomorphic"; that would
//! require group structure on both sides, which `cH` precisely
//! doesn't have. The right word is **isomorphic as sets**.
//!
//! ### What the bijection buys FFT, FRI, and Reed-Solomon
//!
//! Two properties FFT, FRI, and Reed-Solomon care about — both
//! preserved by `φ` even though group structure isn't:
//!
//! - **`n` distinct nonzero points.** Multiplication by the nonzero
//!   `c` is injective on `F_p^*`, so `|cH| = |H| = n` and no element
//!   of `cH` collapses to zero.
//! - **Polynomial evaluation transports cleanly.** Evaluating `p(X)`
//!   on `cH` equals evaluating `p(cX)` on `H`. That is exactly the
//!   "pre-scale the coefficients by powers of `c`" trick the FFT
//!   module relies on — one algorithm, two domain shapes.
//!
//! ### Reed-Solomon distance is set-theoretic
//!
//! The Reed-Solomon minimum-distance theorem only asks for `n`
//! **distinct nonzero** evaluation points; group structure plays no
//! role. By the **polynomial root bound** (a nonzero degree-`<d`
//! polynomial over a field has at most `d - 1` roots), two distinct
//! degree-`<d` polynomials agree on at most `d - 1` of any `n`
//! distinct points. The minimum distance is `n - d + 1` on a
//! subgroup or on a coset — same number, same Berlekamp-Welch
//! decoder, no change.
//!
//! ### FRI folding and the squaring map on cosets
//!
//! FRI halves the domain at each round via the squaring map
//! `x → x^2`. On the coset `cH`,
//!
//! ```text
//! (c · ω^i)^2  =  c^2 · (ω^2)^i,
//! ```
//!
//! which lands on `c^2 · ⟨ω^2⟩` — itself a clean coset of the squared
//! subgroup, of size exactly `n / 2`. No collisions, no fixed points
//! to special-case, no wasted randomness. (On a "pure" subgroup
//! containing `1`, squaring sends both `x` and `-x` to the same point
//! and `1 → 1` is a fixed point; the elegant translate-of-subgroup
//! recursion FRI depends on breaks down.) The offset `c` is what
//! makes the recursion uniform across rounds — an elegant algebraic
//! trick, not a hack.
//!
//! ### What FRI / STARK soundness rests on
//!
//! Soundness of FRI and STARKs does **not** depend on the evaluation
//! domain being a group. It rests on:
//!
//! - The **polynomial root bound / Schwartz-Zippel lemma** — random
//!   challenges hit roots of low-degree error polynomials with
//!   probability `≤ d / |F|`.
//! - **Low-degree algebraic folding** — a cheating prover's freedom
//!   in any round is bounded by the degree of the protocol's
//!   univariate slices, not by any group axioms on the domain.
//! - **Verifier randomness drawn from a sufficiently large `F`**.
//!
//! The coset offset `c` is a public constant (or derived from a public
//! random seed). It adds no algebraic structure an attacker could
//! exploit; what it buys is a clean disjointness between the
//! evaluation domain and the smaller **trace domain** the codeword is
//! interpolating from.
//!
//! ## Why STARKs use cosets
//!
//! The pedagogical answer: **to keep the codeword's evaluation domain
//! disjoint from another, smaller domain you also care about.**
//!
//! Concrete setting in a STARK:
//!
//! - The **trace** (witness data for some computation) is encoded as a
//!   polynomial interpolated through a small subgroup `H_trace = ⟨ω_n⟩`
//!   of size `n`.
//! - The **codeword** the prover commits to is that polynomial evaluated
//!   on a much bigger domain `L`, typically of size `n / ρ` where `ρ` is
//!   the rate (often `1/8`).
//!
//! If `L` is a **subgroup** containing `H_trace` (say `L = ⟨ω_{8n}⟩`, with
//! `H_trace` as a sub-subgroup), then the codeword's evaluation points
//! overlap with the trace points. Not catastrophic, but it complicates
//! protocol checks where the verifier wants to query the codeword at
//! points that *aren't* trace points.
//!
//! If instead `L = c · ⟨ω_{8n}⟩` is a **proper coset** — with `c` chosen
//! so the coset avoids `H_trace` — then the two domains are disjoint.
//! Codeword and trace live in different neighborhoods, which makes:
//!
//! - Constraint-polynomial evaluations cleaner (the verifier never
//!   accidentally hits a trace point).
//! - Soundness analyses tidier (no special-case handling for overlap).
//!
//! The original FRI paper used a subgroup; ethSTARK / Plonky2 / STIR
//! have all moved to cosets.
//!
//! ## In this module
//!
//! - [`EvaluationDomain::new_subgroup`] → `L = ⟨ω⟩`, contains `1`.
//!   Simple; fine for first-pass FRI-style code.
//! - [`EvaluationDomain::new_coset`] → `L = c · ⟨ω⟩`, shifted away from `1`.
//!   What STIR and production STARKs use.
//!
//! ## Smoothness, formally
//!
//! A finite multiplicative subgroup `H ⊆ F_p^*` is called **smooth** (of
//! 2-power order) when `|H| = 2^k` for some integer `k`. Equivalently, the
//! order of every element of `H` divides `2^k`. Goldilocks's
//! `p - 1 = 2^32 · (2^32 - 1)` has **2-adicity 32**, so `F_p^*` contains a
//! smooth subgroup of order `2^k` for every `0 ≤ k ≤ 32` — and none larger.
//!
//! Smoothness is the property that makes the Cooley-Tukey radix-2 FFT and
//! FRI's halving-step folding work. The recursion halves `|H|` at each
//! step by passing to `⟨ω²⟩ ⊂ ⟨ω⟩`, which has order `|H| / 2` (Lemma A in
//! `fft.rs`). For this to land cleanly on `|H| = 1` after `k` halvings,
//! `|H| = 2^k` is **necessary** — a domain of size `12` would jam at size 3.
//!
//! Smoothness has nothing to do with the algebraic structure of `F_p` as a
//! whole; only the existence of large 2-power-sized subgroups matters.
//! A field can be "FFT-friendly" purely because `p - 1` has a high
//! 2-adicity, independent of `p`'s size — see the field-comparison table
//! in `field.rs` for BabyBear vs Goldilocks vs BLS12-381 trade-offs.
//!
//! The FFT (`fft::fft_on_domain`) handles both via the "pre-scale
//! coefficients by powers of `c`" trick: a coset-FFT of `p` on `c · ⟨ω⟩`
//! is just a subgroup-FFT of `p_c(X) = p(cX)` on `⟨ω⟩`. One algorithm,
//! two domain shapes.
//!
//! Most of your testing should use the subgroup case; coset support
//! exists so the FFT abstraction stays general.

use crate::field::Fp;

/// A smooth multiplicative coset `L = offset · <generator>` in `F_p`.
#[derive(Clone, Debug)]
pub struct EvaluationDomain {
    /// `ω`, a primitive `size`-th root of unity in `F_p`.
    generator: Fp,
    /// `n = |L|`. Must be a power of 2.
    size: usize,
    /// The coset offset `c`. `Fp::one()` for a subgroup.
    offset: Fp,
    /// `log_2(size)`. Cached because we'll need it a lot.
    log_size: u32,
}

impl EvaluationDomain {
    /// Construct a smooth multiplicative subgroup of size `2^log_size`.
    ///
    /// The offset is `Fp::one()` so `L = <ω>`.
    ///
    /// Panics if `log_size > field::TWO_ADICITY`.
    pub fn new_subgroup(log_size: u32) -> Self {
        // TODO:
        //   - Use `Fp::primitive_root_of_unity(log_size)` to get the generator.
        //   - size = 1 << log_size.
        //   - offset = Fp::one().
        let generator = Fp::primitive_root_of_unity(log_size);
        let size = 1 << log_size;
        let offset = Fp::one();
        Self { generator, size, offset, log_size }
    }

    /// Construct a coset `L = offset · <ω>` of size `2^log_size`.
    ///
    /// Panics if `offset == Fp::zero()` or `log_size > field::TWO_ADICITY`.
    pub fn new_coset(log_size: u32, offset: Fp) -> Self {
        // TODO:
        //   - Assert offset != Fp::zero() with a clear message.
        //   - Same construction as new_subgroup, but with the given offset.
        //
        // Note: a strict reading of "coset" excludes offset = 1 (that's just
        // the subgroup), but it's harmless to allow it here. Don't add a
        // check against offset == Fp::one().
        assert_ne!(offset, Fp::zero(), "offset must not be zero");
        let generator = Fp::primitive_root_of_unity(log_size);
        let size = 1 << log_size;
        Self { generator, size, offset, log_size }
    }

    /// `n = |L|`.
    pub fn size(&self) -> usize {
        self.size
    }

    /// `log_2(|L|)`.
    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    /// The generator `ω`. A primitive `|L|`-th root of unity.
    pub fn generator(&self) -> Fp {
        self.generator
    }

    /// The coset offset `c`. `Fp::one()` if this domain is a subgroup.
    pub fn offset(&self) -> Fp {
        self.offset
    }

    /// The `i`-th element of `L`.
    ///
    /// By the definition of the domain `L = {c, c·ω, c·ω², ..., c·ω^(n-1)}`,
    /// this is just:
    ///
    /// ```text
    /// L[i] = c · ω^i = offset · generator^i
    /// ```
    ///
    /// Indices **wrap around**: `element(n)` returns `element(0)`,
    /// `element(n + 3)` returns `element(3)`, and so on. That follows from
    /// the cyclic structure — `ω` has order exactly `n`, so `ω^n = 1` and
    /// powers of `ω` only depend on the exponent `mod n`.
    ///
    /// # Example (`F_17` subgroup with `ω = 4`, `c = 1`, `n = 4`)
    ///
    /// ```text
    /// element(0) = 1 · 4^0 = 1
    /// element(1) = 1 · 4^1 = 4
    /// element(2) = 1 · 4^2 = 16
    /// element(3) = 1 · 4^3 = 13
    /// element(4) → wraps to element(0) = 1
    /// ```
    pub fn element(&self, i: usize) -> Fp {
        // TODO: return the `i`-th element of `L = c·⟨ω⟩`.
        //   1. Reduce `i mod n` — the domain wraps cyclically (`ω^n = 1`).
        //   2. Return `offset * generator.pow(i)`, i.e. `c · ω^i`.
        // See the worked example just above (and "What is a coset?" in module
        // docs) for why this is exactly the i-th element of `cH`.
        //
        // Reference implementation below.

        // Reduce i mod n first. Two reasons:
        //
        //   1. Implements the wraparound contract above (element(n) == element(0)).
        //   2. Keeps the exponent small so `pow` does fewer squarings.
        //      Square-and-multiply is O(log exp), so `element(10^9)` shouldn't
        //      take longer than `element(1)`.
        //
        // The reduction is correctness-preserving: ω is a primitive n-th root
        // of unity, hence ω^n = 1, hence ω^i = ω^(i mod n) for any i ≥ 0.
        let new_i = i % self.size;
        self.offset * self.generator.pow(new_i as u64)
    }

    /// Iterator over all `|L|` elements of `L`, in order
    /// `offset, offset·ω, offset·ω^2, ..., offset·ω^{n-1}`.
    ///
    /// Internally walks a running product to avoid `n` separate `pow`
    /// calls — `O(n)` field operations total instead of `O(n log n)`.
    pub fn iter(&self) -> DomainIter {
        // TODO:
        //   Construct a DomainIter at index 0 with current = self.offset.
        DomainIter {
            current: self.offset,
            generator: self.generator,
            remaining: self.size,
        }
    }
}

/// Iterator yielding the elements of an [`EvaluationDomain`] in order.
pub struct DomainIter {
    /// `current` is the next element to yield. Starts at `offset`, multiplied by
    /// `generator` after each `next()` call.
    current: Fp,
    generator: Fp,
    /// How many elements remain to be yielded.
    remaining: usize,
}

impl Iterator for DomainIter {
    type Item = Fp;

    fn next(&mut self) -> Option<Fp> {
        // TODO:
        //   - If remaining == 0, return None.
        //   - Otherwise:
        //       let out = self.current;
        //       self.current = self.current * self.generator;
        //       self.remaining -= 1;
        //       Some(out)
        if self.remaining == 0 {
            return None;
        }
        let out = self.current;
        self.current = self.current * self.generator;
        self.remaining -= 1;
        Some(out)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for DomainIter {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subgroup_first_element_is_one() {
        // TODO: domain.element(0) == Fp::one() when offset == 1.
        let domain = EvaluationDomain::new_subgroup(3);  // size 2^3 = 8
        assert_eq!(domain.element(0), Fp::one());
    }

    #[test]
    fn subgroup_elements_are_powers_of_generator() {
        // TODO: domain of log_size = 3 (size 8). For each i in 0..8,
        // domain.element(i) == generator.pow(i as u64).
        let domain = EvaluationDomain::new_subgroup(3); // size 2^3 = 8
        for i in 0..8 {
            assert_eq!(domain.element(i), domain.generator.pow(i as u64));
        }
    }

    #[test]
    fn iter_matches_element() {
        // TODO: collect the iterator, compare against `(0..size).map(|i| domain.element(i)).collect()`.
        let domain = EvaluationDomain::new_subgroup(3);
        let iter_elements: Vec<Fp> = domain.iter().collect();
        let expected_elements: Vec<Fp> = (0..domain.size()).map(|i| domain.element(i)).collect();
        assert_eq!(iter_elements, expected_elements);
    }

    #[test]
    fn iter_length_matches_size() {
        // TODO: domain.iter().count() == domain.size().
        let domain = EvaluationDomain::new_subgroup(3);
        assert_eq!(domain.iter().count(), domain.size());
    }

    #[test]
    fn iter_implements_exact_size() {
        // TODO: domain.iter().len() == domain.size().
        let domain = EvaluationDomain::new_subgroup(3);
        assert_eq!(domain.iter().len(), domain.size());
    }

    #[test]
    fn coset_first_element_is_offset() {
        // TODO: build a coset with offset != 1 (e.g., offset = Fp::new(7));
        // domain.element(0) == offset.
        let domain = EvaluationDomain::new_coset(3, Fp::new(7));
        assert_eq!(domain.element(0), domain.offset());
    }

    #[test]
    fn generator_has_correct_order() {
        // TODO: domain of log_size 4 (size 16). Verify:
        //   domain.generator().pow(16) == Fp::one()    (it's a 16th root)
        //   domain.generator().pow(8)  == -Fp::one()   (its order is exactly 16, not 8)
        let domain = EvaluationDomain::new_subgroup(4);
        assert_eq!(domain.generator().pow(16), Fp::one());
        assert_eq!(domain.generator().pow(8), -Fp::one());
    }

    #[test]
    fn element_wraps_around() {
        // TODO: domain.element(domain.size()) == domain.element(0).
        let domain = EvaluationDomain::new_subgroup(3);
        assert_eq!(domain.element(domain.size()), domain.element(0));
    }
}
