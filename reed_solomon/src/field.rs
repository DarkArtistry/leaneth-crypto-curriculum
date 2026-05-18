//! The Goldilocks prime field `F_p` with `p = 2^64 - 2^32 + 1`.
//!
//! ## Why Goldilocks?
//!
//! The previous field (the Mersenne prime `2^61 - 1`) worked great for
//! sumcheck because sumcheck only needs random field elements — it never
//! cares about the multiplicative **subgroup structure**.
//!
//! Reed-Solomon codes are different: they need the **Number Theoretic
//! Transform** (the finite-field version of the FFT) to evaluate and
//! interpolate polynomials efficiently. The fast Cooley-Tukey algorithm
//! works beautifully when the transform length `N` is a power of two,
//! `N = 2^k`. For that we need a very specific element in the field: a
//! **primitive `2^k`-th root of unity**.
//!
//! ### What is a primitive `m`-th root of unity?
//!
//! An element `ω ∈ F_p` is an **`m`-th root of unity** if `ω^m = 1`. It is
//! **primitive** if `m` is the *smallest* positive integer with that
//! property — equivalently, the multiplicative **order** of `ω` is
//! exactly `m`.
//!
//! (Familiar analogy: in the complex numbers, `e^(2πi/m)` is a primitive
//! `m`-th root of unity. Its powers give the `m` evenly spaced points
//! around the unit circle — exactly the points an FFT butterflies over.
//! Same idea here, just everything is mod `p`.)
//!
//! ### When does such an element exist?
//!
//! The multiplicative group `F_p^*` (all nonzero field elements) is
//! **cyclic** of order `p - 1`. The relevant fact:
//!
//! > **Theorem.** In a cyclic group of order `N`, an element of order
//! > *exactly* `m` exists **if and only if** `m | N`.
//!
//! This combines two textbook results:
//!
//! - **Lagrange's theorem** (gives the `⇒` direction): in any finite
//!   group, the order of an element divides the order of the group.
//!   So if `h` has order `m` and `|G| = N`, then `m | N`.
//! - **Fundamental theorem of cyclic groups** (gives `⇐`, *and* tells
//!   you how to construct the element): if `G = ⟨g⟩` is cyclic of order
//!   `N` and `m | N`, then `h = g^(N/m)` has order exactly `m`.
//!     - `h^m = g^N = 1`, so `ord(h) | m`.
//!     - If `h^j = 1` for some `0 < j < m`, then `g^(jN/m) = 1` with
//!       `0 < jN/m < N`, contradicting that `g` has order exactly `N`.
//!     - Hence `ord(h) = m`. ✓
//!
//! That construction — `h = g^(N/m)` — is exactly what
//! [`Fp::primitive_root_of_unity`] does (with `N = p - 1`, `m = 2^k`,
//! and `g = MULTIPLICATIVE_GENERATOR`). So:
//!
//! - A primitive `2^k`-th root of unity exists in `F_p` iff `2^k | (p - 1)`.
//! - The largest valid `k` is called the **2-adicity** of `p - 1` — the
//!   highest power of 2 dividing `p - 1`.
//!
//! ### Mersenne flunks the test
//!
//! ```text
//! p     = 2^61 - 1
//! p - 1 = 2^61 - 2 = 2 · (2^60 - 1),    and 2^60 - 1 is odd.
//! ```
//!
//! (Why odd? `2^60` is even; subtract 1 → odd.)
//!
//! 2-adicity = **1**. That means the largest power-of-two FFT length we
//! can support is `N = 2^1 = 2`.
//!
//! Why is that a disaster? Cooley-Tukey works by recursively splitting an
//! `N`-point transform into two `N/2`-point transforms — the famous
//! "butterfly" steps. With `N = 2` you get exactly **one** butterfly and
//! no recursion at all. It's useless for any real workload.
//!
//! #### Concrete example: encoding a Reed-Solomon codeword
//!
//! Suppose we want to encode a message of length 1024 (a polynomial of
//! degree `< 1024`, with 1024 coefficients — exactly what our
//! [`encode::ReedSolomonCode`]'s `degree_bound = 1024` would set).
//! Reed-Solomon adds redundancy by evaluating that polynomial at *more*
//! points than it has coefficients. The **rate** `ρ` measures the
//! trade-off:
//!
//! > `ρ = message length / codeword length`
//!
//! A typical choice in STARK-family proof systems is `ρ = 1/8` — the
//! codeword is 8× longer than the message. For our 1024-coefficient
//! message:
//!
//! ```text
//! codeword length = 1024 / ρ = 1024 · 8 = 8192 = 2^13
//! ```
//!
//! To evaluate the polynomial at all 8192 points via the NTT, we need a
//! primitive `2^13`-th root of unity in `F_p`. Mersenne `2^61 - 1` only
//! has a primitive `2^1`-th root. It falls short by **12 powers of 2** —
//! a factor of `2^12 = 4096×` — nowhere near what real Reed-Solomon
//! codeword sizes demand.
//!
//! [`encode::ReedSolomonCode`]: crate::encode::ReedSolomonCode
//!
//! ### Goldilocks doesn't
//!
//! ```text
//! p     = 2^64 - 2^32 + 1
//! p - 1 = 2^32 · (2^32 - 1)
//!       = 2^32 · 3 · 5 · 17 · 257 · 65537
//! ```
//!
//! 2-adicity = **32** while `p` still fits comfortably in a 64-bit word.
//! That gives FFTs up to length `2^32 ≈ 4 billion` — "just right" for
//! every codeword size in this curriculum, with room to spare.
//!
//! Goldilocks is the field used by Plonky2, ethSTARK, and many production
//! STARK systems. Real implementations use heavily-optimized reduction
//! routines (see "the Goldilocks reduction trick" using `2^64 ≡ 2^32 - 1
//! mod p`); we use the simple `u128` reduction for clarity. The asymptotics
//! are the same.
//!
//! ### How does Goldilocks compare to other production fields?
//!
//! "FFT-friendly" is a spectrum, not a yes/no — different systems pick
//! different points on the trade-off curve between **2-adicity** (max
//! FFT size) and **bit width** (cost per field op). A few production
//! choices for context:
//!
//! ```text
//! Field           | bit width  | 2-adicity | Max FFT size | Used by
//! ----------------|------------|-----------|--------------|---------------------------
//! BabyBear        | 31-bit     |     27    |     2^27     | Plonky3 (small RISC zkVMs)
//! Mersenne 31     | 31-bit     |      1    |      2       | sumcheck only (this crate)
//! Goldilocks      | 64-bit     |     32    |     2^32     | Plonky2, ethSTARK, this crate
//! BN254 scalar    | 254-bit    |     28    |     2^28     | Groth16, PLONK (KZG-based)
//! BLS12-381 scalar| 255-bit    |     32    |     2^32     | Filecoin, Zcash Sapling
//! ```
//!
//! Two trade-offs to notice:
//!
//! - **Smaller fields are cheaper per operation.** BabyBear's 31-bit
//!   arithmetic fits in a single `u32` register (a single 64-bit multiply
//!   produces the product with no overflow handling). Goldilocks needs
//!   `u128` intermediates for `Mul`. Pairing-friendly fields like
//!   BN254/BLS12-381 are 4× wider, so each multiply is ~16× more expensive
//!   than Goldilocks. Smaller is faster *per op*.
//! - **Larger 2-adicity = bigger problems.** BabyBear caps at FFTs of
//!   size `2^27 ≈ 130M`, plenty for small workloads but tight for very
//!   large STARK traces. Goldilocks's `2^32` headroom is overkill for
//!   most applications, which is part of why it's the production default.
//!
//! Goldilocks sits on the **64-bit / `2^32`** point — wide enough to fit
//! every realistic FFT size, narrow enough that a single multiply is
//! one `u128` op. For our curriculum it's "just right" both for the
//! math (2-adicity of 32 is plenty) and for the implementation (one
//! `u128` intermediate, no SIMD/FMA tricks needed).
//!
//! ## Invariant
//!
//! Every `Fp` value is canonical: stored as a `u64` strictly less than
//! `MODULUS`. All constructors and arithmetic ops must preserve this.
//!
//! ## Worked example: a primitive 4th root of unity
//!
//! For `log_n = 2`: `omega = g^((p-1)/4)` where `g = 7`. Two facts:
//!
//! - `omega^4 = g^(p-1) = 1` — the first equality by combining exponents,
//!   the second by **Fermat's little theorem** (`a^(p-1) = 1` in `F_p^*`).
//! - `omega^2 = g^((p-1)/2) = -1` — the unique element of order exactly 2,
//!   which in `F_p` is `p - 1 ≡ -1 mod p`.
//!
//! Both are unit tests below.
//!
//! ## Why `g^((p-1)/2^k)` is a *primitive* `2^k`-th root of unity
//!
//! Two ingredients:
//!
//! 1. **Fermat's little theorem.** For prime `p` and any `a ∈ F_p \ {0}`,
//!    `a^(p-1) = 1`.
//! 2. **`g = 7` is a generator of `F_p^*`** (standard fact for Goldilocks):
//!    `g` has order *exactly* `p - 1`, i.e., `g^j = 1` iff `(p-1) | j`.
//!
//! Set `omega = g^((p-1)/2^k)`. Then:
//!
//! - `omega^(2^k) = g^((p-1)/2^k · 2^k) = g^(p-1) = 1` by Fermat. So
//!   `omega` is *some* `2^k`-th root of unity.
//! - For any `0 < j < 2^k`: if `omega^j = 1`, then `g^(j(p-1)/2^k) = 1`,
//!   so by (2) we'd need `(p-1) | j(p-1)/2^k`, i.e., `2^k | j`. But
//!   `0 < j < 2^k` rules that out.
//!
//! Conclusion: `omega` has order *exactly* `2^k` — that's "primitive".
//!
//! Note (2) is the load-bearing assumption: if `g` were just *some* root of
//! unity rather than a full generator, `omega` could still satisfy
//! `omega^(2^k) = 1` while having a smaller order. Picking a known generator
//! (`7` for Goldilocks) is what makes the construction work.

use rand::Rng;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The Goldilocks prime modulus: `p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001`.
pub const MODULUS: u64 = 0xFFFFFFFF00000001;

/// A multiplicative generator of `F_p^*`. Used to derive primitive roots of unity.
///
/// `7` is a known generator for Goldilocks (standard fact). Trust this — verifying
/// it requires checking that `7^((p-1)/q) ≠ 1` for every prime divisor `q` of
/// `p - 1`. Not hard, but not the point of this crate.
pub const MULTIPLICATIVE_GENERATOR: u64 = 7;

/// Two-adicity of `p - 1`. The largest `k` such that `2^k | (p - 1)`.
///
/// For Goldilocks: `p - 1 = 2^32 · (2^32 - 1)`, so `TWO_ADICITY = 32`.
pub const TWO_ADICITY: u32 = 32;

/// A field element of `F_p` (Goldilocks). Always reduced modulo `MODULUS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fp(u64);

impl Fp {
    /// Construct a new field element from an arbitrary `u64`. Reduces mod `MODULUS`.
    pub fn new(value: u64) -> Self {
        // TODO: reduce `value` mod MODULUS so the stored u64 is in [0, MODULUS).
        //
        // Two ways:
        //   (a) Simple, slow:    `value % MODULUS`
        //   (b) Branchy, faster: since u64::MAX < 2 * MODULUS, a single conditional
        //       subtract suffices: `if value < MODULUS { value } else { value - MODULUS }`.
        //
        // Either is fine for this crate. Pick one and stick with it.
        if value < MODULUS {
            Self(value)
        } else {
            Self(value - MODULUS)
        }
    }

    /// The additive identity, `0`.
    pub fn zero() -> Self {
        // TODO
        Self(0)
    }

    /// The multiplicative identity, `1`.
    pub fn one() -> Self {
        // TODO
        Self(1)
    }

    /// Sample a uniformly random field element.
    ///
    /// Use rejection sampling: draw a `u64`, retry if it's `>= MODULUS`. The
    /// rejection probability is `(2^64 - p) / 2^64 = (2^32 - 1) / 2^64 ≈ 2^-32`,
    /// so almost every draw is accepted on the first try.
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        // TODO: rejection-sample a u64 < MODULUS, wrap in Fp.
        // Hint: `rng.gen::<u64>()` and a `while r >= MODULUS { ... }` loop.
        loop {
            let r = rng.gen::<u64>();
            if r < MODULUS {
                return Self(r);
            }
        }
    }

    /// Get the underlying `u64` representation. For inspection / debug only.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Exponentiation by squaring: returns `self^exp` mod `MODULUS`.
    ///
    /// Standard square-and-multiply.
    pub fn pow(self, mut exp: u64) -> Self {
        // TODO: compute `self^exp` in O(log exp) field multiplications.
        //   1. Walk each bit of `exp` from low to high (square-and-multiply).
        //   2. When the current bit is 1, fold `base` into `result` (`result *= base`).
        //   3. Square `base` each iteration so `base = self^(2^i)` tracks the bit position.
        // See the "Worked example: a primitive 4th root of unity" section above —
        // `g.pow((p-1)/4)` is one place this routine gets hammered.
        //
        // Reference implementation below.
        let mut result = Self::one();
        let mut base = self;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }
        result
    }

    /// Multiplicative inverse via Fermat's little theorem: `self^(p-2)` for `p` prime.
    ///
    /// Returns `None` if `self == zero()`.
    pub fn inverse(self) -> Option<Self> {
        // TODO: special-case zero, then `self.pow(MODULUS - 2)`.
        if self == Self::zero() {
            None
        } else {
            Some(self.pow(MODULUS - 2))
        }
    }

    /// Return a **primitive `2^log_n`-th root of unity** in `F_p`.
    ///
    /// "Primitive" means: order exactly `2^log_n`, not just any root of `X^(2^log_n) - 1`.
    ///
    /// The construction is `omega = g^((p - 1) / 2^log_n)` where
    /// `g = MULTIPLICATIVE_GENERATOR`. Fermat's little theorem gives
    /// `omega^(2^log_n) = g^(p-1) = 1` (so `omega` is a root); the fact
    /// that `g` has order *exactly* `p - 1` rules out any smaller order.
    ///
    /// Panics if `log_n > TWO_ADICITY` (no such root exists in this field).
    ///
    /// # Examples (sanity)
    ///
    /// - `log_n = 0` → `omega = 1` (primitive 1st root).
    /// - `log_n = 1` → `omega = -1 = MODULUS - 1` (primitive 2nd root).
    /// - `log_n = 2` → some `omega` with `omega^4 = 1` and `omega^2 = -1`.
    pub fn primitive_root_of_unity(log_n: u32) -> Self {
        // TODO: produce an element of order exactly `2^log_n` in F_p.
        //   1. Assert `log_n <= TWO_ADICITY` (otherwise no such element exists).
        //   2. Compute `exponent = (p - 1) >> log_n` — an exact division because
        //      `2^log_n | p - 1` by the field's 2-adicity.
        //   3. Return `g.pow(exponent)` for `g = MULTIPLICATIVE_GENERATOR`.
        // See "Why `g^((p-1)/2^k)` is a *primitive* `2^k`-th root of unity" above.
        //
        // Reference implementation below.
        assert!(log_n <= TWO_ADICITY, "log_n must be <= TWO_ADICITY");
        let exponent = (MODULUS - 1) >> log_n;
        Self::new(MULTIPLICATIVE_GENERATOR).pow(exponent)
    }
}

// ============================================================================
// Arithmetic operator impls. Implement all of these. Use `u128` as the
// intermediate type for multiplication. For addition, either u128 or
// overflowing_add + conditional subtract works.
// ============================================================================

impl Add for Fp {
    type Output = Fp;

    fn add(self, rhs: Fp) -> Fp {
        // TODO: addition mod MODULUS.
        // Easiest version (educational): cast both to u128, add, reduce, cast back.
        // Faster version (optional): u64 overflowing_add + conditional subtract.
        let (sum, overflow) = self.0.overflowing_add(rhs.0);

        if overflow || sum >= MODULUS {
            Fp(sum.wrapping_sub(MODULUS))
        } else {
            Fp(sum)
        }
    }
}

impl AddAssign for Fp {
    fn add_assign(&mut self, rhs: Fp) {
        // TODO: trivially in terms of `+`.
        *self = *self + rhs;
    }
}

impl Sub for Fp {
    type Output = Fp;

    fn sub(self, rhs: Fp) -> Fp {
        // TODO: subtraction mod MODULUS.
        //
        // Goldilocks's modulus (2^64 - 2^32 + 1) is too close to u64::MAX
        // to leave headroom for the naive "add MODULUS, then subtract" trick
        // (see commented-out attempt below for what NOT to do). Use
        // `overflowing_sub` instead — let the borrow flag handle the
        // "self.0 < rhs.0" case via u64 wrap-around:
        //
        //   let (diff, borrow) = self.0.overflowing_sub(rhs.0);
        //   if borrow { Fp(diff.wrapping_add(MODULUS)) } else { Fp(diff) }

        // COMMON MISTAKE — DO NOT DO THIS:
        //
        //   if self.0 >= rhs.0 {
        //       Fp(self.0 - rhs.0)
        //   } else {
        //       Fp(self.0 + MODULUS - rhs.0)   // ← `self.0 + MODULUS` overflows u64!
        //   }
        //
        // Why it breaks: `self.0 + MODULUS > u64::MAX` whenever
        // `self.0 >= 2^32 - 1` — most of the field. Concretely, take
        // `self.0 = rhs.0 = MODULUS - 1` (= 2^64 - 2^32). Then
        // `self.0 + MODULUS ≈ 2^65`, which exceeds `u64::MAX = 2^64 - 1`.
        //   Debug build:   panics "attempt to add with overflow".
        //   Release build: silently wraps, garbage result.
        //
        // The naive form works fine for the sumcheck Mersenne `2^61 - 1`
        // (where `2 * MODULUS < 2^64` leaves a full bit of slack), but
        // Goldilocks doesn't have that slack — its 2-adicity-friendly
        // shape pushes `p` right up against u64::MAX.
        let (diff, borrow) = self.0.overflowing_sub(rhs.0);
        if borrow {
            // self.0 < rhs.0: diff = (self.0 - rhs.0) wrapped mod 2^64.
            // We want diff + MODULUS, which lives in (0, MODULUS) — fits in u64.
            Fp(diff.wrapping_add(MODULUS))
        } else {
            Fp(diff)
        }
    }
}

impl SubAssign for Fp {
    fn sub_assign(&mut self, rhs: Fp) {
        // TODO
        *self = *self - rhs;
    }
}

impl Mul for Fp {
    type Output = Fp;

    fn mul(self, rhs: Fp) -> Fp {
        // TODO: multiplication mod MODULUS.
        // Cast both to u128, multiply, reduce mod (MODULUS as u128), cast back.
        //
        // Note: `MODULUS as u128` is fine — MODULUS fits in u64, hence in u128.

        // COMMON MISTAKE — DO NOT DO THIS:
        //
        //   let product = (self.0 as u128) * (rhs.0 as u128);
        //   Fp(product as u64)              // ← just truncates the high 64 bits!
        //
        // Field elements `< p ≈ 2^64` give products up to `(p - 1)^2 ≈ 2^128`.
        // Casting `u128 as u64` keeps only the low 64 bits, which is NOT
        // the same as reducing mod p — the high 64 bits carry information
        // that has to be folded back in by `% MODULUS`.
        //
        // Concrete failure: take `a = b = 2^32` (well within F_p).
        //   product         = 2^64
        //   `as u64`        = 0           (truncated, since 2^64 mod 2^64 = 0)
        //   correct mod p   = 2^32 - 1    (since 2^64 ≡ 2^32 - 1 mod p)
        //
        // The error is *silent* whenever the product stays below MODULUS —
        // truncation happens to give the right answer for small inputs.
        // That's why `pow_matches_repeated_multiplication` (small x, small
        // exp) passed while `inverse_round_trip` failed: the latter calls
        // `pow(p - 2)`, which chains thousands of large products and
        // compounds every truncation error into garbage.
        let product = (self.0 as u128) * (rhs.0 as u128);
        Fp((product % MODULUS as u128) as u64)
    }
}

impl MulAssign for Fp {
    fn mul_assign(&mut self, rhs: Fp) {
        // TODO
        *self = *self * rhs;
    }
}

impl Neg for Fp {
    type Output = Fp;

    fn neg(self) -> Fp {
        // TODO: -x mod MODULUS. Special-case x == 0 to return 0 (not MODULUS).
        if self.0 == 0 {
            Fp(0)
        } else {
            Fp(MODULUS - self.0)
        }
    }
}

impl std::iter::Sum for Fp {
    fn sum<I: Iterator<Item = Fp>>(iter: I) -> Fp {
        // TODO: fold from Fp::zero() with `+`.
        iter.fold(Fp::zero(), |acc, x| acc + x)
    }
}

impl std::iter::Product for Fp {
    fn product<I: Iterator<Item = Fp>>(iter: I) -> Fp {
        // TODO: fold from Fp::one() with `*`.
        iter.fold(Fp::one(), |acc, x| acc * x)
    }
}

// ============================================================================
// Tests. The tests below are designed to be runnable as you implement, in
// order. Don't write your `Fp::new` then jump to FFT — fill in the operator
// impls in the same session and let `cargo test --lib` keep you honest.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulus_constant_is_correct() {
        // (1 << 64) doesn't fit in u64, but in u128 it does. Verify
        // MODULUS == 2^64 - 2^32 + 1 via u128 arithmetic.
        let expected: u128 = (1u128 << 64) - (1u128 << 32) + 1;
        assert_eq!(MODULUS as u128, expected);
    }

    #[test]
    fn zero_is_additive_identity() {
        // TODO: assert Fp::zero() + x == x.
        assert_eq!(Fp::zero() + Fp::new(42), Fp::new(42));
    }

    #[test]
    fn one_is_multiplicative_identity() {
        // TODO
        assert_eq!(Fp::one() * Fp::new(42), Fp::new(42));
    }

    #[test]
    fn add_then_sub_is_identity() {
        // TODO: (x + y) - y == x for some non-trivial x, y.
        assert_eq!((Fp::new(42) + Fp::new(13)) - Fp::new(13), Fp::new(42));
    }

    #[test]
    fn negation_is_additive_inverse() {
        // TODO: x + (-x) == zero.
        assert_eq!(Fp::new(42) + (-Fp::new(42)), Fp::zero());
    }

    #[test]
    fn inverse_round_trip() {
        // TODO: for non-zero x, x * x.inverse().unwrap() == one().
        assert_eq!(Fp::new(42) * Fp::new(42).inverse().unwrap(), Fp::one());
    }

    #[test]
    fn pow_zero_is_one() {
        // TODO: x.pow(0) == one(), even for x = zero.
        assert_eq!(Fp::new(42).pow(0), Fp::one());
        assert_eq!(Fp::zero().pow(0), Fp::one());
    }

    #[test]
    fn pow_matches_repeated_multiplication() {
        // TODO: small x and small exp, check pow against the unrolled product.
        assert_eq!(Fp::new(2).pow(4), Fp::new(2) * Fp::new(2) * Fp::new(2) * Fp::new(2));
        assert_eq!(Fp::new(3).pow(3), Fp::new(3) * Fp::new(3) * Fp::new(3));
    }

    #[test]
    fn primitive_root_of_unity_squares_to_one_at_log_n_one() {
        // TODO: omega = primitive_root_of_unity(1) should equal -one().
        // Therefore omega.pow(2) == one() and omega != one().
        assert_eq!(Fp::primitive_root_of_unity(1), Fp(MODULUS - 1));
        assert_eq!(Fp::primitive_root_of_unity(1).pow(2), Fp::one());
        assert_ne!(Fp::primitive_root_of_unity(1), Fp::one());
    }

    #[test]
    fn primitive_root_of_unity_has_correct_order() {
        // TODO: for log_n = 4, omega = primitive_root_of_unity(4) should satisfy:
        //   omega.pow(16) == one()         (it's a 16th root)
        //   omega.pow(8)  == -one()        (its 8th power is -1, so order is exactly 16, not 8)
        //
        // The second check is the "primitive" part — order divides 16 and isn't 8,
        // hence is exactly 16.
        assert_eq!(Fp::primitive_root_of_unity(4).pow(16), Fp::one());
        assert_eq!(Fp::primitive_root_of_unity(4).pow(8), -Fp::one());
    }

    #[test]
    fn primitive_root_of_unity_at_two_adicity_works() {
        // TODO: log_n = TWO_ADICITY (= 32) should not panic and should produce
        // an element of order exactly 2^32. (Don't actually check the order
        // directly — that would take ~2^32 multiplications. Just check
        // omega.pow(1u64 << 32) == one() and omega.pow(1u64 << 31) == -one().)
        assert_eq!(Fp::primitive_root_of_unity(TWO_ADICITY).pow(1u64 << 32), Fp::one());
        assert_eq!(Fp::primitive_root_of_unity(TWO_ADICITY).pow(1u64 << 31), -Fp::one());
    }

    #[test]
    #[should_panic]
    fn primitive_root_of_unity_above_two_adicity_panics() {
        // TODO: log_n = TWO_ADICITY + 1 must panic.
        // No primitive 2^33-th root of unity exists in F_p.
        let log_n = TWO_ADICITY + 1;
        let _ = Fp::primitive_root_of_unity(log_n);
    }
}
