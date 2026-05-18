//! A toy prime field `F_p` over the Mersenne prime `p = 2^61 - 1`.
//!
//! This field is small enough to debug easily but large enough that sumcheck
//! soundness is meaningful (~2^-61 per round). For real protocols we'd graduate
//! to a 64-bit smooth field like Goldilocks or to `ark-ff`.
//!
//! ## Invariant
//!
//! Every `Fp` value is canonical: stored as a `u64` strictly less than `MODULUS`.
//! All constructors and arithmetic ops must preserve this.

use rand::Rng;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The prime modulus: `2^61 - 1` (a Mersenne prime).
pub const MODULUS: u64 = (1 << 61) - 1;

/// A field element of `F_p`. Always reduced modulo `MODULUS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fp(u64);

impl Fp {
    /// Construct a new field element from an arbitrary `u64`. Reduces mod `MODULUS`.
    pub fn new(value: u64) -> Self {
        // TODO: reduce `value` modulo `MODULUS` and store. Make sure the
        // resulting Fp is canonical (i.e., the stored u64 is strictly less than MODULUS).
        // Hint: `value % MODULUS` is sufficient since both fit in u64.
        Self(value % MODULUS)
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
    /// Use rejection sampling: draw a `u64`, retry if it's `>= MODULUS`.
    /// (For our small field, the bias is small but rejection is the clean fix.)
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        // TODO: use rejection sampling. Hint: `rng.gen::<u64>()`.
        let mut r = rng.gen::<u64>();
        while r >= MODULUS {
            r = rng.gen::<u64>();
        }
        Self(r)
    }

    /// Get the underlying `u64` representation. For inspection / debug only.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Exponentiation by squaring: returns `self^exp` mod `MODULUS`.
    ///
    /// Hint: standard square-and-multiply.
    pub fn pow(self, mut exp: u64) -> Self {
        // TODO: implement square-and-multiply.
        // Pseudocode:
        //   result = Fp::one()
        //   base = self
        //   while exp > 0:
        //     if exp is odd: result = result * base
        //     base = base * base
        //     exp = exp >> 1
        //   return result
        let mut result = Self::one();
        let mut base = self;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result * base;
            }
            base = base * base;
            exp = exp >> 1;
        }

        result
    }

    /// Multiplicative inverse via Fermat's little theorem.
    ///
    /// Returns `None` if `self == zero()` — zero has no inverse in any field.
    ///
    /// # Why `self.pow(MODULUS - 2)` is the inverse
    ///
    /// **Fermat's little theorem.** For a prime `p` and any nonzero
    /// `a ∈ F_p`,
    ///
    /// ```text
    /// a^(p-1) ≡ 1  (mod p).
    /// ```
    ///
    /// **One algebra step.** Split off a factor of `a` on the left:
    ///
    /// ```text
    /// a^(p-1)  =  a · a^(p-2)  ≡  1  (mod p).
    /// ```
    ///
    /// "Multiplying `a` by `a^(p-2)` yields the multiplicative identity"
    /// *is* the definition of "`a^(p-2)` is the multiplicative inverse of
    /// `a` mod `p`". So **the inverse of `a` is `a^(p-2)`**, computed by
    /// the same square-and-multiply [`Fp::pow`] used for any exponentiation.
    /// No extended-Euclidean-algorithm gymnastics needed; the prime modulus
    /// hands us the inverse for free, packaged as an exponentiation.
    ///
    /// # Why zero returns `None`
    ///
    /// Fermat's hypothesis requires `a ≠ 0`. Zero is excluded structurally:
    /// `0 · x = 0` for every `x ∈ F`, so no `x` satisfies `0 · x = 1` — the
    /// inverse genuinely does not exist. We return `None` rather than panic
    /// because callers may legitimately produce zero (e.g., a sum that
    /// happens to cancel, or a Lagrange-denominator collision the caller
    /// wants to detect) and prefer an `Option` over an `unwrap()`.
    pub fn inverse(self) -> Option<Self> {
        // TODO: special-case zero, then return Some(self.pow(MODULUS - 2)).
        if self == Self::zero() {
            None
        } else {
            Some(self.pow(MODULUS - 2))
        }
    }
}

/// Per-thread counters for `Fp` arithmetic operations.
///
/// Every call to `Fp`'s `+`, `-`, `*`, or `Neg` increments the matching
/// per-thread counter. Snapshot via [`op_counter::snapshot`] before and
/// after a region of code, then call [`OpCounts::since`] on the later
/// snapshot to get the ops performed in that region. Used by `bin/demo.rs`
/// to **measure** the prover/verifier's work rather than predict it from
/// formulas.
///
/// Thread-local (not atomic), so parallel tests don't interfere across
/// threads. The increment is a `Cell::set` — no atomic operations and no
/// allocation — so the overhead per `Fp` op is a couple of nanoseconds.
///
/// # Example
///
/// ```ignore
/// use sumcheck::field::{op_counter, Fp};
///
/// let before = op_counter::snapshot();
/// let _ = Fp::new(3) + Fp::new(4);
/// let _ = Fp::new(5) * Fp::new(6);
/// let delta = op_counter::snapshot().since(&before);
/// assert_eq!(delta.adds, 1);
/// assert_eq!(delta.muls, 1);
/// assert_eq!(delta.total(), 2);
/// ```
pub mod op_counter {
    use std::cell::Cell;

    thread_local! {
        static ADDS: Cell<u64> = const { Cell::new(0) };
        static SUBS: Cell<u64> = const { Cell::new(0) };
        static MULS: Cell<u64> = const { Cell::new(0) };
        static NEGS: Cell<u64> = const { Cell::new(0) };
    }

    /// A snapshot of the per-thread `Fp` op counters at one moment.
    ///
    /// Subtract an earlier snapshot via [`OpCounts::since`] to get the
    /// op count between the two snapshots.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct OpCounts {
        /// Number of `Fp + Fp` operations counted.
        pub adds: u64,
        /// Number of `Fp - Fp` operations counted.
        pub subs: u64,
        /// Number of `Fp * Fp` operations counted.
        pub muls: u64,
        /// Number of `-Fp` (unary negation) operations counted.
        pub negs: u64,
    }

    impl OpCounts {
        /// Ops performed since an earlier snapshot.
        pub fn since(&self, earlier: &Self) -> Self {
            Self {
                adds: self.adds - earlier.adds,
                subs: self.subs - earlier.subs,
                muls: self.muls - earlier.muls,
                negs: self.negs - earlier.negs,
            }
        }

        /// Sum across all op categories (`adds + subs + muls + negs`).
        pub fn total(&self) -> u64 {
            self.adds + self.subs + self.muls + self.negs
        }
    }

    /// Snapshot the current thread's `Fp` op counters.
    pub fn snapshot() -> OpCounts {
        OpCounts {
            adds: ADDS.with(|c| c.get()),
            subs: SUBS.with(|c| c.get()),
            muls: MULS.with(|c| c.get()),
            negs: NEGS.with(|c| c.get()),
        }
    }

    /// Reset all of this thread's counters to zero. Useful for isolating
    /// a region of code from earlier accumulated activity, though
    /// [`snapshot`] + [`OpCounts::since`] is usually preferable.
    pub fn reset() {
        ADDS.with(|c| c.set(0));
        SUBS.with(|c| c.set(0));
        MULS.with(|c| c.set(0));
        NEGS.with(|c| c.set(0));
    }

    // --------- Internal increment helpers, called by Fp impls. ---------

    #[inline]
    pub(super) fn bump_add() {
        ADDS.with(|c| c.set(c.get() + 1));
    }
    #[inline]
    pub(super) fn bump_sub() {
        SUBS.with(|c| c.set(c.get() + 1));
    }
    #[inline]
    pub(super) fn bump_mul() {
        MULS.with(|c| c.set(c.get() + 1));
    }
    #[inline]
    pub(super) fn bump_neg() {
        NEGS.with(|c| c.set(c.get() + 1));
    }
}

// ============================================================================
// Arithmetic operator impls. You must implement all of these. Use `u128` as
// an intermediate to avoid overflow during multiplication.
//
// Every primitive op below calls `op_counter::bump_*` so `bin/demo.rs` can
// measure the actual operation count for any region of code.
// ============================================================================

impl Add for Fp {
    type Output = Fp;

    fn add(self, rhs: Fp) -> Fp {
        // TODO: addition mod MODULUS. Hint: u64 sum can overflow if both are
        // close to MODULUS, but using u128 intermediate avoids that.
        op_counter::bump_add();
        let sum = (self.0 as u128) + (rhs.0 as u128);
        Fp::new((sum % MODULUS as u128) as u64)
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
        // Hint: equivalent to self + (-rhs), but you can also do
        //   if self.0 >= rhs.0 { self.0 - rhs.0 } else { self.0 + MODULUS - rhs.0 }
        op_counter::bump_sub();
        if self.0 >= rhs.0 {
            Fp::new(self.0 - rhs.0)
        } else {
            Fp::new(self.0 + MODULUS - rhs.0)
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
        // Hint: cast both to u128, multiply, then take mod and cast back.
        op_counter::bump_mul();
        let product = (self.0 as u128) * (rhs.0 as u128);
        Fp::new((product % MODULUS as u128) as u64)
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
        // TODO: -x mod MODULUS. Special-case x == 0.
        op_counter::bump_neg();
        if self == Self::zero() {
            Self::zero()
        } else {
            Self::new(MODULUS - self.0)
        }
    }
}

impl std::iter::Sum for Fp {
    fn sum<I: Iterator<Item = Fp>>(iter: I) -> Fp {
        // TODO: fold from Fp::zero() with `+`.
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

// ============================================================================
// Tests — leave these empty for now; you'll add them as you implement.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_additive_identity() {
        // TODO: assert Fp::zero() + x == x for some sample x.
        let x = Fp::new(5);
        assert_eq!(Fp::zero() + x, x);
    }

    #[test]
    fn one_is_multiplicative_identity() {
        // TODO
        assert_eq!(Fp::one() * Fp::one(), Fp::one());
    }

    #[test]
    fn inverse_round_trip() {
        // TODO: for non-zero x, x * x.inverse().unwrap() == one()
        let x = Fp::new(7);
        let inv = x.inverse().unwrap();
        assert_eq!(x * inv, Fp::one());
    }

    #[test]
    fn add_then_sub_is_identity() {
        // TODO: (x + y) - y == x
        let x = Fp::new(5);
        let y = Fp::new(3);
        assert_eq!(x + y - y, x);
    }

    #[test]
    fn pow_zero_is_one() {
        // TODO: x.pow(0) == one()
        let x = Fp::new(5);
        assert_eq!(x.pow(0), Fp::one());
    }

    #[test]
    fn pow_matches_repeated_multiplication() {
        // TODO: for a small x and small exp, check pow against the repeated product.
        let x = Fp::new(5);
        let exp = 3;
        assert_eq!(x.pow(exp), x * x * x);
    }

    #[test]
    fn pow_matches_plain_u64_for_small_values() {
        let x = Fp::new(7);
        let exp = 5;

        assert_eq!(x.pow(exp).as_u64(), 7u64.pow(exp as u32));
    }
}
