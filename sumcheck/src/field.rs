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

    /// Multiplicative inverse via Fermat's little theorem: `self^(p-2)` for `p` prime.
    ///
    /// Returns `None` if `self == zero()`.
    pub fn inverse(self) -> Option<Self> {
        // TODO: special-case zero, then return Some(self.pow(MODULUS - 2)).
        if self == Self::zero() {
            None
        } else {
            Some(self.pow(MODULUS - 2))
        }
    }
}

// ============================================================================
// Arithmetic operator impls. You must implement all of these. Use `u128` as
// an intermediate to avoid overflow during multiplication.
// ============================================================================

impl Add for Fp {
    type Output = Fp;

    fn add(self, rhs: Fp) -> Fp {
        // TODO: addition mod MODULUS. Hint: u64 sum can overflow if both are
        // close to MODULUS, but using u128 intermediate avoids that.
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
