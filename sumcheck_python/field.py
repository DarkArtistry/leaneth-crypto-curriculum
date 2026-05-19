"""Prime field F_p, dynamic over any prime modulus.

The default modulus is the Mersenne prime p = 2^61 - 1 (matches the Rust
reference implementation), but any prime can be passed to `Field(prime)`,
which returns a fresh `Fp` class bound to that prime. Two `Fp` values from
different fields cannot be combined arithmetically.

Mathematical contract:
    Every `Fp` value is canonical: stored as an int in `[0, modulus)`.
    All constructors and arithmetic ops preserve this.
"""

from __future__ import annotations

import secrets
from typing import Type


DEFAULT_MODULUS: int = (1 << 61) - 1  # Mersenne prime 2^61 - 1


class Fp:
    """A field element of F_p. Subclass-of-class-factory bound to a modulus.

    Do not instantiate this base class directly. Use `Field(prime)` to get a
    concrete `Fp` subclass, or use the default `MersenneFp` exposed below.
    """

    MODULUS: int = 0  # overridden in concrete subclasses

    __slots__ = ("value",)

    def __init__(self, value: int):
        if self.MODULUS == 0:
            raise TypeError(
                "Fp is abstract; use Field(prime) to create a concrete Fp class"
            )
        # Canonicalise. Python's `%` returns a non-negative result for positive
        # modulus, so negative inputs are handled correctly.
        self.value = value % self.MODULUS

    # ----- constructors -----

    @classmethod
    def zero(cls) -> "Fp":
        return cls(0)

    @classmethod
    def one(cls) -> "Fp":
        return cls(1)

    @classmethod
    def random(cls) -> "Fp":
        """Uniform random element via rejection sampling on a cryptographic RNG."""
        # `secrets.randbelow(modulus)` is already uniform on [0, modulus).
        return cls(secrets.randbelow(cls.MODULUS))

    @classmethod
    def from_int(cls, value: int) -> "Fp":
        return cls(value)

    # ----- arithmetic -----

    def _check_same_field(self, other: "Fp") -> None:
        if type(self) is not type(other):
            raise TypeError(
                f"cannot combine field elements from different fields: "
                f"{type(self).__name__}(p={self.MODULUS}) vs "
                f"{type(other).__name__}(p={other.MODULUS})"
            )

    def __add__(self, other: "Fp") -> "Fp":
        self._check_same_field(other)
        return type(self)(self.value + other.value)

    def __sub__(self, other: "Fp") -> "Fp":
        self._check_same_field(other)
        return type(self)(self.value - other.value)

    def __mul__(self, other: "Fp") -> "Fp":
        self._check_same_field(other)
        return type(self)(self.value * other.value)

    def __neg__(self) -> "Fp":
        return type(self)(-self.value)

    def __pow__(self, exp: int) -> "Fp":
        """Exponentiation via Python's built-in modular pow."""
        if exp < 0:
            return self.inverse() ** (-exp)
        return type(self)(pow(self.value, exp, self.MODULUS))

    def inverse(self) -> "Fp":
        """Multiplicative inverse via Fermat's little theorem: a^(p-2) = a^{-1}.

        Why this works (one algebra step):
            Fermat: a^(p-1) = 1 mod p for nonzero a.
            Therefore a * a^(p-2) = 1 mod p, i.e. a^(p-2) = a^{-1}.

        Raises `ZeroDivisionError` if `self == 0` — zero has no inverse.
        """
        if self.value == 0:
            raise ZeroDivisionError("0 has no multiplicative inverse in a field")
        return type(self)(pow(self.value, self.MODULUS - 2, self.MODULUS))

    # ----- equality / hash / repr -----

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fp):
            return NotImplemented
        return type(self) is type(other) and self.value == other.value

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.value))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value})"

    # ----- inspection -----

    def as_int(self) -> int:
        return self.value


def Field(prime: int) -> Type[Fp]:
    """Construct a concrete `Fp` subclass bound to the given prime.

    The returned class is a fully usable field-element type:
        Fp = Field(2**61 - 1)
        x = Fp(3)
        y = Fp(5)
        z = x + y      # Fp(8)
        w = x.inverse()
    """
    if prime < 2:
        raise ValueError(f"prime must be >= 2, got {prime}")
    # Primality is the caller's responsibility — for educational use we trust
    # the input. A small sanity check (no even prime > 2) catches the common
    # typo.
    if prime > 2 and prime % 2 == 0:
        raise ValueError(f"{prime} is not prime (even and > 2)")

    cls_name = f"Fp_{prime}"
    new_cls = type(cls_name, (Fp,), {"MODULUS": prime})
    return new_cls


# The default field used by the demo and most tests: Mersenne F_p with p = 2^61 - 1.
MersenneFp: Type[Fp] = Field(DEFAULT_MODULUS)
