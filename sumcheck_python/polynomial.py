"""Multivariate polynomial in evaluation form on a product domain `S^n`.

A `MultivariatePoly` is `n_vars` variables stored as `|S|^n_vars` field
evaluations on the grid `S^n_vars`, where `S` is any user-provided `Domain`.
The boolean hypercube (`S = {0, 1}`) is just one particular domain; the
storage layout, the "fix one variable" operation, and `evaluate` all
generalise to any finite `S ⊆ F`.

## What "evaluation form on a product domain" means

Let `g: F^n -> F` be a multivariate polynomial whose per-variable degree is
strictly less than `k = |S|`. By the **multivariate Lagrange-interpolation
theorem**, `g` is uniquely pinned down by the `k^n` numbers `g(s_1, ..., s_n)`
for `(s_1, ..., s_n) ∈ S^n`. Storing those `k^n` numbers *is* the polynomial,
in evaluation form on `S^n`.

When `S = {0, 1}` and `k = 2`, "degree < k per variable" is just
"multilinear" (per-variable degree ≤ 1), so this generalises the classical
Multilinear Extension (MLE) story to general `S`.

## Mixed-radix LSB-first indexing

An index `i ∈ [0, k^n)` decomposes into `n` digits

    i = i_1 + i_2·k + i_3·k^2 + ... + i_n·k^{n-1},     i_j ∈ [0, k)

and `evals[i]` stores `g(elements[i_1], elements[i_2], ..., elements[i_n])`.
Equivalently, digit `j` is `i_j = (i // k^{j-1}) % k`. Variable `x_1` is the
LEAST significant digit — adjacent indices `(0, 1, ..., k-1)` sweep `x_1`
through `S` while holding `x_2, ..., x_n` fixed. That contiguity makes
`fix_first_variable` cache-friendly.

For `S = {0, 1}` (k = 2) this collapses to ordinary LSB-first binary; for
`S` of size 3 it's base-3.

## Fixing `x_1 = r` — Lagrange along one axis

For fixed `(x_2, ..., x_n) = rest`, the function `x_1 -> g(x_1, rest)` is a
polynomial of degree `< k` in one variable. By Lagrange interpolation:

    g(r, rest) = sum over j in 0..k of L_j(r) · g(elements[j], rest)

where `L_j(x) = prod_{i != j} (x - elements[i]) / (elements[j] - elements[i])`
is the Lagrange basis polynomial that is 1 at `elements[j]` and 0 at every
other element of `S`. For the boolean case `S = {0, 1}` this collapses to
`(1 - r) · g(0, rest) + r · g(1, rest)` — the familiar line interpolation.
"""

from __future__ import annotations

from typing import List

from .domain import Domain
from .field import Fp


def lagrange_basis_at(points: List[Fp], i: int, target: Fp) -> Fp:
    """Evaluate the Lagrange basis polynomial `L_i` at `target`.

    The basis is built over `points`:

        L_i(target) = prod_{j != i} (target - points[j]) / (points[i] - points[j]).

    `L_i` is the unique degree-`< len(points)` polynomial that is 1 at
    `points[i]` and 0 at every other `points[j]`.

    Caller must ensure that `points` are pairwise distinct (so each
    denominator factor is nonzero). `Domain` enforces this at construction.
    """
    field_cls = type(points[0])
    numerator = field_cls.one()
    denominator = field_cls.one()
    xi = points[i]
    for j, xj in enumerate(points):
        if j == i:
            continue
        numerator = numerator * (target - xj)
        denominator = denominator * (xi - xj)
    # Safe: denominator nonzero because elements are pairwise distinct.
    return numerator * denominator.inverse()


def _pow_int(base: int, exp: int) -> int:
    """`base ** exp` for integer base — used only for the length check `|S|^n`."""
    acc = 1
    for _ in range(exp):
        acc *= base
    return acc


class MultivariatePoly:
    """A multivariate polynomial in `n_vars` variables, stored as its
    evaluations on the product domain `domain^n_vars`.

    Evaluations are laid out in mixed-radix LSB-first order: index
    `i = i_1 + i_2·k + ... + i_n·k^{n-1}` (with `k = |domain|`) corresponds to
    the input `(elements[i_1], ..., elements[i_n])`, so the first variable
    `x_1` is least significant and the `k` evaluations sharing a fixed
    `(x_2, ..., x_n)` rest sit at contiguous indices.
    """

    __slots__ = ("domain", "n_vars", "evals")

    def __init__(self, domain: Domain, n_vars: int, evals: List[Fp]):
        if n_vars < 0:
            raise ValueError(f"n_vars must be >= 0, got {n_vars}")

        k = domain.size()
        expected_len = _pow_int(k, n_vars)
        if len(evals) != expected_len:
            raise ValueError(
                f"evals length must be |domain|^n_vars = {k}^{n_vars} = "
                f"{expected_len}, got {len(evals)}"
            )

        # Validate the field type of every eval matches the domain's field.
        field_cls = domain.field()
        for i, e in enumerate(evals):
            if type(e) is not field_cls:
                raise TypeError(
                    f"evals[{i}] field {type(e).__name__} does not match "
                    f"domain field {field_cls.__name__}"
                )

        self.domain: Domain = domain
        self.n_vars: int = n_vars
        self.evals: List[Fp] = list(evals)

    @classmethod
    def from_callable(cls, domain: Domain, n_vars: int, g) -> "MultivariatePoly":
        """Build the evaluation table by evaluating `g` on the grid `S^n_vars`.

        `g` is any callable `g(x_1, x_2, ..., x_n) -> Fp` accepting `n_vars`
        positional `Fp` arguments. Evaluations are laid out in mixed-radix
        LSB-first order, matching the storage convention.

        Useful in the demo to define `g` symbolically and have the evaluation
        table built for you.
        """
        elements = domain.elements()
        k = domain.size()
        total = _pow_int(k, n_vars)

        evals: List[Fp] = []
        for i in range(total):
            # Decompose i into base-k digits, LSB first.
            digits = []
            rem = i
            for _ in range(n_vars):
                digits.append(rem % k)
                rem //= k
            point = tuple(elements[d] for d in digits)
            evals.append(g(*point))
        return cls(domain, n_vars, evals)

    def sum_over_domain(self) -> Fp:
        """Compute `H = sum over x in S^n_vars of g(x)`.

        `O(|S|^n_vars)` — just sum the stored evaluations. This is the
        quantity sumcheck proves.
        """
        if len(self.evals) == 0:
            return self.domain.field().zero()
        # `sum(..., start=...)` in Python — start from the field zero so the
        # accumulator type is correct from the get-go.
        return sum(self.evals[1:], start=self.evals[0])

    def fix_first_variable(self, r: Fp) -> "MultivariatePoly":
        """Replace `x_1` with the field element `r`, returning the resulting
        polynomial in `n_vars - 1` variables on the same domain.

        Mathematically:

            g'(x_2, ..., x_n) = g(r, x_2, ..., x_n)

        Implementation: precompute the `k` Lagrange weights `L_j(r)` once
        (they don't depend on `rest`), then for each `rest_idx` build the
        new evaluation as a `k`-term dot product against the contiguous
        slice `evals[rest_idx*k : rest_idx*k + k]`.

        For the boolean case `S = {0, 1}` (`k = 2`, elements `[0, 1]`) the
        Lagrange weights collapse to `L_0(r) = 1 - r, L_1(r) = r`, recovering
        the familiar line update `(1 - r)·A + r·B`.

        Raises `ValueError` if `n_vars == 0`.
        """
        if self.n_vars == 0:
            raise ValueError("cannot fix a variable in a zero-variable polynomial")

        elements = self.domain.elements()
        k = len(elements)

        # Precompute Lagrange weights L_j(r) for j ∈ 0..k. They depend only
        # on `r` and the domain, not on the `rest` slice.
        weights = [lagrange_basis_at(elements, j, r) for j in range(k)]

        # Output has |S|^{n - 1} evaluations.
        new_n_vars = self.n_vars - 1
        new_len = _pow_int(k, new_n_vars)
        new_evals: List[Fp] = []

        zero = self.domain.field().zero()
        for rest_idx in range(new_len):
            # The `k` evaluations sharing this rest_idx lie at indices
            # rest_idx*k + j for j ∈ 0..k — a contiguous slice, since x_1 is
            # the least-significant mixed-radix digit.
            base = rest_idx * k
            acc = zero
            for j in range(k):
                acc = acc + weights[j] * self.evals[base + j]
            new_evals.append(acc)

        return MultivariatePoly(self.domain, new_n_vars, new_evals)

    def evaluate(self, point: List[Fp]) -> Fp:
        """Evaluate `g` at `(r_1, ..., r_n) ∈ F^n` via repeated `fix_first_variable`.

        Cost: `O(|S|^{n_vars + 1})` field ops — the geometric sum of `|S|^m`
        for `m = 1..n_vars`. For `|S| = 2` that's `O(2^n)`. This is the same
        DP that drives sumcheck itself.
        """
        if len(point) != self.n_vars:
            raise ValueError(
                f"evaluate expects a length-{self.n_vars} point, got {len(point)}"
            )

        current = self
        for r in point:
            current = current.fix_first_variable(r)
        # After n_vars folds, current is a 0-variable polynomial: one stored value.
        return current.evals[0]

    def __repr__(self) -> str:
        return (
            f"MultivariatePoly(domain={self.domain!r}, "
            f"n_vars={self.n_vars}, evals_len={len(self.evals)})"
        )
