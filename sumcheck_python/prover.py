"""Sumcheck prover, generic over any per-variable summation domain `S`.

## What the prover does, in one paragraph

The prover holds the public polynomial `g: F^n -> F` and the per-variable
domain `S` (so `k = |S|`). It walks through `n` rounds, in each round (a)
sending the verifier a **univariate slice** of the current polynomial along
its first remaining variable, and (b) receiving back a random challenge `r`
that fixes that variable. The "univariate slice" is the function
`X -> sum over the remaining cube of g(..., X, ...)` — by the per-variable
degree bound this is a polynomial of degree `< k`, so it's determined by its
`k` values on `S`.

## The round message: `[s_i(elements[0]), ..., s_i(elements[k-1])]`

At the start of round `i`, the prover's internal state is a polynomial in
`n_remaining = n - i` variables (variables `x_1, ..., x_i` have been fixed
to verifier challenges `r_1, ..., r_i`). Its evaluations on `S^n_remaining`
are stored in `self.polynomial.evals`, length `k^n_remaining`, mixed-radix
LSB-first. Define

    s_i(X) = sum over (b_2, ..., b_{n_remaining}) in S^{n_remaining - 1}
               of self.polynomial(X, b_2, ..., b_{n_remaining}).

`compute_round_message` returns the `k`-vector
`[s_i(elements[0]), ..., s_i(elements[k-1])]`. The verifier Lagrange-
interpolates these `k` points to recover `s_i(r)` at the random challenge.

## Deriving the partition from mixed-radix indexing

Under mixed-radix LSB-first, `x_1` is the least-significant base-`k` digit.
With `block_size = k^{n_remaining - 1}`,

    s_i(elements[j]) = sum over rest in 0..block_size of evals[j + rest * k].

For `k = 2`, this is the familiar even/odd partition. For `k = 3`, three
groups under base-3 LSB. For any `k`, the same formula `j + rest * k`.
"""

from __future__ import annotations

from typing import List

from .field import Fp
from .polynomial import MultivariatePoly


class SumcheckProver:
    """Sumcheck prover state, holding the current polynomial and accumulated challenges.

    Round-by-round the prover's `polynomial` shrinks by one variable; the
    `challenges` accumulated drive that reduction. The original variable count
    `n` is saved as `initial_n_vars` so `is_done` knows when all `n` rounds
    have been processed.
    """

    __slots__ = ("polynomial", "initial_n_vars", "_challenges")

    def __init__(self, polynomial: MultivariatePoly):
        if not isinstance(polynomial, MultivariatePoly):
            raise TypeError("SumcheckProver requires a MultivariatePoly")
        self.polynomial: MultivariatePoly = polynomial
        self.initial_n_vars: int = polynomial.n_vars
        self._challenges: List[Fp] = []

    def current_round(self) -> int:
        """The 0-indexed round we are about to send a message for."""
        return len(self._challenges)

    def is_done(self) -> bool:
        """True iff all `initial_n_vars` rounds have been processed."""
        return self.current_round() == self.initial_n_vars

    def challenges(self) -> List[Fp]:
        """The challenges accumulated so far."""
        return list(self._challenges)

    def compute_round_message(self) -> List[Fp]:
        """Compute round-`i` univariate message:
        `[s_i(elements[0]), ..., s_i(elements[k-1])]`.

        With `k = |domain|` and `block_size = k^{n_remaining - 1}`:

            for j in 0..k:
              s_i(elements[j]) = sum over rest in 0..block_size of evals[j + rest * k]

        Returns a list of length `k`. Companion verifier will Lagrange-
        interpolate these points at a random challenge.
        """
        if self.is_done():
            raise RuntimeError(
                "SumcheckProver.compute_round_message: protocol is already done"
            )

        domain = self.polynomial.domain
        k = domain.size()
        # block_size = k^{n_remaining - 1} = len(evals) / k. The number of
        # `rest` tuples we sum over per fixed value of x_1.
        block_size = len(self.polynomial.evals) // k

        field_cls = domain.field()
        zero = field_cls.zero()

        msg: List[Fp] = []
        for j in range(k):
            # Sum evals[j + rest * k] for rest = 0..block_size.
            # k = 2 → even/odd partition; k = 3 → three base-3 LSB groups.
            acc = zero
            for rest in range(block_size):
                acc = acc + self.polynomial.evals[j + rest * k]
            msg.append(acc)
        return msg

    def receive_challenge(self, r: Fp) -> None:
        """Receive verifier's challenge `r` for the current variable and reduce
        the polynomial to one fewer variable.

        Internally calls `MultivariatePoly.fix_first_variable(r)`, the generic
        Lagrange-along-one-axis step.
        """
        if self.is_done():
            raise RuntimeError(
                "SumcheckProver.receive_challenge: protocol is already done"
            )
        # Validate that `r` belongs to the polynomial's field.
        if type(r) is not self.polynomial.domain.field():
            raise TypeError(
                f"challenge r is in field {type(r).__name__} but polynomial uses "
                f"{self.polynomial.domain.field().__name__}"
            )
        self._challenges.append(r)
        self.polynomial = self.polynomial.fix_first_variable(r)
