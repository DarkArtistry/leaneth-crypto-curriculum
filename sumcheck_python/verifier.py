"""Sumcheck verifier, generic over any per-variable summation domain `S`.

## What the verifier does, in one paragraph

The verifier sees the public polynomial `g`, the per-variable domain `S` (so
`k = |S|`), and the prover's claim `H = sum_{x in S^n} g(x)`. In each round
it receives a `k`-vector `msg = [s_i(elements[0]), ..., s_i(elements[k-1])]`,
checks that the sum-over-`S` matches the running expectation, samples a
uniformly random `r ∈ F`, and updates the running expectation to `s_i(r)` —
the value of the univariate slice at the random challenge, obtained by
`k`-point Lagrange interpolation. After `n` rounds, it evaluates `g` at the
accumulated challenges and compares against the running expectation.

## Two generic invariants

1.  Per-round sum-over-`S` check:
        sum over j in 0..k of msg[j] == current_claim.

    Why: msg[j] = s_i(elements[j]) and s_i is the partial sum of `g` along all
    but its first variable, so summing s_i over S equals the full sum over the
    remaining cube, which is what current_claim should be.

2.  Lagrange interpolation update:
        current_claim = sum over j in 0..k of L_j(r) · msg[j].

    The unique polynomial of degree `< k` through the `k` points
    (elements[j], msg[j]) is s_i(X) = sum_j L_j(X)·msg[j]; plug in X = r.

For `S = {0, 1}` both formulas collapse to the classic boolean-only versions:
sum check is `msg[0] + msg[1] == current_claim` and interpolation is
`(1 - r)·msg[0] + r·msg[1]`. No special case — boolean is just `k = 2`.

## Soundness

Per-round failure probability ≤ `(k-1)/|F|` by the polynomial-root bound
(a nonzero polynomial of degree `< k` has at most `k - 1` roots). Union bound
across `n` rounds: total `≤ n·(k-1)/|F|`. For `|F| = 2^61 - 1`, `k = 2`,
`n = 20`: about `2^{-56.5}`. For `k = 3` it's `2·n/|F| ≈ 2^{-55.7}`.
"""

from __future__ import annotations

from typing import List, Optional

from .domain import Domain
from .field import Fp
from .polynomial import MultivariatePoly, lagrange_basis_at


class VerificationError(Exception):
    """Raised when a sumcheck verification step fails."""


class SumcheckVerifier:
    """Sumcheck verifier state.

    Tracks the running claim, sampled challenges, and decides accept/reject
    after all rounds plus the final `g`-evaluation check.
    """

    __slots__ = ("domain", "claimed_sum", "n_vars", "current_claim", "_challenges")

    def __init__(self, domain: Domain, claimed_sum: Fp, n_vars: int):
        if not isinstance(domain, Domain):
            raise TypeError("SumcheckVerifier requires a Domain")
        if n_vars < 0:
            raise ValueError(f"n_vars must be >= 0, got {n_vars}")
        if type(claimed_sum) is not domain.field():
            raise TypeError(
                f"claimed_sum field {type(claimed_sum).__name__} does not match "
                f"domain field {domain.field().__name__}"
            )

        self.domain: Domain = domain
        self.claimed_sum: Fp = claimed_sum
        self.n_vars: int = n_vars
        self.current_claim: Fp = claimed_sum
        self._challenges: List[Fp] = []

    def challenges(self) -> List[Fp]:
        """The challenges sampled so far."""
        return list(self._challenges)

    def is_done(self) -> bool:
        """True iff all `n_vars` round messages have been processed."""
        return len(self._challenges) == self.n_vars

    def process_round_message(
        self, msg: List[Fp], r: Optional[Fp] = None
    ) -> Fp:
        """Process the round-`i` message and return the sampled challenge `r`.

        Three checks, then a state update:
            1. Length: msg.len() == |domain|.
            2. Sum-over-S: sum(msg) == current_claim.
            3. Lagrange interp: sample r ∈ F (or use the provided `r` for
               deterministic testing), set current_claim = sum_j L_j(r)*msg[j].

        If a check fails, raises `VerificationError` and leaves state otherwise
        unchanged.

        The optional `r` parameter exists for testing — pass a fixed challenge
        to make the run reproducible. In real use the verifier samples its own
        random `r`.
        """
        if self.is_done():
            raise RuntimeError(
                "SumcheckVerifier.process_round_message: protocol is already done"
            )

        elements = self.domain.elements()
        k = self.domain.size()

        if len(msg) != k:
            raise VerificationError(
                f"round message length {len(msg)} does not match |domain| = {k}"
            )

        for i, m in enumerate(msg):
            if type(m) is not self.domain.field():
                raise TypeError(
                    f"msg[{i}] field {type(m).__name__} does not match domain "
                    f"field {self.domain.field().__name__}"
                )

        # Sum-over-S check. For k = 2 this is msg[0] + msg[1].
        field_cls = self.domain.field()
        total = field_cls.zero()
        for m in msg:
            total = total + m
        if total != self.current_claim:
            raise VerificationError(
                f"round message does not sum to current claim: "
                f"sum(msg) = {total}, current_claim = {self.current_claim}"
            )

        # Sample r (verifier's only source of soundness). Allow caller override.
        if r is None:
            r = field_cls.random()
        else:
            if type(r) is not field_cls:
                raise TypeError(
                    f"provided r is in field {type(r).__name__} but domain uses "
                    f"{field_cls.__name__}"
                )

        # Lagrange-interpolate s_i(r) = sum_j L_j(r) * msg[j] through the `k`
        # known points (elements[j], msg[j]).
        s_at_r = field_cls.zero()
        for j in range(k):
            s_at_r = s_at_r + lagrange_basis_at(elements, j, r) * msg[j]

        self.current_claim = s_at_r
        self._challenges.append(r)
        return r

    def final_check(self, g: MultivariatePoly) -> bool:
        """Final consistency check: `g(r_1, ..., r_n) == current_claim`.

        `g` must be a `MultivariatePoly` on the **same** `Domain` this verifier
        was constructed with (asserted explicitly).

        Returns True if the verifier accepts. Raises `VerificationError` if
        called before all rounds have been processed, or if the claim doesn't
        match.
        """
        if len(self._challenges) != self.n_vars:
            raise RuntimeError(
                "SumcheckVerifier.final_check: must process all rounds first "
                f"({len(self._challenges)}/{self.n_vars} done)"
            )
        if g.domain != self.domain:
            raise VerificationError(
                "final_check: polynomial domain does not match verifier domain"
            )
        if g.n_vars != self.n_vars:
            raise VerificationError(
                f"final_check: polynomial has n_vars = {g.n_vars}, "
                f"verifier expected {self.n_vars}"
            )

        g_at_challenges = g.evaluate(self._challenges)
        if g_at_challenges != self.current_claim:
            raise VerificationError(
                f"final_check: g(r_1, ..., r_n) = {g_at_challenges} != "
                f"current_claim = {self.current_claim}"
            )
        return True
