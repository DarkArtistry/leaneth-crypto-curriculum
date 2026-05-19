"""End-to-end sumcheck orchestration, dynamic over any summation domain.

The orchestrator is the shortest layer in the package: it wires
`SumcheckProver` to `SumcheckVerifier` and runs the round loop. The
per-variable domain `S` is implicit in the polynomial parameter — both
`run_sumcheck` and `run_sumcheck_with_claim` take a `MultivariatePoly` and
propagate that domain to both parties.

## One round, concretely

Let `k = |S|`, so the prover's message is a length-`k` vector of field elements.

1.  Prover computes the round message (one entry per element of `S`).
2.  Verifier checks `sum_j msg[j] == current_claim`.
3.  Verifier samples `r ∈ F`, sets `current_claim = sum_j L_j(r) * msg[j]`,
    and returns `r`.
4.  Prover applies `fix_first_variable(r)`, shrinking by one variable.

Repeat `n` times, then evaluate `g` at the accumulated challenges and compare
against `current_claim`.

## Sumcheck protocol — class wrapper

`SumcheckProtocol` bundles the protocol around a given polynomial. The
constructor accepts the polynomial and validates parameters; `.run()` drives
the round loop and returns the trace plus a boolean accept/reject.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .domain import Domain
from .field import Fp
from .polynomial import MultivariatePoly
from .prover import SumcheckProver
from .verifier import SumcheckVerifier, VerificationError


@dataclass
class SumcheckTrace:
    """Diagnostic record of one sumcheck run.

    `round_messages[i]` is the prover's round-`i` message (length `|S|`).
    `challenges[i]` is the verifier's round-`i` challenge.
    `claimed_sum` is the value the prover initially claimed for `sum_S^n g`.
    `accepted` is True iff every round check and the final check passed.
    """
    claimed_sum: Fp
    round_messages: List[List[Fp]] = field(default_factory=list)
    challenges: List[Fp] = field(default_factory=list)
    accepted: bool = False
    error: Optional[str] = None


class SumcheckProtocol:
    """Orchestrate a sumcheck run for a given polynomial.

    Parameters from the caller:
        polynomial — a `MultivariatePoly` on a `Domain` of any size.
        claimed_sum — optional override. If omitted, derived honestly via
            `polynomial.sum_over_domain()`. Pass a different value to inject a
            wrong claim (the verifier should reject).

    Optional knobs (mainly for testing/determinism):
        fixed_challenges — pass a list of pre-generated challenges to make
            the run reproducible. If given, must have length == polynomial.n_vars.
            If None, the verifier samples fresh challenges per round.

    The protocol validates these arguments against the polynomial:
        - claimed_sum must be in the same field as the polynomial domain.
        - fixed_challenges (if given) must all be in that field.
        - polynomial domain must be a valid Domain instance.
    """

    def __init__(
        self,
        polynomial: MultivariatePoly,
        claimed_sum: Optional[Fp] = None,
        fixed_challenges: Optional[List[Fp]] = None,
    ):
        if not isinstance(polynomial, MultivariatePoly):
            raise TypeError("polynomial must be a MultivariatePoly")
        if not isinstance(polynomial.domain, Domain):
            raise TypeError("polynomial.domain must be a Domain")

        field_cls = polynomial.domain.field()

        if claimed_sum is None:
            claimed_sum = polynomial.sum_over_domain()
        elif type(claimed_sum) is not field_cls:
            raise TypeError(
                f"claimed_sum field {type(claimed_sum).__name__} does not match "
                f"polynomial field {field_cls.__name__}"
            )

        if fixed_challenges is not None:
            if len(fixed_challenges) != polynomial.n_vars:
                raise ValueError(
                    f"fixed_challenges length {len(fixed_challenges)} != "
                    f"polynomial.n_vars {polynomial.n_vars}"
                )
            for i, c in enumerate(fixed_challenges):
                if type(c) is not field_cls:
                    raise TypeError(
                        f"fixed_challenges[{i}] field {type(c).__name__} does "
                        f"not match polynomial field {field_cls.__name__}"
                    )

        self.polynomial: MultivariatePoly = polynomial
        self.claimed_sum: Fp = claimed_sum
        self.fixed_challenges: Optional[List[Fp]] = fixed_challenges

    def run(self) -> SumcheckTrace:
        """Execute the full sumcheck protocol and return a trace.

        Does not raise — protocol failures (wrong claim, bad message length,
        final mismatch) populate `trace.error` and set `trace.accepted = False`.
        Truly programmer errors (wrong types, etc.) still raise.
        """
        polynomial = self.polynomial
        trace = SumcheckTrace(claimed_sum=self.claimed_sum)

        prover = SumcheckProver(polynomial)
        verifier = SumcheckVerifier(
            polynomial.domain, self.claimed_sum, polynomial.n_vars
        )

        round_idx = 0
        while not prover.is_done():
            msg = prover.compute_round_message()
            trace.round_messages.append(list(msg))

            fixed_r = (
                self.fixed_challenges[round_idx]
                if self.fixed_challenges is not None
                else None
            )

            try:
                r = verifier.process_round_message(msg, r=fixed_r)
            except VerificationError as e:
                trace.error = f"round {round_idx}: {e}"
                trace.accepted = False
                return trace

            trace.challenges.append(r)
            prover.receive_challenge(r)
            round_idx += 1

        # Final check.
        try:
            verifier.final_check(polynomial)
            trace.accepted = True
        except VerificationError as e:
            trace.error = f"final check: {e}"
            trace.accepted = False

        return trace


def run_sumcheck(
    polynomial: MultivariatePoly,
    fixed_challenges: Optional[List[Fp]] = None,
) -> SumcheckTrace:
    """Convenience: run sumcheck with the honest claim derived from the polynomial."""
    return SumcheckProtocol(polynomial, fixed_challenges=fixed_challenges).run()


def run_sumcheck_with_claim(
    polynomial: MultivariatePoly,
    claimed_sum: Fp,
    fixed_challenges: Optional[List[Fp]] = None,
) -> SumcheckTrace:
    """Convenience: run sumcheck with the caller-supplied claim.

    Pass a wrong `claimed_sum` to verify the verifier rejects.
    """
    return SumcheckProtocol(
        polynomial,
        claimed_sum=claimed_sum,
        fixed_challenges=fixed_challenges,
    ).run()
