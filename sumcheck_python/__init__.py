"""Dynamic sumcheck protocol — Python educational implementation.

Key difference from the Rust reference: the per-variable summation domain is
**dynamic**. There are no marker types like `BooleanHypercube` or `Interval3`;
just a single `Domain` class that accepts any list of distinct field elements.

Example — boolean hypercube:
    from sumcheck_python import MersenneFp as Fp, Domain, MultivariatePoly, SumcheckProtocol
    boolean = Domain([Fp(0), Fp(1)])
    poly = MultivariatePoly(boolean, 2, [Fp(1), Fp(2), Fp(3), Fp(4)])
    trace = SumcheckProtocol(poly).run()
    assert trace.accepted

Example — non-boolean S = {0, 1, 2}:
    interval3 = Domain([Fp(0), Fp(1), Fp(2)])
    g = lambda x, y: x + y
    poly = MultivariatePoly.from_callable(interval3, 2, g)
    trace = SumcheckProtocol(poly).run()
    assert trace.accepted

Example — arbitrary 4-element domain {3, 7, 11, 17}:
    quad = Domain([Fp(3), Fp(7), Fp(11), Fp(17)])
    # ... build a polynomial with per-variable degree < 4 on `quad^n` ...
"""

from .field import DEFAULT_MODULUS, Field, Fp, MersenneFp
from .domain import Domain
from .polynomial import MultivariatePoly, lagrange_basis_at
from .prover import SumcheckProver
from .verifier import SumcheckVerifier, VerificationError
from .protocol import (
    SumcheckProtocol,
    SumcheckTrace,
    run_sumcheck,
    run_sumcheck_with_claim,
)

__all__ = [
    "DEFAULT_MODULUS",
    "Field",
    "Fp",
    "MersenneFp",
    "Domain",
    "MultivariatePoly",
    "lagrange_basis_at",
    "SumcheckProver",
    "SumcheckVerifier",
    "VerificationError",
    "SumcheckProtocol",
    "SumcheckTrace",
    "run_sumcheck",
    "run_sumcheck_with_claim",
]
