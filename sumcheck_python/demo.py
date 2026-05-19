"""End-to-end sumcheck demos — boolean hypercube AND general dynamic hypercube.

Run:  python -m sumcheck_python.demo

Demonstrates that the same protocol code handles any user-supplied summation
domain. Nothing in the protocol is hard-coded to S = {0, 1}: pass any list of
distinct field elements and it just works.

We run three scenarios in sequence:
    (1) Boolean hypercube      S = {0, 1},          k = 2,  n = 3
    (2) Three-element domain   S = {0, 1, 2},       k = 3,  n = 2
    (3) Arbitrary 4-elem set   S = {3, 7, 11, 17},  k = 4,  n = 2

In every case the protocol class is the same `SumcheckProtocol(poly)` call —
the dynamic Domain is the only thing that varies.
"""

from __future__ import annotations

from .domain import Domain
from .field import MersenneFp as Fp
from .polynomial import MultivariatePoly
from .protocol import SumcheckProtocol, run_sumcheck_with_claim


HR = "=" * 72
SUBHR = "-" * 72


def section(title: str) -> None:
    print()
    print(HR)
    print(f" {title}")
    print(HR)


def subsection(title: str) -> None:
    print()
    print(SUBHR)
    print(f" {title}")
    print(SUBHR)


def show_trace(label: str, trace) -> None:
    """Pretty-print a sumcheck trace."""
    print(f"  [{label}] claimed_sum = {trace.claimed_sum.as_int()}")
    for i, (msg, r) in enumerate(zip(trace.round_messages, trace.challenges)):
        msg_ints = [m.as_int() for m in msg]
        print(f"    round {i}: msg = {msg_ints}, r = {r.as_int()}")
    if trace.accepted:
        print(f"  [{label}] verdict: ACCEPT")
    else:
        print(f"  [{label}] verdict: REJECT — {trace.error}")


def demo_boolean_hypercube() -> None:
    section("Demo 1 — Boolean hypercube  S = {0, 1},  n = 3")

    boolean = Domain([Fp(0), Fp(1)])
    print(f"  domain    : {boolean}")
    print(f"  |S|       : {boolean.size()}")

    # g(x1, x2, x3) = 1 + 2*x1 + 3*x2 + 4*x3   (multilinear, per-var degree 1 < |S| = 2).
    def g(x1, x2, x3):
        return Fp(1) + Fp(2) * x1 + Fp(3) * x2 + Fp(4) * x3

    poly = MultivariatePoly.from_callable(boolean, 3, g)

    print(f"  poly      : g(x1, x2, x3) = 1 + 2*x1 + 3*x2 + 4*x3")
    print(f"  n_vars    : {poly.n_vars}")
    print(f"  |S|^n     : {len(poly.evals)}  (= 2^3 = 8 stored evaluations)")

    H = poly.sum_over_domain()
    print(f"  H = sum_{{S^3}} g(x) = {H.as_int()}  (analytic check: 44 — see "
          f"Thaler §4.1 numeric anchor)")

    subsection("(a) Honest run — verifier should ACCEPT")
    trace = SumcheckProtocol(poly).run()
    show_trace("honest", trace)
    assert trace.accepted, "honest run unexpectedly rejected"

    subsection("(b) Lying prover — claim H + 1, verifier should REJECT")
    bad_trace = run_sumcheck_with_claim(poly, H + Fp(1))
    show_trace("lying ", bad_trace)
    assert not bad_trace.accepted, "lying run unexpectedly accepted"


def demo_three_element_domain() -> None:
    section("Demo 2 — Three-element domain  S = {0, 1, 2},  n = 2")

    interval3 = Domain([Fp(0), Fp(1), Fp(2)])
    print(f"  domain    : {interval3}")
    print(f"  |S|       : {interval3.size()}")

    # g(x1, x2) = x1 + x2   (per-var degree 1 < |S| = 3, OK).
    def g(x1, x2):
        return x1 + x2

    poly = MultivariatePoly.from_callable(interval3, 2, g)

    print(f"  poly      : g(x1, x2) = x1 + x2")
    print(f"  n_vars    : {poly.n_vars}")
    print(f"  |S|^n     : {len(poly.evals)}  (= 3^2 = 9 stored evaluations)")

    H = poly.sum_over_domain()
    # Hand-check: sum_{x,y ∈ {0,1,2}} (x + y) = 18  (Thaler §4.1 numeric anchor).
    print(f"  H = sum_{{S^2}} g(x) = {H.as_int()}  (analytic check: 18)")

    subsection("(a) Honest run — verifier should ACCEPT")
    trace = SumcheckProtocol(poly).run()
    show_trace("honest", trace)
    assert trace.accepted, "honest run unexpectedly rejected"

    subsection("(b) Wrong claim — verifier should REJECT")
    bad_trace = run_sumcheck_with_claim(poly, H + Fp(1))
    show_trace("lying ", bad_trace)
    assert not bad_trace.accepted, "lying run unexpectedly accepted"


def demo_arbitrary_four_element_domain() -> None:
    section("Demo 3 — Arbitrary 4-element domain  S = {3, 7, 11, 17},  n = 2")

    # The protocol does NOT require S = {0, 1, ..., k-1}. Any distinct field
    # elements work — the Lagrange interpolation handles them transparently.
    weird = Domain([Fp(3), Fp(7), Fp(11), Fp(17)])
    print(f"  domain    : {weird}")
    print(f"  |S|       : {weird.size()}")

    # g(x1, x2) = x1 * x2 + 5  (per-var degree 1 < |S| = 4, OK; in fact any
    # bivariate polynomial of per-var degree < 4 is faithfully stored here).
    def g(x1, x2):
        return x1 * x2 + Fp(5)

    poly = MultivariatePoly.from_callable(weird, 2, g)

    print(f"  poly      : g(x1, x2) = x1*x2 + 5")
    print(f"  n_vars    : {poly.n_vars}")
    print(f"  |S|^n     : {len(poly.evals)}  (= 4^2 = 16 stored evaluations)")

    # Sanity-compute H by hand: sum_{x,y in {3,7,11,17}} (xy + 5).
    #   sum xy = (sum x)(sum y) = 38 * 38 = 1444
    #   sum 5  = 16 * 5         = 80
    #   total                    = 1524
    H = poly.sum_over_domain()
    print(f"  H = sum_{{S^2}} g(x) = {H.as_int()}  (analytic check: 1524 = "
          f"(3+7+11+17)^2 + 16*5)")

    subsection("(a) Honest run — verifier should ACCEPT")
    trace = SumcheckProtocol(poly).run()
    show_trace("honest", trace)
    assert trace.accepted, "honest run unexpectedly rejected"

    subsection("(b) Wrong claim — verifier should REJECT")
    bad_trace = run_sumcheck_with_claim(poly, H + Fp(1))
    show_trace("lying ", bad_trace)
    assert not bad_trace.accepted, "lying run unexpectedly accepted"


def demo_dynamic_parameters_showcase() -> None:
    """One last showcase: the same `SumcheckProtocol` class accepts whatever
    domain you hand it — no separate code path per domain size."""

    section("Demo 4 — Dynamic-parameters showcase (same code, three domains)")

    print("  All three runs go through the IDENTICAL `SumcheckProtocol(poly).run()`")
    print("  call. Only the Domain (and therefore the polynomial's storage shape)")
    print("  changes.\n")

    cases = [
        ("S = {0, 1}",          Domain([Fp(0), Fp(1)])),
        ("S = {0, 1, 2}",       Domain([Fp(0), Fp(1), Fp(2)])),
        ("S = {0, 1, 2, 3, 4}", Domain([Fp(0), Fp(1), Fp(2), Fp(3), Fp(4)])),
    ]

    # Use the constant polynomial g(x1, x2) = 7. Per-var degree is 0 < |S| in
    # all three cases, so it's faithfully stored on every domain.
    for label, domain in cases:
        poly = MultivariatePoly.from_callable(domain, 2, lambda x1, x2: Fp(7))
        H = poly.sum_over_domain()
        trace = SumcheckProtocol(poly).run()
        expected = 7 * domain.size() ** 2
        print(f"    {label:28s}  H = {H.as_int():6d}  "
              f"(expected {expected:6d})  "
              f"verdict: {'ACCEPT' if trace.accepted else 'REJECT'}")
        assert trace.accepted
        assert H.as_int() == expected


def main() -> None:
    print(HR)
    print(" Dynamic sumcheck protocol — Python demo")
    print(HR)
    print(" Same protocol code, three different per-variable domains.")
    print(" The Domain class just accepts a list of distinct field elements.")
    print(HR)

    demo_boolean_hypercube()
    demo_three_element_domain()
    demo_arbitrary_four_element_domain()
    demo_dynamic_parameters_showcase()

    print()
    print(HR)
    print(" All demos completed successfully.")
    print(HR)


if __name__ == "__main__":
    main()
