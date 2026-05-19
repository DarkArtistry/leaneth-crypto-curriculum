"""Per-variable summation domain `S ⊆ F`, dynamic over an arbitrary element list.

Unlike the Rust reference, which provides fixed marker types `BooleanHypercube`
(S = {0, 1}) and `Interval3` (S = {0, 1, 2}), this Python `Domain` is purely
data-driven: pass it any list of field elements and it'll just work, provided
the elements are pairwise distinct (required for Lagrange interpolation to be
well-defined).

Mathematical contract:
    A `Domain` is a finite set `S ⊆ F` of pairwise-distinct field elements.
    The order of elements matters for storage layout — the index `j` of an
    element is its base-`|S|` "digit value" in the mixed-radix indexing used
    by `MultivariatePoly`.

Example (boolean hypercube):
    Fp = MersenneFp
    boolean = Domain([Fp(0), Fp(1)])
    assert boolean.size() == 2

Example (general 3-element domain over `F_p`):
    interval3 = Domain([Fp(0), Fp(1), Fp(2)])
    weird = Domain([Fp(7), Fp(13), Fp(42)])  # equally valid
"""

from __future__ import annotations

from typing import List

from .field import Fp


class Domain:
    """A finite per-variable summation domain over a field `F_p`.

    The constructor validates:
        - `elements` is non-empty (`|S| >= 1`).
        - all elements are instances of the same `Fp` subclass.
        - elements are pairwise distinct (required so Lagrange interpolation
          across the domain has nonzero denominators).

    Note: `|S| = 1` is allowed but degenerate (every polynomial of "degree < 1"
    is a constant; sumcheck collapses). Tests still pass, but real protocols
    use `|S| >= 2`.
    """

    __slots__ = ("_elements", "_field_cls")

    def __init__(self, elements: List[Fp]):
        if len(elements) == 0:
            raise ValueError("Domain must contain at least one element")

        field_cls = type(elements[0])
        for i, e in enumerate(elements):
            if not isinstance(e, Fp):
                raise TypeError(
                    f"Domain element [{i}] is not an Fp value: {e!r}"
                )
            if type(e) is not field_cls:
                raise TypeError(
                    f"Domain elements must all belong to the same field; "
                    f"element [0] is {field_cls.__name__} but element [{i}] "
                    f"is {type(e).__name__}"
                )

        # Pairwise distinctness check (O(|S|) via a set).
        seen = set()
        for e in elements:
            if e in seen:
                raise ValueError(
                    f"Domain elements must be pairwise distinct; "
                    f"{e!r} appears more than once"
                )
            seen.add(e)

        # Store an immutable copy so callers can't mutate behind our back.
        self._elements: List[Fp] = list(elements)
        self._field_cls = field_cls

    def elements(self) -> List[Fp]:
        """The elements of `S`, in the order fixed at construction."""
        return self._elements

    def size(self) -> int:
        """`|S|` — the number of points in the domain along each axis."""
        return len(self._elements)

    def field(self) -> type:
        """The `Fp` subclass these elements belong to."""
        return self._field_cls

    def __repr__(self) -> str:
        return f"Domain(size={self.size()}, elements={self._elements})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Domain):
            return NotImplemented
        return self._elements == other._elements

    def __hash__(self) -> int:
        return hash(tuple(self._elements))
