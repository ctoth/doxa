"""Opinion-valued bipolar argumentation semantics — a gradual QBAF whose
carried value is a Jøsang opinion.

A bipolar argument graph: each argument carries an intrinsic Opinion (its own
evidence before supporters/attackers; tau = a is intrinsic[x].a), edges carry
support/attack opinions (edge strength/trust). The graph is evaluated
bottom-up over a DAG to a per-argument Opinion. A leaf resolves to its
intrinsic opinion — that is where belief originates; a move node carries a
vacuous intrinsic and falls back to E = tau when unargued. Acyclic only; a
cycle raises CyclicGraphError.

Pure: zero runtime dependencies, no I/O, no framework coupling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from doxa.opinion import Opinion

# Match the kernel's dogmatic threshold (doxa.opinion._TOL). A fused result
# with u below this is dogmatic and must be re-stamped with
# allow_dogmatic=True so the Opinion constructor accepts it.
_TOL = 1e-9


class CyclicGraphError(ValueError):
    """Raised by ``evaluate`` when the argument graph contains a cycle.

    The bottom-up topological traversal is defined only for a DAG. A cycle
    has no topological order, so evaluation cannot proceed. Subclasses
    ``ValueError`` so existing ``except ValueError`` handlers still catch it.
    """


@dataclass(frozen=True)
class BipolarOpinionGraph:
    """A bipolar argument graph with opinion-valued edges.

    Construction validates the graph fully (see ``__post_init__``); an
    already-constructed instance is guaranteed well-formed. Acyclicity is
    NOT checked at construction — it is checked by ``evaluate`` so the cycle
    diagnostic carries the traversal context.
    """

    arguments: frozenset[str]
    intrinsic: Mapping[str, Opinion]                  # per-argument own opinion; tau = intrinsic[x].a
    supports: frozenset[tuple[str, str]]              # (supporter, target)
    attacks: frozenset[tuple[str, str]]               # (attacker, target)
    edge_opinions: Mapping[tuple[str, str], Opinion]  # per-edge strength/trust

    def __post_init__(self) -> None:
        """Run the six locked construction-time validations, in order.

        Every failure is an explicit ``raise ValueError`` (never ``assert``
        — asserts vanish under ``python -O``). Acyclicity is the one check
        deferred to ``evaluate``. The base-rate-range check of the prior
        (Gate A) design is gone: ``tau`` is now ``intrinsic[x].a``, already
        validated to ``(0, 1)`` by ``Opinion.__post_init__`` when the
        intrinsic ``Opinion`` is constructed — a graph-level check would be
        dead code.
        """
        # --- Check 1: intrinsic covers exactly arguments ---
        intrinsic_keys = frozenset(self.intrinsic)
        if intrinsic_keys != self.arguments:
            missing = self.arguments - intrinsic_keys
            extra = intrinsic_keys - self.arguments
            raise ValueError(
                "intrinsic must cover exactly the declared arguments; "
                f"missing intrinsic opinions for {sorted(missing)}, "
                f"intrinsic opinions for undeclared arguments {sorted(extra)}"
            )

        # --- Check 2: support edges reference only declared arguments ---
        for src, dst in self.supports:
            if src not in self.arguments or dst not in self.arguments:
                raise ValueError(
                    f"support edge {(src, dst)!r} references an "
                    "undeclared argument"
                )

        # --- Check 3: attack edges reference only declared arguments ---
        for src, dst in self.attacks:
            if src not in self.arguments or dst not in self.arguments:
                raise ValueError(
                    f"attack edge {(src, dst)!r} references an "
                    "undeclared argument"
                )

        # --- Check 4: supports and attacks are disjoint ---
        overlap = self.supports & self.attacks
        if overlap:
            raise ValueError(
                "an edge cannot be both a support and an attack; "
                f"overlapping edges: {sorted(overlap)}"
            )

        # --- Check 5: edge_opinions keys exactly cover supports ∪ attacks ---
        all_edges = self.supports | self.attacks
        opinion_keys = frozenset(self.edge_opinions)
        if opinion_keys != all_edges:
            missing_op = all_edges - opinion_keys
            extra_op = opinion_keys - all_edges
            raise ValueError(
                "edge_opinions must have exactly one opinion per edge; "
                f"edges with no opinion: {sorted(missing_op)}, "
                f"opinions for non-edges: {sorted(extra_op)}"
            )

        # --- Check 6: no self-loops ---
        for src, dst in all_edges:
            if src == dst:
                raise ValueError(
                    f"argument {src!r} has a self-loop {(src, dst)!r}; "
                    "an argument cannot support or attack itself"
                )


def evaluate(graph: BipolarOpinionGraph) -> dict[str, Opinion]:
    """Per-argument opinion, computed bottom-up over the DAG.

    Resolves every argument with Kahn's algorithm over a sorted
    (smallest-name-first) ready set — deterministic and independent of
    ``frozenset`` / ``Mapping`` iteration order. Each argument is computed
    only once all its children (supporters and attackers) are resolved,
    then accrued via the locked CCF operator (see ``_accrue``).

    Returns a dict mapping every argument name to its resolved ``Opinion``.
    Raises ``CyclicGraphError`` if the graph contains a cycle.
    """
    all_edges = graph.supports | graph.attacks

    # --- Step 1: build the dependency structure ---
    # in_count[x] = number of incoming support+attack edges (its children).
    # dependents[x] = arguments that have x as a child (edges out of x).
    in_count: dict[str, int] = {arg: 0 for arg in graph.arguments}
    dependents: dict[str, list[str]] = {arg: [] for arg in graph.arguments}
    for src, dst in all_edges:
        in_count[dst] += 1
        dependents[src].append(dst)

    # --- Step 2: seed the ready set with every leaf, in sorted order ---
    ready = sorted(arg for arg in graph.arguments if in_count[arg] == 0)

    omega: dict[str, Opinion] = {}

    # --- Step 3: process smallest-named ready argument first ---
    while ready:
        x = ready.pop(0)
        omega[x] = _accrue(graph, x, omega)
        for y in dependents[x]:
            in_count[y] -= 1
            if in_count[y] == 0:
                # Insert y keeping `ready` sorted (smallest-name-first).
                _insert_sorted(ready, y)

    # --- Step 4: cycle detection ---
    if len(omega) < len(graph.arguments):
        unresolved = sorted(graph.arguments - frozenset(omega))
        raise CyclicGraphError(
            "the argument graph contains a cycle; unresolved arguments "
            f"(on or downstream of the cycle): {unresolved}"
        )

    # --- Step 5: return the resolved opinions ---
    return omega


def _insert_sorted(worklist: list[str], item: str) -> None:
    """Insert ``item`` into the already-sorted ``worklist`` in place."""
    lo, hi = 0, len(worklist)
    while lo < hi:
        mid = (lo + hi) // 2
        if worklist[mid] < item:
            lo = mid + 1
        else:
            hi = mid
    worklist.insert(lo, item)


def _accrue(
    graph: BipolarOpinionGraph,
    x: str,
    omega: Mapping[str, Opinion],
) -> Opinion:
    """Compute the resolved opinion for argument ``x``.

    Implements the Gate A2 §2 corrected "Model C" accrual with the locked
    CCF operator:

    1. Per-edge discounting — discount each child's resolved opinion through
       its edge opinion (``edge_opinion.discount(child_opinion)``).
    2. Negate each discounted attacker (``~``): an attacker's belief in its
       own claim becomes disbelief in ``x``.
    3. Assemble the source pool: discounted supporters followed by negated
       discounted attackers. Model C — the node's *intrinsic* opinion leads
       the pool **iff it is non-vacuous** (``intrinsic.u < 1.0 - _TOL``). A
       non-vacuous intrinsic (a leaf's own evidence, an objection's
       severity) is genuine evidence and must enter the CCF accrual; a
       vacuous intrinsic (a move node, no own evidence) carries nothing and
       is dropped — including it would only corrupt CCF's min-consensus at
       arity >= 3 (Gate A2 §2-3).
    4. Accrue: empty pool → ``Opinion.vacuous(tau)`` (only a vacuous-
       intrinsic node with no edges reaches this); single source → that
       source; N → one ``Opinion.ccf(*pool)`` call.
    5. Re-stamp ``a = tau_x`` (the intrinsic prior, never fused), carrying
       ``allow_dogmatic`` so a dogmatic fused result is accepted.
    """
    intrinsic_x = graph.intrinsic[x]
    tau_x = intrinsic_x.a

    sources: list[Opinion] = []

    # Edges are iterated in sorted order so the source list is built
    # identically regardless of ``frozenset`` iteration order (which varies
    # with ``PYTHONHASHSEED``). CCF is symmetric in its arguments, but a
    # fixed order keeps float arithmetic bit-reproducible (Gate A §3).

    # Step 1 — discounted supporters. The edge opinion is the trust
    # (``discount``'s receiver); the argument is the child's resolved opinion.
    for src, dst in sorted(graph.supports):
        if dst == x:
            s_disc = graph.edge_opinions[(src, dst)].discount(omega[src])
            sources.append(s_disc)

    # Steps 1 + 2 — discounted attackers, negated.
    for src, dst in sorted(graph.attacks):
        if dst == x:
            o_disc = graph.edge_opinions[(src, dst)].discount(omega[src])
            sources.append(~o_disc)

    # Step 3 — assemble the pool. The intrinsic leads iff it is non-vacuous:
    # vacuous(tau).u == 1.0 exactly, so ``< 1.0 - _TOL`` cleanly separates a
    # vacuous intrinsic (dropped) from a near-vacuous deliberate weak prior
    # (kept). (Gate A2 §2 step 3.)
    if intrinsic_x.u < 1.0 - _TOL:
        pool = [intrinsic_x, *sources]
    else:
        pool = sources

    # Step 4 — accrue with the locked CCF operator.
    if not pool:
        # Only reachable for a move node (vacuous intrinsic) with no edges
        # → total ignorance, projecting to exactly tau.
        return Opinion.vacuous(tau_x)
    if len(pool) == 1:
        # A single source is itself (Opinion.ccf(x) returns x unchanged);
        # spelled out to avoid the needless call and document intent.
        fused = pool[0]
    else:
        # CCF is not associative — one N-source call over the full pool.
        fused = Opinion.ccf(*pool)

    # Step 5 — re-stamp a = tau_x; carry the dogmatic flag.
    return Opinion(
        fused.b,
        fused.d,
        fused.u,
        tau_x,
        allow_dogmatic=fused.u < _TOL,
    )
