"""Perturbation sensitivity of doxa's outputs.

This module answers a single question in two settings: *how much does an
output opinion depend on one of its inputs?* It is the sensitivity primitive
the dialectical-checkers principled-lookahead build consumes (lookahead design
L-D5) — the frontier-priority heuristic ("expand the move whose opinion most
depends on an unresolved input") and the calibration step both need it.

Two functions, each pure and deterministic:

- :func:`opinion_sensitivity` — the finite-difference sensitivity of a
  Weighted Belief Fusion's *expectation* to a perturbation in one source
  opinion's uncertainty ``u``. Lifted near-as-is from propstore's
  ``fragility_scoring.opinion_sensitivity`` and cleaned of propstore-isms: it
  depends only on :mod:`doxa.opinion` (``Opinion`` + ``Opinion.wbf``), with no
  propstore imports, no ``CelExpr``/``QueryableAssumption`` coupling, and no
  lazy in-function imports.

- :func:`graph_perturbation_sensitivity` — written fresh for doxa. Given a
  :class:`~doxa.argumentation.BipolarOpinionGraph` and a target argument, it
  perturbs the graph (drop a witness/leaf node, or remove one edge),
  re-:func:`~doxa.argumentation.evaluate`\\ s it, and returns the magnitude of
  the change in the target argument's resolved-opinion *expectation*. It
  depends only on :mod:`doxa.argumentation` and :mod:`doxa.opinion`.

Pure: no I/O, no persistence, no framework coupling.
"""

from __future__ import annotations

from collections.abc import Sequence

from doxa.argumentation import BipolarOpinionGraph, evaluate
from doxa.opinion import Opinion

# Numerical tolerance for "a perturbed component left the open interval".
# Matches doxa.opinion._TOL so a perturbation that would produce a dogmatic
# or out-of-range opinion is rejected with the same grid the kernel uses.
_TOL = 1e-9


def _try_perturb(opinion: Opinion, delta_u: float, a: float) -> Opinion | None:
    """Perturb ``opinion``'s uncertainty by ``delta_u``, holding ``E`` fixed.

    The expectation ``E = b + a*u`` is held constant: the perturbed opinion
    has ``u' = u + delta_u`` and ``b' = E - a*u'``, ``d' = 1 - u' - b'``.
    Returns ``None`` if the perturbation would push ``u`` out of the open
    interval ``(0, 1)`` or drive ``b``/``d`` negative — i.e. when the
    finite difference is not well-defined at this point.

    Lifted from propstore ``fragility_scoring._try_perturb``; identical
    arithmetic, with the lazy ``from propstore.opinion import Opinion``
    replaced by the module-level :class:`doxa.opinion.Opinion` import.
    """
    expectation = opinion.expectation()
    u_new = opinion.u + delta_u
    if u_new < _TOL or u_new > 1.0 - _TOL:
        return None
    b_new = expectation - a * u_new
    d_new = 1.0 - u_new - b_new
    if b_new < -_TOL or d_new < -_TOL:
        return None
    b_new = max(0.0, b_new)
    d_new = max(0.0, d_new)
    try:
        return Opinion(b_new, d_new, u_new, a)
    except ValueError:
        return None


def opinion_sensitivity(
    opinions: Sequence[Opinion],
    index: int,
    *,
    delta: float = 0.01,
) -> float | None:
    """Finite-difference sensitivity of a ``wbf`` fusion to one source's ``u``.

    Fuses ``opinions`` with Weighted Belief Fusion, then measures how much the
    fused *expectation* moves when source ``index``'s uncertainty ``u`` is
    perturbed by ``+/-delta`` (holding that source's expectation fixed). The
    result is the magnitude of the (central, where possible) finite-difference
    quotient ``|E(perturbed_plus) - E(perturbed_minus)| / (2*delta)``.

    Returns ``None`` when the sensitivity is not well-defined:

    - fewer than two opinions (a one-source fusion has nothing to perturb
      against);
    - any source is dogmatic (``u < _TOL``) — a dogmatic source cannot be
      ``u``-perturbed and WBF rejects dogmatic inputs;
    - every perturbation attempt (down to ``delta/4``) leaves the valid
      region or makes ``wbf`` raise.

    A larger return value means the fused expectation is more fragile to that
    source's uncertainty.

    Lifted near-as-is from propstore ``fragility_scoring.opinion_sensitivity``.
    Propstore-isms removed: the lazy ``from propstore.opinion import wbf``
    import becomes a call to the :meth:`doxa.opinion.Opinion.wbf` classmethod
    (doxa exposes ``wbf`` as a classmethod, propstore as a free function); the
    helper :func:`_try_perturb` is lifted alongside it. The finite-difference
    arithmetic — central difference with two one-sided fallbacks and up to two
    ``delta`` halvings — is unchanged.
    """
    if len(opinions) < 2:
        return None
    for opinion in opinions:
        if opinion.u < _TOL:
            return None

    target = opinions[index]

    def _try_fuse(candidate: Opinion) -> float | None:
        mutable = list(opinions)
        mutable[index] = candidate
        try:
            return Opinion.wbf(*mutable).expectation()
        except ValueError:
            return None

    current_delta = delta
    for _attempt in range(3):
        minus = _try_perturb(target, -current_delta, target.a)
        plus = _try_perturb(target, current_delta, target.a)
        if minus is not None and plus is not None:
            expectation_minus = _try_fuse(minus)
            expectation_plus = _try_fuse(plus)
            if expectation_minus is not None and expectation_plus is not None:
                return abs(expectation_plus - expectation_minus) / (
                    2.0 * current_delta
                )
        if plus is not None:
            expectation_base = _try_fuse(target)
            expectation_plus = _try_fuse(plus)
            if expectation_base is not None and expectation_plus is not None:
                return abs(expectation_plus - expectation_base) / current_delta
        if minus is not None:
            expectation_base = _try_fuse(target)
            expectation_minus = _try_fuse(minus)
            if expectation_base is not None and expectation_minus is not None:
                return abs(expectation_base - expectation_minus) / current_delta
        current_delta /= 2.0
    return None


def graph_perturbation_sensitivity(
    graph: BipolarOpinionGraph,
    target: str,
    *,
    drop_node: str | None = None,
    remove_edge: tuple[str, str] | None = None,
) -> float:
    """Sensitivity of one argument's resolved opinion to a graph perturbation.

    Evaluates ``graph`` once, applies a single structural perturbation,
    re-evaluates the perturbed graph, and returns the magnitude of the change
    in ``target``'s resolved-opinion *expectation*::

        |E(omega_perturbed[target]) - E(omega_baseline[target])|

    This is "how much does this argument's opinion depend on that input" — a
    large value means dropping the input materially moves the target's
    opinion; zero means the input is irrelevant to the target.

    Perturbations (at most one; passing both is a usage error):

    - ``drop_node`` — remove an argument and every edge incident to it. The
      argument must not be the ``target`` (a dropped target has no opinion
      to compare). Use this to ask "how much does the target depend on this
      witness/leaf?"
    - ``remove_edge`` — remove a single support or attack edge, leaving both
      endpoints in place. Use this to ask "how much does the target depend
      on this particular support/attack relation?"

    With neither perturbation the graph is re-evaluated unchanged and the
    delta is exactly ``0.0`` — the documented unperturbed-yields-zero
    behaviour.

    Raises :class:`ValueError` for usage errors: an unknown ``target``, both
    perturbations supplied, dropping the ``target`` itself, or naming a node
    or edge that is not in the graph. A perturbation that disconnects the
    target without removing it is fine — it simply yields a (possibly large)
    finite delta.

    Depends only on :mod:`doxa.argumentation` and :mod:`doxa.opinion`. The
    propstore ``imps_rev`` analogue is argumentation-DF-QuAD-bound and is a
    reference only — this is a fresh implementation over doxa's own
    ``BipolarOpinionGraph`` / ``evaluate``.
    """
    if target not in graph.arguments:
        raise ValueError(
            f"target {target!r} is not an argument of the graph"
        )
    if drop_node is not None and remove_edge is not None:
        raise ValueError(
            "pass at most one perturbation — drop_node and remove_edge are "
            "mutually exclusive"
        )

    baseline = evaluate(graph)
    base_expectation = baseline[target].expectation()

    if drop_node is None and remove_edge is None:
        perturbed_graph = graph
    elif drop_node is not None:
        perturbed_graph = _drop_node(graph, target, drop_node)
    else:
        assert remove_edge is not None  # narrowed by the branch above
        perturbed_graph = _remove_edge(graph, remove_edge)

    perturbed = evaluate(perturbed_graph)
    perturbed_expectation = perturbed[target].expectation()
    return abs(perturbed_expectation - base_expectation)


def _drop_node(
    graph: BipolarOpinionGraph,
    target: str,
    node: str,
) -> BipolarOpinionGraph:
    """Return ``graph`` with ``node`` and every incident edge removed."""
    if node not in graph.arguments:
        raise ValueError(
            f"drop_node {node!r} is not an argument of the graph"
        )
    if node == target:
        raise ValueError(
            f"cannot drop the target argument {target!r} — it would have no "
            "resolved opinion to compare against"
        )

    arguments = graph.arguments - {node}
    intrinsic = {arg: op for arg, op in graph.intrinsic.items() if arg != node}
    supports = frozenset(
        edge for edge in graph.supports if node not in edge
    )
    attacks = frozenset(
        edge for edge in graph.attacks if node not in edge
    )
    surviving_edges = supports | attacks
    edge_opinions = {
        edge: op
        for edge, op in graph.edge_opinions.items()
        if edge in surviving_edges
    }
    return BipolarOpinionGraph(
        arguments=arguments,
        intrinsic=intrinsic,
        supports=supports,
        attacks=attacks,
        edge_opinions=edge_opinions,
    )


def _remove_edge(
    graph: BipolarOpinionGraph,
    edge: tuple[str, str],
) -> BipolarOpinionGraph:
    """Return ``graph`` with the single ``edge`` removed (endpoints kept)."""
    if edge not in graph.supports and edge not in graph.attacks:
        raise ValueError(
            f"remove_edge {edge!r} is neither a support nor an attack edge "
            "of the graph"
        )

    supports = graph.supports - {edge}
    attacks = graph.attacks - {edge}
    edge_opinions = {
        key: op for key, op in graph.edge_opinions.items() if key != edge
    }
    return BipolarOpinionGraph(
        arguments=graph.arguments,
        intrinsic=dict(graph.intrinsic),
        supports=supports,
        attacks=attacks,
        edge_opinions=edge_opinions,
    )
