"""doxa — a pure Subjective Logic library.

`doxa` implements Jøsang's Subjective Logic (Jøsang 2001) as a small,
dependency-free kernel: opinions over Beta distributions, the Beta-evidence
mapping, and the multi-source belief fusion operators of van der Heijden et al.
(2018).

The kernel is intentionally pure — no provenance, no persistence, no framework
coupling. It is usable by anyone who needs an honest algebra of uncertain
probability.

Public API
----------
- ``Opinion`` — a subjective opinion ω = (b, d, u, a), with negation,
  conjunction/disjunction, consensus, discounting, uncertainty maximization,
  ordering, the Beta-evidence conversion, and the ``wbf``/``ccf``/``fuse``
  multi-source fusion operators.
- ``BetaEvidence`` — the (r, s, a) Beta-evidence representation.
- ``BipolarOpinionGraph``, ``evaluate``, ``CyclicGraphError`` — an
  opinion-valued bipolar argument graph: each argument carries an intrinsic
  ``Opinion`` (its own evidence; ``tau = a`` is ``intrinsic[x].a``), edges
  carry support/attack opinions, and ``evaluate`` computes a per-argument
  ``Opinion`` bottom-up over the DAG. A leaf resolves to its intrinsic
  opinion; a move node's intrinsic is ``Opinion.vacuous(tau)``.
- ``opinion_sensitivity``, ``graph_perturbation_sensitivity`` — perturbation
  sensitivity of doxa's outputs: how much a fused expectation, or an
  argument's resolved opinion, depends on one of its inputs.
- ``backup_ccf``, ``backup_minimax`` — the opinion-backup operator: propagate
  opinions up a game-search tree (CCF-over-replies primary; hard
  opinion-minimax control).
"""

from doxa.argumentation import BipolarOpinionGraph, CyclicGraphError, evaluate
from doxa.backup import backup_ccf, backup_minimax
from doxa.opinion import BetaEvidence, Opinion
from doxa.sensitivity import graph_perturbation_sensitivity, opinion_sensitivity

__all__ = [
    "BetaEvidence",
    "BipolarOpinionGraph",
    "CyclicGraphError",
    "Opinion",
    "backup_ccf",
    "backup_minimax",
    "evaluate",
    "graph_perturbation_sensitivity",
    "opinion_sensitivity",
]
