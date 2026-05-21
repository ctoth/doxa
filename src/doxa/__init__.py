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
"""

from doxa.opinion import BetaEvidence, Opinion

__all__ = ["BetaEvidence", "Opinion"]
