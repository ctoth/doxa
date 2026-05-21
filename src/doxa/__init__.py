"""doxa — a pure Subjective Logic library.

`doxa` implements Jøsang's Subjective Logic (Jøsang 2001) as a small,
dependency-free kernel: opinions over Beta distributions, the Beta-evidence
mapping, and the multi-source belief fusion operators of van der Heijden et al.
(2018).

The kernel is intentionally pure — no provenance, no persistence, no framework
coupling. It is usable by anyone who needs an honest algebra of uncertain
probability.

The `Opinion` type and fusion operators are not yet defined in this scaffold
release; they arrive in a subsequent kernel commit.
"""
