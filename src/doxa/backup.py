"""The opinion-backup operator — propagate opinions up a search tree.

To search a game tree of opinions, a child opinion must be *backed up* into
its parent: the opponent's reply opinions at a node collapse into the single
opinion of the move that led there. This module supplies that operator for the
dialectical-checkers principled-lookahead build (lookahead design L-D4). It
does **not** build the tree or drive the recursion — those belong to the
step-3 search; here are only the pure ``list[Opinion] -> Opinion`` backup
functions.

Two variants, from the opinion-backup design space (research report §4):

- :func:`backup_ccf` — **PRIMARY**, design **A2 + B3** with a **B1** floor.
  CCF-fuse the opponent's reply opinions, then negate (``~``) the fused
  result for the cross-side flip. CCF "converts disagreement between sources
  into uncertainty" — when the opponent's plausible replies disagree about
  the outcome, the backed-up opinion gets honestly higher ``u`` (the
  endogenous B3 depth-uncertainty). An optional per-ply ``trust`` opinion
  applies the small B1 trust-discount floor.

- :func:`backup_minimax` — **CONTROL**, design **A1 + B1**. Hard
  opinion-minimax: the opponent picks the reply they most prefer (the
  ``Opinion`` ordering argmax), and the backed-up opinion is the negation of
  exactly that reply, with the same optional B1 per-ply trust discount.

**Perspective convention.** Every reply ``Opinion`` is from the perspective of
the *mover at the reply node* — the opponent. ``~omega`` swaps ``b<->d`` and
``a<->1-a`` while leaving ``u`` fixed, so "good for the opponent" reads as
"bad for me" and uncertainty is preserved across the side flip.

**Forced lines.** A single reply (a forced line) backs up to exactly
``~reply`` for **every** variant when ``trust`` is left at its full-trust
default: ``Opinion.ccf`` of one source returns it unchanged, the argmax of a
one-element list is that element, and the default full-trust discount is the
identity. The B1 floor is opt-in via the ``trust`` argument precisely so the
forced-line collapse stays exact.

Pure: no I/O, no persistence, no framework coupling.
"""

from __future__ import annotations

from collections.abc import Sequence

from doxa.opinion import Opinion

# The default per-ply trust opinion. ``dogmatic_true(0.5).discount(child)``
# leaves the child's (b, d, u) unchanged — a full-trust edge is the identity
# of trust discounting (Jøsang Def 14; doxa.opinion.Opinion.discount). With
# this default the B1 floor is a no-op, so a single forced reply backs up to
# exactly ``~reply``. Callers pass a slightly-uncertain trust to engage the
# real per-ply discount.
_FULL_TRUST = Opinion.dogmatic_true(0.5)


def _apply_trust_floor(backed_up: Opinion, trust: Opinion) -> Opinion:
    """Apply the B1 per-ply trust-discount floor to a backed-up opinion.

    ``trust`` is the trust opinion (the receiver of :meth:`Opinion.discount`);
    ``backed_up`` is the opinion being discounted. With the default
    full-trust opinion this is the identity. A slightly-uncertain ``trust``
    bleeds a small, fixed fraction of belief mass into ``u`` at every ply —
    the depth-discount floor of design B1.
    """
    return trust.discount(backed_up)


def backup_ccf(
    replies: Sequence[Opinion],
    *,
    trust: Opinion = _FULL_TRUST,
) -> Opinion:
    """Back opinions up via CCF over the opponent's replies (A2 + B3 + B1).

    PRIMARY backup rule. The opponent's reply opinions are fused with
    Consensus & Compromise Fusion — a single N-source ``Opinion.ccf`` call,
    since CCF is not associative — and the fused opinion is negated for the
    cross-side flip. CCF turns disagreement among the replies into
    uncertainty (design B3, endogenous depth-uncertainty); the optional
    per-ply ``trust`` opinion then applies the B1 trust-discount floor.

    A single reply collapses to exactly ``~reply`` (CCF of one source is the
    identity, and the default ``trust`` discount is the identity).

    Parameters
    ----------
    replies:
        The opponent's reply opinions at the node, each from the
        mover-at-the-reply-node (opponent) perspective. Must be non-empty.
    trust:
        The per-ply trust opinion for the B1 floor. Defaults to a full-trust
        opinion (the floor is then a no-op).

    Raises
    ------
    ValueError
        If ``replies`` is empty.
    """
    if len(replies) == 0:
        raise ValueError("backup_ccf needs at least one reply opinion")

    fused = Opinion.ccf(*replies)
    backed_up = ~fused
    return _apply_trust_floor(backed_up, trust)


def backup_minimax(
    replies: Sequence[Opinion],
    *,
    trust: Opinion = _FULL_TRUST,
) -> Opinion:
    """Back opinions up via hard opinion-minimax (A1 + B1).

    CONTROL backup rule — the minimax-correct baseline to differential-test
    the CCF rule against. The opponent picks the reply they most prefer: the
    argmax over the :class:`Opinion` total order ``(E, -u, -a)`` (Jøsang Def
    10). The backed-up opinion is the negation of exactly that single reply;
    every non-selected reply is discarded. The optional per-ply ``trust``
    opinion applies the B1 trust-discount floor.

    A single reply collapses to exactly ``~reply`` (the argmax of a
    one-element list is that element, and the default ``trust`` discount is
    the identity).

    Parameters
    ----------
    replies:
        The opponent's reply opinions at the node, each from the
        mover-at-the-reply-node (opponent) perspective. Must be non-empty.
    trust:
        The per-ply trust opinion for the B1 floor. Defaults to a full-trust
        opinion (the floor is then a no-op).

    Raises
    ------
    ValueError
        If ``replies`` is empty.
    """
    if len(replies) == 0:
        raise ValueError("backup_minimax needs at least one reply opinion")

    # The opponent prefers the reply with the greatest ordering value;
    # `max` over the Opinion total order is deterministic and independent of
    # the list order (ties broken by the full (E, -u, -a) key).
    opponent_best = max(replies)
    backed_up = ~opponent_best
    return _apply_trust_floor(backed_up, trust)
