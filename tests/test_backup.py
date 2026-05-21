"""Test suite for ``doxa.backup`` — the opinion-backup operator.

``doxa.backup`` propagates opinions up a game-search tree (dialectical-checkers
principled-lookahead design L-D4). Given the opponent's reply opinions at a
node, it derives the single opinion of the move leading to that node. Two pure
``list[Opinion] -> Opinion`` variants:

- ``backup_ccf`` — PRIMARY (research design A2 + B3, with a B1 floor): CCF-fuse
  the opponent's reply opinions, each negated ``~`` for the cross-side flip;
  disagreement among replies becomes uncertainty (endogenous B3). An optional
  per-ply ``trust`` opinion applies the B1 trust-discount floor.
- ``backup_minimax`` — CONTROL (A1 + B1): hard opinion-minimax — ``~`` of the
  reply the opponent most prefers (the ``Opinion`` ordering argmax over the
  opponent's own view), with the same optional B1 per-ply trust discount.

Both variants collapse to ``~reply`` for a single forced reply when ``trust``
is left at its full-trust default.

Markers: ``unit`` for focused contract tests, ``property`` for hypothesis-based
invariant tests — per doxa's ``pyproject.toml`` marker registry.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from doxa import Opinion
from doxa.backup import backup_ccf, backup_minimax


# ── Strategies ─────────────────────────────────────────────────────


@st.composite
def nondogmatic_opinions(draw, min_uncertainty=0.05):
    """Generate valid non-dogmatic opinions (u >= min_uncertainty)."""
    u = draw(st.floats(min_value=min_uncertainty, max_value=1.0 - 0.05))
    remaining = 1.0 - u
    b = draw(st.floats(min_value=0.0, max_value=remaining))
    d = max(0.0, remaining - b)
    a = draw(st.floats(min_value=0.1, max_value=0.9))
    assume(abs(b + d + u - 1.0) < 1e-9)
    assume(b >= 0.0 and d >= 0.0 and u >= 0.0)
    return Opinion(b, d, u, a)


def _assert_valid_opinion(op):
    """Assert ``op`` is a structurally valid Opinion (b+d+u==1, ranges)."""
    assert isinstance(op, Opinion)
    assert abs(op.b + op.d + op.u - 1.0) < 1e-6
    assert -1e-9 <= op.b <= 1.0 + 1e-9
    assert -1e-9 <= op.d <= 1.0 + 1e-9
    assert -1e-9 <= op.u <= 1.0 + 1e-9
    assert 0.0 < op.a < 1.0


# ── Empty input ────────────────────────────────────────────────────


class TestBackupEmptyInput:
    """Both variants reject an empty reply list."""

    @pytest.mark.unit
    def test_ccf_rejects_empty(self):
        with pytest.raises(ValueError):
            backup_ccf([])

    @pytest.mark.unit
    def test_minimax_rejects_empty(self):
        with pytest.raises(ValueError):
            backup_minimax([])


# ── Single-reply collapse (forced line) ────────────────────────────


class TestSingleReplyCollapse:
    """A single forced reply backs up to exactly ``~reply`` for EVERY variant.

    With the default full-trust per-ply opinion, no discount is applied and
    the cross-side step is exactly negation.
    """

    _REPLY = Opinion(0.6, 0.3, 0.1, 0.4)

    @pytest.mark.unit
    def test_ccf_single_reply_is_negation(self):
        result = backup_ccf([self._REPLY])
        assert result == ~self._REPLY

    @pytest.mark.unit
    def test_minimax_single_reply_is_negation(self):
        result = backup_minimax([self._REPLY])
        assert result == ~self._REPLY

    @pytest.mark.unit
    def test_ccf_single_dogmatic_reply_is_negation(self):
        reply = Opinion.dogmatic_true(0.4)
        assert backup_ccf([reply]) == ~reply

    @pytest.mark.unit
    def test_minimax_single_dogmatic_reply_is_negation(self):
        reply = Opinion.dogmatic_false(0.4)
        assert backup_minimax([reply]) == ~reply

    @pytest.mark.property
    @given(nondogmatic_opinions())
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_single_reply_collapses_for_both_variants(self, reply):
        assert backup_ccf([reply]) == ~reply
        assert backup_minimax([reply]) == ~reply


# ── Validity of the result ─────────────────────────────────────────


class TestBackupResultValidity:
    """The backed-up result is always a valid Opinion."""

    @pytest.mark.property
    @given(st.lists(nondogmatic_opinions(), min_size=1, max_size=5))
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_ccf_result_is_valid_opinion(self, replies):
        _assert_valid_opinion(backup_ccf(replies))

    @pytest.mark.property
    @given(st.lists(nondogmatic_opinions(), min_size=1, max_size=5))
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_minimax_result_is_valid_opinion(self, replies):
        _assert_valid_opinion(backup_minimax(replies))

    @pytest.mark.property
    @given(st.lists(nondogmatic_opinions(), min_size=1, max_size=5))
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_ccf_with_discount_floor_is_valid(self, replies):
        """The B1 trust-discount floor still yields a valid Opinion."""
        trust = Opinion(0.9, 0.0, 0.1, 0.5)
        _assert_valid_opinion(backup_ccf(replies, trust=trust))

    @pytest.mark.property
    @given(st.lists(nondogmatic_opinions(), min_size=1, max_size=5))
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_minimax_with_discount_floor_is_valid(self, replies):
        trust = Opinion(0.9, 0.0, 0.1, 0.5)
        _assert_valid_opinion(backup_minimax(replies, trust=trust))


# ── A2: disagreement raises uncertainty ────────────────────────────


class TestCCFDisagreementRaisesUncertainty:
    """CCF (A2) converts reply disagreement into backed-up uncertainty."""

    @pytest.mark.unit
    def test_disagreeing_replies_raise_u_above_agreeing(self):
        """Two replies that disagree about the outcome fuse to higher u.

        Agreeing replies (both good for the opponent) fuse to low u; two
        replies that point opposite ways fuse to high u — A2's core claim.
        """
        agree_a = Opinion(0.7, 0.1, 0.2, 0.5)
        agree_b = Opinion(0.7, 0.1, 0.2, 0.5)
        agreeing = backup_ccf([agree_a, agree_b])

        disagree_a = Opinion(0.8, 0.0, 0.2, 0.5)
        disagree_b = Opinion(0.0, 0.8, 0.2, 0.5)
        disagreeing = backup_ccf([disagree_a, disagree_b])

        assert disagreeing.u > agreeing.u

    @pytest.mark.unit
    def test_identical_replies_fuse_idempotently_then_negate(self):
        """Identical replies → CCF is idempotent → result is ~reply.

        ``Opinion.ccf(r, r) == r`` (CCF self-idempotence), so backing up a
        node where every reply is the same opinion is exactly ``~reply``.
        """
        reply = Opinion(0.5, 0.3, 0.2, 0.5)
        result = backup_ccf([reply, reply, reply])
        assert result.b == pytest.approx((~reply).b, abs=1e-9)
        assert result.d == pytest.approx((~reply).d, abs=1e-9)
        assert result.u == pytest.approx((~reply).u, abs=1e-9)

    @pytest.mark.property
    @given(nondogmatic_opinions())
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_ccf_agreeing_replies_preserve_u(self, reply):
        """When every reply is identical, CCF is idempotent: u is preserved.

        Negation preserves ``u``, so backing up N identical replies yields
        exactly the single reply's uncertainty — disagreement, not arity,
        is what raises ``u`` (the A2 thesis).
        """
        result = backup_ccf([reply, reply, reply])
        assert result.u == pytest.approx(reply.u, abs=1e-9)


# ── CCF order independence ─────────────────────────────────────────


class TestCCFOrderIndependence:
    """``backup_ccf`` is invariant under permutation of the reply list."""

    @pytest.mark.property
    @given(
        nondogmatic_opinions(),
        nondogmatic_opinions(),
        nondogmatic_opinions(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_ccf_backup_is_permutation_invariant(self, a, b, c):
        abc = backup_ccf([a, b, c])
        bca = backup_ccf([b, c, a])
        cab = backup_ccf([c, a, b])
        for other in (bca, cab):
            assert math.isclose(abc.b, other.b, abs_tol=1e-6)
            assert math.isclose(abc.d, other.d, abs_tol=1e-6)
            assert math.isclose(abc.u, other.u, abs_tol=1e-6)
            assert math.isclose(abc.a, other.a, abs_tol=1e-6)


# ── A1: minimax picks the opponent's preferred reply ───────────────


class TestMinimaxPicksOpponentArgmax:
    """``backup_minimax`` negates the reply the opponent most prefers.

    Reply opinions are from the mover-at-that-node's perspective. The
    opponent — the side to move at the reply node — prefers the reply with
    the greatest ``Opinion`` ordering value. The backed-up opinion is the
    negation of that single reply.
    """

    @pytest.mark.unit
    def test_minimax_negates_the_max_ordered_reply(self):
        weak = Opinion(0.2, 0.6, 0.2, 0.5)
        strong = Opinion(0.8, 0.0, 0.2, 0.5)
        # strong has the greater expectation → opponent's choice.
        assert strong > weak
        result = backup_minimax([weak, strong])
        assert result == ~strong

    @pytest.mark.unit
    def test_minimax_ignores_dominated_replies(self):
        """Only the opponent's best reply matters; the rest are discarded."""
        best = Opinion(0.9, 0.0, 0.1, 0.5)
        others = [Opinion(0.1, 0.7, 0.2, 0.5), Opinion(0.3, 0.5, 0.2, 0.5)]
        result = backup_minimax([*others, best])
        assert result == ~best

    @pytest.mark.property
    @given(st.lists(nondogmatic_opinions(), min_size=1, max_size=5))
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_minimax_is_negation_of_max(self, replies):
        result = backup_minimax(replies)
        assert result == ~max(replies)

    @pytest.mark.property
    @given(st.lists(nondogmatic_opinions(), min_size=1, max_size=5))
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_minimax_is_permutation_invariant(self, replies):
        """argmax over the ordering does not depend on list order."""
        import random

        shuffled = list(replies)
        random.Random(0).shuffle(shuffled)
        assert backup_minimax(replies) == backup_minimax(shuffled)


# ── B1 floor: per-ply trust discount ───────────────────────────────


class TestB1TrustDiscountFloor:
    """An uncertain per-ply ``trust`` opinion inflates the backed-up ``u``."""

    @pytest.mark.unit
    def test_ccf_discount_floor_raises_u(self):
        replies = [Opinion(0.7, 0.1, 0.2, 0.5), Opinion(0.6, 0.2, 0.2, 0.5)]
        no_floor = backup_ccf(replies)
        trust = Opinion(0.9, 0.0, 0.1, 0.5)
        floored = backup_ccf(replies, trust=trust)
        assert floored.u > no_floor.u

    @pytest.mark.unit
    def test_minimax_discount_floor_raises_u(self):
        replies = [Opinion(0.7, 0.1, 0.2, 0.5), Opinion(0.6, 0.2, 0.2, 0.5)]
        no_floor = backup_minimax(replies)
        trust = Opinion(0.9, 0.0, 0.1, 0.5)
        floored = backup_minimax(replies, trust=trust)
        assert floored.u > no_floor.u

    @pytest.mark.unit
    def test_full_trust_is_a_no_op(self):
        """The default full-trust opinion leaves the backup unchanged."""
        replies = [Opinion(0.7, 0.1, 0.2, 0.5), Opinion(0.5, 0.3, 0.2, 0.5)]
        full_trust = Opinion.dogmatic_true(0.5)
        assert backup_ccf(replies) == backup_ccf(replies, trust=full_trust)
        assert backup_minimax(replies) == backup_minimax(
            replies, trust=full_trust
        )

    @pytest.mark.property
    @given(st.lists(nondogmatic_opinions(), min_size=1, max_size=4))
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_discount_never_lowers_u(self, replies):
        """The B1 floor is monotone: discounting cannot reduce uncertainty."""
        trust = Opinion(0.85, 0.05, 0.1, 0.5)
        plain = backup_ccf(replies)
        floored = backup_ccf(replies, trust=trust)
        assert floored.u >= plain.u - 1e-9


# ── Determinism ────────────────────────────────────────────────────


class TestBackupDeterminism:
    """Both variants are pure deterministic functions."""

    @pytest.mark.property
    @given(st.lists(nondogmatic_opinions(), min_size=1, max_size=5))
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_ccf_deterministic(self, replies):
        assert backup_ccf(replies) == backup_ccf(replies)

    @pytest.mark.property
    @given(st.lists(nondogmatic_opinions(), min_size=1, max_size=5))
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_minimax_deterministic(self, replies):
        assert backup_minimax(replies) == backup_minimax(replies)
