"""Hypothesis property suite for the doxa subjective-logic kernel (G2).

This file is the new property-based specification added during the ``doxa``
extraction. It complements the ported example tests in ``test_opinion.py`` by
asserting the algebraic laws the kernel docstrings already claim — over
randomized inputs rather than hand-picked examples.

Laws covered (all hard directives from the doxa extraction plan):
- negation involution               ``~~ω == ω``
- conjunction commutativity         ``x.conjunction(y) == y.conjunction(x)``
- consensus associativity           (non-dogmatic opinions)
- CCF self-idempotence              ``Opinion.ccf(ω, ω) == ω``
- fusion symmetry                   WBF and CCF invariant under permutation
- mass-sum invariant                every produced opinion has ``b + d + u == 1``
- van der Heijden Table I exact values (WBF and CCF columns, p. 7)

Grounding:
- Jøsang 2001, "A Logic for Uncertain Probabilities."
- van der Heijden et al. 2018, "Multi-Source Fusion Operations in
  Subjective Logic" — Table I, p. 7 (verified against
  papers/vanderHeijden_2018_MultiSourceFusionOperationsSubjectiveLogic/
  pngs/page-007.png).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from doxa import Opinion


# ── Strategies ─────────────────────────────────────────────────────


@st.composite
def nondogmatic_opinions(draw, min_uncertainty=0.01):
    """Generate valid non-dogmatic opinions (u >= min_uncertainty).

    Non-dogmatic opinions can always be fused via ``consensus_pair``;
    the ``min_uncertainty`` floor keeps fusion well-conditioned.
    """
    u = draw(st.floats(min_value=min_uncertainty, max_value=1.0 - 1e-6))
    remaining = 1.0 - u
    b = draw(st.floats(min_value=0.0, max_value=remaining))
    d = max(0.0, remaining - b)
    a = draw(st.floats(min_value=0.01, max_value=0.99))
    assume(abs(b + d + u - 1.0) < 1e-9)
    assume(b >= 0.0 and d >= 0.0 and u >= 0.0)
    return Opinion(b, d, u, a)


@st.composite
def any_opinions(draw):
    """Generate any valid opinion, dogmatic or non-dogmatic.

    With probability ~1/4 the draw is dogmatic (u = 0); a dogmatic
    opinion requires ``allow_dogmatic=True`` at construction. This
    strategy exercises both branches of the kernel's invariants.
    """
    if draw(st.booleans()) and draw(st.booleans()):
        # Dogmatic: u = 0, b + d = 1.
        b = draw(st.floats(min_value=0.0, max_value=1.0))
        d = max(0.0, 1.0 - b)
        a = draw(st.floats(min_value=0.01, max_value=0.99))
        assume(abs(b + d - 1.0) < 1e-9)
        return Opinion(b, d, 0.0, a, allow_dogmatic=True)
    return draw(nondogmatic_opinions())


# ── Negation involution ────────────────────────────────────────────


class TestNegationInvolutionProperty:
    """``~~ω == ω`` for every valid opinion (Jøsang 2001 Theorem 6)."""

    @pytest.mark.property
    @given(any_opinions())
    @settings(deadline=None)
    def test_double_negation_is_identity(self, op):
        result = ~~op
        assert math.isclose(result.b, op.b, abs_tol=1e-9)
        assert math.isclose(result.d, op.d, abs_tol=1e-9)
        assert math.isclose(result.u, op.u, abs_tol=1e-9)
        assert math.isclose(result.a, op.a, abs_tol=1e-9)
        # Quantized equality must also hold.
        assert result == op


# ── Conjunction commutativity ──────────────────────────────────────


class TestConjunctionCommutativityProperty:
    """``x.conjunction(y) == y.conjunction(x)`` (Jøsang 2001 Theorem 3)."""

    @pytest.mark.property
    @given(any_opinions(), any_opinions())
    @settings(deadline=None)
    def test_conjunction_is_commutative(self, x, y):
        xy = x.conjunction(y)
        yx = y.conjunction(x)
        assert math.isclose(xy.b, yx.b, abs_tol=1e-9)
        assert math.isclose(xy.d, yx.d, abs_tol=1e-9)
        assert math.isclose(xy.u, yx.u, abs_tol=1e-9)
        assert math.isclose(xy.a, yx.a, abs_tol=1e-9)
        assert xy == yx

    @pytest.mark.property
    @given(any_opinions(), any_opinions())
    @settings(deadline=None)
    def test_disjunction_is_commutative(self, x, y):
        """Dual of conjunction commutativity (Jøsang 2001 Theorem 4)."""
        xy = x.disjunction(y)
        yx = y.disjunction(x)
        assert math.isclose(xy.b, yx.b, abs_tol=1e-9)
        assert math.isclose(xy.d, yx.d, abs_tol=1e-9)
        assert math.isclose(xy.u, yx.u, abs_tol=1e-9)
        assert math.isclose(xy.a, yx.a, abs_tol=1e-9)
        assert xy == yx


# ── Consensus associativity (non-dogmatic) ─────────────────────────


class TestConsensusAssociativityProperty:
    """``(a ⊕ b) ⊕ c == a ⊕ (b ⊕ c)`` for non-dogmatic opinions.

    Jøsang's consensus operator is associative. ``consensus_pair``
    raises on two dogmatic inputs, so the strategy is restricted to
    non-dogmatic opinions; ``assume`` further guards the intermediate
    folds against drift into the dogmatic regime.
    """

    @pytest.mark.property
    @given(
        nondogmatic_opinions(min_uncertainty=0.02),
        nondogmatic_opinions(min_uncertainty=0.02),
        nondogmatic_opinions(min_uncertainty=0.02),
    )
    @settings(deadline=None)
    def test_consensus_pair_is_associative(self, a, b, c):
        assume(a.u > 1e-6 and b.u > 1e-6 and c.u > 1e-6)
        left = a.consensus_pair(b).consensus_pair(c)
        right = a.consensus_pair(b.consensus_pair(c))
        assert math.isclose(left.b, right.b, abs_tol=1e-6)
        assert math.isclose(left.d, right.d, abs_tol=1e-6)
        assert math.isclose(left.u, right.u, abs_tol=1e-6)
        assert math.isclose(left.a, right.a, abs_tol=1e-6)


# ── CCF self-idempotence ───────────────────────────────────────────


class TestCCFSelfIdempotenceProperty:
    """``Opinion.ccf(ω, ω) == ω`` for every opinion.

    Self-fusion drives all per-actor residuals to zero, triggering the
    ``b^comp_sum ≈ 0`` edge case of van der Heijden 2018 Definition 5.
    The kernel routes the residual missing mass straight into ``u``,
    which makes CCF self-fusion exactly idempotent.
    """

    @pytest.mark.property
    @given(any_opinions())
    @settings(deadline=None)
    def test_ccf_self_fusion_is_idempotent(self, op):
        result = Opinion.ccf(op, op)
        assert math.isclose(result.b, op.b, abs_tol=1e-9), (
            f"b: {result.b} vs {op.b}"
        )
        assert math.isclose(result.d, op.d, abs_tol=1e-9), (
            f"d: {result.d} vs {op.d}"
        )
        assert math.isclose(result.u, op.u, abs_tol=1e-9), (
            f"u: {result.u} vs {op.u}"
        )


# ── Fusion symmetry (argument permutation) ─────────────────────────


class TestFusionSymmetryProperty:
    """WBF and CCF are invariant under permutation of their arguments.

    Both van der Heijden 2018 Definitions 4 and 5 use symmetric sums
    and products over the actor set, so a direct N-source call returns
    the same opinion regardless of argument order.
    """

    @pytest.mark.property
    @given(
        nondogmatic_opinions(min_uncertainty=0.05),
        nondogmatic_opinions(min_uncertainty=0.05),
        nondogmatic_opinions(min_uncertainty=0.05),
    )
    @settings(deadline=None)
    def test_wbf_is_permutation_invariant(self, a, b, c):
        abc = Opinion.wbf(a, b, c)
        bca = Opinion.wbf(b, c, a)
        cab = Opinion.wbf(c, a, b)
        for other in (bca, cab):
            assert math.isclose(abc.b, other.b, abs_tol=1e-6)
            assert math.isclose(abc.d, other.d, abs_tol=1e-6)
            assert math.isclose(abc.u, other.u, abs_tol=1e-6)
            assert math.isclose(abc.a, other.a, abs_tol=1e-6)

    @pytest.mark.property
    @given(any_opinions(), any_opinions(), any_opinions())
    @settings(deadline=None)
    def test_ccf_is_permutation_invariant(self, a, b, c):
        abc = Opinion.ccf(a, b, c)
        bca = Opinion.ccf(b, c, a)
        cab = Opinion.ccf(c, a, b)
        for other in (bca, cab):
            assert math.isclose(abc.b, other.b, abs_tol=1e-6)
            assert math.isclose(abc.d, other.d, abs_tol=1e-6)
            assert math.isclose(abc.u, other.u, abs_tol=1e-6)
            assert math.isclose(abc.a, other.a, abs_tol=1e-6)


# ── Mass-sum invariant ─────────────────────────────────────────────


class TestMassSumInvariantProperty:
    """Every opinion produced by any kernel operation satisfies b+d+u==1.

    The constructor enforces this for inputs; these properties confirm
    every operation preserves it on its output.
    """

    @staticmethod
    def _assert_sum_one(op):
        assert abs(op.b + op.d + op.u - 1.0) < 1e-6, (
            f"mass sum != 1: b={op.b}, d={op.d}, u={op.u}"
        )

    @pytest.mark.property
    @given(any_opinions())
    @settings(deadline=None)
    def test_negation_preserves_sum(self, op):
        self._assert_sum_one(~op)

    @pytest.mark.property
    @given(any_opinions())
    @settings(deadline=None)
    def test_maximize_uncertainty_preserves_sum(self, op):
        self._assert_sum_one(op.maximize_uncertainty())

    @pytest.mark.property
    @given(any_opinions(), any_opinions())
    @settings(deadline=None)
    def test_conjunction_preserves_sum(self, a, b):
        self._assert_sum_one(a.conjunction(b))

    @pytest.mark.property
    @given(any_opinions(), any_opinions())
    @settings(deadline=None)
    def test_disjunction_preserves_sum(self, a, b):
        self._assert_sum_one(a.disjunction(b))

    @pytest.mark.property
    @given(any_opinions(), any_opinions())
    @settings(deadline=None)
    def test_discount_preserves_sum(self, trust, source):
        self._assert_sum_one(trust.discount(source))

    @pytest.mark.property
    @given(nondogmatic_opinions(), nondogmatic_opinions())
    @settings(deadline=None)
    def test_consensus_pair_preserves_sum(self, a, b):
        self._assert_sum_one(a.consensus_pair(b))

    @pytest.mark.property
    @given(nondogmatic_opinions(min_uncertainty=0.05), nondogmatic_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_wbf_preserves_sum(self, a, b):
        self._assert_sum_one(Opinion.wbf(a, b))

    @pytest.mark.property
    @given(any_opinions(), any_opinions())
    @settings(deadline=None)
    def test_ccf_preserves_sum(self, a, b):
        self._assert_sum_one(Opinion.ccf(a, b))

    @pytest.mark.property
    @given(nondogmatic_opinions(min_uncertainty=0.05), nondogmatic_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_fuse_preserves_sum(self, a, b):
        self._assert_sum_one(Opinion.fuse(a, b, method="auto"))


# ── van der Heijden Table I exact values ───────────────────────────


# van der Heijden et al. 2018, Table I (p. 7). Three sources, shared
# base rate a = 0.5. Verified against the paper page image
# papers/vanderHeijden_2018_MultiSourceFusionOperationsSubjectiveLogic/
# pngs/page-007.png:
#
#                A1     A2     A3      WBF      CCF
#   b(x)        0.10   0.40   0.70    0.562    0.629
#   b(~x)       0.30   0.20   0.10    0.146    0.182
#   u           0.60   0.40   0.20    0.292    0.189
#   a(x)        0.5    0.5    0.5     0.5      0.5
#
# The WBF column is rounded to 3 decimals in the paper; the kernel
# (van der Heijden Definition 4) produces the exact rationals
# 0.562162..., 0.145946..., 0.291892..., which match the paper's
# rounded values. The CCF column 0.629/0.182/0.189 is the same
# regression vector already asserted in test_opinion.py.
_TABLE_I_A1 = (0.10, 0.30, 0.60)
_TABLE_I_A2 = (0.40, 0.20, 0.40)
_TABLE_I_A3 = (0.70, 0.10, 0.20)
_TABLE_I_BASE_RATE = 0.5
_TABLE_I_WBF = (0.562, 0.146, 0.292)
_TABLE_I_CCF = (0.629, 0.182, 0.189)


class TestVanDerHeijdenTableI:
    """Regression of the kernel fusion operators against the paper's Table I.

    van der Heijden et al. 2018, Table I (p. 7). These are exact-value
    ground-truth tests, not property tests.
    """

    @staticmethod
    def _table_I_sources():
        a1 = Opinion(_TABLE_I_A1[0], _TABLE_I_A1[1], _TABLE_I_A1[2], _TABLE_I_BASE_RATE)
        a2 = Opinion(_TABLE_I_A2[0], _TABLE_I_A2[1], _TABLE_I_A2[2], _TABLE_I_BASE_RATE)
        a3 = Opinion(_TABLE_I_A3[0], _TABLE_I_A3[1], _TABLE_I_A3[2], _TABLE_I_BASE_RATE)
        return a1, a2, a3

    def test_wbf_matches_table_I(self):
        """WBF column of Table I: b=0.562, d=0.146, u=0.292, a=0.5."""
        a1, a2, a3 = self._table_I_sources()
        r = Opinion.wbf(a1, a2, a3)
        assert r.b == pytest.approx(_TABLE_I_WBF[0], abs=5e-4)
        assert r.d == pytest.approx(_TABLE_I_WBF[1], abs=5e-4)
        assert r.u == pytest.approx(_TABLE_I_WBF[2], abs=5e-4)
        assert r.a == pytest.approx(_TABLE_I_BASE_RATE, abs=1e-9)

    def test_wbf_matches_table_I_exact_rationals(self):
        """WBF Table I to full precision — the kernel is deterministic.

        Definition 4 yields exact rationals for this input; the values
        below are 37/65.85..., not floating-point noise. Tighter than
        the rounded paper column to lock the kernel arithmetic.
        """
        a1, a2, a3 = self._table_I_sources()
        r = Opinion.wbf(a1, a2, a3)
        assert r.b == pytest.approx(0.5621621621621622, abs=1e-12)
        assert r.d == pytest.approx(0.14594594594594595, abs=1e-12)
        assert r.u == pytest.approx(0.29189189189189185, abs=1e-12)

    def test_ccf_matches_table_I(self):
        """CCF column of Table I: b=0.629, d=0.182, u=0.189, a=0.5."""
        a1, a2, a3 = self._table_I_sources()
        r = Opinion.ccf(a1, a2, a3)
        assert r.b == pytest.approx(_TABLE_I_CCF[0], abs=5e-4)
        assert r.d == pytest.approx(_TABLE_I_CCF[1], abs=5e-4)
        assert r.u == pytest.approx(_TABLE_I_CCF[2], abs=5e-4)
        assert r.a == pytest.approx(_TABLE_I_BASE_RATE, abs=1e-9)

    def test_table_I_mass_sums_are_one(self):
        """Both fused Table I results satisfy the b+d+u==1 invariant."""
        a1, a2, a3 = self._table_I_sources()
        for r in (Opinion.wbf(a1, a2, a3), Opinion.ccf(a1, a2, a3)):
            assert abs(r.b + r.d + r.u - 1.0) < 1e-9

    def test_wbf_and_ccf_table_I_are_distinct(self):
        """WBF and CCF are different operators — Table I shows different columns.

        van der Heijden 2018, §III: CCF is not a fallback for WBF; on
        the same non-dogmatic inputs they produce materially different
        fused opinions.
        """
        a1, a2, a3 = self._table_I_sources()
        wbf_r = Opinion.wbf(a1, a2, a3)
        ccf_r = Opinion.ccf(a1, a2, a3)
        assert abs(wbf_r.b - ccf_r.b) > 1e-3
        assert abs(wbf_r.u - ccf_r.u) > 1e-3
