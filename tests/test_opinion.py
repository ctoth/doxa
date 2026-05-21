"""Tests for doxa.opinion — Jøsang 2001 subjective logic operations.

Ported from propstore/tests/test_opinion.py for the pure ``doxa`` kernel
extraction (G2). The kernel carries NO provenance: every free function has
been lifted to a method or classmethod on ``Opinion``, and the old
``from propstore.opinion import (... ccf, consensus, ...)`` free-function
imports are replaced by ``Opinion.ccf(...)`` / ``trust.discount(source)`` /
etc. No provenance assertions existed in the propstore suite, so none were
deleted; only call sites were rewritten to the lifted API.

Page-image grounding:
- Josang 2001, "A Logic for Uncertain Probabilities" — pp. 4, 6, 24.
- van der Heijden et al. 2018, "Multi-Source Fusion Operations in
  Subjective Logic" — p. 4 (Def 4 WBF), Table I (p. 7).
"""

import math
from dataclasses import MISSING, fields

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from doxa import BetaEvidence, Opinion
from doxa.opinion import _TOL


# --- Helpers ---

def approx(val, abs=1e-7):
    return pytest.approx(val, abs=abs)


class TestNoSilentBaseRateDefaults:
    def test_opinion_requires_explicit_base_rate(self):
        defaults = {field.name: field.default for field in fields(Opinion)}

        assert defaults["a"] is MISSING

    def test_beta_evidence_requires_explicit_base_rate(self):
        defaults = {field.name: field.default for field in fields(BetaEvidence)}

        assert defaults["a"] is MISSING

    def test_convenience_constructors_require_explicit_base_rate(self):
        with pytest.raises(TypeError):
            Opinion.vacuous()
        with pytest.raises(TypeError):
            Opinion.dogmatic_true()
        with pytest.raises(TypeError):
            Opinion.dogmatic_false()
        with pytest.raises(TypeError):
            Opinion.from_evidence(0, 0)
        with pytest.raises(TypeError):
            Opinion.from_probability(0.5, 0)


# --- 1. b + d + u == 1 for all valid opinions ---

class TestBDUSum:
    def test_vacuous(self):
        o = Opinion.vacuous(0.5)
        assert o.b + o.d + o.u == approx(1.0)

    def test_dogmatic_true(self):
        o = Opinion.dogmatic_true(0.5)
        assert o.b + o.d + o.u == approx(1.0)

    def test_dogmatic_false(self):
        o = Opinion.dogmatic_false(0.5)
        assert o.b + o.d + o.u == approx(1.0)

    def test_arbitrary(self):
        o = Opinion(0.3, 0.2, 0.5, 0.6)
        assert o.b + o.d + o.u == approx(1.0)

    def test_rejects_invalid_sum(self):
        with pytest.raises(ValueError, match="b \\+ d \\+ u"):
            Opinion(0.5, 0.5, 0.5, 0.5)


# --- 2. 0 < a < 1 for all non-degenerate opinions ---

class TestBaseRate:
    def test_a_zero_rejected(self):
        with pytest.raises(ValueError, match="a="):
            Opinion(0.5, 0.3, 0.2, 0.0)

    def test_a_one_rejected(self):
        with pytest.raises(ValueError, match="a="):
            Opinion(0.5, 0.3, 0.2, 1.0)

    def test_a_valid(self):
        o = Opinion(0.5, 0.3, 0.2, 0.7)
        assert 0.0 < o.a < 1.0


# --- 3. E(ω) = b + a*u is in [0, 1] ---

class TestExpectation:
    def test_vacuous(self):
        o = Opinion.vacuous(a=0.3)
        assert o.expectation() == approx(0.3)

    def test_dogmatic_true(self):
        assert Opinion.dogmatic_true(0.5).expectation() == approx(1.0)

    def test_dogmatic_false(self):
        assert Opinion.dogmatic_false(0.5).expectation() == approx(0.0)

    def test_in_unit_interval(self):
        o = Opinion(0.3, 0.2, 0.5, 0.6)
        e = o.expectation()
        assert 0.0 <= e <= 1.0
        assert e == approx(0.3 + 0.6 * 0.5)


# --- 4. E(ω_x ∧ ω_y) == E(ω_x) * E(ω_y) ---

class TestConjunctionExpectation:
    """E(x AND y) == E(x)*E(y) (Jøsang Proof 5, p.17).

    The binary conjunction expectation identity holds when operands are
    dogmatic (u=0, so E=b) or when base rates are 0.5 with the adjusted
    conjunction base rate. We test the structural formula directly and
    verify the identity for dogmatic and evidence-saturated cases.
    """

    def test_product_rule_dogmatic(self):
        """Dogmatic opinions: u=0 so E=b, conjunction b = bx*by."""
        x = Opinion(0.7, 0.3, 0.0, 0.5, allow_dogmatic=True)
        y = Opinion(0.4, 0.6, 0.0, 0.5, allow_dogmatic=True)
        conj = x & y
        assert conj.expectation() == approx(x.expectation() * y.expectation())

    def test_product_rule_high_evidence(self):
        """With very high evidence (low u), expectation approaches b, product rule approximately holds."""
        x = Opinion.from_probability(0.75, 1000, 0.5)
        y = Opinion.from_probability(0.60, 1000, 0.5)
        conj = x & y
        # With near-zero u, this should be very close
        assert conj.expectation() == approx(x.expectation() * y.expectation(), abs=0.01)

    def test_conjunction_produces_valid_opinion(self):
        x = Opinion(0.6, 0.1, 0.3, 0.5)
        y = Opinion(0.4, 0.2, 0.4, 0.5)
        conj = x & y
        assert conj.b + conj.d + conj.u == approx(1.0)
        assert 0.0 <= conj.expectation() <= 1.0


# --- 5. E(ω_x ∨ ω_y) == E(ω_x) + E(ω_y) - E(ω_x)*E(ω_y) ---

class TestDisjunctionExpectation:
    """E(x OR y) == E(x) + E(y) - E(x)*E(y) (Jøsang Proof 5, p.17).

    Like conjunction, the identity holds exactly for dogmatic opinions
    and approximately for high-evidence opinions.
    """

    def test_inclusion_exclusion_dogmatic(self):
        x = Opinion(0.7, 0.3, 0.0, 0.5, allow_dogmatic=True)
        y = Opinion(0.4, 0.6, 0.0, 0.5, allow_dogmatic=True)
        disj = x | y
        ex, ey = x.expectation(), y.expectation()
        assert disj.expectation() == approx(ex + ey - ex * ey)

    def test_inclusion_exclusion_high_evidence(self):
        x = Opinion.from_probability(0.75, 1000, 0.5)
        y = Opinion.from_probability(0.60, 1000, 0.5)
        disj = x | y
        ex, ey = x.expectation(), y.expectation()
        assert disj.expectation() == approx(ex + ey - ex * ey, abs=0.01)

    def test_disjunction_produces_valid_opinion(self):
        x = Opinion(0.6, 0.1, 0.3, 0.5)
        y = Opinion(0.4, 0.2, 0.4, 0.5)
        disj = x | y
        assert disj.b + disj.d + disj.u == approx(1.0)
        assert 0.0 <= disj.expectation() <= 1.0


# --- 6. Discounting with vacuous trust produces vacuous opinion ---

class TestDiscountVacuousTrust:
    def test_vacuous_trust(self):
        trust = Opinion.vacuous(0.5)
        source = Opinion(0.7, 0.1, 0.2, 0.6)
        result = trust.discount(source)
        assert result.b == approx(0.0)
        assert result.d == approx(0.0)
        assert result.u == approx(1.0)

    def test_vacuous_trust_preserves_base_rate(self):
        trust = Opinion.vacuous(0.5)
        source = Opinion(0.7, 0.1, 0.2, 0.6)
        result = trust.discount(source)
        assert result.a == approx(source.a)


# --- 7. Negation field mapping ---

class TestNegationInvolution:
    def test_negation_swaps_b_d(self):
        o = Opinion(0.3, 0.2, 0.5, 0.6)
        neg = ~o
        assert neg.b == approx(o.d)
        assert neg.d == approx(o.b)
        assert neg.u == approx(o.u)
        assert neg.a == approx(1.0 - o.a)


# --- 8. Round-trip Opinion -> BetaEvidence -> Opinion ---

class TestRoundTrip:
    def test_roundtrip_vacuous(self):
        o = Opinion.vacuous(a=0.7)
        rt = o.to_beta_evidence().to_opinion()
        assert rt == o

    def test_dogmatic_raises(self):
        with pytest.raises(ValueError, match="dogmatic"):
            Opinion.dogmatic_true(0.5).to_beta_evidence()


# --- 9. from_evidence(0, 0) produces vacuous opinion ---

class TestFromEvidenceVacuous:
    def test_zero_evidence(self):
        o = Opinion.from_evidence(0, 0, 0.5)
        assert o.b == approx(0.0)
        assert o.d == approx(0.0)
        assert o.u == approx(1.0)


# --- 10. from_probability(0.5, 0) produces vacuous opinion ---

class TestFromProbabilityVacuous:
    def test_zero_sample_size(self):
        o = Opinion.from_probability(0.5, 0, 0.5)
        assert o.b == approx(0.0)
        assert o.d == approx(0.0)
        assert o.u == approx(1.0)


# --- 11. from_probability(0.7, 100) produces narrow opinion near 0.7 ---

class TestFromProbabilityNarrow:
    def test_narrow_opinion(self):
        o = Opinion.from_probability(0.7, 100, 0.5)
        assert o.expectation() == approx(0.7, abs=0.02)
        # With 100 observations, uncertainty should be small
        assert o.u < 0.05

    def test_evidence_counts(self):
        o = Opinion.from_probability(0.7, 100, 0.5)
        be = o.to_beta_evidence()
        assert be.r == approx(70.0)
        assert be.s == approx(30.0)


# --- Additional edge-case tests ---

class TestConsensusErrors:
    def test_two_dogmatic_raises(self):
        with pytest.raises(ValueError, match="dogmatic"):
            Opinion.dogmatic_true(0.5).consensus_pair(Opinion.dogmatic_false(0.5))

    def test_single_opinion_consensus(self):
        o = Opinion(0.3, 0.2, 0.5, 0.5)
        assert Opinion.consensus(o) == o

    def test_empty_consensus_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Opinion.consensus()


class TestBetaEvidenceValidation:
    def test_negative_r_rejected(self):
        with pytest.raises(ValueError):
            BetaEvidence(-1, 0, 0.5)

    def test_negative_s_rejected(self):
        with pytest.raises(ValueError):
            BetaEvidence(0, -1, 0.5)


class TestUncertaintyInterval:
    def test_vacuous(self):
        o = Opinion.vacuous(0.5)
        lo, hi = o.uncertainty_interval()
        assert lo == approx(0.0)
        assert hi == approx(1.0)

    def test_dogmatic_true(self):
        o = Opinion.dogmatic_true(0.5)
        lo, hi = o.uncertainty_interval()
        assert lo == approx(1.0)
        assert hi == approx(1.0)


# ── Hypothesis strategies for opinions ─────────────────────────────


@st.composite
def valid_opinions(draw, min_uncertainty=0.01):
    """Generate random valid opinions with u > 0.

    Opinions with u=0 are dogmatic and cannot be fused via consensus
    (raises ValueError).  The min_uncertainty bound ensures fusibility.
    """
    # Draw b, d, u such that b + d + u = 1, all >= 0, u >= min_uncertainty
    u = draw(st.floats(min_value=min_uncertainty, max_value=1.0 - 1e-6))
    remaining = 1.0 - u
    b = draw(st.floats(min_value=0.0, max_value=remaining))
    d = remaining - b
    # Clamp to avoid floating-point drift
    d = max(0.0, d)
    # Base rate in (0, 1)
    a = draw(st.floats(min_value=0.01, max_value=0.99))
    assume(abs(b + d + u - 1.0) < 1e-9)
    assume(b >= 0.0 and d >= 0.0 and u >= 0.0)
    return Opinion(b, d, u, a)


# ── F29: Property tests for consensus algebra ──────────────────────


class TestConsensusPropertyTests:
    """Hypothesis property tests for consensus fusion (F29).

    These fill coverage gaps from audit-opinion-algebra GAP 1:
    commutativity and vacuous identity were only tested with
    concrete examples, not with randomized inputs.
    """

    @pytest.mark.property
    @given(valid_opinions(), valid_opinions())
    @settings(deadline=None)
    def test_commutativity(self, op_a, op_b):
        """fuse(a, b) == fuse(b, a) — Jøsang Theorem 7 is symmetric.

        The consensus operator must be commutative for all valid
        non-dogmatic opinion pairs.
        """
        ab = op_a.consensus_pair(op_b)
        ba = op_b.consensus_pair(op_a)
        assert math.isclose(ab.b, ba.b, abs_tol=1e-9), (
            f"b mismatch: {ab.b} vs {ba.b}"
        )
        assert math.isclose(ab.d, ba.d, abs_tol=1e-9), (
            f"d mismatch: {ab.d} vs {ba.d}"
        )
        assert math.isclose(ab.u, ba.u, abs_tol=1e-9), (
            f"u mismatch: {ab.u} vs {ba.u}"
        )
        assert math.isclose(ab.a, ba.a, abs_tol=1e-9), (
            f"a mismatch: {ab.a} vs {ba.a}"
        )

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_vacuous_identity(self, op):
        """fuse(a, vacuous) ≈ a — vacuous opinion is the identity element.

        Per Jøsang 2001 Theorem 7: fusing with (0, 0, 1, a_v) should
        return the original opinion unchanged (for b, d, u components).
        The base rate of the result equals the original's base rate
        when the other operand is vacuous (verified algebraically in
        audit-opinion-algebra Finding 15).
        """
        vacuous = Opinion.vacuous(a=op.a)  # match base rate to avoid denom_a edge case
        fused = op.consensus_pair(vacuous)
        assert math.isclose(fused.b, op.b, abs_tol=1e-9), (
            f"b: fused={fused.b}, original={op.b}"
        )
        assert math.isclose(fused.d, op.d, abs_tol=1e-9), (
            f"d: fused={fused.d}, original={op.d}"
        )
        assert math.isclose(fused.u, op.u, abs_tol=1e-9), (
            f"u: fused={fused.u}, original={op.u}"
        )
        assert math.isclose(fused.a, op.a, abs_tol=1e-9), (
            f"a: fused={fused.a}, original={op.a}"
        )

    @pytest.mark.property
    @given(valid_opinions(), valid_opinions())
    @settings(deadline=None)
    def test_consensus_reduces_uncertainty(self, op_a, op_b):
        """Consensus never increases uncertainty (Josang 2001, p.25-26).

        For any two non-dogmatic opinions, the fused uncertainty must be
        <= min(a.u, b.u). This is a guard test to maintain the invariant.

        Phase 1 Red: this property must hold for the consensus operator
        to be valid for fusing corpus and categorical opinions.
        """
        fused = op_a.consensus_pair(op_b)
        assert fused.u <= min(op_a.u, op_b.u) + 1e-9, (
            f"Consensus increased uncertainty: fused.u={fused.u}, "
            f"min(a.u={op_a.u}, b.u={op_b.u})={min(op_a.u, op_b.u)}"
        )

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_vacuous_identity_different_base_rate(self, op):
        """fuse(a, vacuous_with_different_a) still preserves b, d, u of a.

        The base rate of the result should equal op.a per the consensus
        formula derivation (audit-opinion-algebra Finding 15 verified this
        algebraically for the non-vacuous case).
        """
        # Use a different base rate for vacuous
        vac_a = 1.0 - op.a  # guaranteed different and in (0, 1)
        assume(0.01 < vac_a < 0.99)
        vacuous = Opinion.vacuous(a=vac_a)
        fused = op.consensus_pair(vacuous)
        # b, d, u should be preserved regardless of vacuous base rate
        assert math.isclose(fused.b, op.b, abs_tol=1e-9), (
            f"b: fused={fused.b}, original={op.b}"
        )
        assert math.isclose(fused.d, op.d, abs_tol=1e-9), (
            f"d: fused={fused.d}, original={op.d}"
        )
        assert math.isclose(fused.u, op.u, abs_tol=1e-9), (
            f"u: fused={fused.u}, original={op.u}"
        )
        # Base rate should be original's base rate (vacuous contributes no information)
        assert math.isclose(fused.a, op.a, abs_tol=1e-9), (
            f"a: fused={fused.a}, original={op.a}"
        )


# ── Uncertainty Maximization (Jøsang 2001 Def 16, p.30) ─────────


class TestUncertaintyMaximization:
    """Jøsang 2001 p.30: maximize u while preserving E(x)."""

    def test_concrete_example(self):
        op = Opinion(0.7, 0.1, 0.2, 0.5)
        result = op.maximize_uncertainty()
        assert abs(result.expectation() - op.expectation()) < 1e-9
        assert result.uncertainty >= op.uncertainty

    def test_vacuous_unchanged(self):
        op = Opinion.vacuous(0.5)
        result = op.maximize_uncertainty()
        assert abs(result.uncertainty - 1.0) < 1e-9

    def test_dogmatic_true(self):
        op = Opinion(1.0, 0.0, 0.0, 0.5, allow_dogmatic=True)
        result = op.maximize_uncertainty()
        # E = 1.0, u_max = min(1/0.5, 0/0.5) = min(2, 0) = 0
        # Can't increase uncertainty when E=1
        assert abs(result.uncertainty - 0.0) < 1e-9

    def test_dogmatic_false(self):
        op = Opinion(0.0, 1.0, 0.0, 0.5, allow_dogmatic=True)
        result = op.maximize_uncertainty()
        # E = 0.0, u_max = min(0/0.5, 1/0.5) = min(0, 2) = 0
        # Can't increase uncertainty when E=0
        assert abs(result.uncertainty - 0.0) < 1e-9

    def test_b_becomes_zero_case(self):
        """When E/a <= (1-E)/(1-a), b should become 0."""
        op = Opinion(0.2, 0.6, 0.2, 0.5)
        # E = 0.2 + 0.5*0.2 = 0.3
        # E/a = 0.6, (1-E)/(1-a) = 0.7/0.5 = 1.4
        # u_max = 0.6, b = 0, d = 1 - 0.6 = 0.4
        result = op.maximize_uncertainty()
        assert abs(result.b) < 1e-9
        assert abs(result.expectation() - op.expectation()) < 1e-9

    def test_d_becomes_zero_case(self):
        """When (1-E)/(1-a) < E/a, d should become 0."""
        op = Opinion(0.7, 0.1, 0.2, 0.5)
        # E = 0.7 + 0.5*0.2 = 0.8
        # E/a = 1.6, (1-E)/(1-a) = 0.2/0.5 = 0.4
        # u_max = 0.4, d = 0, b = 0.8 - 0.5*0.4 = 0.6
        result = op.maximize_uncertainty()
        assert abs(result.d) < 1e-9
        assert abs(result.expectation() - op.expectation()) < 1e-9

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_preserves_expectation(self, op):
        result = op.maximize_uncertainty()
        assert abs(result.expectation() - op.expectation()) < 1e-9

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_uncertainty_never_decreases(self, op):
        result = op.maximize_uncertainty()
        assert result.uncertainty >= op.uncertainty - 1e-9

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_idempotent(self, op):
        """Maximizing twice should give same result as once."""
        once = op.maximize_uncertainty()
        twice = once.maximize_uncertainty()
        assert abs(twice.b - once.b) < 1e-9
        assert abs(twice.d - once.d) < 1e-9
        assert abs(twice.u - once.u) < 1e-9


# ── Opinion Ordering (Jøsang 2001 Def 10, p.9) ─────────────────


class TestOpinionOrdering:
    """Jøsang 2001 Def 10, p.9: opinion ordering by E(x), then u, then a."""

    def test_higher_expectation_is_greater(self):
        high = Opinion(0.8, 0.1, 0.1, 0.5)  # E = 0.85
        low = Opinion(0.3, 0.5, 0.2, 0.5)   # E = 0.4
        assert high > low

    def test_same_expectation_less_uncertainty_is_greater(self):
        # E = 0.8 for both
        certain = Opinion(0.8, 0.2, 0.0, 0.5, allow_dogmatic=True)    # E = 0.8, u = 0
        uncertain = Opinion(0.3, 0.0, 0.7, 5/7)   # E = 0.3 + 5/7*0.7 = 0.8, u = 0.7
        assert certain > uncertain

    def test_same_e_same_u_less_a_is_greater(self):
        # Same E and u, different a
        # b + a*u = E, b + d + u = 1
        # Pick u=0.4, E=0.5
        # a=0.25: b = 0.5 - 0.25*0.4 = 0.4, d = 0.2
        # a=0.75: b = 0.5 - 0.75*0.4 = 0.2, d = 0.4
        low_a = Opinion(0.4, 0.2, 0.4, 0.25)   # E=0.5, u=0.4, a=0.25
        high_a = Opinion(0.2, 0.4, 0.4, 0.75)  # E=0.5, u=0.4, a=0.75
        assert low_a > high_a

    def test_equality(self):
        a = Opinion(0.3, 0.2, 0.5, 0.5)
        b = Opinion(0.3, 0.2, 0.5, 0.5)
        assert a <= b
        assert a >= b
        assert not (a < b)
        assert not (a > b)

    @pytest.mark.property
    @given(valid_opinions(), valid_opinions())
    @settings(deadline=None)
    def test_total_ordering(self, a, b):
        """Ordering is total: either a <= b or b <= a."""
        assert (a <= b) or (b <= a)

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_reflexive(self, a):
        assert a <= a


# ── Phase 8B: Additional Hypothesis property tests ───────────────


class TestConsensusAssociativityProperty:
    """Hypothesis property test for consensus associativity."""

    @pytest.mark.property
    @given(valid_opinions(), valid_opinions(), valid_opinions())
    @settings(deadline=None)
    def test_consensus_associative(self, a, b, c):
        """consensus(a, consensus(b, c)) ≈ consensus(consensus(a, b), c)"""
        # Skip when any pair would be two dogmatic opinions (u=0 for both)
        # because consensus is undefined for that case
        # (valid_opinions strategy already ensures u >= 0.01, but guard anyway)
        assume(a.u > 1e-6 and b.u > 1e-6 and c.u > 1e-6)
        ab_c = a.consensus_pair(b).consensus_pair(c)
        a_bc = a.consensus_pair(b.consensus_pair(c))
        assert abs(ab_c.b - a_bc.b) < 1e-6, f"b: {ab_c.b} vs {a_bc.b}"
        assert abs(ab_c.d - a_bc.d) < 1e-6, f"d: {ab_c.d} vs {a_bc.d}"
        assert abs(ab_c.u - a_bc.u) < 1e-6, f"u: {ab_c.u} vs {a_bc.u}"


class TestConjunctionPreservesSum:
    """Conjunction must preserve b + d + u = 1."""

    @pytest.mark.property
    @given(valid_opinions(), valid_opinions())
    @settings(deadline=None)
    def test_conjunction_preserves_sum(self, a, b):
        result = a & b
        assert abs(result.b + result.d + result.u - 1.0) < 1e-9


class TestDisjunctionPreservesSum:
    """Disjunction must preserve b + d + u = 1."""

    @pytest.mark.property
    @given(valid_opinions(), valid_opinions())
    @settings(deadline=None)
    def test_disjunction_preserves_sum(self, a, b):
        result = a | b
        assert abs(result.b + result.d + result.u - 1.0) < 1e-9


class TestNegationInvolutionProperty:
    """Negation involution: ~~op ≈ op for all valid opinions."""

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_negation_involution(self, op):
        """~~op ≈ op"""
        result = ~~op
        assert abs(result.b - op.b) < 1e-9
        assert abs(result.d - op.d) < 1e-9
        assert abs(result.u - op.u) < 1e-9
        assert abs(result.a - op.a) < 1e-9


class TestBetaEvidenceRoundTripProperty:
    """Round-trip Opinion -> BetaEvidence -> Opinion for non-dogmatic opinions."""

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_beta_evidence_round_trip(self, op):
        """op.to_beta_evidence().to_opinion() ≈ op (when u > 0)"""
        assume(op.u > 1e-6)  # dogmatic opinions can't round-trip
        evidence = op.to_beta_evidence()
        restored = evidence.to_opinion()
        assert abs(restored.b - op.b) < 1e-6
        assert abs(restored.d - op.d) < 1e-6
        assert abs(restored.u - op.u) < 1e-6


class TestDiscountVacuousTrustProperty:
    """Discounting with vacuous trust yields vacuous result."""

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_discount_vacuous_trust_yields_vacuous(self, op):
        """Discounting with vacuous trust yields vacuous result."""
        vacuous_trust = Opinion.vacuous(0.5)
        result = vacuous_trust.discount(op)
        assert abs(result.u - 1.0) < 1e-9


# ── WBF tests (van der Heijden 2018 Definition 4) ───────────────────


class TestWBF:
    """Weighted Belief Fusion — confidence-weighted evidence fusion.

    Grounding: van der Heijden et al. 2018, Definition 4 (p. 4-5).
    """

    def test_wbf_single_opinion(self):
        """WBF of a single opinion returns it unchanged."""
        op = Opinion(0.3, 0.2, 0.5, 0.6)
        result = Opinion.wbf(op)
        assert result == op

    def test_wbf_empty_raises(self):
        """WBF with no opinions raises ValueError."""
        with pytest.raises(ValueError):
            Opinion.wbf()

    def test_wbf_two_source_is_distinct_from_consensus_pair(self):
        """WBF and Jøsang consensus are distinct operators.

        van der Heijden et al. 2018 defines WBF separately from
        consensus & compromise fusion. For conflicting two-source
        inputs, WBF preserves more uncertainty than consensus_pair.
        """
        a = Opinion(0.0, 0.5, 0.5, 0.5)
        b = Opinion(0.5, 0.0, 0.5, 0.5)
        result_wbf = Opinion.wbf(a, b)
        result_cp = a.consensus_pair(b)

        assert result_wbf.b == pytest.approx(0.25, abs=1e-9)
        assert result_wbf.d == pytest.approx(0.25, abs=1e-9)
        assert result_wbf.u == pytest.approx(0.5, abs=1e-9)
        assert result_cp.u == pytest.approx(1.0 / 3.0, abs=1e-9)

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05), valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_wbf_commutativity(self, a, b):
        """WBF(a, b) == WBF(b, a)."""
        r1 = Opinion.wbf(a, b)
        r2 = Opinion.wbf(b, a)
        assert abs(r1.b - r2.b) < 1e-6
        assert abs(r1.d - r2.d) < 1e-6
        assert abs(r1.u - r2.u) < 1e-6
        assert abs(r1.a - r2.a) < 1e-6

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_wbf_vacuous_identity(self, a):
        """WBF(a, vacuous) == a — vacuous contributes nothing."""
        vacuous = Opinion.vacuous(a.a)
        result = Opinion.wbf(a, vacuous)
        assert abs(result.b - a.b) < 1e-6
        assert abs(result.d - a.d) < 1e-6
        assert abs(result.u - a.u) < 1e-6

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05), valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_wbf_uncertainty_stays_within_input_range_for_two_sources(self, a, b):
        """WBF uncertainty is not guaranteed to drop below min input u.

        The paper's Table I (p.7) has WBF u=0.292 while one source has
        u=0.200. The stronger invariant for two non-dogmatic inputs is
        that fused uncertainty remains inside the input uncertainty
        interval.
        """
        result = Opinion.wbf(a, b)
        assert min(a.u, b.u) - _TOL <= result.u <= max(a.u, b.u) + _TOL

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05), valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_wbf_sum_invariant(self, a, b):
        """b + d + u == 1 for fused result.

        van der Heijden 2018 Definition 4 (p. 4-5).
        """
        result = Opinion.wbf(a, b)
        assert abs(result.b + result.d + result.u - 1.0) < 1e-6

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05), valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_wbf_expectation_valid(self, a, b):
        """Fused expectation is in [0, 1].

        Note: E(wbf(a,b)) is NOT necessarily between E(a) and E(b) because
        fusion concentrates belief mass, reducing uncertainty and shifting
        the expectation. This is correct behavior per Jøsang Theorem 7.
        """
        result = Opinion.wbf(a, b)
        assert -_TOL <= result.expectation() <= 1.0 + _TOL

    def test_wbf_single_dogmatic_source_controls_result(self):
        """WBF Definition 4 Case 2: one dogmatic source controls the result.

        van der Heijden et al. 2018, Definition 4 (p. 5, Remark 1).
        """
        dogmatic = Opinion.dogmatic_true(0.5)
        normal = Opinion(0.3, 0.2, 0.5, 0.5)
        result = Opinion.wbf(dogmatic, normal)
        assert result == dogmatic

    @pytest.mark.property
    @given(a=st.floats(min_value=0.01, max_value=0.99))
    @settings(deadline=None)
    def test_wbf_all_vacuous_inputs_stay_vacuous(self, a):
        """WBF Definition 4 Case 3: all no-evidence sources fuse to no evidence.

        van der Heijden et al. 2018, Definition 4 (p. 5, Remark 3).
        """
        result = Opinion.wbf(Opinion.vacuous(a), Opinion.vacuous(a))
        assert result.b == pytest.approx(0.0, abs=1e-9)
        assert result.d == pytest.approx(0.0, abs=1e-9)
        assert result.u == pytest.approx(1.0, abs=1e-9)
        assert result.a == pytest.approx(a, abs=1e-9)

    def test_wbf_three_sources(self):
        """Concrete 3-source WBF: result is valid and bounded."""
        a = Opinion(0.5, 0.1, 0.4, 0.5)
        b = Opinion(0.3, 0.3, 0.4, 0.5)
        c = Opinion(0.6, 0.2, 0.2, 0.5)
        result = Opinion.wbf(a, b, c)
        assert abs(result.b + result.d + result.u - 1.0) < 1e-6
        assert result.u <= max(a.u, b.u, c.u) + _TOL
        assert result.b >= 0.0
        assert result.d >= 0.0
        assert result.u >= 0.0


# ── CCF tests (van der Heijden 2018 Definition 5) ───────────────────


class TestCCF:
    """Cumulative & Compromise Fusion — handles dogmatic sources.

    Grounding: van der Heijden et al. 2018, Definition 5 (p. 6-7).
    """

    def test_ccf_single_opinion(self):
        """CCF of a single opinion returns it unchanged."""
        op = Opinion(0.3, 0.2, 0.5, 0.6)
        result = Opinion.ccf(op)
        assert result == op

    def test_ccf_empty_raises(self):
        """CCF with no opinions raises ValueError."""
        with pytest.raises(ValueError):
            Opinion.ccf()

    def test_ccf_handles_dogmatic(self):
        """CCF can fuse two dogmatic opinions without raising."""
        dt = Opinion.dogmatic_true(0.5)
        df = Opinion.dogmatic_false(0.5)
        result = Opinion.ccf(dt, df)
        assert abs(result.b + result.d + result.u - 1.0) < 1e-6

    def test_ccf_all_dogmatic_same(self):
        """CCF of identical dogmatic opinions returns the same opinion."""
        dt = Opinion.dogmatic_true(0.5)
        result = Opinion.ccf(dt, dt)
        assert abs(result.b - 1.0) < 1e-6
        assert abs(result.d - 0.0) < 1e-6
        assert abs(result.u - 0.0) < 1e-6

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05), valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_ccf_sum_invariant(self, a, b):
        """b + d + u == 1 for CCF result.

        van der Heijden 2018 Definition 5 (p. 6-7).
        """
        result = Opinion.ccf(a, b)
        assert abs(result.b + result.d + result.u - 1.0) < 1e-6

    @pytest.mark.property
    @given(a=st.floats(min_value=0.01, max_value=0.99))
    @settings(deadline=None)
    def test_ccf_all_vacuous_inputs_stay_vacuous(self, a):
        """CCF of all no-evidence sources remains vacuous.

        van der Heijden et al. 2018, Definition 5 (p. 6-7).
        """
        result = Opinion.ccf(Opinion.vacuous(a), Opinion.vacuous(a), Opinion.vacuous(a))
        assert result.b == pytest.approx(0.0, abs=1e-9)
        assert result.d == pytest.approx(0.0, abs=1e-9)
        assert result.u == pytest.approx(1.0, abs=1e-9)
        assert result.a == pytest.approx(a, abs=1e-9)


# ── CCF Definition 5: old simplified tests deleted (review-2026-04-14) ──
#
# TestCCFDefinition5 was removed because its expected values encoded the
# pre-fix simplification, not the paper. TestCCFDefinition5Real (below)
# is the replacement: Table I verification plus four hand-derived worked
# examples that match Def 5 exactly.


class TestCCFProperties:
    """CCF properties from van der Heijden 2018."""

    @pytest.mark.property
    @given(a=valid_opinions(), b=valid_opinions())
    @settings(deadline=None)
    def test_ccf_commutativity(self, a, b):
        """CCF(a, b) == CCF(b, a)."""
        r1 = Opinion.ccf(a, b)
        r2 = Opinion.ccf(b, a)
        assert abs(r1.b - r2.b) < 1e-6
        assert abs(r1.d - r2.d) < 1e-6
        assert abs(r1.u - r2.u) < 1e-6

    @pytest.mark.property
    @given(a=valid_opinions(), b=valid_opinions(), c=valid_opinions())
    @settings(deadline=None)
    def test_ccf_commutativity_three(self, a, b, c):
        """CCF(a, b, c) is the same regardless of argument order."""
        result_abc = Opinion.ccf(a, b, c)
        result_bca = Opinion.ccf(b, c, a)
        result_cab = Opinion.ccf(c, a, b)
        assert abs(result_abc.b - result_bca.b) < 1e-6
        assert abs(result_abc.b - result_cab.b) < 1e-6
        assert abs(result_abc.d - result_bca.d) < 1e-6
        assert abs(result_abc.d - result_cab.d) < 1e-6

    @pytest.mark.property
    @given(a=valid_opinions())
    @settings(deadline=None)
    def test_ccf_single_identity(self, a):
        """CCF of a single opinion is itself."""
        result = Opinion.ccf(a)
        assert result == a

    @pytest.mark.property
    @given(a=valid_opinions(), b=valid_opinions())
    @settings(deadline=None)
    def test_ccf_sum_invariant_property(self, a, b):
        """Fused opinion satisfies b + d + u = 1."""
        result = Opinion.ccf(a, b)
        assert abs(result.b + result.d + result.u - 1.0) < 1e-6

    @pytest.mark.property
    @given(a=valid_opinions(), b=valid_opinions())
    @settings(deadline=None)
    def test_ccf_non_negative(self, a, b):
        """Fused opinion has non-negative components."""
        result = Opinion.ccf(a, b)
        assert result.b >= -1e-9
        assert result.d >= -1e-9
        assert result.u >= -1e-9

    # `test_ccf_nondogmatic_matches_wbf` was deleted — it encoded
    # propstore's old "delegate to WBF for non-dogmatic" heuristic,
    # which contradicts van der Heijden 2018 (CCF and WBF are distinct
    # operators even on non-dogmatic inputs; Table I shows different
    # columns).


# ── Fuse dispatcher tests ───────────────────────────────────────────


class TestFuse:
    """fuse() dispatcher: auto selects WBF or CCF."""

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05), valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_fuse_auto_selects_wbf(self, a, b):
        """For non-dogmatic inputs, fuse(auto) == wbf()."""
        result_fuse = Opinion.fuse(a, b, method="auto")
        result_wbf = Opinion.wbf(a, b)
        assert abs(result_fuse.b - result_wbf.b) < 1e-9
        assert abs(result_fuse.d - result_wbf.d) < 1e-9
        assert abs(result_fuse.u - result_wbf.u) < 1e-9

    def test_fuse_auto_handles_dogmatic_wbf_case(self):
        """For dogmatic inputs, fuse(auto) uses WBF Definition 4 Case 2.

        van der Heijden et al. 2018, Definition 4 (p. 5, Remark 1).
        """
        dt = Opinion.dogmatic_true(0.5)
        df = Opinion.dogmatic_false(0.5)
        result = Opinion.fuse(dt, df, method="auto")
        assert abs(result.b + result.d + result.u - 1.0) < 1e-6

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05), valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_fuse_explicit_wbf(self, a, b):
        """fuse(method='wbf') == wbf()."""
        r1 = Opinion.fuse(a, b, method="wbf")
        r2 = Opinion.wbf(a, b)
        assert abs(r1.b - r2.b) < 1e-9
        assert abs(r1.d - r2.d) < 1e-9

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05), valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_fuse_explicit_ccf(self, a, b):
        """fuse(method='ccf') == ccf()."""
        r1 = Opinion.fuse(a, b, method="ccf")
        r2 = Opinion.ccf(a, b)
        assert abs(r1.b - r2.b) < 1e-9
        assert abs(r1.d - r2.d) < 1e-9


# ── Additional fusion property tests (audit-2026-03-28 gaps) ──────


class TestWBFAdditionalProperties:
    """WBF properties identified as missing in audit-2026-03-28."""

    @pytest.mark.property
    @given(
        valid_opinions(min_uncertainty=0.05),
        valid_opinions(min_uncertainty=0.05),
        valid_opinions(min_uncertainty=0.05),
    )
    @settings(deadline=None)
    def test_wbf_three_source_commutativity(self, a, b, c):
        """WBF(a, b, c) is the same regardless of argument order.

        Jøsang Theorem 7 generalizes to N sources — the multi-source
        formula uses symmetric sums so order should not matter.
        """
        r_abc = Opinion.wbf(a, b, c)
        r_bca = Opinion.wbf(b, c, a)
        r_cab = Opinion.wbf(c, a, b)
        assert abs(r_abc.b - r_bca.b) < 1e-6, f"b: {r_abc.b} vs {r_bca.b}"
        assert abs(r_abc.b - r_cab.b) < 1e-6, f"b: {r_abc.b} vs {r_cab.b}"
        assert abs(r_abc.d - r_bca.d) < 1e-6, f"d: {r_abc.d} vs {r_bca.d}"
        assert abs(r_abc.d - r_cab.d) < 1e-6, f"d: {r_abc.d} vs {r_cab.d}"
        assert abs(r_abc.u - r_bca.u) < 1e-6, f"u: {r_abc.u} vs {r_bca.u}"
        assert abs(r_abc.u - r_cab.u) < 1e-6, f"u: {r_abc.u} vs {r_cab.u}"

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_wbf_self_fusion_preserves_belief_disbelief_ratio(self, a):
        """When fusing identical opinions, the b:d ratio is preserved.

        WBF self-fusion concentrates belief mass (reduces u) while
        maintaining the relative proportion of b and d. Expectation
        is NOT preserved in general because E = b + a*u and u shrinks.
        """
        assume(a.b + a.d > 1e-6)  # skip near-vacuous
        result = Opinion.wbf(a, a)
        # b/d ratio should be preserved
        if a.b > 1e-9 and a.d > 1e-9:
            orig_ratio = a.b / a.d
            result_ratio = result.b / result.d
            assert abs(orig_ratio - result_ratio) < 1e-4, (
                f"b:d ratio changed: {orig_ratio} -> {result_ratio}"
            )

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_wbf_self_fusion_reduces_uncertainty(self, a):
        """Fusing an opinion with itself concentrates belief: u decreases."""
        result = Opinion.wbf(a, a)
        assert result.u <= a.u + _TOL

    @pytest.mark.property
    @given(valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_wbf_self_fusion_preserves_base_rate_without_clamp(self, a):
        """Definition 4 computes base rate directly from confidence weights.

        WBF does not use the CCF base-rate clamp. Self-fusion therefore
        preserves the input base rate up to floating-point arithmetic.
        """
        result = Opinion.wbf(a, a)
        assert result.a == pytest.approx(a.a, abs=1e-12)


class TestCCFAdditionalProperties:
    """CCF properties identified as missing in audit-2026-03-28.

    ``test_ccf_associativity`` was deleted: CCF is NOT associative in
    general (van der Heijden 2018, §I, explicitly notes this). The
    direct N-source call form ``Opinion.ccf(a, b, c)`` is order-invariant
    (see ``TestCCFDefinition5Real.test_commutativity_three``), but
    pairwise folding ``ccf(ccf(a, b), c)`` is not equivalent.
    """

    @pytest.mark.property
    @given(a=valid_opinions(min_uncertainty=0.05))
    @settings(deadline=None)
    def test_ccf_self_fusion_is_idempotent(self, a):
        """``ccf(a, a) == a`` for every opinion.

        Self-fusion triggers the ``b^comp_sum ≈ 0`` edge case in Def 5
        (all residuals are zero when sources are identical). The kernel
        handles it by putting the residual missing mass directly into
        ``u``, which makes self-fusion idempotent.
        """
        assume(a.b + a.d > 1e-6)  # skip near-vacuous
        result = Opinion.ccf(a, a)
        assert result.b == pytest.approx(a.b, abs=1e-9)
        assert result.d == pytest.approx(a.d, abs=1e-9)
        assert result.u == pytest.approx(a.u, abs=1e-9)

    @pytest.mark.property
    @given(a=valid_opinions())
    @settings(deadline=None)
    def test_ccf_base_rate_clamping_observable(self, a):
        """Document that CCF clamps base rates to [0.01, 0.99].

        Known deviation from literature (audit-2026-03-28 bug #2).
        """
        result = Opinion.ccf(a, a)
        assert result.a >= 0.01
        assert result.a <= 0.99


class TestConjunctionDisjunctionDeMorgan:
    """Conjunction/disjunction should satisfy De Morgan's law via negation.

    Per Jøsang 2001: ~(A & B) should equal (~A | ~B) for independent opinions.
    This is a consequence of the dual construction of conjunction/disjunction.
    """

    @pytest.mark.property
    @given(valid_opinions(), valid_opinions())
    @settings(deadline=None)
    def test_de_morgan_conjunction(self, a, b):
        """~(a & b) ≈ (~a | ~b)."""
        lhs = ~(a & b)
        rhs = (~a) | (~b)
        assert abs(lhs.b - rhs.b) < 1e-9, f"b: {lhs.b} vs {rhs.b}"
        assert abs(lhs.d - rhs.d) < 1e-9, f"d: {lhs.d} vs {rhs.d}"
        assert abs(lhs.u - rhs.u) < 1e-9, f"u: {lhs.u} vs {rhs.u}"

    @pytest.mark.property
    @given(valid_opinions(), valid_opinions())
    @settings(deadline=None)
    def test_de_morgan_disjunction(self, a, b):
        """~(a | b) ≈ (~a & ~b)."""
        lhs = ~(a | b)
        rhs = (~a) & (~b)
        assert abs(lhs.b - rhs.b) < 1e-9, f"b: {lhs.b} vs {rhs.b}"
        assert abs(lhs.d - rhs.d) < 1e-9, f"d: {lhs.d} vs {rhs.d}"
        assert abs(lhs.u - rhs.u) < 1e-9, f"u: {lhs.u} vs {rhs.u}"


class TestDiscountProperties:
    """Discount operator properties from Jøsang 2001 Def 14."""

    @pytest.mark.property
    @given(valid_opinions(), valid_opinions())
    @settings(deadline=None)
    def test_discount_preserves_base_rate(self, trust, source):
        """Discounting preserves the source's base rate (Jøsang Def 14)."""
        result = trust.discount(source)
        assert abs(result.a - source.a) < 1e-9

    @pytest.mark.property
    @given(valid_opinions(), valid_opinions())
    @settings(deadline=None)
    def test_discount_sum_invariant(self, trust, source):
        """Discounted opinion satisfies b + d + u = 1."""
        result = trust.discount(source)
        assert abs(result.b + result.d + result.u - 1.0) < 1e-9

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_discount_full_trust_is_identity(self, source):
        """Discounting with dogmatic trust returns the source unchanged.

        Per Jøsang Def 14: when trust = (1, 0, 0, a), the discount
        formula yields b=1*source.b, d=1*source.d, u=0+0+1*source.u = source.u.
        """
        full_trust = Opinion.dogmatic_true(0.5)
        result = full_trust.discount(source)
        assert abs(result.b - source.b) < 1e-9
        assert abs(result.d - source.d) < 1e-9
        assert abs(result.u - source.u) < 1e-9


class TestCCFDefinition5Real:
    """CCF matches van der Heijden 2018 Definition 5 literally.

    Ground-truth regression tests against Table I of the paper plus four
    hand-derived worked examples. The old propstore implementation was a
    simplification that produced fractional-belief "fake consensus" for
    dogmatic disagreement (e.g. ``ccf(dt, df) → (0.5, 0.5, 0)``). Per Def
    5, disagreement gets converted into uncertainty via the ``b^comp(X)``
    disagreement term, so the correct answer is vacuous.
    """

    def test_table_I_three_source(self):
        """Paper Table I (p. 7): three sources, a=0.5 shared, CCF column.

        Inputs:
          A1: b=0.10, d=0.30, u=0.60
          A2: b=0.40, d=0.20, u=0.40
          A3: b=0.70, d=0.10, u=0.20

        Paper-published CCF output (rounded):
          b=0.629, d=0.182, u=0.189
        """
        a1 = Opinion(0.10, 0.30, 0.60, 0.5)
        a2 = Opinion(0.40, 0.20, 0.40, 0.5)
        a3 = Opinion(0.70, 0.10, 0.20, 0.5)
        r = Opinion.ccf(a1, a2, a3)
        assert r.b == pytest.approx(0.629, abs=5e-4)
        assert r.d == pytest.approx(0.182, abs=5e-4)
        assert r.u == pytest.approx(0.189, abs=5e-4)
        assert r.a == pytest.approx(0.5, abs=1e-9)

    def test_dogmatic_total_disagreement_is_vacuous(self):
        """CCF(dogmatic_true, dogmatic_false) → vacuous.

        Both sources have b^cons = 0; all residual mass flows into
        b^comp(X) = 1 (the disagreement term), which then becomes
        uncertainty after normalization. This is the correct Def 5
        semantic and the entire point of the reviewer's Issue 2:
        disagreement converts to ignorance, not fractional belief.
        """
        a = Opinion.dogmatic_true(0.5)
        b = Opinion.dogmatic_false(0.5)
        r = Opinion.ccf(a, b)
        assert r.b == pytest.approx(0.0, abs=1e-9)
        assert r.d == pytest.approx(0.0, abs=1e-9)
        assert r.u == pytest.approx(1.0, abs=1e-9)

    def test_dogmatic_majority_still_vacuous(self):
        """Two-true + one-false also goes vacuous under Def 5.

        CCF is NOT a voting operator — it doesn't preserve
        majority-count information. The paper (§III.C / Remark 4):
        "generates vague belief from conflicting belief on singleton
        values." For binomial, "vague belief" IS uncertainty.
        Callers who want voting semantics should use WBF or plain
        averaging, not CCF.
        """
        dt = Opinion.dogmatic_true(0.5)
        df = Opinion.dogmatic_false(0.5)
        r = Opinion.ccf(dt, dt, df)
        assert r.b == pytest.approx(0.0, abs=1e-9)
        assert r.d == pytest.approx(0.0, abs=1e-9)
        assert r.u == pytest.approx(1.0, abs=1e-9)

    def test_dogmatic_agreement_preserves_dogmatic(self):
        """CCF(dt, dt) = dt — b^comp_sum = 0 edge case.

        When all residuals are zero (full agreement on all singletons),
        there is no compromise mass to distribute. The paper's Def 5
        is not explicit about this degenerate case (Remark 4 assumes
        ``b^comp_sum > 0``). The kernel handles it by putting residual
        missing mass directly into u, which makes self-fusion
        idempotent for every input.
        """
        dt = Opinion.dogmatic_true(0.5)
        r = Opinion.ccf(dt, dt)
        assert r.b == pytest.approx(1.0, abs=1e-9)
        assert r.d == pytest.approx(0.0, abs=1e-9)
        assert r.u == pytest.approx(0.0, abs=1e-9)

    def test_dogmatic_plus_uncertain(self):
        """CCF((1,0,0), (0.4,0.2,0.4)) = (0.8, 0, 0.2).

        Hand-derived:
          b^cons(x)=min(1,0.4)=0.4, b^cons(¬x)=0, b^cons_sum=0.4
          r_A=(0.6, 0), r_B=(0, 0.2); u^pre=0
          b^comp(x)  = 0.6·0.4 + 0·0 + 0.6·0   = 0.24
          b^comp(¬x) = 0·0.4   + 0.2·0 + 0·0.2 = 0
          b^comp(X)  = (0.6)(0.2) − 0·0 − 0·0.2 = 0.12
          b^comp_sum = 0.36
          η = (1 − 0.4 − 0) / 0.36 = 5/3
          u^fused    = 0 + (5/3)·0.12 = 0.2
          b^fused(x) = 0.4 + (5/3)·0.24 = 0.8
          b^fused(¬x)= 0
        """
        a = Opinion.dogmatic_true(0.5)
        b = Opinion(0.4, 0.2, 0.4, 0.5)
        r = Opinion.ccf(a, b)
        assert r.b == pytest.approx(0.8, abs=1e-9)
        assert r.d == pytest.approx(0.0, abs=1e-9)
        assert r.u == pytest.approx(0.2, abs=1e-9)

    def test_nondogmatic_partial_disagreement(self):
        """CCF((0.6,0.2,0.2), (0.2,0.6,0.2)) = (0.34, 0.34, 0.32).

        Disagreement boosts u from 0.2 → 0.32 by converting the
        cross-residual mass into uncertainty. This is the non-dogmatic
        analogue of the dogmatic vacuous-disagreement case.

        Hand-derived:
          b^cons(x)=0.2, b^cons(¬x)=0.2, b^cons_sum=0.4
          r_A=(0.4, 0), r_B=(0, 0.4); u^pre=0.04
          b^comp(x) = 0.4·0.2 + 0·0.2 + 0.4·0 = 0.08
          b^comp(¬x)= 0·0.2 + 0.4·0.2 + 0·0.4 = 0.08
          b^comp(X) = (0.4)(0.4) − 0·0 − 0·0.4 = 0.16
          b^comp_sum= 0.32
          η = (1 − 0.4 − 0.04) / 0.32 = 1.75
          u^fused   = 0.04 + 1.75·0.16 = 0.32
          b^fused(x)= 0.2 + 1.75·0.08 = 0.34
        """
        a = Opinion(0.6, 0.2, 0.2, 0.5)
        b = Opinion(0.2, 0.6, 0.2, 0.5)
        r = Opinion.ccf(a, b)
        assert r.b == pytest.approx(0.34, abs=1e-9)
        assert r.d == pytest.approx(0.34, abs=1e-9)
        assert r.u == pytest.approx(0.32, abs=1e-9)

    def test_self_fusion_is_idempotent(self):
        """CCF(a, a) == a for any opinion — b^comp_sum = 0 handling."""
        for op in [
            Opinion(0.3, 0.2, 0.5, 0.5),
            Opinion(0.1, 0.8, 0.1, 0.3),
            Opinion.vacuous(0.5),
            Opinion.dogmatic_true(0.7),
            Opinion.dogmatic_false(0.2),
        ]:
            r = Opinion.ccf(op, op)
            assert r.b == pytest.approx(op.b, abs=1e-9), f"b differs for {op}"
            assert r.d == pytest.approx(op.d, abs=1e-9), f"d differs for {op}"
            assert r.u == pytest.approx(op.u, abs=1e-9), f"u differs for {op}"

    def test_vacuous_identity(self):
        """CCF(a, vacuous) preserves a's b,d,u up to u^pre scaling.

        Unlike WBF (which treats vacuous as identity), CCF's
        multiplication by u^pre = u_a · 1 = u_a leaves the u component
        unchanged for the a=vacuous case. For a non-vacuous, CCF
        produces a valid opinion but not strictly identical to a —
        this test only checks that vacuous fusion doesn't corrupt
        sum invariants or produce nonsense.
        """
        a = Opinion(0.3, 0.3, 0.4, 0.5)
        vac = Opinion.vacuous(0.5)
        r = Opinion.ccf(a, vac)
        assert abs(r.b + r.d + r.u - 1.0) < 1e-9
        # Residuals from a flow through; vacuous contributes nothing
        # meaningful beyond its u=1.

    @pytest.mark.property
    @given(a=valid_opinions(), b=valid_opinions())
    @settings(deadline=None)
    def test_sum_invariant_property(self, a, b):
        """CCF always produces valid opinions (Remark 4)."""
        r = Opinion.ccf(a, b)
        assert abs(r.b + r.d + r.u - 1.0) < 1e-9
        assert r.b >= -1e-9
        assert r.d >= -1e-9
        assert r.u >= -1e-9

    @pytest.mark.property
    @given(a=valid_opinions(), b=valid_opinions())
    @settings(deadline=None)
    def test_commutativity_binary(self, a, b):
        """CCF(a, b) == CCF(b, a) — formulas are symmetric in actors."""
        r1 = Opinion.ccf(a, b)
        r2 = Opinion.ccf(b, a)
        assert abs(r1.b - r2.b) < 1e-9
        assert abs(r1.d - r2.d) < 1e-9
        assert abs(r1.u - r2.u) < 1e-9

    @pytest.mark.property
    @given(a=valid_opinions(), b=valid_opinions(), c=valid_opinions())
    @settings(deadline=None)
    def test_commutativity_three(self, a, b, c):
        """CCF is multi-source-symmetric (order-independent across actors).

        Distinct from associativity — CCF is NOT associative in general
        (the paper's Section I notes this), so ``ccf(ccf(a, b), c)`` is
        not the same as ``ccf(a, ccf(b, c))``. But a direct N-source
        call is invariant under permutation of its arguments because
        Def 5's sums and products are symmetric in A.
        """
        r_abc = Opinion.ccf(a, b, c)
        r_bca = Opinion.ccf(b, c, a)
        r_cab = Opinion.ccf(c, a, b)
        assert abs(r_abc.b - r_bca.b) < 1e-9
        assert abs(r_abc.b - r_cab.b) < 1e-9
        assert abs(r_abc.d - r_bca.d) < 1e-9
        assert abs(r_abc.d - r_cab.d) < 1e-9
        assert abs(r_abc.u - r_bca.u) < 1e-9
        assert abs(r_abc.u - r_cab.u) < 1e-9


# ── Review fixes (review-2026-04-14) ───────────────────────────────


class TestHashEqConsistency:
    """`__eq__` and `__hash__` must agree on the same quantization grid.

    Regression test for the review-2026-04-14 hash contract violation:
    the old implementation used tolerance-based equality but 8-decimal
    rounding for hashing, so two opinions could be ``==`` yet produce
    different hashes — a silent dict/set corruption bug.
    """

    def test_straddle_rounding_boundary(self):
        """Values on opposite sides of an 8-decimal rounding boundary
        but within ``_TOL`` of each other must hash identically.

        b1 = 0.5 + 4.6e-9 rounds *down* to 0.5
        b2 = 0.5 + 5.4e-9 rounds *up*   to 0.50000001
        |b1 - b2| = 8e-10 < _TOL, so __eq__ returns True.
        The old hash(round(·, 8)) impl produced different hashes here.
        """
        b1 = 0.5 + 4.6e-9
        b2 = 0.5 + 5.4e-9
        d = 0.3
        o1 = Opinion(b1, d, 1.0 - b1 - d, 0.5)
        o2 = Opinion(b2, d, 1.0 - b2 - d, 0.5)
        # Precondition: they must actually be __eq__ (tol-based).
        assert o1 == o2
        # The bug: hashes differed.  Hash contract says a == b ⇒ hash(a) == hash(b).
        assert hash(o1) == hash(o2)

    def test_set_membership_respects_equality(self):
        """If ``o1 == o2``, a set containing o1 must also contain o2."""
        b1 = 0.5 + 4.6e-9
        b2 = 0.5 + 5.4e-9
        d = 0.3
        o1 = Opinion(b1, d, 1.0 - b1 - d, 0.5)
        o2 = Opinion(b2, d, 1.0 - b2 - d, 0.5)
        assert o1 == o2
        s = {o1}
        assert o2 in s, "Hash contract violation: o1 == o2 but o2 not in {o1}"

    def test_dict_lookup_respects_equality(self):
        """Dict keyed on o1 must resolve lookups by o2 when o1 == o2."""
        b1 = 0.5 + 4.6e-9
        b2 = 0.5 + 5.4e-9
        d = 0.3
        o1 = Opinion(b1, d, 1.0 - b1 - d, 0.5)
        o2 = Opinion(b2, d, 1.0 - b2 - d, 0.5)
        d_map = {o1: "value"}
        assert d_map[o2] == "value"

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_identical_opinions_hash_equal(self, o):
        """Trivial reflexivity: equal opinions hash equal."""
        other = Opinion(o.b, o.d, o.u, o.a)
        assert o == other
        assert hash(o) == hash(other)

    @pytest.mark.property
    @given(valid_opinions())
    @settings(deadline=None)
    def test_tol_perturbation_preserves_hash(self, o):
        """Perturbing any component by less than _TOL/4 must preserve
        both equality and hash (property covers all quantization grid
        points, not just the straddle example).
        """
        eps = _TOL / 4
        # Perturb b upward and d downward to preserve sum constraint.
        b2 = o.b + eps
        d2 = o.d - eps
        if b2 < 0 or b2 > 1 or d2 < 0 or d2 > 1:
            return
        other = Opinion(b2, d2, o.u, o.a)
        assume(o == other)
        assert hash(o) == hash(other), (
            f"hash mismatch on tol-equal opinions: "
            f"{o} vs {other}"
        )


class TestOpinionBool:
    """``__bool__`` must raise ``TypeError`` to prevent the ``and``/``or`` trap.

    Python's ``and``/``or`` keywords short-circuit on truthiness and do
    NOT dispatch to ``__and__``/``__or__``.  A user writing
    ``if op1 and op2`` would silently get a truthy Opinion back rather
    than the subjective-logic conjunction.  Raising ``TypeError`` from
    ``__bool__`` converts that landmine into a runtime error at the
    call site.
    """

    def test_bool_on_vacuous_raises(self):
        with pytest.raises(TypeError, match="not truthy|conjunction|expectation"):
            bool(Opinion.vacuous(0.5))

    def test_bool_on_dogmatic_true_raises(self):
        with pytest.raises(TypeError):
            bool(Opinion.dogmatic_true(0.5))

    def test_bool_on_dogmatic_false_raises(self):
        with pytest.raises(TypeError):
            bool(Opinion.dogmatic_false(0.5))

    def test_bool_on_arbitrary_raises(self):
        with pytest.raises(TypeError):
            bool(Opinion(0.3, 0.2, 0.5, 0.6))

    def test_if_opinion_trap_raises(self):
        """The classic trap: ``if opinion:`` must error loudly."""
        op = Opinion(0.3, 0.2, 0.5, 0.6)
        with pytest.raises(TypeError):
            if op:
                pass

    def test_and_keyword_trap_raises(self):
        """``opinion and opinion`` forces truthiness evaluation."""
        op = Opinion(0.3, 0.2, 0.5, 0.6)
        with pytest.raises(TypeError):
            op and op

    def test_is_none_check_still_works(self):
        """Explicit ``is None`` / ``is not None`` must NOT trigger bool."""
        op = Opinion(0.3, 0.2, 0.5, 0.6)
        assert op is not None
        assert not (op is None)  # `not (X is None)` does not call __bool__ on X

    def test_conditional_expression_with_is_not_none(self):
        """``x if op is not None else y`` is the correct idiom."""
        op = Opinion(0.3, 0.2, 0.5, 0.6)
        val = op.b if op is not None else 0.0
        assert val == 0.3


class TestExplicitConjunctionDisjunction:
    """Explicit method names for subjective-logic conjunction/disjunction.

    Provides keyword-safe alternatives to ``&``/``|`` so callers aren't
    forced to rely on operator overloads (review-2026-04-14). The
    operators remain as aliases with identical semantics.
    """

    def test_conjunction_equals_and_operator(self):
        a = Opinion(0.6, 0.2, 0.2, 0.4)
        b = Opinion(0.3, 0.5, 0.2, 0.7)
        assert a.conjunction(b) == a & b

    def test_disjunction_equals_or_operator(self):
        a = Opinion(0.6, 0.2, 0.2, 0.4)
        b = Opinion(0.3, 0.5, 0.2, 0.7)
        assert a.disjunction(b) == a | b

    def test_conjunction_commutative(self):
        a = Opinion(0.6, 0.2, 0.2, 0.4)
        b = Opinion(0.3, 0.5, 0.2, 0.7)
        assert a.conjunction(b) == b.conjunction(a)

    def test_disjunction_commutative(self):
        a = Opinion(0.6, 0.2, 0.2, 0.4)
        b = Opinion(0.3, 0.5, 0.2, 0.7)
        assert a.disjunction(b) == b.disjunction(a)

    def test_conjunction_with_vacuous_reduces_belief(self):
        """ω ∧ vacuous should still satisfy b + d + u = 1."""
        op = Opinion(0.6, 0.2, 0.2, 0.4)
        result = op.conjunction(Opinion.vacuous(0.5))
        assert abs(result.b + result.d + result.u - 1.0) < 1e-9


class TestConsensusPairNearVacuous:
    """``consensus_pair`` must stay numerically sane near the vacuous
    boundary (review-2026-04-14 Issue 3).

    The base-rate denominator ``u_a + u_b - 2*u_a*u_b = u_a*v_b + u_b*v_a``
    (where v = 1-u) shrinks as both sources approach u=1. The old guard
    only caught ``|denom_a| < _TOL``, which is a strict-equality check.
    The contract these tests enforce:

    1. Two exactly-vacuous opinions with distinct base rates fall back
       cleanly (no NaN/inf, no ValueError).
    2. Near-vacuous opinions produce a base rate inside the convex hull
       of the inputs.
    3. The fused opinion is always a valid Opinion (construction
       succeeds).
    """

    def test_both_exactly_vacuous_with_distinct_base_rates(self):
        """Fallback path: two vacuous opinions fuse to a mid-range base rate."""
        o1 = Opinion.vacuous(0.2)
        o2 = Opinion.vacuous(0.8)
        r = o1.consensus_pair(o2)
        assert 0.0 < r.a < 1.0
        assert 0.2 <= r.a <= 0.8
        # b, d must remain zero (both inputs are b=d=0)
        assert r.b == 0.0
        assert r.d == 0.0
        # u must be 1.0 — consensus of two vacuous is vacuous
        assert r.u == pytest.approx(1.0, abs=1e-9)

    def test_near_vacuous_same_u_distinct_base_rates(self):
        """Equal near-vacuous uncertainties → roughly midpoint base rate."""
        u = 1 - 1e-8
        o1 = Opinion(0.0, 1 - u, u, 0.2)
        o2 = Opinion(0.0, 1 - u, u, 0.8)
        r = o1.consensus_pair(o2)
        # With equal confidence weights, the fused base rate is the mean.
        assert r.a == pytest.approx(0.5, abs=1e-6)
        assert 0.2 <= r.a <= 0.8

    def test_near_vacuous_asymmetric_u(self):
        """Asymmetric near-vacuous: more confident source dominates."""
        # o1 has smaller u → more certain → should pull a toward 0.2
        o1 = Opinion(0.0, 1 - 0.999, 0.999, 0.2)
        o2 = Opinion(0.0, 1 - 0.99999, 0.99999, 0.8)
        r = o1.consensus_pair(o2)
        # o1 is ~100x more confident, so fused a should be very close to 0.2
        assert 0.2 <= r.a <= 0.8
        assert r.a < 0.5, f"expected a < 0.5 (o1 dominates), got {r.a}"

    @pytest.mark.property
    @given(
        a1=st.floats(min_value=0.01, max_value=0.99),
        a2=st.floats(min_value=0.01, max_value=0.99),
        v1=st.floats(min_value=1e-12, max_value=1e-4),
        v2=st.floats(min_value=1e-12, max_value=1e-4),
    )
    @settings(deadline=None, max_examples=500)
    def test_near_vacuous_property_base_rate_in_hull(self, a1, a2, v1, v2):
        """For any near-vacuous pair, fused base rate ∈ [min(a1,a2), max(a1,a2)]."""
        u1 = 1.0 - v1
        u2 = 1.0 - v2
        o1 = Opinion(0.0, v1, u1, a1)
        o2 = Opinion(0.0, v2, u2, a2)
        r = o1.consensus_pair(o2)
        lo = min(a1, a2)
        hi = max(a1, a2)
        # Allow tiny float drift via 1e-9 margin.
        assert lo - 1e-9 <= r.a <= hi + 1e-9, (
            f"a={r.a} not in [{lo}, {hi}] for u=({u1}, {u2}), a=({a1}, {a2})"
        )
        # Fused opinion must be well-formed (constructor already ran).
        assert 0.0 < r.a < 1.0
        assert abs(r.b + r.d + r.u - 1.0) < 1e-9

    def test_exactly_vacuous_one_side(self):
        """One vacuous + one non-vacuous: base rate should come from non-vacuous."""
        vac = Opinion.vacuous(0.1)
        op = Opinion(0.3, 0.3, 0.4, 0.9)
        r = vac.consensus_pair(op)
        # Vacuous side has zero confidence, so non-vacuous dominates.
        # Allow 1e-9 float drift around the convex hull.
        assert 0.1 - 1e-9 <= r.a <= 0.9 + 1e-9


# TestCCFAveragePreservesUncertainty was removed. It was added in my
# first review-fix pass and it locked in the claim ``fused.u ==
# mean(u_i)``, which is the PROPSTORE SIMPLIFICATION of Def 5, not Def
# 5 itself. The real Def 5 (see TestCCFDefinition5Real) produces a
# very different ``u`` because the ``b^comp(X)`` disagreement term
# converts residual disagreement mass into uncertainty via η, which
# is the opposite of arithmetic averaging. Keeping the old tests
# would prevent any correct Def 5 implementation from shipping.


class TestOpinionRepr:
    """The dataclass-generated repr must survive the custom ``__eq__`` /
    ``__hash__`` overrides (review-2026-04-14 sanity check).
    """

    def test_repr_contains_all_fields(self):
        op = Opinion(0.3, 0.2, 0.5, 0.6)
        r = repr(op)
        assert "b=" in r
        assert "d=" in r
        assert "u=" in r
        assert "a=" in r

    def test_repr_is_eval_able_shape(self):
        """The auto-generated repr has the ``Opinion(...)`` shape."""
        op = Opinion(0.3, 0.2, 0.5, 0.6)
        assert repr(op).startswith("Opinion(")
        assert repr(op).endswith(")")
