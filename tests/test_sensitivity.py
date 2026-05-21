"""Test suite for ``doxa.sensitivity`` — opinion / argument-graph sensitivity.

``doxa.sensitivity`` is the perturbation-sensitivity layer added for the
dialectical-checkers principled-lookahead build (lookahead design L-D5). It
answers "how much does an output opinion depend on one of its inputs":

- ``opinion_sensitivity`` — finite-difference sensitivity of a ``wbf`` fusion's
  expectation to a perturbation in one source opinion's uncertainty ``u``.
  Lifted near-as-is from propstore's ``fragility_scoring.opinion_sensitivity``,
  cleaned of propstore-isms (it now depends only on ``doxa.opinion``).
- ``graph_perturbation_sensitivity`` — fresh: perturb a ``BipolarOpinionGraph``
  (drop a witness/leaf node, or remove one edge), re-``evaluate``, and return
  the resolved-opinion delta at a target argument.

Markers: ``unit`` for focused contract tests, ``property`` for hypothesis-based
invariant tests — per doxa's ``pyproject.toml`` marker registry.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from doxa import BipolarOpinionGraph, Opinion
from doxa.sensitivity import graph_perturbation_sensitivity, opinion_sensitivity


# ── Strategies ─────────────────────────────────────────────────────


@st.composite
def nondogmatic_opinions(draw, min_uncertainty=0.05):
    """Generate valid non-dogmatic opinions (u >= min_uncertainty).

    The ``min_uncertainty`` floor keeps the finite-difference perturbation
    well-conditioned — ``opinion_sensitivity`` declines (returns ``None``)
    when any source is dogmatic.
    """
    u = draw(st.floats(min_value=min_uncertainty, max_value=1.0 - 0.05))
    remaining = 1.0 - u
    b = draw(st.floats(min_value=0.0, max_value=remaining))
    d = max(0.0, remaining - b)
    a = draw(st.floats(min_value=0.1, max_value=0.9))
    assume(abs(b + d + u - 1.0) < 1e-9)
    assume(b >= 0.0 and d >= 0.0 and u >= 0.0)
    return Opinion(b, d, u, a)


# ── opinion_sensitivity — unit tests ───────────────────────────────


class TestOpinionSensitivityContract:
    """Focused contract tests for ``opinion_sensitivity``."""

    @pytest.mark.unit
    def test_fewer_than_two_opinions_returns_none(self):
        """A fusion of one source has no sensitivity to define — None."""
        single = [Opinion(0.4, 0.2, 0.4, 0.5)]
        assert opinion_sensitivity(single, 0) is None
        assert opinion_sensitivity([], 0) is None

    @pytest.mark.unit
    def test_dogmatic_source_returns_none(self):
        """A dogmatic source cannot be u-perturbed — the call declines."""
        opinions = [
            Opinion(0.4, 0.2, 0.4, 0.5),
            Opinion.dogmatic_true(0.5),
        ]
        assert opinion_sensitivity(opinions, 0) is None

    @pytest.mark.unit
    def test_returns_finite_nonnegative_float(self):
        """A well-conditioned fusion yields a finite, non-negative float."""
        opinions = [
            Opinion(0.10, 0.30, 0.60, 0.5),
            Opinion(0.40, 0.20, 0.40, 0.5),
            Opinion(0.70, 0.10, 0.20, 0.5),
        ]
        s = opinion_sensitivity(opinions, 1)
        assert s is not None
        assert math.isfinite(s)
        assert s >= 0.0

    @pytest.mark.unit
    def test_deterministic(self):
        """The same inputs always produce the same sensitivity."""
        opinions = [
            Opinion(0.10, 0.30, 0.60, 0.5),
            Opinion(0.40, 0.20, 0.40, 0.5),
            Opinion(0.70, 0.10, 0.20, 0.5),
        ]
        first = opinion_sensitivity(opinions, 2)
        second = opinion_sensitivity(opinions, 2)
        assert first == second

    @pytest.mark.unit
    def test_central_difference_value(self):
        """Sensitivity equals the central finite difference of wbf().E().

        Re-derive the expected value directly from the documented
        formula: perturb the target opinion's ``u`` by ``+/-delta``
        holding ``E`` fixed, fuse, and take ``|E_plus - E_minus| / 2delta``.
        """
        opinions = [
            Opinion(0.10, 0.30, 0.60, 0.5),
            Opinion(0.40, 0.20, 0.40, 0.5),
            Opinion(0.70, 0.10, 0.20, 0.5),
        ]
        delta = 0.01
        target = opinions[1]
        a = target.a
        e = target.expectation()

        def perturbed(du):
            u_new = target.u + du
            b_new = e - a * u_new
            d_new = 1.0 - u_new - b_new
            return Opinion(b_new, d_new, u_new, a)

        minus_list = list(opinions)
        minus_list[1] = perturbed(-delta)
        plus_list = list(opinions)
        plus_list[1] = perturbed(+delta)
        e_minus = Opinion.wbf(*minus_list).expectation()
        e_plus = Opinion.wbf(*plus_list).expectation()
        expected = abs(e_plus - e_minus) / (2.0 * delta)

        got = opinion_sensitivity(opinions, 1, delta=delta)
        assert got is not None
        assert got == pytest.approx(expected, abs=1e-12)


# ── opinion_sensitivity — property tests ───────────────────────────


class TestOpinionSensitivityProperties:
    """Property-based invariants for ``opinion_sensitivity``."""

    @pytest.mark.property
    @given(
        nondogmatic_opinions(),
        nondogmatic_opinions(),
        nondogmatic_opinions(),
        st.integers(min_value=0, max_value=2),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_finite_and_nonnegative(self, a, b, c, index):
        """When defined, the sensitivity is a finite, non-negative float."""
        opinions = [a, b, c]
        s = opinion_sensitivity(opinions, index)
        if s is not None:
            assert math.isfinite(s)
            assert s >= 0.0

    @pytest.mark.property
    @given(
        nondogmatic_opinions(),
        nondogmatic_opinions(),
        nondogmatic_opinions(),
        st.integers(min_value=0, max_value=2),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_deterministic(self, a, b, c, index):
        """The function is a pure deterministic computation."""
        opinions = [a, b, c]
        assert opinion_sensitivity(opinions, index) == opinion_sensitivity(
            opinions, index
        )

    @pytest.mark.property
    @given(
        nondogmatic_opinions(),
        nondogmatic_opinions(),
        nondogmatic_opinions(),
        st.integers(min_value=0, max_value=2),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_bounded_by_inverse_delta(self, a, b, c, index):
        """E stays in [0, 1], so a central difference cannot exceed 1/delta.

        ``|E_plus - E_minus| <= 1`` and the divisor is ``2*delta`` (or a
        halved delta on the fallback branch), so the result is bounded by
        the largest possible ``1/delta`` the routine can use.
        """
        delta = 0.01
        opinions = [a, b, c]
        s = opinion_sensitivity(opinions, index, delta=delta)
        if s is not None:
            # Worst case divisor is delta/4 (two halvings on the
            # one-sided fallback): bound 1 / (delta/4) = 4/delta.
            assert s <= 4.0 / delta + 1e-6


# ── graph_perturbation_sensitivity — helpers ───────────────────────


def _full_trust_edge() -> Opinion:
    """A fully-trusted edge — ``dogmatic_true(0.5)`` passes the child through."""
    return Opinion.dogmatic_true(0.5)


def _two_witness_graph() -> BipolarOpinionGraph:
    """A move node ``m`` supported by witness ``w1`` and attacked by ``w2``.

    ``m`` carries a vacuous intrinsic (a move node); ``w1``/``w2`` are
    leaves with their own intrinsic evidence.
    """
    return BipolarOpinionGraph(
        arguments=frozenset({"m", "w1", "w2"}),
        intrinsic={
            "m": Opinion.vacuous(0.55),
            "w1": Opinion(0.7, 0.1, 0.2, 0.6),
            "w2": Opinion(0.4, 0.3, 0.3, 0.5),
        },
        supports=frozenset({("w1", "m")}),
        attacks=frozenset({("w2", "m")}),
        edge_opinions={
            ("w1", "m"): _full_trust_edge(),
            ("w2", "m"): _full_trust_edge(),
        },
    )


# ── graph_perturbation_sensitivity — unit tests ────────────────────


class TestGraphPerturbationSensitivityContract:
    """Focused contract tests for ``graph_perturbation_sensitivity``."""

    @pytest.mark.unit
    def test_drop_node_changes_target_opinion(self):
        """Dropping a real witness moves the target's resolved opinion."""
        graph = _two_witness_graph()
        delta = graph_perturbation_sensitivity(graph, "m", drop_node="w1")
        assert math.isfinite(delta)
        assert delta > 0.0

    @pytest.mark.unit
    def test_remove_edge_changes_target_opinion(self):
        """Removing the support edge moves the target's resolved opinion."""
        graph = _two_witness_graph()
        delta = graph_perturbation_sensitivity(graph, "m", remove_edge=("w1", "m"))
        assert math.isfinite(delta)
        assert delta > 0.0

    @pytest.mark.unit
    def test_unperturbed_graph_yields_zero_delta(self):
        """No perturbation argument → identical graph → zero delta."""
        graph = _two_witness_graph()
        delta = graph_perturbation_sensitivity(graph, "m")
        assert delta == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.unit
    def test_removing_an_irrelevant_edge_for_disconnected_target(self):
        """Perturbing an input that does not feed the target gives 0 delta.

        ``w1``'s resolved opinion is its own intrinsic; removing the
        ``w2 -> m`` edge cannot change ``w1``.
        """
        graph = _two_witness_graph()
        delta = graph_perturbation_sensitivity(
            graph, "w1", remove_edge=("w2", "m")
        )
        assert delta == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.unit
    def test_drop_node_and_remove_edge_are_mutually_exclusive(self):
        """Passing both perturbations is a usage error."""
        graph = _two_witness_graph()
        with pytest.raises(ValueError):
            graph_perturbation_sensitivity(
                graph, "m", drop_node="w1", remove_edge=("w2", "m")
            )

    @pytest.mark.unit
    def test_unknown_target_raises(self):
        """A target not in the graph is a usage error."""
        graph = _two_witness_graph()
        with pytest.raises(ValueError):
            graph_perturbation_sensitivity(graph, "nonexistent")

    @pytest.mark.unit
    def test_drop_unknown_node_raises(self):
        """Dropping an argument not in the graph is a usage error."""
        graph = _two_witness_graph()
        with pytest.raises(ValueError):
            graph_perturbation_sensitivity(graph, "m", drop_node="ghost")

    @pytest.mark.unit
    def test_remove_unknown_edge_raises(self):
        """Removing an edge not in the graph is a usage error."""
        graph = _two_witness_graph()
        with pytest.raises(ValueError):
            graph_perturbation_sensitivity(graph, "m", remove_edge=("w1", "w2"))

    @pytest.mark.unit
    def test_dropping_the_target_itself_raises(self):
        """The target must survive the perturbation to have a delta."""
        graph = _two_witness_graph()
        with pytest.raises(ValueError):
            graph_perturbation_sensitivity(graph, "m", drop_node="m")

    @pytest.mark.unit
    def test_deterministic(self):
        """Re-running the same perturbation yields the same delta."""
        graph = _two_witness_graph()
        first = graph_perturbation_sensitivity(graph, "m", drop_node="w2")
        second = graph_perturbation_sensitivity(graph, "m", drop_node="w2")
        assert first == second


# ── graph_perturbation_sensitivity — property tests ────────────────


@st.composite
def two_witness_graphs(draw):
    """Generate a ``m <- w1, w2`` graph with random witness intrinsics."""
    w1 = draw(nondogmatic_opinions())
    w2 = draw(nondogmatic_opinions())
    tau = draw(st.floats(min_value=0.1, max_value=0.9))
    return BipolarOpinionGraph(
        arguments=frozenset({"m", "w1", "w2"}),
        intrinsic={
            "m": Opinion.vacuous(tau),
            "w1": w1,
            "w2": w2,
        },
        supports=frozenset({("w1", "m")}),
        attacks=frozenset({("w2", "m")}),
        edge_opinions={
            ("w1", "m"): _full_trust_edge(),
            ("w2", "m"): _full_trust_edge(),
        },
    )


class TestGraphPerturbationSensitivityProperties:
    """Property-based invariants for ``graph_perturbation_sensitivity``."""

    @pytest.mark.property
    @given(two_witness_graphs())
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_finite_and_bounded(self, graph):
        """The delta is finite and bounded by the expectation range [0, 1]."""
        for perturb in (
            {"drop_node": "w1"},
            {"drop_node": "w2"},
            {"remove_edge": ("w1", "m")},
            {"remove_edge": ("w2", "m")},
        ):
            delta = graph_perturbation_sensitivity(graph, "m", **perturb)
            assert math.isfinite(delta)
            assert 0.0 <= delta <= 1.0 + 1e-9

    @pytest.mark.property
    @given(two_witness_graphs())
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_unperturbed_is_zero(self, graph):
        """An unperturbed re-evaluation yields exactly zero delta."""
        assert graph_perturbation_sensitivity(graph, "m") == pytest.approx(
            0.0, abs=1e-12
        )

    @pytest.mark.property
    @given(two_witness_graphs())
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_deterministic(self, graph):
        """Repeated perturbation queries are deterministic."""
        first = graph_perturbation_sensitivity(graph, "m", drop_node="w1")
        second = graph_perturbation_sensitivity(graph, "m", drop_node="w1")
        assert first == second
