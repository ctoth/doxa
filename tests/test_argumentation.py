"""RED test suite for ``doxa.argumentation`` — opinion-valued bipolar
argumentation semantics.

Corrected at Gate B-fix of the foreman-coordinated gauntlet to the
**intrinsic-opinion model locked in Gate A2**
(``dialectical-chess/reports/doxa-argumentation-gateA2.md``). The earlier
Gate B suite encoded Gate A's belief-sterile model: each argument carried only
a scalar base rate, so every leaf resolved to ``vacuous`` and belief could
never originate. Gate A2 supersedes that — each argument now carries an
intrinsic ``Opinion`` (``BipolarOpinionGraph.intrinsic``), a leaf resolves to
its intrinsic opinion, and accrual follows "Model C" (the intrinsic enters the
CCF source pool iff it is non-vacuous).

This file encodes the Gate A2 design exactly: the ``BipolarOpinionGraph``
dataclass with its five fields (``intrinsic: Mapping[str, Opinion]`` replacing
``base_rates``), the six construction-time validations, the ``evaluate``
traversal (Kahn's algorithm with a sorted ready set, cycle detection via
``CyclicGraphError``), the locked CCF accrual operator, and the exact-value
worked example.

Numeric grounding — every concrete value below is the Gate A2-locked computed
value (Gate A2 §§4-6, computed against the installed ``doxa`` package):
- Worked example: ``tau=0.55``, supporter leaf intrinsic ``(0.7, 0.1, 0.2,
  0.6)``, objection leaf intrinsic ``(0.4, 0.3, 0.3, 0.5)``, both edges
  ``dogmatic_true`` → ``omega_m = Opinion(b=0.516, d=0.208, u=0.276, a=0.55)``,
  ``E ≈ 0.6678``.
- Balanced conflict ``u = 0.496`` vs agreement ``u = 0.200`` — the decisive
  CCF property (Gate A2 §5 SC4).

This suite is RED against the current (old-model) ``argumentation.py``, whose
``BipolarOpinionGraph`` still has the ``base_rates`` field — passing
``intrinsic=`` raises ``TypeError``. Gate C-fix makes it green by adopting the
intrinsic-opinion model.

Markers: ``unit`` for focused contract tests, ``property`` for hypothesis-based
invariant tests — per doxa's ``pyproject.toml`` marker registry.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from doxa import Opinion

from doxa.argumentation import (  # noqa: E402
    BipolarOpinionGraph,
    CyclicGraphError,
    evaluate,
)

# ── Constants ──────────────────────────────────────────────────────

# Locked worked-example inputs (Gate A2 §4 "The recomputed worked example").
_TAU = 0.55
_SUPPORTER = (0.7, 0.1, 0.2, 0.6)
_OBJECTION = (0.4, 0.3, 0.3, 0.5)

# Locked worked-example result (Gate A2 §4 "New locked exact values").
_OMEGA_M = (0.516, 0.208, 0.276, 0.55)
_E_WORKED = 0.6678


# ── Helpers ────────────────────────────────────────────────────────


def approx(val, abs=1e-7):
    """pytest.approx with the doxa-conventional default tolerance."""
    return pytest.approx(val, abs=abs)


def _full_trust_edge() -> Opinion:
    """A fully-trusted edge — modelled as ``dogmatic_true(0.5)``.

    ``dogmatic_true.discount(child)`` leaves the child's ``(b, d, u)``
    unchanged (Gate A2 §4; verified in ``opinion.py:399-401``), so a
    full-trust edge passes its child's opinion through untouched.
    """
    return Opinion.dogmatic_true(0.5)


def single_node_graph(tau: float = _TAU) -> BipolarOpinionGraph:
    """An unargued one-node move graph — no supports, no attacks.

    The single node is a move node: its intrinsic opinion is
    ``Opinion.vacuous(tau)`` (no own evidence), so it resolves to
    ``(0, 0, 1, tau)`` and ``E = tau`` (Gate A2 §5 SC1).
    """
    return BipolarOpinionGraph(
        arguments=frozenset({"m"}),
        intrinsic={"m": Opinion.vacuous(tau)},
        supports=frozenset(),
        attacks=frozenset(),
        edge_opinions={},
    )


# ── Hypothesis strategies ──────────────────────────────────────────


@st.composite
def valid_opinions(draw, min_uncertainty=0.01):
    """Generate valid non-dogmatic opinions (u >= min_uncertainty).

    Matches the strategy in ``test_opinion.py`` / ``test_opinion_properties.py``
    so edge opinions and intrinsic opinions used in property tests are
    well-conditioned.
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
def base_rates_strategy(draw):
    """A base rate strictly inside ``(0, 1)``."""
    return draw(st.floats(min_value=0.02, max_value=0.98))


@st.composite
def star_graphs(draw):
    """Generate a random one-target graph: target ``m`` with N children.

    Each child is a leaf supporter or attacker carrying a genuine
    (drawn, non-vacuous) intrinsic ``Opinion`` — so property tests
    actually exercise belief flow (Gate A2 §6.A). The target ``m`` is a
    move node with a ``vacuous`` intrinsic. Each child reaches ``m``
    through a random valid edge opinion. This is a DAG by construction
    (one layer of leaves into one target).
    """
    n = draw(st.integers(min_value=0, max_value=4))
    args = ["m"] + [f"c{i}" for i in range(n)]
    intrinsic = {"m": Opinion.vacuous(draw(base_rates_strategy()))}
    for i in range(n):
        intrinsic[f"c{i}"] = draw(valid_opinions(min_uncertainty=0.02))
    supports = set()
    attacks = set()
    edge_opinions = {}
    for i in range(n):
        child = f"c{i}"
        is_support = draw(st.booleans())
        edge = (child, "m")
        if is_support:
            supports.add(edge)
        else:
            attacks.add(edge)
        edge_opinions[edge] = draw(valid_opinions(min_uncertainty=0.02))
    return BipolarOpinionGraph(
        arguments=frozenset(args),
        intrinsic=intrinsic,
        supports=frozenset(supports),
        attacks=frozenset(attacks),
        edge_opinions=edge_opinions,
    )


# ════════════════════════════════════════════════════════════════════
# Construction-time validations — Gate A2 §1, the six checks
# ════════════════════════════════════════════════════════════════════


class TestConstructionValidation:
    """``BipolarOpinionGraph.__post_init__`` — the 6 locked validations.

    Gate A2 §1: Gate A's seven validations become six — the base-rate-range
    check is removed (now enforced by ``Opinion.__post_init__`` when the
    intrinsic ``Opinion`` is built). Each remaining check is an explicit
    ``raise ValueError`` (never ``assert`` — asserts vanish under
    ``python -O``). The error message must name the offending element so
    the failure is diagnosable.
    """

    def test_valid_graph_constructs(self):
        """A well-formed graph constructs without raising — control case."""
        graph = BipolarOpinionGraph(
            arguments=frozenset({"m", "s"}),
            intrinsic={
                "m": Opinion.vacuous(0.5),
                "s": Opinion.vacuous(0.5),
            },
            supports=frozenset({("s", "m")}),
            attacks=frozenset(),
            edge_opinions={("s", "m"): _full_trust_edge()},
        )
        assert graph.arguments == frozenset({"m", "s"})

    # --- Check 1: intrinsic covers exactly arguments ---

    @pytest.mark.unit
    def test_intrinsic_missing_key_rejected(self):
        """A declared argument with no intrinsic opinion raises ``ValueError``."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"m", "s"}),
                intrinsic={"m": Opinion.vacuous(0.5)},  # 's' missing
                supports=frozenset(),
                attacks=frozenset(),
                edge_opinions={},
            )

    @pytest.mark.unit
    def test_intrinsic_extra_key_rejected(self):
        """An intrinsic opinion for a non-declared argument raises ``ValueError``."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"m"}),
                intrinsic={
                    "m": Opinion.vacuous(0.5),
                    "ghost": Opinion.vacuous(0.5),  # 'ghost' not declared
                },
                supports=frozenset(),
                attacks=frozenset(),
                edge_opinions={},
            )

    @pytest.mark.unit
    def test_intrinsic_mismatch_names_offending_argument(self):
        """The ``ValueError`` names the symmetric-difference argument."""
        with pytest.raises(ValueError, match="s"):
            BipolarOpinionGraph(
                arguments=frozenset({"m", "s"}),
                intrinsic={"m": Opinion.vacuous(0.5)},
                supports=frozenset(),
                attacks=frozenset(),
                edge_opinions={},
            )

    # --- Base-rate range: now enforced by Opinion.__post_init__ ---
    #
    # Gate A2 §1 / §6.D: the base rate is now ``intrinsic[x].a``. A bad base
    # rate is rejected by the ``Opinion`` constructor when the intrinsic
    # ``Opinion`` is built — before it can ever reach the graph. The check
    # moved from ``BipolarOpinionGraph.__post_init__`` to
    # ``Opinion.__post_init__``; there is no graph-level base-rate check.

    @pytest.mark.unit
    def test_base_rate_zero_rejected(self):
        """A base rate of exactly ``0.0`` is rejected by ``Opinion``."""
        with pytest.raises(ValueError):
            Opinion.vacuous(0.0)

    @pytest.mark.unit
    def test_base_rate_one_rejected(self):
        """A base rate of exactly ``1.0`` is rejected by ``Opinion``."""
        with pytest.raises(ValueError):
            Opinion.vacuous(1.0)

    @pytest.mark.unit
    def test_base_rate_above_one_rejected(self):
        """A base rate above ``1.0`` is rejected by ``Opinion``."""
        with pytest.raises(ValueError):
            Opinion.vacuous(1.5)

    @pytest.mark.unit
    def test_base_rate_out_of_range_names_argument(self):
        """The ``Opinion`` ``ValueError`` names the offending base rate.

        ``Opinion.__post_init__`` raises ``ValueError(f"a={self.a} not in
        (0, 1)")`` — the message carries the offending value.
        """
        with pytest.raises(ValueError, match="a="):
            Opinion(0.0, 0.0, 1.0, 0.0)

    # --- Check 2: support edges reference only declared arguments ---

    @pytest.mark.unit
    def test_support_edge_unknown_source_rejected(self):
        """A support edge from an undeclared argument raises ``ValueError``."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"m"}),
                intrinsic={"m": Opinion.vacuous(0.5)},
                supports=frozenset({("ghost", "m")}),
                attacks=frozenset(),
                edge_opinions={("ghost", "m"): _full_trust_edge()},
            )

    @pytest.mark.unit
    def test_support_edge_unknown_target_rejected(self):
        """A support edge into an undeclared argument raises ``ValueError``."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"s"}),
                intrinsic={"s": Opinion.vacuous(0.5)},
                supports=frozenset({("s", "ghost")}),
                attacks=frozenset(),
                edge_opinions={("s", "ghost"): _full_trust_edge()},
            )

    # --- Check 3: attack edges reference only declared arguments ---

    @pytest.mark.unit
    def test_attack_edge_unknown_source_rejected(self):
        """An attack edge from an undeclared argument raises ``ValueError``."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"m"}),
                intrinsic={"m": Opinion.vacuous(0.5)},
                supports=frozenset(),
                attacks=frozenset({("ghost", "m")}),
                edge_opinions={("ghost", "m"): _full_trust_edge()},
            )

    @pytest.mark.unit
    def test_attack_edge_unknown_target_rejected(self):
        """An attack edge into an undeclared argument raises ``ValueError``."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"o"}),
                intrinsic={"o": Opinion.vacuous(0.5)},
                supports=frozenset(),
                attacks=frozenset({("o", "ghost")}),
                edge_opinions={("o", "ghost"): _full_trust_edge()},
            )

    # --- Check 4: supports and attacks are disjoint ---

    @pytest.mark.unit
    def test_support_attack_overlap_rejected(self):
        """An ordered pair that is both a support and an attack raises."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"m", "s"}),
                intrinsic={
                    "m": Opinion.vacuous(0.5),
                    "s": Opinion.vacuous(0.5),
                },
                supports=frozenset({("s", "m")}),
                attacks=frozenset({("s", "m")}),  # same edge in both
                edge_opinions={("s", "m"): _full_trust_edge()},
            )

    @pytest.mark.unit
    def test_support_attack_overlap_names_edge(self):
        """The ``ValueError`` for an overlapping edge names the source."""
        with pytest.raises(ValueError, match="s"):
            BipolarOpinionGraph(
                arguments=frozenset({"m", "s"}),
                intrinsic={
                    "m": Opinion.vacuous(0.5),
                    "s": Opinion.vacuous(0.5),
                },
                supports=frozenset({("s", "m")}),
                attacks=frozenset({("s", "m")}),
                edge_opinions={("s", "m"): _full_trust_edge()},
            )

    # --- Check 5: edge_opinions keys exactly cover supports ∪ attacks ---

    @pytest.mark.unit
    def test_edge_opinion_missing_for_support_rejected(self):
        """A support edge with no opinion in ``edge_opinions`` raises."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"m", "s"}),
                intrinsic={
                    "m": Opinion.vacuous(0.5),
                    "s": Opinion.vacuous(0.5),
                },
                supports=frozenset({("s", "m")}),
                attacks=frozenset(),
                edge_opinions={},  # ('s','m') has no opinion
            )

    @pytest.mark.unit
    def test_edge_opinion_missing_for_attack_rejected(self):
        """An attack edge with no opinion in ``edge_opinions`` raises."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"m", "o"}),
                intrinsic={
                    "m": Opinion.vacuous(0.5),
                    "o": Opinion.vacuous(0.5),
                },
                supports=frozenset(),
                attacks=frozenset({("o", "m")}),
                edge_opinions={},  # ('o','m') has no opinion
            )

    @pytest.mark.unit
    def test_edge_opinion_for_nonexistent_edge_rejected(self):
        """An ``edge_opinions`` key that is not an edge raises ``ValueError``."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"m", "s"}),
                intrinsic={
                    "m": Opinion.vacuous(0.5),
                    "s": Opinion.vacuous(0.5),
                },
                supports=frozenset(),
                attacks=frozenset(),
                edge_opinions={("s", "m"): _full_trust_edge()},  # no such edge
            )

    # --- Check 6: no self-loops ---

    @pytest.mark.unit
    def test_self_loop_support_rejected(self):
        """A support edge ``(x, x)`` raises ``ValueError`` at construction."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"m"}),
                intrinsic={"m": Opinion.vacuous(0.5)},
                supports=frozenset({("m", "m")}),
                attacks=frozenset(),
                edge_opinions={("m", "m"): _full_trust_edge()},
            )

    @pytest.mark.unit
    def test_self_loop_attack_rejected(self):
        """An attack edge ``(x, x)`` raises ``ValueError`` at construction."""
        with pytest.raises(ValueError):
            BipolarOpinionGraph(
                arguments=frozenset({"m"}),
                intrinsic={"m": Opinion.vacuous(0.5)},
                supports=frozenset(),
                attacks=frozenset({("m", "m")}),
                edge_opinions={("m", "m"): _full_trust_edge()},
            )

    @pytest.mark.unit
    def test_self_loop_names_argument(self):
        """The self-loop ``ValueError`` names the offending argument."""
        with pytest.raises(ValueError, match="m"):
            BipolarOpinionGraph(
                arguments=frozenset({"m"}),
                intrinsic={"m": Opinion.vacuous(0.5)},
                supports=frozenset({("m", "m")}),
                attacks=frozenset(),
                edge_opinions={("m", "m"): _full_trust_edge()},
            )

    # --- Validations use raise, not assert ---

    @pytest.mark.unit
    def test_validation_uses_explicit_raise_not_assert(self):
        """Construction validation must survive ``python -O``.

        Gate A2 §1: every failure is an explicit ``raise``, never ``assert``
        (asserts are stripped under ``-O``). A bare ``AssertionError`` here
        would mean the check was an ``assert`` — only ``ValueError`` is
        acceptable.
        """
        with pytest.raises(ValueError) as exc_info:
            BipolarOpinionGraph(
                arguments=frozenset({"m", "s"}),
                intrinsic={"m": Opinion.vacuous(0.5)},  # 's' missing
                supports=frozenset(),
                attacks=frozenset(),
                edge_opinions={},
            )
        assert not isinstance(exc_info.value, AssertionError)


# ════════════════════════════════════════════════════════════════════
# Rationality sanity check 1 — unargued argument → E = tau exactly
# ════════════════════════════════════════════════════════════════════


class TestSanityUnargued:
    """Gate A2 §5 SC1: a move argument with no support and no attack and a
    ``vacuous`` intrinsic resolves to ``Opinion.vacuous(tau)`` and
    ``expectation() == tau`` exactly."""

    @pytest.mark.unit
    def test_unargued_argument_is_vacuous(self):
        """No supporters, no attackers → ``(b, d, u) = (0, 0, 1)``."""
        result = evaluate(single_node_graph(0.55))
        omega = result["m"]
        assert omega.b == approx(0.0)
        assert omega.d == approx(0.0)
        assert omega.u == approx(1.0)

    @pytest.mark.unit
    def test_unargued_argument_base_rate_is_tau(self):
        """The unargued node carries ``a = tau`` exactly."""
        result = evaluate(single_node_graph(0.55))
        assert result["m"].a == approx(0.55)

    @pytest.mark.unit
    def test_unargued_argument_expectation_equals_tau(self):
        """``expectation() == tau`` exactly for an unargued argument."""
        for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
            result = evaluate(single_node_graph(tau))
            assert result["m"].expectation() == approx(tau)


# ════════════════════════════════════════════════════════════════════
# Rationality sanity check 2 — one strong supporter → E strictly above tau
# ════════════════════════════════════════════════════════════════════


class TestSanityOneSupporter:
    """Gate A2 §5 SC2: one strong, undisputed supporter pulls ``E`` strictly
    above ``tau``.

    Under the corrected (intrinsic-opinion) model the supporter ``s`` is a
    leaf carrying a genuine intrinsic opinion — belief originates there. A
    strong supporter leaf intrinsic ``(0.8, 0.05, 0.15, 0.6)`` through a
    full-trust edge gives ``omega_m = (0.800, 0.050, 0.150, 0.55)``,
    ``E = 0.8825`` (Gate A2 §5 SC2, §6.C).
    """

    @staticmethod
    def _one_supporter_graph(tau=_TAU):
        """A move ``m`` with one strong supporter leaf ``s``."""
        return BipolarOpinionGraph(
            arguments=frozenset({"m", "s"}),
            intrinsic={
                "m": Opinion.vacuous(tau),
                "s": Opinion(0.8, 0.05, 0.15, 0.6),
            },
            supports=frozenset({("s", "m")}),
            attacks=frozenset(),
            edge_opinions={("s", "m"): _full_trust_edge()},
        )

    @pytest.mark.unit
    def test_one_strong_supporter_raises_expectation_above_tau(self):
        """A strong supporter leaf through a full-trust edge → ``E > tau``."""
        result = evaluate(self._one_supporter_graph())
        assert result["m"].expectation() > _TAU

    @pytest.mark.unit
    def test_one_strong_supporter_with_belief_raises_e_above_tau(self):
        """A strong supporter leaf → ``E == 0.8825`` (Gate A2 §5 SC2).

        The supporter ``s`` is a leaf carrying intrinsic ``(0.8, 0.05, 0.15,
        0.6)``; a full-trust edge passes it through unchanged, so
        ``omega_m = (0.800, 0.050, 0.150, 0.55)`` and
        ``E = 0.800 + 0.55 * 0.150 = 0.8825``.
        """
        result = evaluate(self._one_supporter_graph())
        assert result["m"].expectation() == pytest.approx(0.8825, abs=1e-4)


# ════════════════════════════════════════════════════════════════════
# Rationality sanity check 3 — one strong objection → E strictly below tau
# ════════════════════════════════════════════════════════════════════


class TestSanityOneObjection:
    """Gate A2 §5 SC3: one strong, undisputed objection pulls ``E`` strictly
    below ``tau``.

    Under the corrected model the objection ``o`` is a leaf carrying a
    genuine intrinsic opinion. A strong objection leaf intrinsic
    ``(0.8, 0.05, 0.15, 0.5)`` through a full-trust edge gives
    ``omega_m = (0.050, 0.800, 0.150, 0.55)``, ``E = 0.1325`` (Gate A2 §5
    SC3, §6.C).
    """

    @pytest.mark.unit
    def test_one_strong_objection_lowers_expectation_below_tau(self):
        """A strong objection leaf → ``E == 0.1325`` and ``< tau``."""
        graph = BipolarOpinionGraph(
            arguments=frozenset({"m", "o"}),
            intrinsic={
                "m": Opinion.vacuous(_TAU),
                "o": Opinion(0.8, 0.05, 0.15, 0.5),
            },
            supports=frozenset(),
            attacks=frozenset({("o", "m")}),
            edge_opinions={("o", "m"): _full_trust_edge()},
        )
        result = evaluate(graph)
        assert result["m"].expectation() == pytest.approx(0.1325, abs=1e-4)
        assert result["m"].expectation() < _TAU


# ════════════════════════════════════════════════════════════════════
# Rationality sanity check 4 — balanced support+objection raises u
#   (the decisive CCF property — Gate A2 §5 SC4)
# ════════════════════════════════════════════════════════════════════


class TestSanityBalancedConflictRaisesUncertainty:
    """Gate A2 §5 SC4 — THE discriminating test.

    Balanced support and objection of equally strong arguments must produce
    a *higher* ``u`` than agreement of two equally strong supporters.
    Disagreement becomes honest uncertainty, not fake confidence. Gate A2
    locked the exact contrast: conflict ``u = 0.496`` vs agreement
    ``u = 0.200``.

    Under the corrected model the supporter / objection nodes are leaves
    carrying genuine intrinsic opinions — belief originates there. The
    ``gs``/``go``/``g`` grounding scaffolding of the old (belief-sterile)
    suite is deleted.
    """

    @staticmethod
    def _conflict_graph():
        """One supporter leaf and one objection leaf, equally strong."""
        return BipolarOpinionGraph(
            arguments=frozenset({"m", "s", "o"}),
            intrinsic={
                "m": Opinion.vacuous(_TAU),
                "s": Opinion(0.7, 0.1, 0.2, 0.6),
                "o": Opinion(0.7, 0.1, 0.2, 0.6),
            },
            supports=frozenset({("s", "m")}),
            attacks=frozenset({("o", "m")}),
            edge_opinions={
                ("s", "m"): Opinion.dogmatic_true(0.5),
                ("o", "m"): Opinion.dogmatic_true(0.5),
            },
        )

    @staticmethod
    def _agreement_graph():
        """Two supporter leaves of equal strength."""
        return BipolarOpinionGraph(
            arguments=frozenset({"m", "s1", "s2"}),
            intrinsic={
                "m": Opinion.vacuous(_TAU),
                "s1": Opinion(0.7, 0.1, 0.2, 0.6),
                "s2": Opinion(0.7, 0.1, 0.2, 0.6),
            },
            supports=frozenset({("s1", "m"), ("s2", "m")}),
            attacks=frozenset(),
            edge_opinions={
                ("s1", "m"): Opinion.dogmatic_true(0.5),
                ("s2", "m"): Opinion.dogmatic_true(0.5),
            },
        )

    @pytest.mark.unit
    def test_balanced_conflict_uncertainty_is_0496(self):
        """Balanced 1-support + 1-objection → ``u ≈ 0.496`` (Gate A2 locked)."""
        result = evaluate(self._conflict_graph())
        assert result["m"].u == pytest.approx(0.496, abs=1e-6)

    @pytest.mark.unit
    def test_agreement_uncertainty_is_0200(self):
        """Two agreeing supporters → ``u ≈ 0.200`` (Gate A2 locked)."""
        result = evaluate(self._agreement_graph())
        assert result["m"].u == pytest.approx(0.200, abs=1e-6)

    @pytest.mark.unit
    def test_conflict_uncertainty_strictly_exceeds_agreement(self):
        """The decisive CCF property: conflict ``u`` > agreement ``u``.

        This is the single check that rejected WBF and BetaEvidence
        summation in Gate A — they hold ``u`` flat under conflict. CCF
        routes conflicting belief into uncertainty.
        """
        conflict = evaluate(self._conflict_graph())["m"]
        agreement = evaluate(self._agreement_graph())["m"]
        assert conflict.u > agreement.u + 1e-3

    @pytest.mark.unit
    def test_balanced_conflict_expectation_near_tau(self):
        """Balanced equal-strength conflict → ``E`` near ``tau``.

        With ``b ≈ d`` after fusion and ``a = tau``, the expectation
        ``b + a·u`` sits close to ``tau`` — the contested move is honestly
        uncertain, not falsely confident. Gate A2 §5 SC4 / §6.B locked
        ``E = 0.252 + 0.55*0.496 ≈ 0.5248``.
        """
        result = evaluate(self._conflict_graph())["m"]
        # b == d == 0.252, u == 0.496, a == 0.55 → E = 0.252 + 0.55*0.496
        assert result.expectation() == pytest.approx(
            0.252 + 0.55 * 0.496, abs=1e-6
        )


# ════════════════════════════════════════════════════════════════════
# Rationality sanity check 5 — every node a valid Opinion, E ∈ [0, 1]
# ════════════════════════════════════════════════════════════════════


class TestSanityValidOpinions:
    """Gate A2 §5 SC5: every resolved node is a valid ``Opinion`` and every
    ``expectation()`` lies in ``[0, 1]``."""

    @pytest.mark.property
    @given(star_graphs())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_every_node_is_a_valid_opinion(self, graph):
        """Every value of ``evaluate`` is an ``Opinion`` with ``b+d+u==1``."""
        result = evaluate(graph)
        for name, omega in result.items():
            assert isinstance(omega, Opinion), f"{name} is not an Opinion"
            assert abs(omega.b + omega.d + omega.u - 1.0) < 1e-6, (
                f"{name}: mass sum != 1"
            )

    @pytest.mark.property
    @given(star_graphs())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_every_expectation_in_unit_interval(self, graph):
        """Every node's ``expectation()`` lies within ``[0, 1]``."""
        result = evaluate(graph)
        for name, omega in result.items():
            e = omega.expectation()
            assert -1e-9 <= e <= 1.0 + 1e-9, f"{name}: E={e} out of [0,1]"

    @pytest.mark.property
    @given(star_graphs())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_evaluate_covers_every_argument(self, graph):
        """``evaluate`` returns a key for every declared argument."""
        result = evaluate(graph)
        assert set(result) == set(graph.arguments)

    @pytest.mark.property
    @given(star_graphs())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_every_node_base_rate_is_its_tau(self, graph):
        """Each resolved node carries ``a = intrinsic[node].a`` exactly.

        Gate A2 §2 step 5: ``tau`` (= ``intrinsic[x].a``) is re-stamped,
        never fused from children.
        """
        result = evaluate(graph)
        for name, omega in result.items():
            assert omega.a == pytest.approx(
                graph.intrinsic[name].a, abs=1e-9
            )


# ════════════════════════════════════════════════════════════════════
# Rationality sanity check 6 — weak/uncertain edge raises u, no fabrication
# ════════════════════════════════════════════════════════════════════


class TestSanityWeakEdge:
    """Gate A2 §5 SC6: a weak / uncertain edge raises ``u`` and never
    fabricates belief.

    Compared head-to-head against a full-trust edge over the SAME supporter
    leaf: the weak edge must leave the target with strictly more ``u`` and
    no more ``b`` than the full-trust edge. Under the corrected model the
    supporter ``s`` is a leaf carrying a genuine intrinsic opinion — the
    ``g`` grounding node of the old suite is deleted.
    """

    @staticmethod
    def _graph_with_support_edge(edge: Opinion) -> BipolarOpinionGraph:
        """One supporter leaf carrying a strong intrinsic opinion, reaching
        ``m`` through the supplied edge opinion (Gate A2 §6.B)."""
        return BipolarOpinionGraph(
            arguments=frozenset({"m", "s"}),
            intrinsic={
                "m": Opinion.vacuous(_TAU),
                "s": Opinion(0.9, 0.0, 0.1, 0.6),
            },
            supports=frozenset({("s", "m")}),
            attacks=frozenset(),
            edge_opinions={("s", "m"): edge},
        )

    @pytest.mark.unit
    def test_weak_edge_raises_uncertainty_vs_full_trust(self):
        """A weak edge leaves the target with more ``u`` than a full edge.

        Gate A2 §5 SC6 / §6.B: weak edge → ``m.u = 0.820``, full-trust edge
        → ``m.u = 0.100``.
        """
        weak = evaluate(self._graph_with_support_edge(Opinion(0.2, 0.1, 0.7, 0.5)))
        strong = evaluate(
            self._graph_with_support_edge(Opinion.dogmatic_true(0.5))
        )
        assert weak["m"].u > strong["m"].u

    @pytest.mark.unit
    def test_weak_edge_does_not_fabricate_belief(self):
        """A weak edge yields no more belief than a full-trust edge.

        Gate A2 §6.C: weak ``m.b = 0.180`` ≤ full-trust ``m.b = 0.900``.
        """
        weak = evaluate(self._graph_with_support_edge(Opinion(0.2, 0.1, 0.7, 0.5)))
        strong = evaluate(
            self._graph_with_support_edge(Opinion.dogmatic_true(0.5))
        )
        assert weak["m"].b <= strong["m"].b + 1e-9

    @pytest.mark.unit
    def test_fully_vacuous_edge_contributes_nothing(self):
        """A vacuous edge → the supporter contributes nothing → ``E = tau``.

        Gate A2 §6.C: the supporter ``s`` is a leaf with a non-vacuous
        intrinsic, but a vacuous *edge* discounts any child to vacuous, so
        ``m`` falls back to ``E = tau``.
        """
        result = evaluate(
            self._graph_with_support_edge(Opinion.vacuous(0.5))
        )
        assert result["m"].u == approx(1.0)
        assert result["m"].expectation() == approx(_TAU)


# ════════════════════════════════════════════════════════════════════
# Rationality sanity check 7 — topological correctness
# ════════════════════════════════════════════════════════════════════


class TestSanityTopologicalCorrectness:
    """Gate A2 §5 SC7 / Gate A §3: a child is fully resolved before its
    parent.

    A multi-layer DAG resolves correctly only if traversal honours
    dependency order — a leaf's opinion must already be available when its
    parent is computed. Under the corrected model the chain root ``g``
    carries a genuine intrinsic opinion, so belief genuinely propagates
    ``g → s → m``.
    """

    @staticmethod
    def _chain_graph() -> BipolarOpinionGraph:
        """Chain ``g → s → m``; ``g`` is a leaf with a strong intrinsic."""
        return BipolarOpinionGraph(
            arguments=frozenset({"m", "s", "g"}),
            intrinsic={
                "m": Opinion.vacuous(0.5),
                "s": Opinion.vacuous(0.5),
                "g": Opinion(0.8, 0.05, 0.15, 0.6),
            },
            supports=frozenset({("s", "m"), ("g", "s")}),
            attacks=frozenset(),
            edge_opinions={
                ("s", "m"): Opinion.dogmatic_true(0.5),
                ("g", "s"): Opinion.dogmatic_true(0.5),
            },
        )

    @pytest.mark.unit
    def test_three_layer_chain_resolves_bottom_up(self):
        """Chain ``g → s → m``: each node resolves from its resolved child.

        Gate A2 §5 SC7: ``g`` is a leaf with intrinsic ``(0.8, 0.05, 0.15,
        0.6)``; full-trust edges propagate belief, so ``g``, ``s`` and ``m``
        each resolve to ``(0.800, 0.050, 0.150)`` — belief reaches every
        layer.
        """
        result = evaluate(self._chain_graph())
        assert result["g"].b == approx(0.8)
        assert result["s"].b == approx(0.8)
        assert result["m"].b == approx(0.8)

    @pytest.mark.unit
    def test_belief_propagates_through_chain(self):
        """A leaf with a genuine intrinsic propagates belief up a full-trust
        chain.

        Gate A2 §5 SC7 / §6.C: ``g`` is a leaf carrying intrinsic
        ``(0.8, 0.05, 0.15, 0.6)``; belief genuinely propagates, so ``s``
        and ``m`` each carry ``b > 0`` — the parent sees the resolved child,
        not a vacuous placeholder.
        """
        result = evaluate(self._chain_graph())
        assert result["s"].b > 0.0
        assert result["m"].b > 0.0

    @pytest.mark.unit
    def test_diamond_dag_resolves(self):
        """A diamond DAG (one node feeds two parents that feed a sink)."""
        graph = BipolarOpinionGraph(
            arguments=frozenset({"sink", "left", "right", "src"}),
            intrinsic={
                "sink": Opinion.vacuous(0.5),
                "left": Opinion.vacuous(0.5),
                "right": Opinion.vacuous(0.5),
                "src": Opinion(0.8, 0.05, 0.15, 0.6),
            },
            supports=frozenset(
                {
                    ("left", "sink"),
                    ("right", "sink"),
                    ("src", "left"),
                    ("src", "right"),
                }
            ),
            attacks=frozenset(),
            edge_opinions={
                ("left", "sink"): _full_trust_edge(),
                ("right", "sink"): _full_trust_edge(),
                ("src", "left"): _full_trust_edge(),
                ("src", "right"): _full_trust_edge(),
            },
        )
        result = evaluate(graph)
        assert set(result) == {"sink", "left", "right", "src"}


# ════════════════════════════════════════════════════════════════════
# The exact-value worked example — Gate A2 §4 "recomputed worked example"
# ════════════════════════════════════════════════════════════════════


class TestWorkedExample:
    """Gate A2 §4: the locked worked example, an exact-value regression.

    Move ``m`` with ``tau = 0.55``; one supporter leaf with intrinsic
    ``(0.7, 0.1, 0.2, 0.6)``, one objection leaf with intrinsic
    ``(0.4, 0.3, 0.3, 0.5)``; both edges ``dogmatic_true``.
    Locked result: ``omega_m = Opinion(b=0.516, d=0.208, u=0.276, a=0.55)``,
    ``E ≈ 0.6678``.

    Under the corrected (intrinsic-opinion) model the supporter and
    objection are leaf arguments carrying intrinsic opinions — belief
    originates there. The move ``m`` has a ``vacuous`` intrinsic (no own
    evidence), dropped from the CCF pool (Model C). The ``gs``/``go``
    grounding scaffolding of the old (belief-sterile) suite is deleted.
    """

    @staticmethod
    def _worked_graph() -> BipolarOpinionGraph:
        """The Gate A2 worked-example graph (§4).

        ``s`` and ``o`` are leaf arguments carrying intrinsic opinions:
        ``intrinsic["s"] = (0.7, 0.1, 0.2, 0.6)``,
        ``intrinsic["o"] = (0.4, 0.3, 0.3, 0.5)``. ``m`` is a move node
        with ``intrinsic["m"] = vacuous(0.55)``. Both edges are
        ``dogmatic_true`` (fully trusted).
        """
        return BipolarOpinionGraph(
            arguments=frozenset({"m", "s", "o"}),
            intrinsic={
                "m": Opinion.vacuous(_TAU),
                "s": Opinion(_SUPPORTER[0], _SUPPORTER[1],
                             _SUPPORTER[2], _SUPPORTER[3]),
                "o": Opinion(_OBJECTION[0], _OBJECTION[1],
                             _OBJECTION[2], _OBJECTION[3]),
            },
            supports=frozenset({("s", "m")}),
            attacks=frozenset({("o", "m")}),
            edge_opinions={
                ("s", "m"): Opinion.dogmatic_true(0.5),
                ("o", "m"): Opinion.dogmatic_true(0.5),
            },
        )

    @pytest.mark.unit
    def test_worked_example_belief(self):
        """``omega_m.b ≈ 0.516``."""
        result = evaluate(self._worked_graph())
        assert result["m"].b == pytest.approx(_OMEGA_M[0], abs=1e-4)

    @pytest.mark.unit
    def test_worked_example_disbelief(self):
        """``omega_m.d ≈ 0.208``."""
        result = evaluate(self._worked_graph())
        assert result["m"].d == pytest.approx(_OMEGA_M[1], abs=1e-4)

    @pytest.mark.unit
    def test_worked_example_uncertainty(self):
        """``omega_m.u ≈ 0.276`` — disagreement keeps ``u`` substantial."""
        result = evaluate(self._worked_graph())
        assert result["m"].u == pytest.approx(_OMEGA_M[2], abs=1e-4)

    @pytest.mark.unit
    def test_worked_example_base_rate_is_tau(self):
        """``omega_m.a == 0.55`` exactly — ``tau`` is re-stamped, not fused."""
        result = evaluate(self._worked_graph())
        assert result["m"].a == pytest.approx(_TAU, abs=1e-9)

    @pytest.mark.unit
    def test_worked_example_expectation(self):
        """``omega_m.expectation() ≈ 0.6678``."""
        result = evaluate(self._worked_graph())
        assert result["m"].expectation() == pytest.approx(_E_WORKED, abs=1e-4)

    @pytest.mark.unit
    def test_worked_example_strength_above_tau(self):
        """The strong supporter pulls ``E`` above ``tau``."""
        result = evaluate(self._worked_graph())
        assert result["m"].expectation() > _TAU


# ════════════════════════════════════════════════════════════════════
# Cycle detection — CyclicGraphError, catchable as ValueError
# ════════════════════════════════════════════════════════════════════


class TestCycleDetection:
    """Gate A §3: ``evaluate`` raises ``CyclicGraphError`` on a cyclic graph.

    Acyclicity is NOT checked at construction (Gate A2 §1; Gate A §2, §6
    item 5) — the graph dataclass constructs fine for a cyclic graph; the
    cycle is detected inside ``evaluate`` via Kahn's algorithm.
    """

    @staticmethod
    def _two_cycle_graph() -> BipolarOpinionGraph:
        """``a → b → a`` — a two-node support cycle."""
        return BipolarOpinionGraph(
            arguments=frozenset({"a", "b"}),
            intrinsic={
                "a": Opinion.vacuous(0.5),
                "b": Opinion.vacuous(0.5),
            },
            supports=frozenset({("a", "b"), ("b", "a")}),
            attacks=frozenset(),
            edge_opinions={
                ("a", "b"): _full_trust_edge(),
                ("b", "a"): _full_trust_edge(),
            },
        )

    @staticmethod
    def _three_cycle_graph() -> BipolarOpinionGraph:
        """``a → b → c → a`` — a three-node mixed support/attack cycle."""
        return BipolarOpinionGraph(
            arguments=frozenset({"a", "b", "c"}),
            intrinsic={
                "a": Opinion.vacuous(0.5),
                "b": Opinion.vacuous(0.5),
                "c": Opinion.vacuous(0.5),
            },
            supports=frozenset({("a", "b"), ("b", "c")}),
            attacks=frozenset({("c", "a")}),
            edge_opinions={
                ("a", "b"): _full_trust_edge(),
                ("b", "c"): _full_trust_edge(),
                ("c", "a"): _full_trust_edge(),
            },
        )

    @pytest.mark.unit
    def test_cyclic_graph_constructs_without_raising(self):
        """A cyclic graph constructs fine — the cycle check is in evaluate."""
        graph = self._two_cycle_graph()
        assert graph.arguments == frozenset({"a", "b"})

    @pytest.mark.unit
    def test_two_cycle_raises_cyclic_graph_error(self):
        """``evaluate`` on a 2-cycle raises ``CyclicGraphError``."""
        with pytest.raises(CyclicGraphError):
            evaluate(self._two_cycle_graph())

    @pytest.mark.unit
    def test_three_cycle_raises_cyclic_graph_error(self):
        """``evaluate`` on a 3-cycle raises ``CyclicGraphError``."""
        with pytest.raises(CyclicGraphError):
            evaluate(self._three_cycle_graph())

    @pytest.mark.unit
    def test_cyclic_graph_error_is_a_value_error(self):
        """``CyclicGraphError`` subclasses ``ValueError`` (Gate A §2).

        An existing ``except ValueError`` handler still catches it.
        """
        assert issubclass(CyclicGraphError, ValueError)

    @pytest.mark.unit
    def test_cycle_catchable_as_value_error(self):
        """A cycle is catchable through a bare ``except ValueError``."""
        with pytest.raises(ValueError):
            evaluate(self._two_cycle_graph())

    @pytest.mark.unit
    def test_cycle_error_names_unresolved_arguments(self):
        """The ``CyclicGraphError`` message names the unresolved arguments.

        Gate A §3 step 4: the diagnostic carries the set of arguments on
        (or downstream of) the cycle.
        """
        with pytest.raises(CyclicGraphError) as exc_info:
            evaluate(self._two_cycle_graph())
        message = str(exc_info.value)
        assert "a" in message and "b" in message

    @pytest.mark.unit
    def test_partial_cycle_still_raises(self):
        """A graph with an acyclic part plus a cycle still raises.

        ``leaf → m`` is acyclic; ``a ↔ b`` is a cycle. The cycle must still
        be detected.
        """
        graph = BipolarOpinionGraph(
            arguments=frozenset({"m", "leaf", "a", "b"}),
            intrinsic={
                "m": Opinion.vacuous(0.5),
                "leaf": Opinion.vacuous(0.5),
                "a": Opinion.vacuous(0.5),
                "b": Opinion.vacuous(0.5),
            },
            supports=frozenset(
                {("leaf", "m"), ("a", "b"), ("b", "a")}
            ),
            attacks=frozenset(),
            edge_opinions={
                ("leaf", "m"): _full_trust_edge(),
                ("a", "b"): _full_trust_edge(),
                ("b", "a"): _full_trust_edge(),
            },
        )
        with pytest.raises(CyclicGraphError):
            evaluate(graph)


# ════════════════════════════════════════════════════════════════════
# Boundary behaviours — Gate A2 §2, §6.C
# ════════════════════════════════════════════════════════════════════


class TestBoundaryBehaviours:
    """Gate A2 §2 / §6.C: vacuous edges, dogmatic leaves.

    - A fully-vacuous edge contributes nothing.
    - A dogmatic-true supporter leaf alone → ``(1, 0, 0, tau)``.
    - A dogmatic supporter leaf vs a dogmatic attacker leaf → vacuous,
      ``E = tau``.
    """

    @pytest.mark.unit
    def test_vacuous_edge_contributes_nothing(self):
        """A supporter leaf reached only through a vacuous edge → ``E = tau``.

        Gate A2 §6.C: ``intrinsic["s"]`` is a genuine ``Opinion``, but the
        vacuous *edge* discounts it to vacuous, so ``m`` falls back to
        ``E = tau``.
        """
        graph = BipolarOpinionGraph(
            arguments=frozenset({"m", "s"}),
            intrinsic={
                "m": Opinion.vacuous(_TAU),
                "s": Opinion(0.8, 0.05, 0.15, 0.6),
            },
            supports=frozenset({("s", "m")}),
            attacks=frozenset(),
            edge_opinions={("s", "m"): Opinion.vacuous(0.5)},
        )
        result = evaluate(graph)
        assert result["m"].u == approx(1.0)
        assert result["m"].expectation() == approx(_TAU)

    @pytest.mark.unit
    def test_dogmatic_supporter_alone_gives_full_belief(self):
        """A single dogmatic-true supporter leaf → ``omega_m = (1, 0, 0, tau)``.

        Gate A2 §6.C: under the corrected model a leaf's intrinsic can be
        ``dogmatic_true``. ``intrinsic["s"] = dogmatic_true(0.6)``; the
        ``(s, m)`` edge is ``dogmatic_true`` and passes it through. Computed
        result: ``omega_m = (1.0, 0.0, 0.0, 0.55)``.
        """
        graph = BipolarOpinionGraph(
            arguments=frozenset({"m", "s"}),
            intrinsic={
                "m": Opinion.vacuous(_TAU),
                "s": Opinion.dogmatic_true(0.6),
            },
            supports=frozenset({("s", "m")}),
            attacks=frozenset(),
            edge_opinions={("s", "m"): Opinion.dogmatic_true(0.5)},
        )
        result = evaluate(graph)
        assert result["m"].b == approx(1.0)
        assert result["m"].d == approx(0.0)
        assert result["m"].u == approx(0.0)
        assert result["m"].a == approx(_TAU)

    @pytest.mark.unit
    def test_dogmatic_supporter_vs_dogmatic_attacker_is_vacuous(self):
        """A dogmatic supporter leaf against a dogmatic attacker leaf → vacuous.

        Gate A2 §6.C / §5: total conflict between certainties is honest
        ignorance — ``omega_m = (0, 0, 1, tau)``, ``E = tau``. Both ``s``
        and ``o`` are leaves with ``dogmatic_true`` intrinsic opinions; the
        symmetric dogmatic support and dogmatic attack cancel into
        uncertainty.
        """
        graph = BipolarOpinionGraph(
            arguments=frozenset({"m", "s", "o"}),
            intrinsic={
                "m": Opinion.vacuous(_TAU),
                "s": Opinion.dogmatic_true(0.5),
                "o": Opinion.dogmatic_true(0.5),
            },
            supports=frozenset({("s", "m")}),
            attacks=frozenset({("o", "m")}),
            edge_opinions={
                ("s", "m"): Opinion.dogmatic_true(0.5),
                ("o", "m"): Opinion.dogmatic_true(0.5),
            },
        )
        result = evaluate(graph)
        assert result["m"].u == approx(1.0)
        assert result["m"].expectation() == approx(_TAU)


# ════════════════════════════════════════════════════════════════════
# Determinism — Gate A §3 (sorted ready set)
# ════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Gate A §3: ``evaluate`` is deterministic.

    Kahn's algorithm with a sorted (smallest-name-first) ready set makes the
    output bit-reproducible and independent of ``frozenset`` / ``Mapping``
    iteration order. CCF is float arithmetic — a fixed traversal order keeps
    the exact-value tests stable.
    """

    @staticmethod
    def _example_graph() -> BipolarOpinionGraph:
        """The Gate A2 worked-example graph (§4, §6.B) — ``s`` and ``o`` are
        leaves carrying genuine intrinsic opinions."""
        return BipolarOpinionGraph(
            arguments=frozenset({"m", "s", "o"}),
            intrinsic={
                "m": Opinion.vacuous(_TAU),
                "s": Opinion(0.7, 0.1, 0.2, 0.6),
                "o": Opinion(0.4, 0.3, 0.3, 0.5),
            },
            supports=frozenset({("s", "m")}),
            attacks=frozenset({("o", "m")}),
            edge_opinions={
                ("s", "m"): Opinion.dogmatic_true(0.5),
                ("o", "m"): Opinion.dogmatic_true(0.5),
            },
        )

    @pytest.mark.unit
    def test_evaluate_is_repeatable(self):
        """Two ``evaluate`` calls on the same graph give identical results."""
        graph = self._example_graph()
        first = evaluate(graph)
        second = evaluate(graph)
        assert set(first) == set(second)
        for name in first:
            assert first[name] == second[name], f"{name} differs across runs"

    @pytest.mark.unit
    def test_evaluate_is_independent_of_input_ordering(self):
        """Equal graphs built with differently-ordered inputs evaluate equal.

        ``frozenset`` and ``dict`` carry no semantic order, but to lock the
        intent: two ``BipolarOpinionGraph`` instances whose edge sets and
        intrinsic dicts were assembled in different orders must produce
        byte-identical results.
        """
        intrinsic_ops = {
            "m": Opinion.vacuous(_TAU),
            "s": Opinion(0.7, 0.1, 0.2, 0.6),
            "o": Opinion(0.4, 0.3, 0.3, 0.5),
        }
        edge_ops = {
            ("s", "m"): Opinion.dogmatic_true(0.5),
            ("o", "m"): Opinion.dogmatic_true(0.5),
        }

        def build(intrinsic_order, support_order):
            intrinsic = {k: v for k, v in intrinsic_order}
            return BipolarOpinionGraph(
                arguments=frozenset({"m", "s", "o"}),
                intrinsic=intrinsic,
                supports=frozenset(support_order),
                attacks=frozenset({("o", "m")}),
                edge_opinions=dict(edge_ops),
            )

        intrinsic_forward = [
            ("m", intrinsic_ops["m"]),
            ("s", intrinsic_ops["s"]),
            ("o", intrinsic_ops["o"]),
        ]
        intrinsic_reverse = list(reversed(intrinsic_forward))
        supports_forward = [("s", "m")]
        supports_reverse = list(reversed(supports_forward))
        g1 = build(intrinsic_forward, supports_forward)
        g2 = build(intrinsic_reverse, supports_reverse)
        r1 = evaluate(g1)
        r2 = evaluate(g2)
        for name in r1:
            assert r1[name] == r2[name], (
                f"{name} differs across input orderings"
            )

    @pytest.mark.unit
    def test_worked_example_deterministic_exact(self):
        """The worked example evaluates to the locked value on every run.

        Gate A2 §6.B: ``m`` components ``== (0.516, 0.208, 0.276, 0.55)``
        over 5 runs.
        """
        graph = self._example_graph()
        for _ in range(5):
            result = evaluate(graph)
            assert result["m"].b == pytest.approx(_OMEGA_M[0], abs=1e-4)
            assert result["m"].d == pytest.approx(_OMEGA_M[1], abs=1e-4)
            assert result["m"].u == pytest.approx(_OMEGA_M[2], abs=1e-4)
            assert result["m"].a == pytest.approx(_TAU, abs=1e-9)

    @pytest.mark.property
    @given(star_graphs())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_evaluate_deterministic_for_random_graphs(self, graph):
        """``evaluate`` is deterministic over arbitrary star graphs."""
        first = evaluate(graph)
        second = evaluate(graph)
        for name in first:
            assert first[name] == second[name]


# ════════════════════════════════════════════════════════════════════
# Single-source accrual — Gate A2 §2 step 4
# ════════════════════════════════════════════════════════════════════


class TestSingleSourceAccrual:
    """Gate A2 §2 step 4: a single evidence source is itself, re-stamped.

    With exactly one supporter (and no attacker) of a vacuous-intrinsic move
    node, the accrued opinion's ``(b, d, u)`` equals the discounted
    supporter's, with ``a`` re-stamped to ``tau``.
    """

    @pytest.mark.property
    @given(
        edge=valid_opinions(min_uncertainty=0.02),
        intrinsic=valid_opinions(min_uncertainty=0.02),
        tau=base_rates_strategy(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_single_supporter_restamps_base_rate(self, edge, intrinsic, tau):
        """One supporter leaf through one edge → ``omega_m.a == tau`` exactly.

        The supporter ``s`` is a leaf carrying a genuine intrinsic opinion;
        ``edge.discount(intrinsic)`` is the single accrual source for the
        vacuous-intrinsic move ``m``. Whatever its ``(b, d, u)``, the
        accrued ``a`` must be ``tau`` (re-stamped), never the discounted
        source's ``a``.
        """
        graph = BipolarOpinionGraph(
            arguments=frozenset({"m", "s"}),
            intrinsic={
                "m": Opinion.vacuous(tau),
                "s": intrinsic,
            },
            supports=frozenset({("s", "m")}),
            attacks=frozenset(),
            edge_opinions={("s", "m"): edge},
        )
        result = evaluate(graph)
        assert result["m"].a == pytest.approx(tau, abs=1e-9)

    @pytest.mark.unit
    def test_single_objection_only_lowers_or_holds_expectation(self):
        """One objection alone (supporter absent) does not raise ``E``.

        A lone negated objection as the single source cannot push ``E``
        above ``tau``. With an objection leaf carrying a genuine intrinsic
        opinion, the negated objection holds ``E`` at or below ``tau``.
        """
        graph = BipolarOpinionGraph(
            arguments=frozenset({"m", "o"}),
            intrinsic={
                "m": Opinion.vacuous(_TAU),
                "o": Opinion(0.7, 0.1, 0.2, 0.5),
            },
            supports=frozenset(),
            attacks=frozenset({("o", "m")}),
            edge_opinions={("o", "m"): _full_trust_edge()},
        )
        result = evaluate(graph)
        assert result["m"].expectation() <= _TAU + 1e-9
