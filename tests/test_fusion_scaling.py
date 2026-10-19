from __future__ import annotations

import math

import pytest

from doxa import Opinion
from doxa.argumentation import BipolarOpinionGraph, evaluate


def _pool(copies: int) -> list[Opinion]:
    return [Opinion(.6, .3, .1, .5)] * copies + [
        Opinion(.4, .3, .3, .5)
    ] * copies


@pytest.mark.parametrize("copies", [1, 6, 8, 200])
def test_wbf_preserves_the_answer_when_the_profile_is_replicated(copies: int) -> None:
    fused = Opinion.wbf(*_pool(copies))

    assert (fused.b, fused.d, fused.u, fused.a) == pytest.approx(
        (19 / 34, .3, 12 / 85, .5)
    )


@pytest.mark.parametrize("copies", [6, 7, 8, 200])
def test_ccf_normalizes_positive_compromise_mass_at_any_scale(copies: int) -> None:
    pool = _pool(copies)
    fused = Opinion.ccf(*pool)
    # Every residual disbelief is zero. All compromise mass therefore goes
    # to belief, leaving only the product of uncertainties outside consensus.
    uncertainty = math.prod(opinion.u for opinion in pool)

    assert fused.b == pytest.approx(.7 - uncertainty)
    assert fused.d == pytest.approx(.3)
    assert fused.u == pytest.approx(uncertainty, rel=1e-10, abs=1e-300)


@pytest.mark.parametrize("copies", [8, 200])
def test_auto_retains_wbf_for_large_nondogmatic_profiles(copies: int) -> None:
    fused = Opinion.fuse(*_pool(copies))

    assert (fused.b, fused.d, fused.u) == pytest.approx((19 / 34, .3, 12 / 85))


@pytest.mark.parametrize("copies", [8, 300])
def test_ccf_preserves_ratios_when_all_compromise_products_are_tiny(copies: int) -> None:
    pool = [Opinion(.6, .2, .2, .5)] * copies + [
        Opinion(.4, .4, .2, .5)
    ] * copies
    fused = Opinion.ccf(*pool)
    uncertainty = math.prod(opinion.u for opinion in pool)
    remaining = .4 - uncertainty
    # The common product cancels to compromise weights copies,copies,1.
    denominator = 2 * copies + 1

    assert (fused.b, fused.d, fused.u) == pytest.approx(
        (
            .4 + remaining * copies / denominator,
            .2 + remaining * copies / denominator,
            uncertainty + remaining / denominator,
        )
    )


def test_graph_accrual_normalizes_a_fourteen_parent_compromise() -> None:
    pool = _pool(7)
    leaves = tuple(str(index) for index in range(len(pool)))
    edges = frozenset((leaf, "target") for leaf in leaves)
    graph = BipolarOpinionGraph(
        arguments=frozenset((*leaves, "target")),
        intrinsic={
            **dict(zip(leaves, pool)),
            "target": Opinion.vacuous(.5),
        },
        supports=edges,
        attacks=frozenset(),
        edge_opinions={edge: Opinion.dogmatic_true(.5) for edge in edges},
    )

    fused = evaluate(graph)["target"]

    assert fused.b == pytest.approx(.69999999997813)
    assert fused.d == pytest.approx(.3)
    assert fused.u == pytest.approx(2.187e-11, rel=1e-10, abs=1e-300)
