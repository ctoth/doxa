"""Subjective Logic opinion module — Jøsang 2001.

Implements Opinion and BetaEvidence dataclasses with operators:
negation, conjunction, disjunction, consensus, discounting.

This is a pure kernel with ZERO external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Prior weight for non-informative prior (Jøsang p.20)
W = 2

_TOL = 1e-9

# Clamp range for fused base rates in the kernel-specific operators.
#
# WBF no longer uses this clamp: van der Heijden et al. 2018 Definition
# 4 (p.5) already computes a confidence-weighted base rate from valid
# input base rates, so the result stays inside the open interval. CCF
# still uses the clamp because its base-rate handling is a kernel
# convention layered over the paper's binomial reduction.
_BASE_RATE_CLAMP = (0.01, 0.99)


def _clamp_base_rate(a: float) -> float:
    """Clamp a fused base rate to ``_BASE_RATE_CLAMP``."""
    lo, hi = _BASE_RATE_CLAMP
    return max(lo, min(hi, a))


@dataclass(frozen=True)
class Opinion:
    """A subjective opinion ω = (b, d, u, a).

    b: belief, d: disbelief, u: uncertainty, a: base rate (atomicity).
    Constraint: b + d + u ≈ 1.0, all in [0,1], a in (0,1).
    """

    b: float
    d: float
    u: float
    a: float
    allow_dogmatic: bool = False

    def __post_init__(self) -> None:
        for name, val in [("b", self.b), ("d", self.d), ("u", self.u)]:
            if val < -_TOL or val > 1.0 + _TOL:
                raise ValueError(f"{name}={val} not in [0, 1]")
        if self.a <= 0.0 or self.a >= 1.0:
            raise ValueError(f"a={self.a} not in (0, 1)")
        total = self.b + self.d + self.u
        if abs(total - 1.0) > _TOL:
            raise ValueError(
                f"b + d + u = {total}, expected 1.0"
            )
        if self.u < _TOL and not self.allow_dogmatic:
            raise ValueError("dogmatic opinions require allow_dogmatic=True")

    # --- Aliases for readability ---

    @property
    def uncertainty(self) -> float:
        """Alias for ``u`` (uncertainty component)."""
        return self.u

    @property
    def base_rate(self) -> float:
        """Alias for ``a`` (base rate / atomicity component)."""
        return self.a

    # --- Special constructors ---

    @classmethod
    def vacuous(cls, a: float) -> Opinion:
        """Total ignorance."""
        return cls(0.0, 0.0, 1.0, a)

    @classmethod
    def dogmatic_true(cls, a: float) -> Opinion:
        """Absolute belief."""
        return cls(
            1.0,
            0.0,
            0.0,
            a,
            allow_dogmatic=True,  # tautology citation: Josang 2001 dogmatic opinion has u=0.
        )

    @classmethod
    def dogmatic_false(cls, a: float) -> Opinion:
        """Absolute disbelief."""
        return cls(
            0.0,
            1.0,
            0.0,
            a,
            allow_dogmatic=True,  # tautology citation: Josang 2001 dogmatic opinion has u=0.
        )

    @classmethod
    def from_evidence(cls, r: float, s: float, a: float) -> Opinion:
        """Create opinion from evidence counts."""
        return BetaEvidence(r, s, a).to_opinion()

    @classmethod
    def from_probability(cls, p: float, n: float, a: float) -> Opinion:
        """Create opinion from calibrated probability p with effective sample size n.

        r = p * n, s = (1-p) * n.
        """
        return cls.from_evidence(p * n, (1.0 - p) * n, a)

    # --- Core methods ---

    def expectation(self) -> float:
        """E(ω) = b + a * u  (Jøsang Def 6, p.5)."""
        return self.b + self.a * self.u

    def uncertainty_interval(self) -> tuple[float, float]:
        """[Bel, Pl] = (b, 1 - d)."""
        return (self.b, 1.0 - self.d)

    # --- Conversion ---

    def to_beta_evidence(self) -> BetaEvidence:
        """Opinion -> BetaEvidence (Jøsang Def 12, p.20-21).

        Raises ValueError for dogmatic opinions (u == 0).
        """
        if self.u < _TOL:
            raise ValueError("Cannot convert dogmatic opinion (u≈0) to BetaEvidence")
        r = W * self.b / self.u
        s = W * self.d / self.u
        return BetaEvidence(r, s, self.a)

    # --- Operators ---

    def __invert__(self) -> Opinion:
        """Negation: ~ω = Opinion(d, b, u, 1 - a)  (Jøsang Theorem 6, p.18)."""
        return Opinion(
            self.d,
            self.b,
            self.u,
            1.0 - self.a,
            allow_dogmatic=self.allow_dogmatic,
        )

    def conjunction(self, other: Opinion) -> Opinion:
        """Subjective-logic conjunction (Jøsang 2001 Theorem 3, p.14).

        Binomial (binary frame) conjunction of two opinions on
        independent propositions. The base-rate formula ``a1 * a2``
        assumes binary frames; generalizations to k-nomial frames
        require the hyper-opinion machinery from van der Heijden 2018
        and are not implemented here.

        Prefer this explicit method over ``&`` in callers that might
        otherwise be confused with Python's ``and`` keyword — ``op1
        and op2`` short-circuits on truthiness and does NOT call
        ``__and__`` (see ``__bool__``).
        """
        b = self.b * other.b
        d = self.d + other.d - self.d * other.d
        u = self.b * other.u + self.u * other.b + self.u * other.u
        a = self.a * other.a
        return Opinion(
            b,
            d,
            u,
            a,
            allow_dogmatic=u < _TOL,  # tautology citation: Josang 2001 dogmatic opinion has u=0.
        )

    def disjunction(self, other: Opinion) -> Opinion:
        """Subjective-logic disjunction (Jøsang 2001 Theorem 4, p.14-15).

        Binomial (binary frame) disjunction of two opinions on
        independent propositions. The base-rate formula
        ``a1 + a2 - a1*a2`` is the probabilistic OR of independent
        events and is correct only for binary frames; non-binary
        frames require the generalized fusion in van der Heijden 2018
        and are not implemented here.

        Prefer this explicit method over ``|`` in callers that might
        otherwise be confused with Python's ``or`` keyword.
        """
        b = self.b + other.b - self.b * other.b
        d = self.d * other.d
        u = self.d * other.u + self.u * other.d + self.u * other.u
        a = self.a + other.a - self.a * other.a
        return Opinion(
            b,
            d,
            u,
            a,
            allow_dogmatic=u < _TOL,  # tautology citation: Josang 2001 dogmatic opinion has u=0.
        )

    def __and__(self, other: Opinion) -> Opinion:
        """Alias for :meth:`conjunction`."""
        return self.conjunction(other)

    def __or__(self, other: Opinion) -> Opinion:
        """Alias for :meth:`disjunction`."""
        return self.disjunction(other)

    def __bool__(self) -> bool:
        """Opinions are not truthy — always raises ``TypeError``.

        Python's ``and``/``or`` keywords short-circuit on truthiness
        and never dispatch to ``__and__``/``__or__`` (review-2026-04-14).
        Without this guard, ``if op1 and op2`` silently produces a
        truthy Opinion rather than the subjective-logic conjunction,
        and ``if opinion`` is indistinguishable from ``if opinion is
        not None``. Force callers to be explicit.
        """
        raise TypeError(
            "Opinion is not truthy; use `op is None` / `op is not None` for "
            "presence checks, `op.conjunction(other)` / `op & other` for "
            "subjective-logic conjunction, or compare `op.expectation()` "
            "for probability-like checks."
        )

    def _quantized(self) -> tuple[int, int, int, int]:
        """Project (b, d, u, a) onto the shared ``_TOL`` grid.

        Both ``__eq__`` and ``__hash__`` must use this single quantization
        so the hash contract holds (review-2026-04-14): the older
        tolerance-based ``__eq__`` paired with ``round(·, 8)`` hashing
        could silently disagree on opinions straddling a rounding
        boundary, corrupting dicts and sets.
        """
        return (
            round(self.b / _TOL),
            round(self.d / _TOL),
            round(self.u / _TOL),
            round(self.a / _TOL),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Opinion):
            return NotImplemented
        return self._quantized() == other._quantized()

    def __hash__(self) -> int:
        return hash(self._quantized())

    # --- Ordering (Jøsang 2001 Def 10, p.9) ---

    def _ordering_key(self) -> tuple[float, float, float]:
        """Return (E(x), -u, -a) for total ordering.

        Jøsang 2001 Def 10: compare by E(x) ascending, then by u
        descending (less uncertainty is greater), then by a descending
        (less base rate is greater).
        """
        return (self.expectation(), -self.u, -self.a)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Opinion):
            return NotImplemented
        return self._ordering_key() < other._ordering_key()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Opinion):
            return NotImplemented
        return self._ordering_key() <= other._ordering_key()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Opinion):
            return NotImplemented
        return self._ordering_key() > other._ordering_key()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Opinion):
            return NotImplemented
        return self._ordering_key() >= other._ordering_key()

    # --- Uncertainty Maximization (Jøsang 2001 Def 16, p.30) ---

    def maximize_uncertainty(self) -> Opinion:
        """Maximize u while preserving E(x) = b + a*u.

        Jøsang 2001 p.30: for non-repeatable events, redistribute
        belief mass to maximize uncertainty. Constraints:
          b + d + u = 1, b >= 0, d >= 0, u >= 0
          b + a*u = E (constant)

        u_max = min(E/a, (1-E)/(1-a))

        Precondition: ``0 < a < 1`` (enforced in ``__post_init__``)
        — the divisions by ``a`` and ``1 - a`` below depend on this
        strict bound. Relaxing the constructor's check would require
        an explicit guard here (review-2026-04-14 Issue 7).
        """
        e = self.expectation()
        a = self.a

        # Upper bounds on u from b >= 0 and d >= 0.
        # Safe because __post_init__ enforces 0 < a < 1 strictly.
        u_from_b = e / a            # b = E - a*u >= 0
        u_from_d = (1.0 - e) / (1.0 - a)  # d = 1 - E - u*(1-a) >= 0

        u_max = min(u_from_b, u_from_d)
        # Clamp to [0, 1] for numerical safety
        u_max = max(0.0, min(1.0, u_max))

        b_new = e - a * u_max
        d_new = 1.0 - u_max - b_new

        # Clamp negatives from float drift
        b_new = max(0.0, b_new)
        d_new = max(0.0, d_new)

        return Opinion(
            b_new,
            d_new,
            u_max,
            a,
            allow_dogmatic=u_max < _TOL,  # tautology citation: Josang 2001 dogmatic opinion has u=0.
        )

    # --- Consensus / discounting (Jøsang 2001) ---

    def consensus_pair(self, other: Opinion) -> Opinion:
        """Consensus of two opinions (Jøsang Theorem 7, p.25).

        Base-rate fusion uses the cancellation-free form
        ``a_b·u_a·v_b + a_a·u_b·v_a  /  u_a·v_b + u_b·v_a``
        where ``v = 1 - u`` (review-2026-04-14 Issue 3). The obvious form
        ``u_a + u_b - 2·u_a·u_b`` suffers catastrophic cancellation when
        both ``u`` values approach 1, causing the fused base rate to
        drift tens of ULPs away from the analytically exact answer.
        """
        kappa = self.u + other.u - self.u * other.u
        if abs(kappa) < _TOL:
            raise ValueError("Cannot fuse two dogmatic opinions (κ ≈ 0)")

        b = (self.b * other.u + other.b * self.u) / kappa
        d = (self.d * other.u + other.d * self.u) / kappa
        u = (self.u * other.u) / kappa

        # Base rate fusion — confidence-weighted average, written in the
        # cancellation-free v = 1 - u form so near-vacuous pairs stay stable.
        v_a = 1.0 - self.u
        v_b = 1.0 - other.u
        denom_a = self.u * v_b + other.u * v_a
        if denom_a < _TOL:
            # Both sources are (near-)vacuous → zero confidence in either
            # base rate. Fall back to a plain average; this is the
            # degenerate limit of the confidence-weighted form.
            a = (self.a + other.a) / 2.0
        else:
            a = (other.a * self.u * v_b + self.a * other.u * v_a) / denom_a

        return Opinion(
            b,
            d,
            u,
            a,
            allow_dogmatic=u < _TOL,  # tautology citation: Josang 2001 dogmatic opinion has u=0.
        )

    @classmethod
    def consensus(cls, *opinions: Opinion) -> Opinion:
        """Fold multiple opinions via pairwise consensus (associative)."""
        if len(opinions) == 0:
            raise ValueError("Need at least one opinion")
        result = opinions[0]
        for op in opinions[1:]:
            result = result.consensus_pair(op)
        return result

    def discount(self, source: Opinion) -> Opinion:
        """Trust discounting (Jøsang Def 14, p.24).

        ``self`` is the trust opinion; ``source`` is the opinion being
        discounted. Jøsang's explicit form::

            b = trust.b * source.b
            d = trust.b * source.d
            u = 1 - b - d
              = 1 - trust.b * (source.b + source.d)
              = 1 - trust.b * (1 - source.u)
              = trust.d + trust.u + trust.b * source.u   (since
                                                           trust.b + trust.d
                                                           + trust.u = 1)

        We write the expanded form directly so the identity is visible and
        we don't have to reconstruct u from (1 - b - d) at runtime. The
        derivation only holds because trust is a well-formed Opinion
        (b + d + u = 1).
        """
        trust = self
        b = trust.b * source.b
        d = trust.b * source.d
        u = trust.d + trust.u + trust.b * source.u
        a = source.a
        return Opinion(
            b,
            d,
            u,
            a,
            allow_dogmatic=u < _TOL,  # tautology citation: Josang 2001 dogmatic opinion has u=0.
        )

    # --- N-source fusion (van der Heijden et al. 2018) ---

    @classmethod
    def wbf(cls, *opinions: Opinion) -> Opinion:
        """N-source Weighted Belief Fusion.

        Implements van der Heijden et al. 2018 Definition 4 (p.5). Table I
        on p.8 gives the concrete three-source regression example used by
        the tests.

        Definition 4 has separate cases for dogmatic and no-evidence inputs:
        dogmatic sources dominate finite-evidence sources, while all-vacuous
        inputs remain vacuous.
        """
        if len(opinions) == 0:
            raise ValueError("Need at least one opinion")
        if len(opinions) == 1:
            return opinions[0]

        N = len(opinions)
        dogmatic = [op for op in opinions if op.u < _TOL]
        if dogmatic:
            # van der Heijden et al. 2018 Definition 4 Case 2 / Remark 2
            # (p.5): finite-evidence opinions have negligible weight when
            # dogmatic sources are present. This unweighted API represents
            # the equal-relative-infinity limit for multiple dogmatic inputs.
            count = len(dogmatic)
            b_fused = sum(op.b for op in dogmatic) / count
            d_fused = sum(op.d for op in dogmatic) / count
            a_fused = sum(op.a for op in dogmatic) / count
            return Opinion(
                b_fused,
                d_fused,
                0.0,
                a_fused,
                allow_dogmatic=True,  # tautology citation: Josang 2001 dogmatic opinion has u=0.
            )

        if all(op.u > 1.0 - _TOL for op in opinions):
            return Opinion(
                0.0,
                0.0,
                1.0,
                sum(op.a for op in opinions) / N,
            )

        # Definition 4 Case 1. For each source, the belief/disbelief term is
        # weighted by (1-u_i) times the product of every other uncertainty.
        total_prod = math.prod(op.u for op in opinions)
        prod_except = [total_prod / op.u for op in opinions]
        weighted_terms = [
            (1.0 - op.u) * prod_except[i]
            for i, op in enumerate(opinions)
        ]

        num_b = sum(
            op.b * weighted_terms[i]
            for i, op in enumerate(opinions)
        )
        num_d = sum(
            op.d * weighted_terms[i]
            for i, op in enumerate(opinions)
        )
        num_u = sum(1.0 - op.u for op in opinions) * total_prod

        kappa = sum(weighted_terms)
        if abs(kappa) < _TOL:
            raise ValueError("WBF denominator κ ≈ 0")

        b_fused = num_b / kappa
        d_fused = num_d / kappa
        u_fused = num_u / kappa

        # Clamp for float drift
        b_fused = max(0.0, b_fused)
        d_fused = max(0.0, d_fused)
        u_fused = max(0.0, u_fused)

        # Definition 4 base-rate equation (p.5): confidence-weighted by
        # (1-u_i), not by the full belief/disbelief denominator above.
        weights = [1.0 - op.u for op in opinions]
        total_weight = sum(weights)
        a_fused = sum(
            op.a * w for op, w in zip(opinions, weights)
        ) / total_weight

        if abs((b_fused + d_fused + u_fused) - 1.0) <= 1e-8:
            u_fused = max(0.0, 1.0 - b_fused - d_fused)

        return Opinion(
            b_fused,
            d_fused,
            u_fused,
            a_fused,
            allow_dogmatic=u_fused < _TOL,  # tautology citation: Josang 2001 dogmatic opinion has u=0.
        )

    @classmethod
    def ccf(cls, *opinions: Opinion) -> Opinion:
        """Consensus & Compromise Fusion (van der Heijden 2018, Definition 5).

        Multi-source fusion that handles both dogmatic and non-dogmatic
        inputs uniformly via the three-phase consensus/compromise/normalize
        algorithm. Disagreement between sources is converted into
        uncertainty (not fractional belief) — two dogmatic sources that
        completely disagree fuse to a vacuous opinion, reflecting that
        the kernel honestly does not know which source to trust.

        CCF is NOT associative in general (§I of the paper) — use the
        direct N-source call form ``ccf(a, b, c, ...)`` rather than
        pairwise folding. The result IS symmetric in its arguments.

        CCF is distinct from WBF: for non-dogmatic inputs both operators
        are well-defined but produce different results (see Table I of
        the paper). ``fuse(method="auto")`` prefers WBF for non-dogmatic
        inputs and falls back to CCF on dogmatic ones; call ``ccf()``
        directly when you want Def 5 semantics regardless of input.
        """
        if len(opinions) == 0:
            raise ValueError("Need at least one opinion")
        if len(opinions) == 1:
            return opinions[0]
        return _ccf_binomial(list(opinions))

    @classmethod
    def fuse(cls, *opinions: Opinion, method: str = "auto") -> Opinion:
        """Dispatch to WBF or CCF.

        Parameters
        ----------
        method : str
            "wbf" — always WBF (raises on dogmatic inputs)
            "ccf" — always CCF
            "auto" — try WBF first; fall back to CCF on dogmatic inputs
        """
        if method == "wbf":
            return cls.wbf(*opinions)
        elif method == "ccf":
            return cls.ccf(*opinions)
        elif method == "auto":
            try:
                return cls.wbf(*opinions)
            except ValueError:
                return cls.ccf(*opinions)
        else:
            raise ValueError(f"Unknown fusion method: {method!r}")


@dataclass(frozen=True)
class BetaEvidence:
    """Beta evidence representation: r positive, s negative evidence counts.

    r >= 0, s >= 0, a in (0, 1).
    """

    r: float
    s: float
    a: float

    def __post_init__(self) -> None:
        if self.r < 0:
            raise ValueError(f"r={self.r} must be >= 0")
        if self.s < 0:
            raise ValueError(f"s={self.s} must be >= 0")
        if self.a <= 0.0 or self.a >= 1.0:
            raise ValueError(f"a={self.a} not in (0, 1)")

    def to_opinion(self) -> Opinion:
        """BetaEvidence -> Opinion."""
        denom = self.r + self.s + W
        b = self.r / denom
        d = self.s / denom
        u = W / denom
        return Opinion(b, d, u, self.a)


def _ccf_binomial(opinions: list[Opinion]) -> Opinion:
    """Direct N-source binomial reduction of van der Heijden 2018 Def 5.

    For a binomial frame ``X = {x, ¬x}`` with singleton focal elements
    ``{x}``, ``{¬x}``, and whole-frame mass ``u``, the three-phase
    algorithm specializes to the following closed-form expressions (the
    multi-valued compromise terms 3 and 4 of Eq. 7 collapse because
    ``{x} ∩ {¬x} = ∅`` and ``{x} ∪ {¬x} = X``, so most focal-element
    combinations contribute zero):

    **Step 1 — Consensus (Eq. 6):**
      ``b^cons(x) = min_i b_i``, ``b^cons(¬x) = min_i d_i``
      ``r_i(x)    = b_i − b^cons(x)``, ``r_i(¬x) = d_i − b^cons(¬x)``

    **Step 2 — Compromise (Eq. 7 reduced to binomial):**
      ``b^comp(x)  = Σ_i r_i(x)·Π_{j≠i} u_j  +  Π_i r_i(x)``
      ``b^comp(¬x) = Σ_i r_i(¬x)·Π_{j≠i} u_j +  Π_i r_i(¬x)``
      ``b^comp(X)  = Π_i(r_i(x)+r_i(¬x)) − Π_i r_i(x) − Π_i r_i(¬x)``
      ``u^pre      = Π_i u_i``

    The ``b^comp(X)`` term captures all cross-actor disagreement
    (tuples where union=X and intersection=∅). It is the
    inclusion-exclusion form of "total residual mass minus the
    all-agreeing tuples."

    **Step 3 — Normalization (Eqs. 10-12):**
      ``η = (1 − b^cons_sum − u^pre) / b^comp_sum``
      ``u^fused      = u^pre + η · b^comp(X)``     # disagreement → uncertainty
      ``b^fused(x)   = b^cons(x) + η · b^comp(x)``
      ``b^fused(¬x)  = b^cons(¬x) + η · b^comp(¬x)``

    **Edge case — ``b^comp_sum ≈ 0``**: Remark 4 in the paper assumes
    ``b^comp_sum > 0``. This assumption fails for self-fusion
    (identical inputs → all residuals zero) and for full-agreement
    fusion. We handle it by putting the residual missing mass
    ``1 − b^cons_sum − u^pre`` directly into ``u``, which makes
    self-fusion idempotent: ``ccf(op, op) == op`` for every ``op``.

    **Base rate**: Def 5 takes ``a_X`` as a single shared input across
    all actors — the paper treats base rate as a precondition, not a
    fused field. The kernel routinely fuses opinions with
    calibration-derived ``a`` values that differ per source, so we
    compute a confidence-weighted average of base rates (analogous to
    WBF Case 1 in the paper) and clamp to ``_BASE_RATE_CLAMP``. This
    is a kernel convention layered on top of Def 5, not part of
    the paper's algorithm.

    Verified against Table I of the paper: given the three-source
    example A₁=(0.10, 0.30, 0.60), A₂=(0.40, 0.20, 0.40),
    A₃=(0.70, 0.10, 0.20) with ``a=0.5``, this formula reproduces
    the paper's CCF column ``(0.629, 0.182, 0.189)`` exactly.
    """
    N = len(opinions)
    if N == 1:
        return opinions[0]

    # --- Step 1: Consensus extraction ---
    cons_b = min(op.b for op in opinions)
    cons_d = min(op.d for op in opinions)
    cons_sum = cons_b + cons_d

    # Residual masses per actor per focal element.
    res_b = [op.b - cons_b for op in opinions]
    res_d = [op.d - cons_d for op in opinions]

    # --- Step 2: Compromise phase ---
    # Product of all uncertainties (pre-uncertainty mass).
    u_pre = math.prod(op.u for op in opinions)

    # Term 1 of Eq. 7 (residue weighted by other actors' u) plus term
    # 2 for x ∈ R(X) (joint commitment to the same singleton). For
    # binomial singletons, terms 3 and 4 of Eq. 7 contribute zero
    # when computing b^comp({x}) or b^comp({¬x}).
    sum_b_res_weighted = sum(
        res_b[i] * math.prod(opinions[j].u for j in range(N) if j != i)
        for i in range(N)
    )
    sum_d_res_weighted = sum(
        res_d[i] * math.prod(opinions[j].u for j in range(N) if j != i)
        for i in range(N)
    )
    comp_b = sum_b_res_weighted + math.prod(res_b)
    comp_d = sum_d_res_weighted + math.prod(res_d)

    # b^comp(X) — disagreement term. Inclusion-exclusion form of
    # "all residual-mass tuples minus the all-{x} and all-{¬x} tuples."
    prod_res_total = math.prod(res_b[i] + res_d[i] for i in range(N))
    comp_X = prod_res_total - math.prod(res_b) - math.prod(res_d)

    comp_sum = comp_b + comp_d + comp_X

    # --- Step 3: Normalization ---
    # When comp_sum ≈ 0 (no compromise mass to distribute), put any
    # residual missing mass directly into u. This handles self-fusion
    # and full-agreement edge cases that Def 5 Remark 4 doesn't cover.
    if comp_sum < _TOL:
        b_fused = cons_b
        d_fused = cons_d
        u_fused = max(0.0, 1.0 - cons_sum)
    else:
        eta = (1.0 - cons_sum - u_pre) / comp_sum
        u_fused = u_pre + eta * comp_X
        b_fused = cons_b + eta * comp_b
        d_fused = cons_d + eta * comp_d

    # Clamp tiny float drift from the constructor.
    b_fused = max(0.0, b_fused)
    d_fused = max(0.0, d_fused)
    u_fused = max(0.0, u_fused)

    # Base-rate fusion: kernel convention (see docstring). Use
    # confidence-weighted averaging across sources; fall back to
    # plain averaging when all sources are fully vacuous.
    weights = [1.0 - op.u for op in opinions]
    total_weight = sum(weights)
    if total_weight < _TOL:
        a_fused = sum(op.a for op in opinions) / N
    else:
        a_fused = sum(op.a * w for op, w in zip(opinions, weights)) / total_weight
    a_fused = _clamp_base_rate(a_fused)

    return Opinion(
        b_fused,
        d_fused,
        u_fused,
        a_fused,
        allow_dogmatic=u_fused < _TOL,  # tautology citation: Josang 2001 dogmatic opinion has u=0.
    )
