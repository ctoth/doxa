# doxa

A pure, dependency-free Subjective Logic library for Python.

`doxa` implements Jøsang's Subjective Logic — an algebra of *opinions* over
Beta-distributed uncertain probabilities. An opinion separates belief,
disbelief, and an explicit **uncertainty mass**, so a `doxa` program can say "I
don't know" honestly instead of fabricating a point probability.

## What is Subjective Logic?

Subjective Logic (Jøsang 2001) extends probabilistic logic with a first-class
representation of *uncertainty about the probability itself*. A binomial
opinion is a tuple `ω = (b, d, u, a)`:

- `b` — belief: mass committed to the proposition being true
- `d` — disbelief: mass committed to it being false
- `u` — uncertainty: mass committed to *neither* — honest ignorance
- `a` — base rate (atomicity): the prior probability used to project `u`

with the constraint `b + d + u == 1` and `0 < a < 1`. The *projected
probability* (expectation) `E(ω) = b + a·u` collapses an opinion back to a
single number when one is needed, but the uncertainty mass is preserved
everywhere else. A vacuous opinion `(0, 0, 1, a)` represents total ignorance; a
dogmatic opinion (`u = 0`) is an ordinary probability.

`doxa` provides:

- **`Opinion`** — the binomial opinion type with the full Jøsang operator set:
  negation, conjunction, disjunction, consensus, trust discounting,
  uncertainty maximization, and a total ordering.
- **`BetaEvidence`** — the `(r, s, a)` evidence-count form, with the bijective
  mapping to and from `Opinion`.
- **Multi-source fusion** — N-source Weighted Belief Fusion (WBF) and Consensus
  & Compromise Fusion (CCF) from van der Heijden et al. 2018, plus a `fuse`
  dispatcher.
- **Opinion-valued argumentation** — `BipolarOpinionGraph` and `evaluate`:
  a bipolar (support/attack) argument graph whose edges carry opinions,
  evaluated bottom-up to a per-argument `Opinion`. Disagreement between
  arguments becomes honest uncertainty, not fake confidence.

## Theory and grounding

`doxa` is a direct implementation of two papers, both shipped under
[`papers/`](papers/) with extraction notes:

- **Jøsang, A. (2001).** *A Logic for Uncertain Probabilities.* International
  Journal of Uncertainty, Fuzziness and Knowledge-Based Systems. The original
  definition of subjective logic — opinion tuples, the Beta-distribution
  mapping, and the negation, conjunction, disjunction, consensus, and
  discounting operators.
- **van der Heijden, R. W., Kopp, H., & Kargl, F. (2018).** *Multi-Source
  Fusion Operations in Subjective Logic.* The corrected multi-source fusion
  operators — N-source Weighted Belief Fusion (WBF) and Consensus & Compromise
  Fusion (CCF). The kernel's fusion code is regression-tested against Table I
  of this paper.

Each operator's docstring cites the specific definition or theorem it
implements.

The `argumentation` module is a modest fusion of two mature research lines —
subjective-logic argumentation (which attaches opinions to arguments but
evaluates them with *crisp* semantics) and gradual QBAF semantics (which
propagates argument strength *gradually* but only as scalars) — propagating a
full opinion through a bipolar argument graph using only operators already in
the kernel. CCF (van der Heijden et al. 2018, Definition 5) is the accrual
operator; no new algebra is introduced.

## Install

`doxa` is not on PyPI. It installs straight from git, and requires **Python
>= 3.11** with **zero runtime dependencies**.

```sh
uv add git+https://github.com/ctoth/doxa
```

```sh
pip install git+https://github.com/ctoth/doxa
```

## Quick start

### Constructing opinions

```python
from doxa import Opinion, BetaEvidence

# An opinion is (belief, disbelief, uncertainty, base_rate); b + d + u == 1.
omega = Opinion(b=0.7, d=0.1, u=0.2, a=0.5)

# Named constructors:
ignorant = Opinion.vacuous(a=0.5)          # (0, 0, 1, a) — total ignorance
certain_yes = Opinion.dogmatic_true(a=0.5)  # (1, 0, 0, a) — absolute belief
certain_no = Opinion.dogmatic_false(a=0.5)  # (0, 1, 0, a) — absolute disbelief

# From observed evidence counts (8 positive, 2 negative):
from_obs = Opinion.from_evidence(r=8, s=2, a=0.5)

# From a calibrated probability with an effective sample size:
from_prob = Opinion.from_probability(p=0.8, n=10, a=0.5)
assert from_prob == from_obs  # p*n = 8, (1-p)*n = 2
```

### Projected probability

`expectation()` collapses an opinion to a single probability `E(ω) = b + a·u`:

```python
omega = Opinion(b=0.7, d=0.1, u=0.2, a=0.5)
omega.expectation()   # 0.8  (= 0.7 + 0.5 * 0.2)
```

### Propositional operators

```python
a = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
b = Opinion(b=0.3, d=0.5, u=0.2, a=0.5)

not_a = ~a                 # negation: swaps b/d, complements a
both = a.conjunction(b)    # conjunction; `a & b` is an alias
either = a.disjunction(b)  # disjunction; `a | b` is an alias
```

Prefer the named `conjunction` / `disjunction` methods over `&` / `|` when a
reader might confuse them with Python's `and` / `or` keywords — those keywords
short-circuit on truthiness and never call the operators. `Opinion` is
deliberately not truthy: `bool(opinion)` raises `TypeError`.

### Trust discounting

Discounting weakens a source's opinion by how much you trust that source:

```python
trust = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)   # how much we trust the source
claim = Opinion(b=0.9, d=0.0, u=0.1, a=0.5)   # what the source asserts

discounted = trust.discount(claim)            # the claim, seen through trust
```

### Multi-source fusion

When several sources offer opinions on the same proposition, fuse them:

```python
s1 = Opinion(b=0.10, d=0.30, u=0.60, a=0.5)
s2 = Opinion(b=0.40, d=0.20, u=0.40, a=0.5)
s3 = Opinion(b=0.70, d=0.10, u=0.20, a=0.5)

wbf_result = Opinion.wbf(s1, s2, s3)    # Weighted Belief Fusion
ccf_result = Opinion.ccf(s1, s2, s3)    # Consensus & Compromise Fusion
auto = Opinion.fuse(s1, s2, s3)         # picks WBF, falls back to CCF on dogmatic input
```

For the three sources above, `wbf` yields `(b, d, u) = (0.562, 0.146, 0.292)`
and `ccf` yields `(0.629, 0.182, 0.189)` — the WBF and CCF columns of Table I
in van der Heijden et al. 2018. The two operators are genuinely different: CCF
turns inter-source *disagreement* into uncertainty rather than fractional
belief.

The older pairwise `consensus` operator is also available:

```python
consensus_result = Opinion.consensus(s1, s2, s3)
```

### Beta evidence

`BetaEvidence` is the evidence-count form, bijective with non-dogmatic
opinions:

```python
evidence = BetaEvidence(r=8, s=2, a=0.5)   # 8 positive, 2 negative
opinion = evidence.to_opinion()
back = opinion.to_beta_evidence()           # round-trips to r=8, s=2
```

### Argumentation

A `BipolarOpinionGraph` is a bipolar argument graph: each argument carries a
base rate (`tau = a`, its intrinsic prior), and each support/attack edge
carries an `Opinion` (the edge's strength/trust). `evaluate` resolves it
bottom-up over the DAG to a per-argument `Opinion`:

```python
from doxa import BipolarOpinionGraph, Opinion, evaluate

# Move 'm' (tau = 0.55) with one supporter 's' and one objection 'o'.
# 's' and 'o' are forced dogmatic-true by dogmatic-true children, so each
# edge opinion below IS the discounted argument opinion that reaches 'm'.
graph = BipolarOpinionGraph(
    arguments=frozenset({"m", "s", "o", "gs", "go"}),
    base_rates={"m": 0.55, "s": 0.6, "o": 0.5, "gs": 0.5, "go": 0.5},
    supports=frozenset({("s", "m"), ("gs", "s"), ("go", "o")}),
    attacks=frozenset({("o", "m")}),
    edge_opinions={
        ("s", "m"): Opinion(0.7, 0.1, 0.2, 0.6),   # the supporter's strength
        ("o", "m"): Opinion(0.4, 0.3, 0.3, 0.5),   # the objection's strength
        ("gs", "s"): Opinion.dogmatic_true(0.5),
        ("go", "o"): Opinion.dogmatic_true(0.5),
    },
)

result = evaluate(graph)            # dict[str, Opinion], one entry per argument
omega_m = result["m"]               # Opinion(b≈0.516, d≈0.208, u≈0.276, a=0.55)
omega_m.expectation()               # ≈ 0.668  (the strong supporter pulls E above tau)
```

The strong supporter raises the move's projected strength above its base
rate, the weaker objection holds it down, and `u` stays substantial (≈ 0.276)
because the two arguments disagree — disagreement becomes honest uncertainty.
An unargued argument resolves to `Opinion.vacuous(tau)`, so `expectation()`
falls back to exactly `tau`. `evaluate` raises `CyclicGraphError` if the graph
contains a cycle.

## API overview

### `Opinion(b, d, u, a, allow_dogmatic=False)`

A frozen, hashable binomial opinion. The constructor enforces `b + d + u ≈ 1`,
all of `b, d, u` in `[0, 1]`, and `0 < a < 1`. Dogmatic opinions (`u == 0`)
must pass `allow_dogmatic=True`.

- Properties: `uncertainty` (alias for `u`), `base_rate` (alias for `a`).
- Constructors: `vacuous(a)`, `dogmatic_true(a)`, `dogmatic_false(a)`,
  `from_evidence(r, s, a)`, `from_probability(p, n, a)`.
- Core: `expectation()`, `uncertainty_interval()`, `to_beta_evidence()`,
  `maximize_uncertainty()`.
- Operators: `__invert__` (`~`), `conjunction` / `&`, `disjunction` / `|`,
  total ordering (`<`, `<=`, `>`, `>=`), `==` / `hash` (both quantize
  `b, d, u, a` onto a shared tolerance grid).
- Consensus & trust: `consensus_pair(other)`, `consensus(*opinions)`,
  `discount(source)`.
- Fusion: `wbf(*opinions)`, `ccf(*opinions)`, `fuse(*opinions, method="auto")`.

### `BetaEvidence(r, s, a)`

A frozen evidence-count record: `r >= 0` positive, `s >= 0` negative,
`0 < a < 1`. `to_opinion()` maps it to an `Opinion`. (Converting an `Opinion`
back uses `Opinion.to_beta_evidence()`, which raises for dogmatic opinions.)

### Argumentation

#### `BipolarOpinionGraph(arguments, base_rates, supports, attacks, edge_opinions)`

A frozen bipolar argument graph. Five required fields:

- `arguments: frozenset[str]` — the argument node identifiers.
- `base_rates: Mapping[str, float]` — `tau = a` per argument, each in `(0, 1)`.
- `supports: frozenset[tuple[str, str]]` — support edges `(supporter, target)`.
- `attacks: frozenset[tuple[str, str]]` — attack edges `(attacker, target)`.
- `edge_opinions: Mapping[tuple[str, str], Opinion]` — the per-edge
  strength/trust opinion for every edge.

Construction validates the graph (raising `ValueError`): `base_rates` covers
exactly `arguments`; every base rate in `(0, 1)`; support and attack edges
reference only declared arguments; `supports` and `attacks` are disjoint;
`edge_opinions` has exactly one opinion per edge; no self-loops. Acyclicity is
*not* checked at construction — it is checked by `evaluate`.

#### `evaluate(graph) -> dict[str, Opinion]`

Resolves every argument bottom-up over the DAG (Kahn's algorithm with a sorted
ready set — deterministic), returning a dict mapping each argument name to its
`Opinion`. Each argument's opinion is accrued from its discounted supporters
and negated discounted attackers with the CCF fusion operator, re-stamped with
the argument's own base rate. Raises `CyclicGraphError` (a `ValueError`
subclass) if the graph contains a cycle.

## Typing

`doxa` ships a `py.typed` marker, so type checkers consume its inline
annotations directly with no stubs.

## Development

```sh
uv sync
uv run pytest
uv run pyright
```
