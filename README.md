# doxa

A pure, dependency-free Subjective Logic library for Python.

`doxa` implements Jøsang's Subjective Logic — an algebra of *opinions* over
Beta-distributed uncertain probabilities. An opinion separates belief,
disbelief, and explicit **uncertainty mass**, so a `doxa` program can say "I
don't know" honestly rather than fabricating a point probability.

The distribution name and the import package are both `doxa`.

## Status

Early scaffold. This release ships the package skeleton only — the `Opinion`
kernel and fusion operators arrive in a subsequent commit. The sections below
describe the intended surface; see the source for what is actually implemented.

## Scope

`doxa` is a *pure kernel*: zero runtime dependencies, no persistence, no
provenance, no framework coupling. It is meant to be usable by anyone who needs
an honest algebra of uncertain probability, with no knowledge of any larger
system.

The intended surface:

- **`Opinion`** — a binomial opinion `(b, d, u, a)` with `b + d + u == 1`,
  where `a` is the base rate. Includes the propositional operators (negation,
  conjunction, disjunction) and the projected probability expectation.
- **`BetaEvidence`** — the `(r, s, a)` evidence form, with the bijective
  mapping to and from `Opinion`.
- **Fusion operators** — multi-source belief fusion, including weighted belief
  fusion (WBF) and consensus & compromise fusion (CCF), plus trust discounting.

## Development

```powershell
uv sync
uv run pyright
uv run pytest -vv
```

## References

- Jøsang, A. (2001). *A Logic for Uncertain Probabilities.* International
  Journal of Uncertainty, Fuzziness and Knowledge-Based Systems. The original
  definition of subjective logic — opinion tuples, the Beta-distribution
  mapping, and the consensus and discounting operators.
- van der Heijden, R. W., Kopp, H., & Kargl, F. (2018). *Multi-Source Fusion
  Operations in Subjective Logic.* Corrected multi-source fusion operators,
  including N-source weighted belief fusion (WBF) and consensus & compromise
  fusion (CCF).

See [`papers/`](papers/) for the reference collection and extraction notes.
