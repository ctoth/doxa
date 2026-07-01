# Subjective Logic and Calibration

Scientific claims carry uncertainty. A point estimate like "confidence = 0.7" collapses the distinction between "I have strong evidence for 70%" and "I have no idea, but 70% is a reasonable default." propstore uses subjective logic (Josang 2001) to preserve this distinction: every claim carries a full opinion that separates belief, disbelief, and uncertainty. The system treats "I don't know" as a valid and important signal, not something to be papered over with a fabricated number.

## Opinions

### The Opinion type

An opinion is a tuple `(b, d, u, a)` where:

- **b** (belief) -- evidence-based support for the proposition
- **d** (disbelief) -- evidence-based opposition
- **u** (uncertainty) -- how much evidence is missing
- **a** (base rate) -- prior probability in the absence of evidence

The constraint `b + d + u = 1` is enforced at construction time (`opinion.py`, tolerance 1e-9). The base rate `a` lives in (0, 1) and must be supplied explicitly. When the system lacks a sourced or policy-selected prior for an assertion, the result is `BaseRateUnresolved`, not an `Opinion`.

The **expected probability** (Josang 2001 Def 6, p.5) combines belief with the uncertainty-weighted base rate:

```
E(w) = b + a * u
```

This is the single-number summary when you need one. A vacuous opinion `(0, 0, 1, a)` has expectation `a`; that `a` must come from a resolved base-rate assertion or an explicit policy. A dogmatic opinion `(1, 0, 0, a)` has expectation 1.0 -- certainty ignores the base rate entirely.

The **uncertainty interval** `[b, 1 - d]` = `[Bel, Pl]` gives the range of compatible probabilities (`opinion.py:76`).

### Special opinions

| Name | Tuple | Meaning | Constructor |
|------|-------|---------|-------------|
| Vacuous | `(0, 0, 1, a)` | Total ignorance (Josang 2001, p.8) | `Opinion.vacuous(a)` |
| Dogmatic true | `(1, 0, 0, a)` | Absolute belief | `Opinion.dogmatic_true(a)` |
| Dogmatic false | `(0, 1, 0, a)` | Absolute disbelief | `Opinion.dogmatic_false(a)` |

Vacuous opinions represent total uncertainty after a base rate has already been resolved. They are not a substitute for a missing prior. Missing calibration or missing source trust returns a typed unresolved result so the caller can surface the absence of a prior instead of inventing one.

### Operators

All operators are implemented on the `Opinion` dataclass (`opinion.py`) with citations to Josang 2001 and van der Heijden 2018.

| Operator | Syntax | Definition | Citation |
|----------|--------|------------|----------|
| Negation | `~w` | `Opinion(d, b, u, 1 - a)` | Josang Theorem 6, p.18 |
| Conjunction | `w1 & w2` | `b = b1*b2`, `d = d1+d2-d1*d2`, `u = b1*u2+u1*b2+u1*u2`, `a = a1*a2` | Josang Theorem 3, p.14 |
| Disjunction | `w1 \| w2` | `b = b1+b2-b1*b2`, `d = d1*d2`, `u = d1*u2+u1*d2+u1*u2`, `a = a1+a2-a1*a2` | Josang Theorem 4, p.14-15 |
| Consensus fusion | `consensus(w1, w2)` | Combines independent sources, reduces uncertainty | Josang Theorem 7, p.25 |
| Trust discounting | `discount(trust, source)` | Attenuates opinion by trustworthiness of its source | Josang Def 14, p.24 |
| Ordering | `w1 < w2` | Compare by `(E(w), -u, -a)` -- expectation first, then uncertainty descending, then base rate descending | Josang Def 10, p.9 |
| Uncertainty maximization | `w.maximize_uncertainty()` | Maximize `u` while preserving `E(w)`: `u_max = min(E/a, (1-E)/(1-a))` | Josang Def 16, p.30 |
| Weighted Belief Fusion | `wbf(w1, w2, ...)` | N-source generalization of consensus. Raises on dogmatic inputs. | van der Heijden 2018, Def 4 |
| Cumulative & Compromise Fusion | `ccf(w1, w2, ...)` | Three-phase: consensus extraction, compromise on residuals, normalize. Handles dogmatic sources. Disagreement is converted to uncertainty, so two dogmatic sources that disagree fuse to vacuous. Not associative. | van der Heijden 2018, Def 5 |

The `fuse()` dispatcher (`opinion.py:401`) selects the fusion method: `"wbf"` forces WBF, `"ccf"` forces CCF, and `"auto"` (default) tries WBF first, falling back to CCF when any source is dogmatic.

Negation is an involution: `~~w == w`. Conjunction and disjunction assume independent frames. Consensus fusion is commutative and associative, and always reduces uncertainty relative to either input.

### Evidence mapping

The `BetaEvidence(r, s, a)` dataclass (`opinion.py:192`) represents evidence counts directly:

- `r` -- positive evidence observations (>= 0)
- `s` -- negative evidence observations (>= 0)
- `a` -- base rate

The bijection between evidence and opinions (Josang Def 12, p.20-21) uses prior weight `W = 2`:

```
b = r / (r + s + W)
d = s / (r + s + W)
u = W / (r + s + W)
```

With no observations (r=0, s=0): `b=0, d=0, u=1` -- the vacuous opinion. As evidence accumulates, uncertainty shrinks. With 8 positive and 2 negative observations: `b=0.667, d=0.167, u=0.167` -- strong belief with residual uncertainty.

Convenience functions:
- `from_evidence(r, s, a)` -- `BetaEvidence(r, s, a).to_opinion()` (`opinion.py:223`)
- `from_probability(p, n, a)` -- converts a calibrated probability with effective sample size to an opinion via `from_evidence(p*n, (1-p)*n, a)` (`opinion.py:228`)

## Calibration pipeline

### The problem

Raw LLM outputs are not calibrated. A model labeling a stance as "strong" does not mean `p = 0.7`. Without calibration and a resolved prior, the system cannot map categorical labels to meaningful opinion values. propstore treats that case as `BaseRateUnresolved` and upgrades to evidence-based opinions only when calibration data and a base-rate assertion are available.

### Temperature scaling

`TemperatureScaler` (`calibrate.py:37`) implements Guo et al. 2017, p.5. A single parameter `T` is fit on validation data by minimizing negative log-likelihood via golden section search on [0.01, 10.0]:

```
calibrated = softmax(logits / T)
```

Temperature scaling preserves the ranking of class probabilities while adjusting their sharpness. `T > 1` softens overconfident predictions; `T < 1` sharpens underconfident ones.

### Corpus CDF calibration

`CorpusCalibrator` (`calibrate.py:105`) converts embedding distances to opinions using the observed corpus distribution:

1. **Percentile ranking** -- where does this distance fall in the reference corpus? (`calibrate.py:142`, via `bisect_right`)
2. **Effective sample size** -- local density around the distance, with bandwidth `h = 1/sqrt(n)`. Capped at 50, scaled by corpus confidence factor `min(1.0, (n-1)/9.0)`. Cites Sensoy et al. 2018, p.3-4. (`calibrate.py:150`)
3. **Similarity** -- `1 - percentile`
4. **Opinion** -- `from_probability(similarity, n_eff, base_rate)` maps similarity and local evidence density to an opinion after the corpus base rate has been supplied explicitly

Small corpora produce high-uncertainty opinions (low effective sample size). Large corpora with tight clusters produce high-belief opinions. The mapping is principled: evidence counts from observed data, not heuristic thresholds.

### Categorical-to-opinion mapping

`categorical_to_opinion()` converts LLM strength labels to opinions with two modes:

**Without calibration data or an explicit category prior:** returns `BaseRateUnresolved(reason="missing_base_rate")`. A category label alone is not a prior.

**With calibration data and a resolved prior:** loads historical `(correct, total)` counts from the sidecar's `calibration_counts` table per `(pass_number, category)`. Maps `r = correct`, `s = total - correct` to an evidence-based opinion via `from_evidence(r, s, base_rate)`. More calibration data means lower uncertainty.

When both corpus-distance and categorical opinions are available, they are fused via `fuse()` (consensus fusion) to combine independent evidence sources (`relate.py:219`).

### ECE (Expected Calibration Error)

`expected_calibration_error()` (`calibrate.py:327`) implements Guo et al. 2017, p.1:

```
ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
```

This bins predictions by confidence and measures how well confidence tracks accuracy. Lower ECE means better calibration. Used to evaluate whether temperature scaling or corpus calibration actually improved the system's probability estimates.

## End-to-end flow

This is how opinions flow from LLM output through argumentation to a resolved winner:

```
LLM output (strength label)
    |
    v
categorical_to_opinion()
    |
    |  [if base-rate resolution fails]
    v
BaseRateUnresolved
    |
    |  [if base-rate resolution succeeds]
    v
AssertionOpinion -----> Opinion
    |                     |
    | [if reference_distances]
    | CorpusCalibrator.to_opinion(base_rate) --> fuse() (consensus)
    |                     |
    v                     v
Resolution dict: confidence = Opinion.expectation()
                  opinion_b, opinion_d, opinion_u, opinion_a
    |
    v
Stance proposal YAML file (branch `proposal/stances`, path `stances/*.yaml`)
    |
    v
Sidecar build: relation_edge table (opinion columns)
    |
    v
PrAF: p_relation_from_stance() -> Opinion per edge
    |                                |
    v                                v
MC sampling: Opinion.expectation() = existence probability
    |
    v
Acceptance probabilities per argument
    |
    v
apply_decision_criterion(criterion, pessimism_index)
    |
    v
Resolved winner (or tie)
```

**Step by step:**

1. The LLM classifies a stance with a strength label (strong/moderate/weak/none). `categorical_to_opinion()` converts this to either `BaseRateUnresolved` or an `Opinion` whose prior was supplied explicitly.

2. If corpus embedding distances are available, `CorpusCalibrator.to_opinion()` produces a second independent opinion using the same explicit corpus base rate. The two are fused via consensus to combine both evidence sources.

3. The opinion's `expectation()` becomes the `confidence` field. All four opinion components are written to the stance YAML only when calibration succeeds.

4. During sidecar build, opinion columns are populated on the `relation_edge` table from stance YAML files.

5. `p_relation_from_stance()` extracts the Opinion from each stance's opinion columns. If opinion fields are absent or incomplete, PrAF receives `NoCalibration` rather than a fabricated probability.

6. The MC sampler uses `Opinion.expectation()` as the existence probability for each argument and defeat in each Monte Carlo sample (Li et al. 2012).

7. After MC sampling produces acceptance probabilities, `apply_decision_criterion()` selects or breaks ties using the render policy's criterion and pessimism index.

## Decision criteria

### At render time

Four criteria are implemented in `apply_decision_criterion()` (`world/types.py:348`), following Denoeux 2019:

| Criterion | Formula | Intuition |
|-----------|---------|-----------|
| `pignistic` | `b + a * u` | Default. Best single-point estimate. Distributes uncertainty proportionally to the base rate. |
| `lower_bound` | `b` | Conservative. Only counts what is directly believed. Ignores all uncertainty. |
| `upper_bound` | `1 - d` | Optimistic. Everything not actively disbelieved. |
| `hurwicz` | `alpha * b + (1 - alpha) * (1 - d)` | Tunable pessimism via `pessimism_index` (alpha). At alpha=1.0, equivalent to lower bound. At alpha=0.0, equivalent to upper bound. |

The pignistic criterion is equivalent to `Opinion.expectation()`. The lower and upper bounds give the endpoints of the uncertainty interval `[Bel, Pl]` (Josang 2001, p.4). Hurwicz interpolates between them (Denoeux 2019, p.17).

When opinion columns are NULL, the probabilistic path treats the value as uncalibrated unless an explicit owner-layer policy supplies a prior and provenance.

### Integration with resolution

Decision criteria are used in PrAF resolution (`world/resolution.py:393`) as a tiebreaker when multiple claims have equal acceptance probability after MC sampling. For non-PrAF resolution strategies (recency, sample_size, claim_graph, ASPIC, ATMS), each strategy has its own winner-selection logic and does not use decision criteria.

### CLI usage

```bash
# Default pignistic criterion
pks world resolve domain=argumentation

# Conservative: only count direct belief
pks world resolve domain=argumentation --decision-criterion lower_bound

# Optimistic: everything not disbelieved
pks world resolve domain=argumentation --decision-criterion upper_bound

# Tunable pessimism (0.0 = optimistic, 1.0 = pessimistic)
pks world resolve domain=argumentation --decision-criterion hurwicz --pessimism-index 0.3
```

The `RenderPolicy` dataclass (`world/types.py:177`) stores these settings:
- `decision_criterion: str` (default `"pignistic"`)
- `pessimism_index: float` (default `0.5`)
- `show_uncertainty_interval: bool` (default `False`)

### Known limitations

- Interval dominance (Denoeux 2019) is not implemented.
- Extended Josang operators (deduction, comultiplication, abduction -- Josang & McAnally 2004, Josang 2008) are not implemented.
- The WBF operator raises on dogmatic inputs; use CCF or `fuse(method="auto")` when dogmatic sources are possible.

## References

- Josang, A. (2001). "A Logic for Uncertain Probabilities." *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*, 9(3), pp. 279-311.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." *ICML 2017*.
- Sensoy, M., Kaplan, L., & Kandemir, M. (2018). "Evidential Deep Learning to Quantify Classification Uncertainty." *NeurIPS 2018*.
- Denoeux, T. (2019). "Decision-Making with Belief Functions: A Review." *International Journal of Approximate Reasoning*, 109, pp. 87-110.
- van der Heijden, F. et al. (2018). Weighted Belief Fusion (Def 4) and Cumulative & Compromise Fusion (Def 5).
- Josang, A., & McAnally, D. (2004). Multiplication and comultiplication of beliefs. *International Journal of Approximate Reasoning*, 38(1).
- Josang, A. (2008). Conditional reasoning with subjective logic. *Journal of Multiple-Valued Logic and Soft Computing*, 15(1).
