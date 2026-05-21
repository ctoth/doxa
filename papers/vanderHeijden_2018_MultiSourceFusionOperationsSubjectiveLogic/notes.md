---
title: "Multi-Source Fusion Operations in Subjective Logic"
authors: "Rens Wouter van der Heijden, Henning Kopp, Frank Kargl"
year: 2018
venue: "arXiv preprint (1805.01388)"
doi_url: "https://arxiv.org/abs/1805.01388"
produced_by:
  agent: "claude-opus-4-6"
  skill: "paper-reader"
  timestamp: "2026-03-28T21:46:07Z"
---
# Multi-Source Fusion Operations in Subjective Logic

## One-Sentence Summary
Provides corrected and formally defined multi-source fusion operators for subjective logic, establishing equivalence between weighted belief fusion (WBF) and confidence-weighted averaging of Dirichlet HPDs, and correcting prior formulations of cumulative, averaging, consensus & compromise, and belief constraint fusion for the multi-source case.

## Problem Addressed
Subjective logic defines several fusion operators (cumulative, averaging, weighted belief, consensus & compromise, belief constraint), but some lack associativity, making multi-source fusion (N>2 sources) poorly defined when iterated pairwise. Prior work (Jøsang 2016) presented binary operators, but iterated application produces different results depending on grouping order. This paper provides direct N-source definitions, corrects errors in prior formulations regarding base rate handling, and proves equivalence to operations on Dirichlet hyper-opinion distributions. *(p.1)*

## Key Contributions
- Direct multi-source definitions for all five fusion operators that avoid associativity problems *(p.1)*
- Proof that weighted belief fusion (WBF) for multiple sources is compatible with weighted belief fusion of Dirichlet HPDs *(p.5, Theorem 1)*
- Correction of cumulative belief fusion (CBF) and averaging belief fusion (ABF) base rate formulas from Jøsang 2016 *(p.6)*
- Multi-source consensus & compromise fusion (CCF) generalization *(p.7, Definition 5)*
- Multi-source belief constraint fusion (BCF) generalization *(p.3, Definition 3)*
- Formal mapping between subjective logic opinions and Dirichlet HPDs via the hyper-opinion setting *(p.2-3)*

## Study Design
*Non-empirical: formal definitions, proofs, and corrections.*

## Methodology
The paper establishes a bijection between subjective logic (multinomial opinions) and the Dirichlet/hyper-Dirichlet probability distribution framework. Using this mapping, they prove that fusion operators in opinion space correspond to well-defined operations on Dirichlet HPDs. They define multi-source operators directly (not via iterated binary application) and verify compatibility with the Dirichlet side.

## Key Equations / Statistical Models

### Multinomial Opinion (Definition 1)
$$
\omega_X^A = (b_X^A, u_X^A, a_X)
$$
Where $b_X^A$ is a belief mass distribution over $\mathcal{X}$, $u_X^A \in [0,1]$ is uncertainty, $a_X$ is the base rate distribution. Constraints: $u_X^A + \sum_{x \in \mathcal{X}} b_X^A(x) = 1$ and $\forall x: b_X^A(x) \geq 0, u_X^A \geq 0$. *(p.2)*

### Projected Probability
$$
P_X^A(x) = b_X^A(x) + a_X(x) \cdot u_X^A
$$
*(p.2)*

### Dirichlet HPD Mapping
$$
\text{Dir}_X^A(p_X, \alpha_X^A) \text{ with } \alpha_X^A(x) = r_X^A(x) + a_X(x) \cdot W
$$
Where $r_X^A(x)$ is evidence for outcome $x$, $W$ is the non-informative prior weight (default $W=2$ for binomial). The mapping from opinion to Dirichlet: $r_X^A(x) = \frac{W \cdot b_X^A(x)}{u_X^A}$ and $u_X^A = \frac{W}{W + \sum_{x} r_X^A(x)}$. *(p.2)*

### Hyper-Opinion (Definition 2)
$$
\omega_X^A = (b_X^A, u_X^A, a_X)
$$
Where $b_X^A$ is now defined over the reduced power set $\mathcal{R}(\mathcal{X}) = 2^{\mathcal{X}} \setminus \{\mathcal{X}, \emptyset\}$. The belief additivity constraint becomes $u_X^A + \sum_{x_j \in \mathcal{R}(\mathcal{X})} b_X^A(x_j) = 1$. *(p.2)*

### Belief Constraint Fusion (BCF) — Multi-source (Definition 3)
$$
b_{X}^{A \otimes B}(x) = \frac{\text{Conf}(b_X^A, b_X^B)(x)}{1 - \text{Conf}(b_X^A, b_X^B)(\emptyset)}
$$
Where the conflict function Conf computes the intersection of focal elements. For multi-source: let $\mathcal{A}^{(N)} = \{A_1, \ldots, A_N\}$ be the set of actors, then:
$$
b_X^{A_{1:N}}(x_i) = \frac{1}{\eta} \sum_{\substack{x_{j_1}, \ldots, x_{j_N} \\ x_{j_1} \cap \cdots \cap x_{j_N} = x_i}} \prod_{s=1}^{N} \hat{b}_X^{A_s}(x_{j_s})
$$
where $\hat{b}_X^{A_s}(x_j) = b_X^{A_s}(x_j)$ for $x_j \neq \mathcal{X}$ and $\hat{b}_X^{A_s}(\mathcal{X}) = u_X^{A_s}$. $\eta$ normalizes by excluding the empty-set mass. *(p.3)*

### Weighted Belief Fusion (WBF) — Multi-source (Definition 4)
For a finite set of actors $\mathcal{A}^{(N)}$ with opinions $\omega_X^{A_i} = (b_X^{A_i}, u_X^{A_i}, a_X)$:
$$
\hat{b}_X^{A_{1:N}}(x) = \frac{\sum_{i=1}^{N} \left( \frac{b_X^{A_i}(x)}{u_X^{A_i}} \prod_{j \neq i} u_X^{A_j} \right)}{\sum_{i=1}^{N} \left( \frac{1 - u_X^{A_i}}{u_X^{A_i}} \prod_{j=1}^{N} u_X^{A_j} \right) + \prod_{j=1}^{N} u_X^{A_j}}
$$

$$
\hat{u}_X^{A_{1:N}} = \frac{\prod_{j=1}^{N} u_X^{A_j}}{\sum_{i=1}^{N} \left( \frac{1 - u_X^{A_i}}{u_X^{A_i}} \prod_{j=1}^{N} u_X^{A_j} \right) + \prod_{j=1}^{N} u_X^{A_j}}
$$

All sources must share the same base rate $a_X$. *(p.4-5)*

### Theorem 1: WBF Compatibility with Dirichlet
The weighted belief fusion of multiple opinions is compatible with weighted belief fusion of Dirichlet HPDs. Specifically, WBF corresponds to confidence-weighted averaging of the evidence parameters of all Dirichlet HPDs. *(p.5)*

### CC Fusion (CCF) — Three-phase (Definition 5)
Step 1 (Consensus Phase): The consensus belief $b_Y^{cons}(x)$ is computed as a minimum consensus belief per $x$. The residual beliefs $b_Y^{res,A_i}(x)$ of each actor are the differences between their belief and the consensus belief. The total consensus $C$ is the sum of all consensus beliefs:
$$
b_Y^{cons}(x) = \min_{A_i \in \mathcal{A}^{(N)}} b_X^{A_i}(x) \quad \text{for each } x \in \mathcal{X}
$$
$$
C = \sum_{x \in \mathcal{X}} b_Y^{cons}(x)
$$
*(p.6)*

Step 2 (Compromise Phase): The compromise belief $b_Y^{comp}(x)$ is a frame of $\mathcal{X}$ by adding four components: 1) the residual belief weighted by all other actors' uncertainty; 2) the common belief in sets where the intersection is $x$; 3) the common belief in sets where the union is $x$ but intersection is non-empty; 4) the common belief in sets where the union is $x$ and the intersection is empty. *(p.6)*

Step 3 (Normalization Phase): Scaling $b_Y^{comp}(x)$ and uncertainty: If $u_Y^{N} = 1$, a normalization factor $\psi$ of $1/N$ has to be applied. *(p.6)*

The belief on the entire domain $\mathcal{X}$ is then added to the uncertainty. The intuition behind this is that a belief in $\mathcal{X}$ is the belief that anything will happen. *(p.6)*

### Corrected Cumulative Belief Fusion (CBF)
The authors correct the CBF base rate formula from Jøsang 2016. The original formula for base rates in the multi-source case used division by $N$ for averaging, but this is incorrect when sources have different uncertainties. The corrected version uses confidence-weighted averaging:
$$
a_X^{CBF}(x) = \frac{\text{confidence-weighted combination of base rates}}{\text{total confidence}}
$$
The specific correction: in Jøsang's Eq. (12.2), the base rate should be confidence-weighted rather than simple-averaged. *(p.3-4)*

### Corrected Averaging Belief Fusion (ABF)
Similarly, the ABF base rate formula requires correction. The original used simple averaging; the correction weights by confidence. *(p.3)*

### Remark 1: Dogmatic Opinions
If $u_X^{A_i} = 0$ for exactly one actor and $u_X^{A_j} > 0$ for all others, the fused result uses only the dogmatic actor's belief. *(p.5)*

### Remark 2: Infinite Evidence (All Dogmatic)
When all actors have $u_X^{A_i} = 0$, the WBF definition uses a limit. The paper defines this by setting $u_X^{A_i} = \varepsilon$ per actor with infinite evidence. This is analogous to the notion of relative infinity from Section III. Only non-dogmatic actors will have finite (and therefore negligible) evidence. *(p.5)*

### Remark 3: No Evidence
If no evidence is available (all $u_X^{A_i} = 1$ for all actors), the result is the vacuous opinion: $b_X(x) = 0$ for all $x$, $u_X = 1$. The fused opinion should obviously also have no evidence; this justifies the definition in case 3. *(p.5)*

## Parameters

| Name | Symbol | Units | Default | Range | Page | Notes |
|------|--------|-------|---------|-------|------|-------|
| Non-informative prior weight | $W$ | — | 2 | >0 | 2 | Default for binomial; $W = k$ for $k$-nomial |
| Number of sources | $N$ | — | — | ≥2 | 1 | Multi-source fusion requires at least 2 |
| Uncertainty | $u_X^A$ | — | — | [0,1] | 2 | Per-source uncertainty mass |
| Base rate | $a_X(x)$ | — | — | [0,1], sum=1 | 2 | Must be shared across all sources for WBF |

## Methods & Implementation Details
- WBF requires all sources share the same base rate $a_X$; the base rate of the fused opinion equals this shared base rate *(p.5)*
- For CBF and ABF, the base rate in the fused opinion must be confidence-weighted, not simple-averaged — this is the key correction to Jøsang 2016 *(p.3-4)*
- CC fusion is defined via three phases: consensus extraction, compromise computation, normalization *(p.6)*
- BCF (belief constraint fusion) follows the Dempster-Shafer combination rule pattern *(p.3)*
- The bijection between opinions and Dirichlet HPDs (via hyper-opinions) is the theoretical backbone — Definition 2 establishes the mapping, and the fusion operators are then shown to be compatible on both sides *(p.2-3)*
- Edge case: when uncertainty $u_A$ is approximately 0 (dogmatic opinion), the bijection maps to infinite evidence; special handling is needed via limits *(p.2, Remark 1-2)*
- Multi-source CBF corresponds to confidence-weighted averaging of Dirichlet HPDs (Theorem 1 proves this for WBF) *(p.5)*

## Figures of Interest
- **Fig 1 (p.2):** Bijection between hyper-opinions (left) and Dirichlet HPDs (right), showing the mapping via evidence parameters $r_X^A(x)$
- **Fig 2 (p.3):** Important edge case in the bijective map where uncertainty is 0 (dogmatic opinion) — the map is undefined at this point
- **Fig 3 (p.3):** The belief mass is mapped through Dirichlet combination using $\hat{b}$ notation
- **Table I (p.7):** Running examples for all five fusion operators with specific numerical values, confirming formulas

## Results Summary
- WBF for multiple sources is proven equivalent to confidence-weighted averaging of Dirichlet evidence parameters (Theorem 1) *(p.5)*
- Cumulative and averaging belief fusion base rate formulas from Jøsang 2016 are shown to be incorrect and corrected *(p.3-4)*
- CC fusion is generalized from binary to N-source with a three-phase algorithm *(p.6-7)*
- BCF is generalized to multi-source via N-fold intersection of focal elements *(p.3)*
- Table I provides numerical verification of all operators *(p.7)*

## Limitations
- CCF is specifically defined for subjective logic only — the authors note they do not prove equivalence to operations on Dirichlet HPDs for CCF *(p.7)*
- The paper does not address computational complexity of multi-source CCF (the three-phase algorithm involves products over all actors) *(p.6-7)*
- Only multinomial frames are considered; no extension to continuous domains *(p.2)*

## Arguments Against Prior Work
- Jøsang 2016 (book Ch. 12) multi-source CBF and ABF base rate formulas are incorrect — they use simple averaging rather than confidence-weighted averaging *(p.3-4)*
- Prior binary fusion operators lack associativity, so iterating them pairwise for N>2 sources gives order-dependent results — this is fundamentally broken for multi-source scenarios *(p.1)*
- The paper notes that "some of these operators lack associativity, making multi-source fusion poorly defined" *(p.1)*

## Design Rationale
- Direct N-source definitions are preferred over iterated binary because associativity cannot be guaranteed for all operators *(p.1)*
- The Dirichlet HPD equivalence is used as the verification backbone — if an opinion-space operation matches the corresponding HPD operation, it is correct *(p.5, Theorem 1)*
- Base rates must be shared across all sources for WBF (this is a precondition, not computed) because confidence-weighted averaging of base rates is only meaningful when they agree *(p.5)*
- The three-phase CCF algorithm (consensus, compromise, normalization) is chosen because it handles the multi-source case naturally — the consensus phase finds agreement, compromise distributes disagreement, and normalization ensures valid opinion constraints *(p.6)*

## Testable Properties
- WBF with all $u=1$ (vacuous) must produce vacuous opinion: $b(x)=0, u=1$ *(p.5, Remark 3)*
- WBF with exactly one dogmatic source ($u=0$) must produce that source's belief distribution *(p.5, Remark 1)*
- WBF fused uncertainty $\hat{u}^{A_{1:N}} \leq \min_i u^{A_i}$ — fusion should always reduce uncertainty (more evidence = less uncertain) *(p.5)*
- $b(x) + u = 1$ (belief additivity) must hold for all fused opinions *(p.2)*
- $P(x) = b(x) + a(x) \cdot u$ must hold for projected probability after fusion *(p.2)*
- WBF of Dirichlet evidence = confidence-weighted average of evidence from each source (Theorem 1) *(p.5)*
- CBF/ABF base rates in fused opinion must use confidence-weighted averaging, not simple averaging *(p.3-4)*
- CCF consensus belief $b^{cons}(x) = \min_i b^{A_i}(x)$ for each value $x$ *(p.6)*
- BCF normalization must exclude empty-set mass *(p.3)*
- Table I numerical examples can be used as regression tests for all five operators *(p.7)*

## Relevance to Project
This paper is **directly relevant** to propstore's subjective logic implementation. The project already implements Jøsang 2001 consensus fusion. This paper:
1. Provides corrected multi-source fusion formulas — the project should verify its fusion operators match these corrected versions
2. Establishes the Dirichlet HPD equivalence that can be used to verify correctness of the opinion algebra
3. Identifies specific errors in Jøsang 2016 base rate handling that may be present in the codebase
4. Provides the formal three-phase CCF algorithm that could be implemented

## Open Questions
- [ ] Does propstore's current consensus fusion use the corrected base rate formula or the incorrect Jøsang 2016 version?
- [ ] Should WBF be implemented as the primary multi-source fusion operator (it has the cleanest Dirichlet equivalence)?
- [ ] Is CCF needed for propstore's use cases, or is WBF sufficient?
- [ ] The paper does not prove CCF compatibility with Dirichlet HPDs — is this a concern for the project?

## Related Work Worth Reading
- Jøsang 2016 (book): "Subjective Logic: A Formalism for Reasoning Under Uncertainty" — the reference work that this paper corrects *(cited throughout)*
- Dempster 1968 / Shafer 1976: Dempster-Shafer theory foundations, BCF is a generalization of Dempster's rule *(p.3)*
- Josang & McAnally 2004: Multiplication and comultiplication of beliefs *(p.8, ref [11])*
- Ivanovska et al. 2015: Subjective logic extensions *(p.8, ref [10])*
- Kaplan et al. 2001: Sensor fusion in distributed systems *(p.8, ref [12])*

## Collection Cross-References

### Already in Collection
- [[Josang_2001_LogicUncertainProbabilities]] — foundational subjective logic paper defining opinion algebra, consensus fusion operator, projected probability; this paper extends and corrects fusion for multi-source case
- [[Josang_2010_CumulativeAveragingFusionBeliefs]] — introduces CBF and ABF operators; this paper corrects its base rate formulas for multi-source case and provides direct N-source definitions
- [[Shafer_1976_MathematicalTheoryEvidence]] — Dempster-Shafer foundations; BCF is a generalization of Dempster's combination rule to subjective logic
- [[Sensoy_2018_EvidentialDeepLearningQuantifyClassification]] — uses Dirichlet-based uncertainty with evidence-to-opinion mapping; this paper's Dirichlet HPD bijection formalizes the same mapping
- [[Denoeux_2018_Decision-MakingBeliefFunctionsReview]] — belief function decision theory; related via Dempster-Shafer combination rule that BCF generalizes

### New Leads (Not Yet in Collection)
- Jøsang 2016 (book): "Subjective Logic: A Formalism for Reasoning Under Uncertainty" — the comprehensive reference work, contains the formulas this paper corrects (Ch. 12) → NOW IN COLLECTION: [Subjective Logic: A Formalism for Reasoning Under Uncertainty](../Josang_2016_SubjectiveLogic/notes.md)
- Jøsang & McAnally 2004: "Multiplication and comultiplication of beliefs" — extended operators not yet implemented in propstore
- Ivanovska et al. 2015: Subjective logic extensions for decision support

### Now in Collection (previously listed as leads)
- [Subjective Logic: A Formalism for Reasoning Under Uncertainty](../Josang_2016_SubjectiveLogic/notes.md) — The 2016 textbook whose Ch. 12 multi-source CBF and ABF base-rate formulas this paper identifies as incorrect. The book uses simple averaging across sources; van der Heijden proves confidence-weighted averaging is required for equivalence to Dirichlet HPDF averaging (Theorem 1).

### Supersedes or Recontextualizes
- [[Josang_2010_CumulativeAveragingFusionBeliefs]] — this paper corrects base rate formulas in the multi-source CBF and ABF definitions from Josang 2010; the original simple-averaged base rates are incorrect and should be replaced with confidence-weighted averaging.
- [Subjective Logic: A Formalism for Reasoning Under Uncertainty](../Josang_2016_SubjectiveLogic/notes.md) — same correction applies to the multi-source CBF and ABF formulas in book Ch. 12. propstore implementations should use van der Heijden's confidence-weighted form, not the book's simple-averaged form.

### Conceptual Links (not citation-based)
- [[Margoni_2024_SubjectiveLogicMetaAnalysis]] — uses WBF for multi-source opinion fusion in meta-analysis context; van der Heijden's corrected WBF formula and Dirichlet equivalence proof (Theorem 1) provide the theoretical foundation for Margoni's applied fusion pipeline

### Cited By (in Collection)
- (none found)
