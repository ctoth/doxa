---
title: "A Logic for Uncertain Probabilities"
authors: "Audun Josang"
year: 2001
venue: "International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, Vol. 9, No. 3"
doi_url: "https://doi.org/10.1142/S0218488501000831"
---

# A Logic for Uncertain Probabilities

## One-Sentence Summary
Defines subjective logic — a framework where probabilities are represented as opinion tuples (b, d, u, a) supporting logical operators and evidential combination, providing a complete calculus for reasoning under uncertainty that generalizes both probability theory and binary logic.

## Problem Addressed
Traditional probability theory cannot distinguish between a probability of 0.5 based on strong evidence and 0.5 based on total ignorance. Bare-float probabilities lose the meta-information about how certain we are of the probability estimate itself. This paper provides a formal framework where uncertainty about probabilities is a first-class citizen, enabling rational decision-making (e.g., resolving the Ellsberg paradox). *(p.1)*

## Key Contributions
- Opinion representation as (b, d, u, a) tuples with b + d + u = 1 constraint *(p.7)*
- Standard logical operators (AND, OR, NOT) over opinions preserving probability expectation consistency *(p.14-18)*
- Evidential operators: discounting (trust transitivity) and consensus (cumulative fusion) *(p.24-26)*
- Bijective mapping between opinion space and Beta probability density functions *(p.20-21)*
- Demonstration that subjective logic generalizes both binary logic and probability calculus *(p.30)*
- Resolution of the Ellsberg paradox as a natural consequence of the framework *(p.10-12)*

## Methodology
Built on Dempster-Shafer theory of evidence, the paper defines a focused binary frame of discernment for each proposition, derives belief/disbelief/uncertainty functions from belief mass assignments, then constructs an opinion space with logical and evidential operators. The evidence space (Beta distributions) provides an alternative representation with a proven bijective mapping. *(p.2-7)*

## Key Equations

### Opinion Definition (Def 9)

$$
\omega_x = \{b(x), d(x), u(x), a(x)\}
$$
Where: b = belief, d = disbelief, u = uncertainty, a = relative atomicity (base rate). Constraint: b + d + u = 1, and 0 <= a <= 1.
*(p.7)*

### Belief Function Additivity (Theorem 1)

$$
b(x) + d(x) + u(x) = 1, \quad x \neq \Theta, \quad x \neq \emptyset
$$
Where: b(x) = sum of belief masses of all subsets of x; d(x) = sum of belief masses of all subsets of complement of x; u(x) = sum of belief masses overlapping both x and not-x.
*(p.4)*

### Probability Expectation (Def 6)

$$
E(\omega_x) = b(x) + a(x) \cdot u(x)
$$
Where: E maps an opinion to a single probability value by distributing uncertainty proportionally to relative atomicity.
*(p.5)*

### Kolmogorov Axiom Compliance (Theorem 2)

$$
E(x) \geq 0 \quad \forall x \in 2^\Theta
$$
$$
E(\Theta) = 1
$$
$$
E(x_1 \cup x_2 \cup \ldots) = \sum E(x_i) \quad \text{for pairwise disjoint } x_i
$$
*(p.9)*

### Propositional Conjunction (Theorem 3)

$$
b_{x \wedge y} = b_x b_y
$$
$$
d_{x \wedge y} = d_x + d_y - d_x d_y
$$
$$
u_{x \wedge y} = b_x u_y + u_x b_y + u_x u_y
$$
$$
a_{x \wedge y} = a_x a_y
$$
Where: x and y are propositions on independent binary frames of discernment.
*(p.14)*

### Propositional Disjunction (Theorem 4)

$$
b_{x \vee y} = b_x + b_y - b_x b_y
$$
$$
d_{x \vee y} = d_x d_y
$$
$$
u_{x \vee y} = d_x u_y + u_x d_y + u_x u_y
$$
$$
a_{x \vee y} = a_{x \vee y}
$$
Where: derived relative atomicity for disjunction.
*(p.14-15)*

### Probability Expectation of Conjunction/Disjunction (Proof 5)

$$
E(\omega_{x \wedge y}) = E(\omega_x) \cdot E(\omega_y)
$$
$$
E(\omega_{x \vee y}) = E(\omega_x) + E(\omega_y) - E(\omega_x) \cdot E(\omega_y)
$$
This proves that propositional operators preserve the product and co-product rules of probability.
*(p.17)*

### Negation (Theorem 6)

$$
b_{\neg x} = d_x, \quad d_{\neg x} = b_x, \quad u_{\neg x} = u_x, \quad a_{\neg x} = 1 - a_x
$$
De Morgan's laws hold for propositional conjunction and disjunction of opinions.
*(p.18)*

### Probability Density Function (Def 11)

$$
f(p \mid r, s, a) = \frac{\Gamma(r + s + 2)}{\Gamma(r + 2a) \cdot \Gamma(s + 2(1-a))} \cdot p^{r+2a-1} \cdot (1-p)^{s+2(1-a)-1}
$$
Where: r = positive evidence count, s = negative evidence count, a = relative atomicity. Parameters: p in [0,1], r >= 0, s >= 0, a in (0,1). When a = 0.5 this is the standard Beta(r+1, s+1) distribution.
*(p.19)*

### Mapping: Opinion to Evidence (Def 12)

$$
b = \frac{r}{r + s + 2}, \quad d = \frac{s}{r + s + 2}, \quad u = \frac{2}{r + s + 2}
$$
Inverse mapping:
$$
r = \frac{2b}{u}, \quad s = \frac{2d}{u} \quad (u \neq 0)
$$
Where: r, s are evidence counts; b, d, u are opinion components. The constant 2 in the denominator represents the "weight" of prior ignorance.
*(p.20-21)*

### Probability Expectation from Evidence (Eq 18)

$$
E(p_i) = \frac{n_i + 2/t}{\sum_{j=1}^{t} n_j + 2}
$$
Where: n_i = observation count for outcome i, t = total number of possible outcomes, 1/t = relative atomicity.
*(p.20)*

### Consensus Operator (Theorem 7)

$$
b_{x}^{A \diamond B} = \frac{b_x^A u_x^B + b_x^B u_x^A}{\kappa}
$$
$$
d_{x}^{A \diamond B} = \frac{d_x^A u_x^B + d_x^B u_x^A}{\kappa}
$$
$$
u_{x}^{A \diamond B} = \frac{u_x^A u_x^B}{\kappa}
$$
$$
a_{x}^{A \diamond B} = \frac{a_x^B u_x^A + a_x^A u_x^B - (a_x^A + a_x^B) u_x^A u_x^B}{u_x^A + u_x^B - 2 u_x^A u_x^B}
$$
Where: kappa = u_A + u_B - u_A * u_B. Undefined when both opinions are dogmatic (u_A = u_B = 0). Commutative and associative.
*(p.25)*

### Discounting Operator (Def 14)

$$
\omega_x^{A:B} = \omega_B^A \otimes \omega_x^B
$$
$$
b_{x}^{A:B} = b_B^A \cdot b_x^B, \quad d_{x}^{A:B} = b_B^A \cdot d_x^B
$$
$$
u_{x}^{A:B} = d_B^A + u_B^A + b_B^A \cdot u_x^B, \quad a_{x}^{A:B} = a_x^B
$$
Where: A's opinion about B's trustworthiness discounts B's opinion about x. If A fully distrusts B (b_B^A = 0), the result is vacuous. Relative atomicity is preserved from B's opinion.
*(p.24)*

### Combining Evidence (Def 13)

$$
f(p \mid r_A + r_B, \; s_A + s_B, \; a^{A \diamond B})
$$
$$
a^{A \diamond B} = \frac{a^A (r_A + s_A) + a^B (r_B + s_B)}{(r_A + s_A) + (r_B + s_B)}
$$
Where: evidence counts simply add; relative atomicity is weighted average by total evidence. Observer with most evidence has greatest influence on combined atomicity.
*(p.22)*

### Uncertainty Maximization (Def 16)

$$
b_1 = k_1 - u_a \cdot d_1/(1-a), \quad d_1 = k_1 - u_a \cdot b_1/a
$$
$$
u_1 = k_1 + u_a, \quad a_1 = a
$$
Where: transforms an opinion about a non-repeatable event to maximize uncertainty while preserving E(x). Purpose: one-time events should not have artificially precise beliefs.
*(p.30)*

## Parameters

| Name | Symbol | Units | Default | Range | Page | Notes |
|------|--------|-------|---------|-------|------|-------|
| Belief | b | - | 0 | [0, 1] | 7 | Degree of belief that proposition is true |
| Disbelief | d | - | 0 | [0, 1] | 7 | Degree of belief that proposition is false |
| Uncertainty | u | - | 1 | [0, 1] | 7 | Degree of uncommitted belief mass |
| Relative atomicity | a | - | 0.5 | (0, 1) | 5 | Prior probability / base rate; default for binary frame |
| Positive evidence | r | count | 0 | [0, inf) | 19 | Number of positive observations |
| Negative evidence | s | count | 0 | [0, inf) | 19 | Number of negative observations |
| Kappa (consensus) | kappa | - | - | (0, 1] | 25 | u_A + u_B - u_A*u_B; normalization factor |

## Implementation Details

### Data Structures Needed
- **Opinion tuple**: (b, d, u, a) with float64 components, constraint b + d + u = 1.0, 0 < a < 1 *(p.7)*
- **Evidence tuple**: (r, s, a) with r, s >= 0 non-negative floats (not necessarily integers), a in (0,1) *(p.19)*
- **Special opinions**: vacuous = (0, 0, 1, a), absolute_true = (1, 0, 0, a), absolute_false = (0, 1, 0, a) *(p.8)*

### Key Operations to Implement
1. **Opinion creation from evidence**: b = r/(r+s+2), d = s/(r+s+2), u = 2/(r+s+2) *(p.20-21)*
2. **Evidence extraction from opinion**: r = 2b/u, s = 2d/u (guard: u > 0) *(p.21)*
3. **Probability expectation**: E = b + a*u *(p.5)*
4. **Conjunction**: element-wise per Theorem 3 formulas *(p.14)*
5. **Disjunction**: element-wise per Theorem 4 formulas *(p.14-15)*
6. **Negation**: swap b/d, complement a *(p.18)*
7. **Discounting**: scale b,d by trust belief; absorb rest into uncertainty *(p.24)*
8. **Consensus**: weighted combination per Theorem 7 formulas *(p.25)*
9. **Uncertainty maximization**: redistribute b,d to u preserving E(x) *(p.30)*
10. **Opinion ordering**: compare by E(x), then u (less is better), then a (less is better) *(p.9)*

### Edge Cases
- **Dogmatic opinions** (u = 0): Cannot be combined by consensus operator; cannot be mapped to ppdf *(p.21, 25)*
- **Vacuous opinion** (b = 0, d = 0, u = 1): Maps to uniform Beta(1,1); E(x) = a *(p.21)*
- **Two dogmatic opinions in consensus**: Undefined — kappa = 0 *(p.25)*
- **Zero trust in discounting**: b_B^A = 0 produces vacuous opinion regardless of B's opinion *(p.24)*
- **Non-binary frames**: Relative atomicity a = |x|/|Theta| for uniform prior; focused BMA reduces to binary *(p.5-6)*

### Initialization Procedures
- Start with vacuous opinion (0, 0, 1, a) for any proposition without evidence *(p.8)*
- Set a = 1/t where t is the number of mutually exclusive outcomes (0.5 for binary) *(p.5)*
- Update from evidence: after r positive and s negative observations, compute opinion via Def 12 mapping *(p.20-21)*

## Figures of Interest
- **Fig 4 (p.8):** Opinion triangle — the core visualization. Belief on right, disbelief on left, uncertainty at top. Projector line from opinion point to director (probability axis) gives E(x).
- **Fig 5 (p.10):** Ellsberg paradox — frame of discernment with red/black/yellow balls, showing how opinions capture uncertainty that bare probabilities miss.
- **Fig 9 (p.13):** Four opinions with same E(x) = 0.5 but different uncertainty and atomicity — demonstrates ordering criteria.
- **Fig 10 (p.17):** Conjunctive system Z = X AND Y for reliability analysis.
- **Fig 11 (p.20):** Beta ppdf after 7 positive and 1 negative observations — shows how evidence shapes the density function.
- **Fig 14 (p.26):** Trust network — judge J evaluates witnesses W1, W2, W3 via discounting then consensus.

## Results Summary
The paper proves that subjective logic's probability expectation function satisfies Kolmogorov axioms (Theorem 2), that propositional conjunction/disjunction preserve the product/co-product of probabilities (Proof 5), and that the consensus operator corresponds to evidence accumulation in the Beta distribution space (Sec 4.3). Binary logic is shown to be a special case when all opinions are absolute. The framework successfully resolves the Ellsberg paradox and provides intuitive results in Zadeh's paradox where Dempster's rule fails. *(p.9, 17, 28-29)*

## Limitations
- Propositional conjunction and disjunction require independent frames of discernment — cannot handle dependent propositions *(p.14)*
- Dogmatic opinions (u = 0) cannot be combined via consensus and have no ppdf representation *(p.21, 25)*
- The ppdf expression may only be an approximation for non-binary frames of discernment *(p.23)*
- Propositional conjunction of ppdfs differs slightly from simultaneous pdf computation (same E but different curves) *(p.23)*
- The ordering of opinions (Def 10) uses relative atomicity as tiebreaker, which "needs to be validated by practical experiments on human judgment" *(p.13)*
- Uncertainty maximization for non-repeatable events is motivated but the criteria for when to apply it are left to judgment *(p.29-30)*

## Arguments Against Prior Work
- Traditional probability theory cannot represent second-order uncertainty — "it is also necessary to take into account that beliefs always are held by individuals and that beliefs for this reason are fundamentally subjective" *(p.1)*
- Dempster's rule of combination produces counter-intuitive results in Zadeh's example (assigns 100% guilt to one suspect when both witnesses disagree); the consensus operator gives the intuitive answer (no suspects are guilty) *(p.27-29)*
- Non-normalized Dempster's rule handles Zadeh's example better than standard Dempster's rule, but the consensus operator naturally handles uncertainty introduction *(p.28-29)*
- Choquet capacities and similar models can explain the Ellsberg paradox, but cannot handle the 9-color urn ordering example (Sec 2.6) *(p.12)*

## Design Rationale
- **Binary focus via focused frame**: Rather than operating on the full power set 2^Theta, opinions focus on binary propositions (x vs not-x), making the algebra tractable while preserving all information via relative atomicity *(p.6)*
- **Relative atomicity as fourth parameter**: Needed to distinguish between differently-sized subsets of the frame; without it, E(x) would always use a = 0.5 and lose information about prior distributions *(p.5)*
- **Consensus vs Dempster's rule**: Consensus chosen because it naturally introduces uncertainty (reducing certainty when combining conflicting evidence) whereas Dempster's rule amplifies conflicts *(p.27-29)*
- **Evidence space mapping**: The bijection to Beta distributions grounds opinions in statistical evidence and enables the evidence accumulation interpretation of the consensus operator *(p.20-22)*

## Testable Properties
- b + d + u = 1 for all valid opinions *(p.7)*
- 0 < a < 1 for all non-degenerate opinions *(p.7)*
- E(omega_x) = b + a*u must be in [0, 1] *(p.5)*
- E(omega_{x AND y}) = E(omega_x) * E(omega_y) for independent propositions *(p.17)*
- E(omega_{x OR y}) = E(omega_x) + E(omega_y) - E(omega_x)*E(omega_y) for independent propositions *(p.17)*
- Consensus is commutative: omega_A consensus omega_B = omega_B consensus omega_A *(p.25)*
- Consensus is associative: (A consensus B) consensus C = A consensus (B consensus C) *(p.25)*
- Consensus reduces uncertainty: u_{A consensus B} <= min(u_A, u_B) *(p.25-26)*
- Consensus of infinite non-dogmatic opinions converges to zero uncertainty *(p.26)*
- Discounting with vacuous trust (b_B^A = 0) produces vacuous opinion *(p.24)*
- Discounting preserves relative atomicity: a_{A:B} = a_B *(p.24)*
- Negation is involution: NOT(NOT(omega_x)) = omega_x *(p.18)*
- Round-trip opinion->evidence->opinion preserves all components (when u > 0) *(p.20-21)*
- E(Theta) = 1 (Kolmogorov axiom 2) *(p.9)*

## Relevance to Project
Propstore currently uses bare-float probabilities for claim confidence. Subjective logic provides a principled replacement where:
1. **Vacuous opinions** represent claims with no evidence — currently impossible to distinguish from 50/50 evidence
2. **Consensus operator** provides a mathematically grounded way to combine multiple sources' assessments of the same claim, naturally handling the case where sources disagree
3. **Discounting operator** enables trust chains — if source A cites source B, the chain of trust can be formally modeled
4. **The ATMS connection**: opinions with their uncertainty component map naturally to assumption-labeled belief — an ATMS node's label sets correspond to the evidence (r, s) supporting an opinion
5. **Argumentation integration**: opinion ordering (Def 10) provides a preference relation for ASPIC+ that accounts for uncertainty, not just bare probability
6. **Non-commitment discipline**: vacuous opinions are the formal representation of "I don't know" — they preserve the non-commitment principle at the semantic core

## Open Questions
- [ ] How do opinions compose for dependent propositions (not independent frames)?
- [ ] What is the relationship between opinion uncertainty and ATMS label minimality?
- [ ] How should relative atomicity (a) be determined for propstore claims — from corpus statistics?
- [ ] Is uncertainty maximization appropriate for all propstore claims (since real-world claims are typically non-repeatable)?
- [ ] How does the consensus operator interact with ASPIC+ preference orderings when combining argument strengths?
- [ ] Later Josang work (2016 book "Subjective Logic") extends to multinomial opinions — needed for multi-valued claims? [Partially addressed by Josang_2010_CumulativeAveragingFusionBeliefs — defines multinomial opinions as triple (b_vec, u, a_vec) over K-ary frames with multiplication operator for combining opinions on different frames. Does NOT cover fusion (cumulative/averaging) for combining same-frame opinions from different sources — a separate Josang paper is needed.]

## Related Work Worth Reading
- Josang 1997 "Artificial reasoning with subjective logic" — earlier version, electronic transaction authentication *(p.31)*
- Josang 1999 "Trust-based decision making for electronic transactions" — application to trust networks *(p.31)*
- Shafer 1976 "A Mathematical Theory of Evidence" — foundational Dempster-Shafer theory *(p.31)*
- Smets 1990 "The transferable belief model" — alternative to Dempster-Shafer with open world assumption *(p.31)*
- Walley 1997 "Statistical inferences based on a second-order possibility distribution" — related uncertain probability framework *(p.31)*
- Casella & Berger 1990 "Statistical Inference" — simultaneous pdfs computation method *(p.23)*

## Collection Cross-References

### Already in Collection
- (none — Josang 2001 does not cite any paper currently in the collection)

### New Leads (Not Yet in Collection)
- Shafer (1976) — "A Mathematical Theory of Evidence" — foundational Dempster-Shafer theory; the belief mass assignment framework Josang builds upon
- Smets (1990) — "The transferable belief model" — alternative combination rule philosophy with open world assumption
- Walley (1997) — "Statistical inferences based on a second-order possibility distribution" — competing uncertain probability framework

### Supersedes or Recontextualizes
- (none)

### Cited By (in Collection)
- [[Sensoy_2018_EvidentialDeepLearningQuantifyClassification]] — cites the 2016 book version of subjective logic (Josang 2016) as the formal framework underpinning the evidential deep learning approach to opinion-based uncertainty quantification
- [[Josang_2010_CumulativeAveragingFusionBeliefs]] — cites as ref [1], the foundational binary opinion framework. The 2010 paper extends binary opinions to multinomial opinions over K-ary frames and defines the multiplication operator for combining opinions on different (orthogonal) frames.
- [Subjective Logic: A Formalism for Reasoning Under Uncertainty](../Josang_2016_SubjectiveLogic/notes.md) — the 2016 textbook is a harmonised, corrected, and expanded specification of this 2001 paper. Notably, the 2016 book restores the marginal base rate (MBR) to binomial deduction (Eq. 9.36) — Jøsang explicitly notes (p.165) that the 2001 deduction operator omitted MBR and was inconsistent under repeated inversion. Read Josang 2001 first for the seed; read Josang 2016 for the comprehensive operator catalogue.

### Conceptual Links (not citation-based)

**Dempster-Shafer and belief maintenance:**
- [[Falkenhainer_1987_BeliefMaintenanceSystem]] — **Strong.** Both build on Dempster-Shafer theory for reasoning under uncertainty. Falkenhainer's BMS uses DS belief intervals [s,p] for truth maintenance with Dempster's rule for evidence combination; Josang's subjective logic provides the opinion algebra that generalizes DS belief functions with explicit uncertainty and logical operators. The BMS's evidence combination via Dempster's rule is directly comparable to Josang's consensus operator, which Josang argues produces more intuitive results in Zadeh's paradox. The BMS's design of being "semi-independent of the specific belief system" (p.71-72) suggests subjective logic opinions could replace DS intervals as the BMS's belief formalism.

**Evidential reasoning and neural networks:**
- [[Sensoy_2018_EvidentialDeepLearningQuantifyClassification]] — **Strong.** Sensoy uses Dirichlet distributions (the multinomial generalization of Beta distributions) to represent uncertain probabilities from neural network outputs, mapping directly to subjective logic opinions. This paper (Josang 2001) provides the theoretical foundation: the bijective mapping between opinions and Beta distributions (Def 12) is exactly the mapping Sensoy exploits. Sensoy's evidence parameters (alpha_k) correspond to Josang's (r, s) evidence counts. Together they provide the path from neural network outputs to formal opinion tuples usable in propstore's argumentation layer.

**Decision-making under belief functions:**
- [[Denoeux_2018_Decision-MakingBeliefFunctionsReview]] — **Strong.** Denoeux reviews how to make decisions when uncertainty is represented by DS belief functions — exactly the decision problem that arises at propstore's render layer when opinions (rather than bare probabilities) annotate claims. Josang's probability expectation E(x) = b + a*u corresponds to Denoeux's pignistic transformation for decision-making. Denoeux's taxonomy of decision criteria (Hurwicz, minimax regret, E-admissibility) provides the render-time policy options for converting Josang's opinions into actionable decisions.
- [Decision-Making with Belief Functions and Pignistic Probabilities](../Wilson_1993_Decision-MakingBeliefFunctionsPignistic/notes.md) — **Strong.** Wilson proves (Theorem 5.7) that lower/upper expected utility over pignistic transforms induced by all refinements equals the standard `[Bel, Pl]` envelope expectation. A binomial subjective-logic opinion `(b, d, u)` is structurally a `[b, b+u]` probability interval — exactly `[Bel, Pl]` on a binary frame. Wilson's frame-arbitrariness critique therefore lands on Jøsang's projected-probability `E(x) = b + a*u`: the projection is frame-dependent, while the opinion itself (and its envelope) is the frame-invariant object. Justifies treating SL opinions as imprecise probabilities at the propstore source-of-truth layer rather than collapsing them to projections, with the projection deferred to render-time policy.

**Probabilistic argumentation:**
- [[Li_2011_ProbabilisticArgumentationFrameworks]] — **Moderate.** Li extends Dung's AF with probabilities on argument/defeat existence; Josang provides the opinion algebra for representing those probabilities with explicit uncertainty. Combining these: instead of bare-float probabilities on argument existence, use subjective logic opinions — enabling the distinction between "50% likely to exist based on strong evidence" vs "50% likely based on ignorance" within probabilistic argumentation frameworks.

---

**See also (conceptual link):** [The transferable belief model](../Smets_Kennes_1994_TransferableBeliefModel/notes.md) — Direct semantic ancestor of Subjective Logic. Jøsang's vacuous opinion `b=d=0, u=1` corresponds to Smets' vacuous BBA `m(Ω)=1`; the SL belief-disbelief-uncertainty barycentric coordinates correspond to TBM's `bel`/`pl` envelope (Wilson bound `bel ≤ BetP ≤ pl`). Open-world philosophy from TBM is adopted by SL.

---

**See also (conceptual link):** [Quantifying Beliefs by Belief Functions: An Axiomatic Justification](../Smets_1993_QuantifyingBeliefsBeliefFunctions/notes.md) - Smets's vacuous credibility function I_Omega (I_Omega(Omega)=1, else 0) is the belief-function counterpart of Jøsang's vacuous opinion (b=d=0, u=1). Both formalisms treat total ignorance as a first-class element of the belief space; Smets's closure axiom shows every belief function is generated from I_Omega via refinement, conditioning (unnormalized Dempster), and simplex combinations - structurally analogous to building subjective opinions by accumulating evidence from a vacuous starting point.
