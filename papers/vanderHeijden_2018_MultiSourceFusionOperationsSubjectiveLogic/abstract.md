# Abstract

## Original Text (Verbatim)

The purpose of multi-source fusion is to combine information from more than two evidence sources, or multiple opinions from multiple actors. For subjective logic, a number of different fusion operators have been proposed, each satisfying a specific set of criteria, which make them appropriate for different use cases. Some of these operators are associative, and therefore multi-source fusion is well-defined for these operators. In this paper, we provide multi-source fusion for the remaining operators, including weighted belief fusion (WBF) and consensus & compromise fusion (CCF). By linking the WBF to the notion of confidence-weighted averaging of Dirichlet hyper-probability distributions, we show that the derived multi-source fusion produces valid opinions, and explore the relationship with Dirichlet PDFs. Section IV discusses how multi-source fusion is used for CBF and ABF, as well as belief constraint fusion. The obtained results are extensions of Jøsang's work in Subjective Logic, and show that some of the previously published results (i.e., the base rate equations for the cumulative and averaging operations) are incorrect.

---

## Our Interpretation

The paper addresses a fundamental gap in subjective logic: how to correctly combine opinions from more than two sources when the fusion operators lack associativity. It provides direct N-source definitions, proves WBF's equivalence to Dirichlet confidence-weighted averaging, and importantly corrects errors in Josang's 2016 book regarding base rate computation in cumulative and averaging fusion. This is critical for any implementation of subjective logic fusion, as using the incorrect formulas produces wrong base rates in fused opinions.
