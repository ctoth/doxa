# Doxa issue 3

## Goal
Fix product-scale-dependent WBF/CCF behavior and open a separate draft PR. No Propstore source/pin edits and no merge or ready transition.

## State
Separate branch/worktree fix/fusion-product-scaling-issue-3 at published aa137b0. Global instructions, Doxa source/tests/CI, local notes, and exact Propstore caller chains already read; no local project instructions found.

## Smallest change and decision
Roughly forty source lines of numerical reformulation at the two existing operators. WBF's common product cancels, leaving weights (1-u)/u. CCF normalizes nonnegative compromise terms after subtracting their largest log scale; no change to mathematical formulas, base-rate convention, or genuinely zero-compromise fallback. Each residual belief/disbelief vector contains an exact zero because consensus is its minimum, so their full products vanish and need not be subtracted from the whole-frame product.

## Evidence and scope
At aa137b0 src/doxa/opinion.py:467-486 WBF rejects a positive product-scaled denominator below 1e-9. Lines 692-698 similarly route a positive CCF compromise sum below 1e-9 into the zero branch. The reported 14/16-source examples are representable and occur before underflow. Larger profiles also underflow the direct products, so tests cover both regimes.

## Next step
Initial eleven scaling regressions: eight failed and three passed before implementation. After cancelling WBF's common product and rescaling CCF compromise terms in log space, all eleven passed. Added two more CCF cases in which belief, disbelief, and whole-frame compromise share weights copies,copies,1, including 600 sources where every raw product underflows. The final regression file passes all thirteen in 0.65s.

Command: `uv run --no-project --offline --python C:/Users/Q/code/doxa/.venv/Scripts/python.exe python -B -m pytest -p no:cacheprovider tests/test_fusion_scaling.py -q`, with PYTHONPATH selecting this src and UV_CACHE_DIR selecting this worktree's cache. No dependency installation.

For the reported x=(.6,.3,.1), y=(.4,.3,.3) pools: at fourteen sources, CCF compromise mass is 3.0618e-10; at sixteen sources WBF denominator is 5.94864e-11. Both are positive representable values below 1e-9, so the original failure is a false zero test, not underflow. Actual product underflow is separately covered by the 400/600-source regressions.

Published and fetched draft PR https://github.com/ctoth/doxa/pull/4 at source commit 1c26cd11bd32e15b8e1541f1ddfb564ad390bda0. CI run 39339779362 passed typing and full-suite tests on Python 3.11, 3.12, and 3.13. Base rates, dogmatic routing, and the true-zero fallback remain at their current owners. No merge or ready transition performed. This final note update records those results; verify the note-only final head before handoff.
