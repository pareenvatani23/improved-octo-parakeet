# Outcome-Learning Pilot — Results (honest verdict)

**Data:** 1,260 real Product Hunt launches, topic = Artificial Intelligence,
2022–2024 (fetched via GitHub Actions using the PH API; rate-limited at 1,260).
**Label:** top-decile launch **votes** (a *buzz* proxy — not revenue).
**Features:** our 6-dimension idea taxonomy (keyword-extracted) + `word_count`, `has_ai`.
**Eval:** time-split — train on launches before 2023-12-16, test on later ones.

## Headline (out-of-time, test set)
| Model | ROC-AUC | PR-AUC | top-decile lift |
|---|---|---|---|
| Full (taxonomy + length) | 0.628 | 0.183 | 2.24× |
| **Taxonomy only (no length)** | **0.518** | 0.164 | 1.50× |
| word_count only | 0.612 | 0.174 | 1.99× |

Base rate 10.8%.

## Verdict: NO reliable signal in the idea taxonomy
The full model *looked* promising (2.24× lift), but the ablation shows that edge is
almost entirely **description length** — an effort/promotion confounder, not the
merit of the idea. Remove length and the idea features sit at **ROC 0.518 ≈ chance**.

So on this data, **the idea taxonomy does not predict launch buzz out-of-time.** The
honest read: we should *not* replace the rubric's weights with "learned from Product
Hunt buzz" — there is essentially nothing real to learn here beyond a triviality
(longer, more-detailed launch posts get slightly more upvotes).

This is the eval gate working as designed: it caught a confounder that would
otherwise have produced a falsely encouraging result.

## Why (legitimate reasons, not excuses)
1. **Weak label.** Product Hunt votes = launch-day buzz, driven by the maker's
   existing audience and promotion — not idea merit and certainly not revenue.
2. **Restricted cohort.** AI-only topic ⇒ almost no variance in the "trend"
   dimension (everything is AI), so that feature can't discriminate.
3. **Crude features.** Keyword extraction is noisy; LLM labels would be cleaner —
   but that won't rescue a fundamentally weak label.
4. **Length confound** dominates the little signal that exists.

## What this means for next steps
- **Do not** ship a "learned from buzz" model. The pilot proves the *pipeline*
  works end-to-end and that this *label* has no usable idea-signal.
- To get real signal you need a real **outcome label** — downloads / revenue /
  survival — which requires market data (Sensor Tower / data.ai / Appfigures).
  Broadening beyond AI-only would add variance, but the label is the real blocker.
- LLM feature extraction and a bigger, multi-category cohort are cheap upgrades to
  try *if* a real outcome label is in hand.

## Reproduce
```
# (fetch runs in GitHub Actions -> data/producthunt_cohort.csv)
python3 tools/extract_features.py
python3 tools/train_eval.py
```
