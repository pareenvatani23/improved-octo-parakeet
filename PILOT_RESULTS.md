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

## Update: broadened pilot (2,258 apps, 5+ categories)
Re-ran after adding Productivity, Fintech, Health & Fitness, Developer Tools to
the AI cohort (2,258 apps; ~10% hit rate; same time-split). This adds real
variance to the "trend" dimension, which was the main worry with AI-only.

| Model | ROC-AUC | top-decile lift |
|---|---|---|
| full (taxonomy + length) | 0.620 | 1.61× |
| taxonomy + has_ai (no length) | 0.583 | 1.61× |
| **taxonomy ONLY** | **0.512 ≈ chance** | 1.47× |
| word_count only | 0.590 | 1.47× |

**Same verdict, now confirmed across categories.** The out-of-time signal
decomposes entirely into two confounders:
1. **description length** (`word_count`) — launch effort/detail, ROC 0.590 alone;
2. **has_ai** — an AI-hype/timing artifact (AI apps drew more 2022–23 buzz).

The **idea taxonomy itself** — the 6 dimensions we'd actually use to *generate*
ideas — is at **ROC 0.512 ≈ chance**. Broadening did **not** rescue it, exactly as
predicted: the ceiling is the *label* (buzz), not the features or the cohort size.

**Final pilot conclusion:** Product Hunt buzz cannot teach the model what makes an
idea good. The pipeline is proven and reusable; to get real signal it must be
pointed at a real outcome label (downloads / revenue / survival), not votes.

## Update 2: real install labels (Google Play, 1,200 apps) — data was unusable
Fetched a Google Play cohort via search (real `minInstalls` label) to get a
*real* outcome instead of buzz. The pipeline ran, but the **data itself is not
valid for the question**, and the honest move is to say so:

- **Not a launch cohort.** Search surfaces popular survivors, so the sample spans
  launch years **2010–2026** with a **median of 1,000,000 installs** — essentially
  no real flops are present (survivorship + popularity bias).
- **The label leaks age.** Installs are cumulative, so old apps dominate the top
  decile: **92% of the "install-hits" (100M+) launched before 2023.** The label is
  effectively "how long has this existed," not "is the idea good."
- **Degenerate eval.** After the time-split, the test base rate collapses to 2.8%,
  and the taxonomy-only "3× lift" is ~1 app out of ~10 test hits — **noise, not
  signal.**

**Conclusion:** free Google Play data (search-sourced) cannot answer the question.
A valid test needs a **launch cohort** — every app released in a narrow window,
*including flops*, with installs measured at a fixed time-since-launch. Free web
sources don't provide that; it requires a market-data provider that lists new
releases comprehensively (Sensor Tower / data.ai / AppMagic). This is the crux
flagged in Phase 0, now demonstrated empirically.

## Overall verdict across both pilots
- **Product Hunt (buzz label):** idea taxonomy at chance; signal was length + hype.
- **Google Play (install label):** data unusable — survivorship + age bias.

Neither *free* source can validly show that idea features predict success. The
pipeline is proven and reusable; the blocker is a **representative launch cohort
with a real outcome**, which is paid. Honest recommendation: keep the App-Idea
tool as the transparent *rubric* it now is, and only revisit outcome-learning
with proper market data.

## Update 3: does ICON COLOR predict downloads? (No — it's a brand/age mirage)
Extracted each app's icon average colour + hue bucket (1,200 apps) and joined to
installs. `tools/fetch_icons.mjs` + `tools/analyze_color.py`.

**Raw (confounded) result looked juicy:** red icons had a 20.8% install-hit rate
vs ~9% for blue/green. But it does not survive scrutiny:

- **It's a few famous old brands.** The top "warm-coloured" apps are Google Meet
  (10B installs, yellow), Snapchat & Pinterest (2012), Google Keep, and a cluster
  of ~2013 video editors (InShot, KineMaster, VivaVideo — red/orange). Their colour
  is branding; their installs come from a decade of accumulation.
- **Age confound.** Install-hits launched in **2015** on average vs **2019** for the
  cohort; the warm-colour hits are the *oldest* (median 2013). So "red wins" really
  means "old famous apps happen to be red."
- **Fails the honest test.** Out-of-time, adding colour to `genre+year` *lowered*
  ROC (0.716 → 0.636); colour-only ROC 0.548. Colour adds nothing beyond
  category/age, and the test fold is degenerate anyway (2.8% base rate).

**Verdict:** no, icon colour is not a usable download signal here — the apparent
effect is brand + survivorship + age, not colour. (A real test would need a launch
cohort incl. flops, colour measured at launch, controlled for category — i.e. the
same paid-data requirement as everything else. Note: icon-colour A/B tests *do*
move install *conversion* in industry, but that's a within-app experiment, not
"colour predicts total downloads across apps.")

## Update 4: the clean test — PH launch list (incl. flops) + real installs
The definitive free version of "get apps by category, label flop vs hit by their
numbers." We took the Product Hunt launch list (which **includes flops by design**)
and matched each app to Google Play for its **real install count**
(`tools/enrich_ph_installs.mjs`). 253 of 2,258 PH apps were on Google Play with a
confident name match.

**This fixed the survivorship problem** — the install distribution finally spans
the whole range: min **0**, p25 **100**, median **5,000**, p90 **1,000,000**
(vs. the search-sourced pull whose *median* was 1,000,000 — i.e. no flops at all).

**Result (out-of-time, real install label, flops present):**
| Model | ROC-AUC | top-decile lift |
|---|---|---|
| full (taxonomy + length) | 0.589 | 0.78× |
| **taxonomy ONLY** | **0.457 (below chance)** | 0.00× |
| word_count only | — | 1.55× |

**Verdict: even with the flops in the sample, the idea taxonomy does NOT predict
install success.** Taxonomy-only is *below* chance; only description length has a
weak edge (the effort proxy we've seen throughout). This is the strongest version
of the finding because it's no longer explainable by survivorship — the failures
were included and the idea features still carry no signal.

Caveats: small sample (253; ~76 in the test fold), and only PH apps that shipped
to Google Play are covered. But the direction is consistent with every prior run.

## Overall conclusion (all pilots)
Across buzz, search-sourced installs, icon colour, and now a flop-inclusive
install cohort: **app success is not predictable from idea/creative features alone
in any free data we can obtain.** The only recurring "signal" is description length
(effort), which is not idea merit. Execution, marketing, timing and luck — the
things these features can't see — dominate. To even attempt a real answer you need
the private drivers (retention, spend, k-factor) or at minimum a large, clean
launch cohort with velocity labels (paid market data, e.g. AppTweak/Sensor Tower).

## Reproduce
```
# (fetch runs in GitHub Actions -> data/producthunt_cohort.csv)
python3 tools/extract_features.py
python3 tools/train_eval.py
```
