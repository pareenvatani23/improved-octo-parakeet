# Outcome-Learning Version — Scope & Phase 0 Findings

Turning the App-Idea tool from a **hand-weighted rubric** (my priors) into a model
whose factor weights and synergies are **learned from apps that actually succeeded
vs. flopped**. The RL search layer we already built is reused unchanged — only the
reward function is swapped from rubric → trained model.

---

## Phase 0 — Feasibility spike (findings)

### Data-access reality from the build environment
Empirically tested on 2026-07-04. **Nothing is directly fetchable here:**

| Endpoint | Result |
|---|---|
| Apple iTunes Search API (`itunes.apple.com`) | ❌ 403 (egress policy) — also 403 via WebFetch (Apple bot-block) |
| Apple RSS charts (`rss.applemarketingtools.com`) | ❌ 403 (egress policy) |
| Product Hunt GraphQL (`api.producthunt.com`) | ❌ 403 |
| Google Play (`play.google.com`) | ❌ 403 |
| Crunchbase API (`api.crunchbase.com`) | ❌ 403 |
| Ahrefs / Semrush MCP | ❌ gated (paid plan required) |
| WebSearch tool | ✅ works, but returns article snippets — **not** structured bulk data |

**Conclusion:** the dataset must be **supplied by you** (a CSV/export, or an API key
for a source that a runner can reach). I build the entire pipeline; you provide the
data tap. A pilot can also run wherever you have data access.

### Candidate data sources — labels & features

| Source | Gives | Access | Cost | Survivorship-safe? |
|---|---|---|---|---|
| **Sensor Tower / data.ai** | real downloads + revenue, history | API/export | $$$$ (enterprise; trials exist) | ✅ retains delisted |
| **Appfigures / AppMagic / Apptopia** | downloads est., ratings, history | API | $–$$$ | ✅ mostly |
| **Product Hunt** | launch date, description, upvotes, makers | GraphQL (free token) | free | ✅ (flops stay listed) |
| **Apple iTunes API + RSS** | metadata, genre, price, rating counts, current charts | free | free | ❌ current-only, no history |
| **google-play-scraper (npm)** | current metadata, review counts | scrape | free (ToS-gray) | ❌ current-only |
| **Crunchbase / PitchBook** | funding, stage, survival | API | $$–$$$$ | ✅ |
| **Google Trends** | trend momentum (as-of-launch) | unofficial | free-ish | n/a (feature, not label) |

### The label options (pick one to start)
- **Revenue/downloads threshold @ 12mo** (needs paid data) — cleanest.
- **Reached top-N category chart** within X months — semi-public.
- **Raised Series A+ / alive @ 3yrs** (Crunchbase) — easy, noisy.
- **Product Hunt buzz + later traction proxy** — free, obtainable, cohort-complete.

Recommended start: **binary "hit"** on whichever real metric your chosen source gives.
Base rate will be low (~2–5%); that drives the eval design below.

---

## Recommended pilot (cheapest credible first step)
**One category, ~1–2k apps, one export.** Prove the pipeline + test whether *any*
learnable signal exists **before** anyone pays for Sensor Tower.

- Cohort: all apps launched in a fixed window (e.g. AI-productivity apps 2022–2024),
  **including dead ones** (avoid survivorship bias).
- Free route: Product Hunt launches (cohort + description + buzz) — self-contained.
- Or: a market-data export you already have access to.

### Exact data spec to hand me (CSV, one row per app)
Minimum viable in **bold**; more is better.

```
**app_id**, **name**, **category**, **launch_date**,
**description**            # free text — I LLM-extract the taxonomy from this
price_model                # free | paid | freemium | subscription
has_iap                    # true/false
platform                   # ios | android | web
# --- ONE outcome column (as-of a horizon), plus its date ---
**outcome_metric**         # downloads_12mo | revenue_12mo | peak_chart_rank |
                           #   rating_count | upvotes | funding_usd | still_alive
**outcome_asof_date**
```
I map `description` → our 6 taxonomy dimensions (trend, audience, mechanic, virality
loop, monetization, format) with an LLM, so you don't have to pre-label them.

---

## Full plan (Phases 1–4)

**1. Dataset build** — sample cohort at launch → LLM feature-extract → attach
outcomes measured strictly *after* launch. (~2–3 days once data is in hand.)

**2. Model + evaluation** — gradient-boosted trees + logistic baseline → calibrated
P(hit). Trees learn interactions → **replaces my hand-coded synergies with
discovered ones.** SHAP for interpretable weights; bootstrap for the uncertainty
bands (reuse the band UI we built). (~2 days.)

**3. RL reward swap** — plug learned P(hit) into the existing policy-gradient agent;
re-run the search; ship calibrated probabilities. (~1 day.)

**4. Validation loop** — smoke-test the top picks (landing-page / ad CTR / waitlist);
feed real results back. (ongoing.)

### The honesty gate (what makes this real, not theatre)
- **Time-split validation:** train on launches before date D, test *after*. In-sample
  fit is meaningless; out-of-time lift is the only truth.
- **Metrics:** PR-AUC (classes imbalanced), calibration curve, **top-decile precision**
  vs. base rate. Beat baselines (random / biggest-category / current rubric) or admit it.
- **Biases we kill explicitly:** survivorship (capture launch cohort), look-ahead
  leakage (features as-of-launch, outcomes later), incumbent confounds (drop
  big/funded-at-launch), concept drift (measured by the time-split).
- **Kill criterion (stated upfront):** if out-of-time top-decile lift ≈ 1×, report
  "the signal isn't there" rather than ship a pretty fiction.

### Honest ceiling
App success is genuinely hard to predict — top VCs are wrong most of the time. A
realistic win is **~2–3× top-decile lift** + **interpretable learned factors** +
reliably filtering out obviously-bad ideas — not a crystal ball.

---

## Phase 0 decision needed from you
1. **Which label / data source** (drives everything): paid market data (best), or the
   free Product Hunt pilot (cheapest), or an export you already have?
2. **Budget** for outcome data, if any.

Once you pick, Phase 1 starts. I can also build the pipeline now against a **synthetic
stand-in** so it's ready the moment real data lands.
