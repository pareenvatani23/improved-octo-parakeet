/* App-Idea RL — a creative reinforcement-learning idea finder.
 *
 * An "app idea" is one choice from each of 6 dimensions (trend, audience,
 * mechanic, virality loop, monetization, format). A transparent scoring model
 * turns any combination into a "Hit Score" from demand, virality, competition,
 * monetization, feasibility and discovered cross-feature synergies.
 *
 * The agent is a policy-gradient learner (REINFORCE): it keeps a probability
 * distribution over the values of each dimension, repeatedly *samples* whole
 * ideas, scores them, and nudges the probabilities toward the choices that beat
 * its running average. It is genuinely creative — it invents and tests novel
 * combinations rather than reading a list — and it converges on the highest
 * scoring regions of a ~47,000-idea space.
 *
 * NOTE: the Hit Score is a heuristic model built from assumed factor weights,
 * not real market data. It is a plausible, tunable proxy — not a guarantee.
 *
 * Browser + Node friendly (globalThis.IdeaRL / module.exports).
 */
"use strict";

(function () {

// Each value carries 5 attributes in 0..1:
//   d = demand/trend momentum, v = virality, c = competition/saturation (bad),
//   m = monetization potential, f = feasibility.
function V(label, d, v, c, m, f, note) { return { label, d, v, c, m, f, note: note || "" }; }

const DIMENSIONS = [
  {
    key: "trend", label: "Trend hook", values: [
      V("AI agents that do tasks for you", 0.97, 0.72, 0.82, 0.65, 0.5),
      V("Longevity & healthspan", 0.86, 0.6, 0.5, 0.72, 0.52),
      V("Mental health & ADHD focus", 0.9, 0.66, 0.62, 0.6, 0.62),
      V("Creator economy tools", 0.82, 0.86, 0.7, 0.62, 0.62),
      V("Financial independence", 0.8, 0.5, 0.62, 0.82, 0.6),
      V("Dating & loneliness", 0.76, 0.82, 0.72, 0.6, 0.6),
    ]
  },
  {
    key: "audience", label: "Audience", values: [
      V("Gen Z", 0.85, 0.92, 0.72, 0.42, 0.8),
      V("Creators & solopreneurs", 0.82, 0.82, 0.62, 0.72, 0.72),
      V("Knowledge workers", 0.8, 0.5, 0.72, 0.82, 0.82),
      V("Millennial parents", 0.75, 0.5, 0.5, 0.72, 0.8),
      V("Students", 0.72, 0.82, 0.62, 0.32, 0.82),
      V("SMB owners", 0.72, 0.42, 0.52, 0.88, 0.6),
    ]
  },
  {
    key: "mechanic", label: "Core mechanic", values: [
      V("AI companion / coach", 0.92, 0.62, 0.82, 0.62, 0.5),
      V("Shareable AI generator", 0.86, 0.92, 0.72, 0.55, 0.62),
      V("Habit streaks", 0.72, 0.72, 0.72, 0.52, 0.9),
      V("Gamified challenge", 0.76, 0.86, 0.6, 0.52, 0.72),
      V("Tracker & insights", 0.66, 0.42, 0.6, 0.62, 0.85),
      V("Marketplace / matching", 0.72, 0.6, 0.72, 0.82, 0.5),
    ]
  },
  {
    key: "loop", label: "Virality loop", values: [
      V("Shareable result card", 0.7, 0.92, 0.6, 0.5, 0.85),
      V("Duet / remix / challenge", 0.72, 0.9, 0.62, 0.5, 0.7),
      V("Invite-to-unlock", 0.6, 0.76, 0.5, 0.55, 0.8),
      V("Leaderboard / streak flex", 0.65, 0.72, 0.55, 0.5, 0.82),
      V("Duo / group accountability", 0.62, 0.72, 0.45, 0.55, 0.75),
      V("Referral reward", 0.58, 0.6, 0.5, 0.6, 0.85),
    ]
  },
  {
    key: "money", label: "Monetization", values: [
      V("Freemium + AI credits", 0.78, 0.5, 0.6, 0.78, 0.7),
      V("Subscription", 0.72, 0.4, 0.6, 0.82, 0.75),
      V("B2B seats", 0.7, 0.35, 0.5, 0.9, 0.6),
      V("Marketplace fee", 0.7, 0.5, 0.7, 0.82, 0.55),
      V("Ads + in-app purchases", 0.68, 0.55, 0.7, 0.5, 0.8),
      V("One-time purchase", 0.5, 0.4, 0.4, 0.42, 0.85),
    ]
  },
  {
    key: "format", label: "Format", values: [
      V("AI chat / voice-first", 0.9, 0.6, 0.75, 0.6, 0.55),
      V("Short-video native", 0.82, 0.9, 0.75, 0.5, 0.55),
      V("Camera / AR", 0.78, 0.85, 0.6, 0.5, 0.5),
      V("Lockscreen / widget", 0.7, 0.68, 0.5, 0.5, 0.7),
      V("Wearable companion", 0.72, 0.55, 0.45, 0.62, 0.5),
      V("Simple utility app", 0.6, 0.45, 0.55, 0.55, 0.9),
    ]
  },
];

// Cross-feature synergies: if the listed value labels co-occur, add bonus
// (scaled by the fraction present, so the landscape is smooth enough to learn).
const SYNERGIES = [
  { vals: ["AI agents that do tasks for you", "Creators & solopreneurs", "Shareable AI generator"], bonus: 1.4, why: "AI that produces shareable output for creators is the hottest wedge right now" },
  { vals: ["AI agents that do tasks for you", "SMB owners", "B2B seats"], bonus: 1.2, why: "SMBs pay real money for agents that save labour" },
  { vals: ["Mental health & ADHD focus", "Gen Z", "Gamified challenge"], bonus: 1.0, why: "Gen Z engages with mental-health as playful daily challenges" },
  { vals: ["Habit streaks", "Gen Z", "Leaderboard / streak flex"], bonus: 0.9, why: "streak-flexing drives Gen Z retention loops" },
  { vals: ["Creator economy tools", "Shareable result card", "Short-video native"], bonus: 1.0, why: "result cards are free distribution on short-video feeds" },
  { vals: ["Dating & loneliness", "Gen Z", "Duet / remix / challenge"], bonus: 0.9, why: "social dating formats spread through remix culture" },
  { vals: ["Longevity & healthspan", "Wearable companion", "Subscription"], bonus: 0.8, why: "wearable + subscription is a proven health-monetization pattern" },
  { vals: ["Financial independence", "SMB owners", "B2B seats"], bonus: 0.7, why: "money tools for SMBs convert to seats" },
  { vals: ["Shareable AI generator", "Camera / AR"], bonus: 0.6, why: "camera-native AI generators go viral fastest" },
  { vals: ["AI companion / coach", "Mental health & ADHD focus"], bonus: 0.6, why: "AI coaching fits the mental-health surge" },
];

// --- real-keyword-data wiring ----------------------------------------------
// Each groundable feature maps to a target keyword. When live metrics are
// supplied (globalThis.KEYWORD_DATA = { "<keyword>": {volume, difficulty} },
// e.g. loaded from keyword_data.js populated by Ahrefs/Semrush), the value's
// demand is derived from real search volume and its competition from real
// keyword difficulty. Absent data, the authored estimate is used.
const KW = {
  "AI agents that do tasks for you": "ai agent",
  "Longevity & healthspan": "longevity",
  "Mental health & ADHD focus": "adhd",
  "Creator economy tools": "creator economy",
  "Financial independence": "financial independence",
  "Dating & loneliness": "dating app",
  "AI companion / coach": "ai companion",
  "Shareable AI generator": "ai image generator",
  "Habit streaks": "habit tracker",
  "Gamified challenge": "gamification app",
  "Tracker & insights": "sleep tracker",
  "Marketplace / matching": "marketplace app",
  "AI chat / voice-first": "ai chatbot",
  "Short-video native": "short video app",
  "Camera / AR": "augmented reality app",
  "Lockscreen / widget": "widget app",
  "Wearable companion": "smartwatch app",
  "Simple utility app": "utility app",
};

function demandFromVolume(v) {
  // log-normalise monthly search volume to 0..1 (200 -> ~0, 1M -> ~1)
  const x = Math.log10(Math.max(10, v));
  return Math.max(0.05, Math.min(1, (x - 2.3) / (6 - 2.3)));
}

let GROUNDED = 0;
function applyGrounding() {
  const data = (typeof globalThis !== "undefined" && globalThis.KEYWORD_DATA) || {};
  GROUNDED = 0;
  for (const dim of DIMENSIONS) {
    for (const val of dim.values) {
      val.kw = KW[val.label] || null;
      val.grounded = false;
      const rec = val.kw && data[val.kw];
      if (rec) {
        if (typeof rec.volume === "number") { val.d = demandFromVolume(rec.volume); val.volume = rec.volume; }
        if (typeof rec.difficulty === "number") { val.c = Math.max(0, Math.min(1, rec.difficulty / 100)); val.kd = rec.difficulty; }
        if (typeof rec.volume === "number" || typeof rec.difficulty === "number") { val.grounded = true; GROUNDED++; }
      }
    }
  }
  _cal = null; // force recalibration against grounded numbers
}
function groundingStatus() {
  let total = 0;
  for (const dim of DIMENSIONS) for (const v of dim.values) if (v.kw) total++;
  return { grounded: GROUNDED, groundable: total };
}

const WEIGHTS = { d: 1.7, v: 1.6, c: 1.15, m: 0.95, f: 0.6 };
const SPREAD = 1.5;

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// Raw (uncalibrated) score = weighted attributes + synergies.
function rawCore(choices) {
  const a = ideaAttrs(choices);
  const syn = synergyScore(choices);
  return { raw: WEIGHTS.d * a.demand + WEIGHTS.v * a.viral - WEIGHTS.c * a.comp
    + WEIGHTS.m * a.money + WEIGHTS.f * a.feas + syn.total, a, syn };
}

// Self-calibrate: standardise raw scores across the whole idea space so the
// average idea lands near 50 and Hit Scores spread over a meaningful range.
let _cal = null;
function calibrate() {
  if (_cal) return _cal;
  const sizes = DIMENSIONS.map(d => d.values.length);
  const idx = sizes.map(() => 0);
  let n = 0, sum = 0, sumSq = 0;
  while (true) {
    const r = rawCore(idx).raw; sum += r; sumSq += r * r; n++;
    let k = sizes.length - 1;
    while (k >= 0) { idx[k]++; if (idx[k] < sizes[k]) break; idx[k] = 0; k--; }
    if (k < 0) break;
  }
  const mean = sum / n, std = Math.sqrt(sumSq / n - mean * mean) || 1;
  _cal = { mean, std };
  return _cal;
}

// choices = array of value-indices, one per dimension
function ideaAttrs(choices) {
  const g = (dim, key) => DIMENSIONS[dim].values[choices[dim]][key];
  const demand = 0.4 * g(0, "d") + 0.25 * g(1, "d") + 0.2 * g(2, "d") + 0.15 * g(5, "d");
  const viral = 0.4 * g(3, "v") + 0.25 * g(1, "v") + 0.2 * g(2, "v") + 0.15 * g(5, "v");
  const comp = 0.4 * g(0, "c") + 0.35 * g(2, "c") + 0.25 * g(5, "c");
  const money = 0.6 * g(4, "m") + 0.4 * g(2, "m");
  const feas = 0.5 * g(2, "f") + 0.3 * g(5, "f") + 0.2 * g(4, "f");
  return { demand, viral, comp, money, feas };
}

function labelSet(choices) {
  const s = new Set();
  for (let d = 0; d < DIMENSIONS.length; d++) s.add(DIMENSIONS[d].values[choices[d]].label);
  return s;
}

function synergyScore(choices) {
  const set = labelSet(choices);
  let total = 0; const hits = [];
  for (const syn of SYNERGIES) {
    let present = 0;
    for (const v of syn.vals) if (set.has(v)) present++;
    const frac = present / syn.vals.length;
    if (frac > 0) { total += syn.bonus * frac; if (frac === 1) hits.push(syn); }
  }
  return { total, hits };
}

function scoreIdea(choices) {
  const { raw, a, syn } = rawCore(choices);
  const cal = calibrate();
  const z = SPREAD * (raw - cal.mean) / cal.std;
  const hit = 100 * sigmoid(z);
  const k = 0.5 + 1.5 * a.viral;                 // projected virality coefficient
  const confidence = Math.round(100 * Math.max(0.15, Math.min(0.95, 0.55 + 0.5 * a.feas - 0.4 * a.comp)));
  return { hit, k, confidence, attrs: a, synergies: syn.hits, z };
}

// ------------------------------------------------------------------ agent
class PolicyAgent {
  constructor(cfg = {}) {
    this.lr = cfg.lr ?? 0.35;
    this.entropy = cfg.entropy ?? 0.02;
    this.batch = cfg.batch ?? 24;
    this.baseline = 0; this.baselineInit = false;
    this._seed = (cfg.seed ?? 7) >>> 0 || 1;
    // logits[d][v]
    this.logits = DIMENSIONS.map(dim => new Float64Array(dim.values.length));
  }
  _r() { let x = this._seed; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this._seed = x >>> 0; return (this._seed % 1e7) / 1e7; }
  probs(d) {
    const lg = this.logits[d]; let mx = -Infinity;
    for (const x of lg) mx = Math.max(mx, x);
    const e = new Float64Array(lg.length); let s = 0;
    for (let i = 0; i < lg.length; i++) { e[i] = Math.exp(lg[i] - mx); s += e[i]; }
    for (let i = 0; i < e.length; i++) e[i] /= s;
    return e;
  }
  sampleDim(d) {
    const p = this.probs(d); let r = this._r(), acc = 0;
    for (let i = 0; i < p.length; i++) { acc += p[i]; if (r <= acc) return i; }
    return p.length - 1;
  }
  sampleIdea() { return DIMENSIONS.map((_, d) => this.sampleDim(d)); }
  greedyIdea() { return DIMENSIONS.map((_, d) => { const p = this.probs(d); let bi = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[bi]) bi = i; return bi; }); }

  // one learning step over a batch of self-sampled ideas
  step() {
    const samples = [];
    let mean = 0, bestSc = -1, bestChoices = null;
    for (let b = 0; b < this.batch; b++) {
      const choices = this.sampleIdea();
      const sc = scoreIdea(choices).hit;
      samples.push({ choices, sc });
      mean += sc;
      if (sc > bestSc) { bestSc = sc; bestChoices = choices; }
    }
    mean /= this.batch;
    if (!this.baselineInit) { this.baseline = mean; this.baselineInit = true; }
    else this.baseline = 0.9 * this.baseline + 0.1 * mean;
    // advantage normalisation
    let varr = 0; for (const s of samples) varr += (s.sc - mean) ** 2; varr /= this.batch;
    const std = Math.sqrt(varr) + 1e-6;

    const grad = DIMENSIONS.map(dim => new Float64Array(dim.values.length));
    for (const s of samples) {
      const adv = (s.sc - this.baseline) / std;
      for (let d = 0; d < DIMENSIONS.length; d++) {
        const p = this.probs(d); const chosen = s.choices[d];
        for (let i = 0; i < p.length; i++) {
          const dlog = (i === chosen ? 1 : 0) - p[i];
          grad[d][i] += adv * dlog;
        }
      }
    }
    for (let d = 0; d < DIMENSIONS.length; d++) {
      const p = this.probs(d);
      for (let i = 0; i < this.logits[d].length; i++) {
        const ent = -this.entropy * (Math.log(p[i] + 1e-9) + 1); // entropy regularisation
        this.logits[d][i] += this.lr * (grad[d][i] / this.batch + ent);
      }
    }
    return { mean, best: bestSc, bestChoices, samples };
  }
}

// brute-force helpers (space is small enough to enumerate)
function totalIdeas() { return DIMENSIONS.reduce((a, d) => a * d.values.length, 1); }
function describeIdea(choices) {
  const pick = (d) => DIMENSIONS[d].values[choices[d]].label;
  return `A ${pick(2).toLowerCase()} app in ${pick(5).toLowerCase()} form for ${pick(1)}, built around ${pick(0).toLowerCase()}, that spreads via ${pick(3).toLowerCase()} and earns through ${pick(4).toLowerCase()}.`;
}

applyGrounding();  // consume live keyword data if present; else authored estimates

const IdeaRL = { DIMENSIONS, SYNERGIES, scoreIdea, ideaAttrs, PolicyAgent, totalIdeas, describeIdea, applyGrounding, groundingStatus, KW };
if (typeof globalThis !== "undefined") globalThis.IdeaRL = IdeaRL;
if (typeof module !== "undefined" && module.exports) module.exports = IdeaRL;

})();
