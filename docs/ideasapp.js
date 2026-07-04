/* UI + policy visualisation + narration for the App-Idea RL finder. */
"use strict";

const { DIMENSIONS, scoreIdea, PolicyAgent, totalIdeas, describeIdea } = window.IdeaRL;

let agent = new PolicyAgent({ seed: 7 });
let running = false;
let iters = 0, evaluated = 0;
let bestEver = -1, bestChoices = null;
const topIdeas = [];             // {key, choices, hit, k, confidence, synergies}
const history = [];              // {avg, best}
let lastSample = null, lastScore = 0;

const els = {
  run: document.getElementById("runBtn"),
  reset: document.getElementById("resetBtn"),
  speed: document.getElementById("speed"),
  status: document.getElementById("status"),
  iters: document.getElementById("iters"),
  best: document.getElementById("best"),
  avg: document.getElementById("avg"),
  evaluated: document.getElementById("evaluated"),
  policy: document.getElementById("policy"),
  board: document.getElementById("board"),
  thoughtLast: document.getElementById("thoughtLast"),
  thoughtPlan: document.getElementById("thoughtPlan"),
  thoughtLog: document.getElementById("thoughtLog"),
};
const curve = document.getElementById("curve");
const cctx = curve.getContext("2d");

function keyOf(c) { return c.join(","); }
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

// --- realisation log --------------------------------------------------------
const MAX_LOG = 5;
let realLog = [{ text: "Hit Search and I'll start inventing and testing app-idea combinations, steering probability toward the winners.", ok: null }];
function logIt(text, ok) {
  const top = realLog[realLog.length - 1];
  if (top && top.text === text) return;
  realLog.push({ text, ok });
  if (realLog.length > MAX_LOG) realLog.shift();
}

// --- leaderboard ------------------------------------------------------------
function considerIdea(choices) {
  const key = keyOf(choices);
  if (topIdeas.some(t => t.key === key)) return;
  const s = scoreIdea(choices);
  topIdeas.push({ key, choices, hit: s.hit, k: s.k, confidence: s.confidence, synergies: s.synergies });
  topIdeas.sort((a, b) => b.hit - a.hit);
  if (topIdeas.length > 6) topIdeas.length = 6;

  if (s.hit > bestEver + 1e-9) {
    bestEver = s.hit; bestChoices = choices;
    let why = s.synergies.length ? ` Synergy fired — ${s.synergies[0].why}.` : "";
    logIt(`New #1 (Hit ${s.hit.toFixed(0)}, virality k≈${s.k.toFixed(2)}): ${describeIdea(choices)}${why}`, true);
  }
}

// --- search loop (chunked) --------------------------------------------------
function chunk() {
  if (!running) return;
  const speed = parseInt(els.speed.value, 10);
  const steps = speed * 2;
  let lastMean = 0;
  for (let i = 0; i < steps; i++) {
    const r = agent.step();
    iters++;
    evaluated += agent.batch;
    lastMean = r.mean;
    considerIdea(r.bestChoices);
    lastSample = r.bestChoices; lastScore = scoreIdea(r.bestChoices).hit;
  }
  considerIdea(agent.greedyIdea());
  history.push({ avg: lastMean, best: bestEver });

  els.iters.textContent = iters;
  els.best.textContent = bestEver.toFixed(1);
  els.avg.textContent = lastMean.toFixed(1);
  els.evaluated.textContent = evaluated.toLocaleString();
  els.status.textContent = `Searching… ${iters} learning steps — probability is concentrating on the winners.`;
  drawCurve();
  setTimeout(chunk, 16);
}

function startSearch() {
  if (running) return;
  running = true;
  els.run.disabled = true;
  els.run.textContent = "Searching…";
  chunk();
}

// --- policy visualisation ---------------------------------------------------
function buildPolicyDom() {
  els.policy.innerHTML = "";
  for (let d = 0; d < DIMENSIONS.length; d++) {
    const dim = DIMENSIONS[d];
    const row = document.createElement("div"); row.className = "pdim";
    const h = document.createElement("div"); h.className = "pdim-label"; h.textContent = dim.label;
    row.appendChild(h);
    const bars = document.createElement("div"); bars.className = "pbars";
    for (let v = 0; v < dim.values.length; v++) {
      const b = document.createElement("div"); b.className = "pbar";
      b.innerHTML = `<span class="pname"></span><span class="ptrack"><span class="pfill"></span></span><span class="ppct"></span>`;
      bars.appendChild(b);
    }
    row.appendChild(bars);
    els.policy.appendChild(row);
  }
}
function updatePolicyDom() {
  const rows = els.policy.children;
  for (let d = 0; d < DIMENSIONS.length; d++) {
    const p = agent.probs(d);
    let bi = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[bi]) bi = i;
    const bars = rows[d].children[1].children;
    for (let v = 0; v < DIMENSIONS[d].values.length; v++) {
      const bar = bars[v];
      bar.querySelector(".pname").textContent = DIMENSIONS[d].values[v].label;
      bar.querySelector(".pfill").style.width = (p[v] * 100).toFixed(1) + "%";
      bar.querySelector(".ppct").textContent = (p[v] * 100).toFixed(0) + "%";
      bar.classList.toggle("plead", v === bi);
    }
  }
}

// --- leaderboard render -----------------------------------------------------
function renderBoard() {
  if (!topIdeas.length) { els.board.innerHTML = `<div class="empty">No ideas yet — hit Search.</div>`; return; }
  els.board.innerHTML = topIdeas.map((t, i) => {
    const syn = t.synergies.length ? `<div class="isyn">✦ ${esc(t.synergies.map(s => s.why).join(" · "))}</div>` : "";
    return `<div class="idea">
      <div class="irank">#${i + 1}</div>
      <div class="ibody">
        <div class="idesc">${esc(describeIdea(t.choices))}</div>
        ${syn}
      </div>
      <div class="iscore"><div class="ihit">${t.hit.toFixed(0)}</div><div class="imeta">k≈${t.k.toFixed(2)} · ${t.confidence}% conf</div></div>
    </div>`;
  }).join("");
}

// --- thoughts ---------------------------------------------------------------
function updateThoughts() {
  els.thoughtLast.textContent = lastSample
    ? `I just tested: ${describeIdea(lastSample)} → Hit ${lastScore.toFixed(0)}.`
    : "Waiting to start…";
  // plan: the currently most-likely (greedy) idea
  const g = agent.greedyIdea();
  const gs = scoreIdea(g);
  els.thoughtPlan.textContent = running || iters > 0
    ? `My current front-runner (Hit ${gs.hit.toFixed(0)}, k≈${gs.k.toFixed(2)}): ${describeIdea(g)}`
    : "I'll test thousands of combinations and let the numbers pull me toward the best.";
  const entries = realLog.slice().reverse();
  els.thoughtLog.innerHTML = entries.map((e, i) => {
    const cls = e.ok === true ? "ok" : e.ok === false ? "fail" : "neutral";
    const op = [1, 0.72, 0.52, 0.4, 0.32][i] ?? 0.3;
    return `<div class="entry ${cls}" style="opacity:${op}">${esc(e.text)}</div>`;
  }).join("");
}

// --- learning curve ---------------------------------------------------------
function fit(cv, c) { const dpr = window.devicePixelRatio || 1; const r = cv.getBoundingClientRect(); cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); c.setTransform(dpr, 0, 0, dpr, 0, 0); return r; }
function drawCurve() {
  const rect = fit(curve, cctx);
  cctx.clearRect(0, 0, rect.width, rect.height);
  cctx.fillStyle = "rgba(255,255,255,0.04)"; cctx.fillRect(0, 0, rect.width, rect.height);
  cctx.fillStyle = "#7d8bb5"; cctx.font = "11px system-ui, sans-serif";
  cctx.fillText("Hit 100", 2, 12); cctx.fillText("0", 2, rect.height - 3);
  cctx.fillText("best (green) & batch-avg (blue) Hit Score", rect.width - 240, 12);
  if (history.length < 2) return;
  const plot = (key, col) => {
    cctx.strokeStyle = col; cctx.lineWidth = 2; cctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = (i / (history.length - 1)) * (rect.width - 8) + 4;
      const y = rect.height - 4 - (history[i][key] / 100) * (rect.height - 8);
      if (i === 0) cctx.moveTo(x, y); else cctx.lineTo(x, y);
    }
    cctx.stroke();
  };
  plot("avg", "#1e88e5");
  plot("best", "#3ddc84");
}

// --- periodic UI refresh ----------------------------------------------------
setInterval(() => { updatePolicyDom(); renderBoard(); updateThoughts(); }, 160);

els.run.addEventListener("click", startSearch);
els.reset.addEventListener("click", () => {
  running = false;
  agent = new PolicyAgent({ seed: (Math.random() * 1e9) | 0 || 7 });
  iters = 0; evaluated = 0; bestEver = -1; bestChoices = null;
  topIdeas.length = 0; history.length = 0; lastSample = null;
  realLog = [{ text: "Reset — fresh policy, everything equally likely again. Hit Search.", ok: null }];
  els.iters.textContent = "0"; els.best.textContent = "—"; els.avg.textContent = "—"; els.evaluated.textContent = "0";
  els.run.disabled = false; els.run.textContent = "Search for a hit idea";
  els.status.textContent = `Fresh policy — every one of ${totalIdeas().toLocaleString()} possible ideas is equally likely. Hit Search.`;
  drawCurve(); updatePolicyDom(); renderBoard(); updateThoughts();
});
window.addEventListener("resize", drawCurve);

buildPolicyDom();
els.status.textContent = `Fresh policy — every one of ${totalIdeas().toLocaleString()} possible ideas is equally likely. Hit Search.`;
updatePolicyDom(); renderBoard(); updateThoughts(); drawCurve();
