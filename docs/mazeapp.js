/* UI + isometric 3D rendering + narration for the Maze RL Sim. */
"use strict";

const { MazeEnv, MazeQLearner, ACTION_NAMES, N, S, W, E, UP, DOWN } = window.MazeRL;

const MAZE_SEED = 5;
const world = new MazeEnv({ seed: MAZE_SEED });
let agent = new MazeQLearner(world.nStates, world.nActions);

// --- DOM --------------------------------------------------------------------
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const curve = document.getElementById("curve");
const cctx = curve.getContext("2d");
const els = {
  train: document.getElementById("trainBtn"),
  reset: document.getElementById("resetBtn"),
  greedy: document.getElementById("greedy"),
  speed: document.getElementById("speed"),
  status: document.getElementById("status"),
  episodes: document.getElementById("episodes"),
  success: document.getElementById("success"),
  best: document.getElementById("best"),
  epsilon: document.getElementById("epsilon"),
  thoughtLast: document.getElementById("thoughtLast"),
  thoughtPlan: document.getElementById("thoughtPlan"),
  thoughtLog: document.getElementById("thoughtLog"),
};

// --- canvas sizing ----------------------------------------------------------
function fitCanvas(cv, c) {
  const dpr = window.devicePixelRatio || 1;
  const rect = cv.getBoundingClientRect();
  cv.width = Math.round(rect.width * dpr);
  cv.height = Math.round(rect.height * dpr);
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return rect;
}

// --- isometric layout -------------------------------------------------------
const BASE = { TW: 46, TH: 26, GAP: 118, BOXH: 10, PAD: 26 };
let L = null;

function rawXY(f, r, c) {
  return { x: (c - r) * BASE.TW / 2, y: (c + r) * BASE.TH / 2 - f * BASE.GAP };
}
function computeLayout(rect) {
  let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
  for (let f = 0; f < world.floors; f++)
    for (let r = 0; r < world.rows; r++)
      for (let c = 0; c < world.cols; c++) {
        const p = rawXY(f, r, c);
        minX = Math.min(minX, p.x - BASE.TW / 2); maxX = Math.max(maxX, p.x + BASE.TW / 2);
        minY = Math.min(minY, p.y - BASE.TH / 2 - BASE.BOXH); maxY = Math.max(maxY, p.y + BASE.TH / 2 + BASE.BOXH);
      }
  const wpx = maxX - minX, hpx = maxY - minY;
  const scale = Math.min(1, (rect.width - 2 * BASE.PAD) / wpx, (rect.height - 2 * BASE.PAD) / hpx);
  const ox = BASE.PAD - minX * scale + ((rect.width - 2 * BASE.PAD) - wpx * scale) / 2;
  const oy = BASE.PAD - minY * scale + ((rect.height - 2 * BASE.PAD) - hpx * scale) / 2;
  L = { scale, ox, oy };
}
function iso(f, r, c) {
  const p = rawXY(f, r, c);
  return { x: L.ox + p.x * L.scale, y: L.oy + p.y * L.scale };
}
const S_ = (v) => v * L.scale;

// --- drawing helpers --------------------------------------------------------
function diamond(x, y, hw, hh) {
  ctx.beginPath();
  ctx.moveTo(x, y - hh); ctx.lineTo(x + hw, y); ctx.lineTo(x, y + hh); ctx.lineTo(x - hw, y);
  ctx.closePath();
}
function drawPad(f, r, c, top, side, visited) {
  const p = iso(f, r, c);
  const hw = S_(BASE.TW * 0.40), hh = S_(BASE.TH * 0.40), th = S_(BASE.BOXH);
  // left & right faces (thickness)
  ctx.fillStyle = side;
  ctx.beginPath();
  ctx.moveTo(p.x - hw, p.y); ctx.lineTo(p.x, p.y + hh);
  ctx.lineTo(p.x, p.y + hh + th); ctx.lineTo(p.x - hw, p.y + th);
  ctx.closePath(); ctx.fill();
  ctx.beginPath();
  ctx.moveTo(p.x + hw, p.y); ctx.lineTo(p.x, p.y + hh);
  ctx.lineTo(p.x, p.y + hh + th); ctx.lineTo(p.x + hw, p.y + th);
  ctx.closePath(); ctx.fill();
  // top
  ctx.fillStyle = visited ? "#2f6d8f" : top;
  diamond(p.x, p.y, hw, hh); ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.25)"; ctx.lineWidth = 1; ctx.stroke();
}
function drawConnector(f, a, b) {
  const pa = iso(f, a[0], a[1]), pb = iso(f, b[0], b[1]);
  ctx.strokeStyle = "#3b4e78";
  ctx.lineWidth = S_(BASE.TW * 0.20);
  ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
}

// --- the live escape attempt (smoothly animated) ----------------------------
let demoEnv = new MazeEnv({ seed: MAZE_SEED });
let demoState = demoEnv.reset();
let frc = { f: demoEnv.f, r: demoEnv.r, c: demoEnv.c };   // fractional position for gliding
let visited = new Set([demoState]);
let pauseFrames = 0;
let banner = null;
let demoLastAction = null, demoLastMoved = true;

// per-attempt bookkeeping for narration
let demoSteps = 0, demoRevisits = 0, demoBumps = 0, demoMaxFloor = 0;
let bestSteps = Infinity;

const MAX_LOG = 4;
let realLog = [{ text: "Hit Train, then watch me hunt for the exit. I'll narrate what I'm doing and what I learn.", ok: null, count: 1 }];
function logRealization(text, ok) {
  const top = realLog[realLog.length - 1];
  if (top && top.text === text) top.count += 1;
  else { realLog.push({ text, ok, count: 1 }); if (realLog.length > MAX_LOG) realLog.shift(); }
}

function newAttempt() {
  demoState = demoEnv.reset();
  frc = { f: demoEnv.f, r: demoEnv.r, c: demoEnv.c };
  visited = new Set([demoState]);
  banner = null;
  demoSteps = 0; demoRevisits = 0; demoBumps = 0; demoMaxFloor = 0;
}

function arrived() {
  return Math.abs(frc.f - demoEnv.f) < 0.04 && Math.abs(frc.r - demoEnv.r) < 0.04 && Math.abs(frc.c - demoEnv.c) < 0.04;
}

function doLogicStep() {
  const a = agent.act(demoState, els.greedy.checked);
  const res = demoEnv.step(a);
  demoLastAction = a; demoLastMoved = res.info.moved;
  demoSteps = res.info.t;
  if (!res.info.moved) demoBumps++;
  if (visited.has(res.state) && res.info.moved) demoRevisits++;
  visited.add(res.state);
  demoMaxFloor = Math.max(demoMaxFloor, res.info.f);
  demoState = res.state;

  if (res.done) {
    if (res.info.atExit) {
      bestSteps = Math.min(bestSteps, res.info.t);
      banner = { text: "✓ ESCAPED!", color: "#3ddc84", ok: true };
      logRealization(`Escaped in ${res.info.t} steps! Shortest route I've found is ${bestSteps} steps.`, true);
    } else {
      banner = { text: "✗ STILL TRAPPED", color: "#ff5d5d", ok: false };
      const uniq = visited.size;
      if (demoMaxFloor < world.exit.f) {
        logRealization(`Ran out of moves (${res.info.t}) and never even reached floor ${world.exit.f + 1}. I'm not finding the shafts up — I should explore more, especially upward.`, false);
      } else {
        logRealization(`Ran out of moves (${res.info.t}) without escaping. I re-tread corridors a lot (visited ${uniq} cells, ${demoRevisits} re-visits) — I keep going in circles instead of heading for the exit.`, false);
      }
    }
    pauseFrames = 50;
  }
}

function tickDemo() {
  if (pauseFrames > 0) { pauseFrames--; if (pauseFrames === 0) newAttempt(); return; }
  const speed = parseInt(els.speed.value, 10);
  // glide toward the current target cell; when there, take the next logic step
  const rate = 0.12 + speed * 0.05;
  frc.f += (demoEnv.f - frc.f) * rate;
  frc.r += (demoEnv.r - frc.r) * rate;
  frc.c += (demoEnv.c - frc.c) * rate;
  let guard = 0;
  while (arrived() && pauseFrames === 0 && guard++ < speed) {
    frc = { f: demoEnv.f, r: demoEnv.r, c: demoEnv.c };
    doLogicStep();
  }
}

// --- full scene -------------------------------------------------------------
function drawScene(rect) {
  computeLayout(rect);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const g = ctx.createLinearGradient(0, 0, 0, rect.height);
  g.addColorStop(0, "#0c1226"); g.addColorStop(1, "#161f3c");
  ctx.fillStyle = g; ctx.fillRect(0, 0, rect.width, rect.height);

  for (let f = 0; f < world.floors; f++) {
    // floor label
    const lp = iso(f, world.rows - 1, 0);
    ctx.fillStyle = "rgba(140,160,210,0.5)";
    ctx.font = `${Math.max(10, S_(15))}px system-ui, sans-serif`;
    ctx.textAlign = "right";
    ctx.fillText(`floor ${f + 1}`, lp.x - S_(BASE.TW * 0.6), lp.y);
    ctx.textAlign = "left";

    // corridors (open horizontal edges)
    for (let r = 0; r < world.rows; r++)
      for (let c = 0; c < world.cols; c++) {
        if (c + 1 < world.cols && world.open.has(world.edgeKey(world.idx(f, r, c), world.idx(f, r, c + 1))))
          drawConnector(f, [r, c], [r, c + 1]);
        if (r + 1 < world.rows && world.open.has(world.edgeKey(world.idx(f, r, c), world.idx(f, r + 1, c))))
          drawConnector(f, [r, c], [r + 1, c]);
      }

    // pads, back-to-front
    const cells = [];
    for (let r = 0; r < world.rows; r++) for (let c = 0; c < world.cols; c++) cells.push([r, c]);
    cells.sort((a, b) => (a[0] + a[1]) - (b[0] + b[1]));
    for (const [r, c] of cells) {
      const id = world.idx(f, r, c);
      const isStart = f === 0 && r === 0 && c === 0;
      const isExit = f === world.exit.f && r === world.exit.r && c === world.exit.c;
      let top = "#26324f", side = "#1a2238";
      if (isStart) { top = "#3a6df0"; side = "#274bb0"; }
      drawPad(f, r, c, isExit ? "#1e9e63" : top, isExit ? "#157049" : side, visited.has(id) && !isStart && !isExit);
      // shaft markers
      if (world.shaftUp.has(id)) drawShaft(f, r, c, "#ffd166", "▲");
      if (world.shaftDown.has(id)) drawShaft(f, r, c, "#7fd1ff", "▼");
      if (isExit) drawExit(f, r, c);
    }
  }

  drawRobot();
  drawBanner(rect);
}

function drawShaft(f, r, c, color, glyph) {
  const p = iso(f, r, c);
  ctx.fillStyle = color;
  ctx.font = `bold ${Math.max(9, S_(13))}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText(glyph, p.x, p.y - S_(BASE.TH * 0.55));
  ctx.textAlign = "left";
}
function drawExit(f, r, c) {
  const p = iso(f, r, c);
  const t = (animClock % 60) / 60;
  const pulse = 0.5 + 0.5 * Math.sin(t * Math.PI * 2);
  ctx.strokeStyle = `rgba(61,220,132,${0.4 + 0.4 * pulse})`;
  ctx.lineWidth = 2;
  diamond(p.x, p.y, S_(BASE.TW * 0.5 + pulse * 6), S_(BASE.TH * 0.5 + pulse * 3.5)); ctx.stroke();
  ctx.fillStyle = "#caffe0";
  ctx.font = `bold ${Math.max(10, S_(14))}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText("EXIT", p.x, p.y - S_(BASE.TH * 0.9));
  ctx.textAlign = "left";
}
function drawRobot() {
  const p = iso(frc.f, frc.r, frc.c);
  const bob = Math.sin(animClock * 0.2) * S_(2);
  const rad = S_(BASE.TW * 0.24);
  // shadow
  ctx.fillStyle = "rgba(0,0,0,0.3)";
  diamond(p.x, p.y + S_(2), rad * 0.9, rad * 0.5); ctx.fill();
  // body
  const cy = p.y - rad * 0.7 + bob;
  const grad = ctx.createRadialGradient(p.x - rad * 0.3, cy - rad * 0.3, rad * 0.2, p.x, cy, rad);
  grad.addColorStop(0, "#ffe9a8"); grad.addColorStop(1, "#ff8a3d");
  ctx.fillStyle = grad;
  ctx.beginPath(); ctx.arc(p.x, cy, rad, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.3)"; ctx.lineWidth = 1; ctx.stroke();
  // eye
  ctx.fillStyle = "#222";
  ctx.beginPath(); ctx.arc(p.x + rad * 0.25, cy - rad * 0.1, rad * 0.16, 0, Math.PI * 2); ctx.fill();
}
function drawBanner(rect) {
  if (!banner || pauseFrames <= 0) return;
  ctx.font = "bold 22px system-ui, sans-serif";
  ctx.textAlign = "center";
  const tw = ctx.measureText(banner.text).width;
  ctx.fillStyle = "rgba(8,12,26,0.8)";
  roundRect(rect.width / 2 - tw / 2 - 16, 14, tw + 32, 38, 8); ctx.fill();
  ctx.fillStyle = banner.color;
  ctx.fillText(banner.text, rect.width / 2, 40);
  ctx.textAlign = "left";
}
function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

let animClock = 0;
function loop() {
  const rect = fitCanvas(canvas, ctx);
  tickDemo();
  drawScene(rect);
  animClock++;
  requestAnimationFrame(loop);
}

// --- narration --------------------------------------------------------------
function describeAction(a, moved) {
  if (!moved) return `tried to go ${ACTION_NAMES[a]} but hit a wall`;
  if (a === UP) return "climbed up a shaft to the next floor";
  if (a === DOWN) return "climbed down a shaft";
  return `moved ${ACTION_NAMES[a]}`;
}
function describePlan() {
  const f = demoEnv.f, r = demoEnv.r, c = demoEnv.c;
  const ex = world.exit;
  const exploring = !els.greedy.checked && agent.epsilon > 0.3;
  const { allZero } = agent.bestAction(demoState);
  const onShaft = world.shaftUp.has(demoState) || world.shaftDown.has(demoState);

  let loc = `I'm on floor ${f + 1} of ${world.floors}, at cell (${r},${c}). The exit is on floor ${ex.f + 1}, far corner.`;
  if (onShaft) loc += " There's a shaft here I can take between floors.";

  let intent;
  if (exploring || allZero) intent = "I don't know this area yet, so I'm trying directions at random to map it out.";
  else intent = `Plan: ${describeAction(agent.bestAction(demoState).a, true)}.`;
  return `${loc} ${intent}`;
}
function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function updateThoughts() {
  if (!els.thoughtLast) return;
  els.thoughtLast.textContent =
    demoLastAction == null ? "Waiting to start…" : `I just ${describeAction(demoLastAction, demoLastMoved)}.`;
  els.thoughtPlan.textContent = describePlan();
  const entries = realLog.slice().reverse();
  els.thoughtLog.innerHTML = entries.map((e, i) => {
    const cls = e.ok === true ? "ok" : e.ok === false ? "fail" : "neutral";
    const op = [1, 0.7, 0.5, 0.38][i] ?? 0.3;
    const cnt = e.count > 1 ? ` <span class="cnt">×${e.count}</span>` : "";
    return `<div class="entry ${cls}" style="opacity:${op}">${esc(e.text)}${cnt}</div>`;
  }).join("");
}

// --- training ---------------------------------------------------------------
let training = false;
const history = [];
function runEpisode(env, ag) {
  let s = env.reset(), done = false, info = null;
  while (!done) {
    const a = ag.act(s, false);
    const r = env.step(a);
    ag.update(s, a, r.reward, r.state, r.done);
    s = r.state; done = r.done; info = r.info;
  }
  ag.decayEpsilon();
  return info;
}
function trainChunk(remaining, total, trainEnv, recent) {
  if (!training || remaining <= 0) {
    training = false;
    els.train.disabled = false;
    els.train.textContent = "Train (6000 episodes)";
    els.status.textContent = "Trained ✓ — watch it make a beeline for the exit!";
    els.greedy.checked = true;
    return;
  }
  const chunk = Math.min(40, remaining);
  for (let i = 0; i < chunk; i++) {
    const info = runEpisode(trainEnv, agent);
    recent.push(info.atExit ? 1 : 0);
    if (recent.length > 200) recent.shift();
  }
  const done = total - remaining + chunk;
  const succ = recent.reduce((a, b) => a + b, 0) / recent.length;
  history.push(succ);
  drawCurve();
  els.episodes.textContent = done;
  els.success.textContent = (succ * 100).toFixed(0) + "%";
  els.best.textContent = bestSteps === Infinity ? "—" : bestSteps;
  els.epsilon.textContent = agent.epsilon.toFixed(3);
  els.status.textContent = `Training… ${done}/${total} — watch it improve!`;
  setTimeout(() => trainChunk(remaining - chunk, total, trainEnv, recent), 16);
}
function startTraining() {
  if (training) return;
  training = true;
  history.length = 0;
  bestSteps = Infinity;
  agent = new MazeQLearner(world.nStates, world.nActions);
  els.train.disabled = true;
  els.train.textContent = "Training…";
  els.greedy.checked = false;
  const trainEnv = new MazeEnv({ seed: MAZE_SEED });
  trainChunk(6000, 6000, trainEnv, []);
}

function drawCurve() {
  const rect = fitCanvas(curve, cctx);
  cctx.clearRect(0, 0, rect.width, rect.height);
  cctx.fillStyle = "rgba(255,255,255,0.04)";
  cctx.fillRect(0, 0, rect.width, rect.height);
  cctx.fillStyle = "#7d8bb5";
  cctx.font = "11px system-ui, sans-serif";
  cctx.fillText("100%", 2, 12);
  cctx.fillText("0%", 2, rect.height - 3);
  cctx.fillText("escape rate vs. episode", rect.width - 150, 12);
  if (history.length < 2) return;
  cctx.strokeStyle = "#3ddc84";
  cctx.lineWidth = 2;
  cctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const x = (i / (history.length - 1)) * (rect.width - 8) + 4;
    const y = rect.height - 4 - history[i] * (rect.height - 8);
    if (i === 0) cctx.moveTo(x, y); else cctx.lineTo(x, y);
  }
  cctx.stroke();
}

// --- live stats / wire up ---------------------------------------------------
setInterval(() => {
  els.best.textContent = bestSteps === Infinity ? "—" : bestSteps;
  updateThoughts();
}, 150);

els.train.addEventListener("click", startTraining);
els.reset.addEventListener("click", () => {
  training = false;
  agent = new MazeQLearner(world.nStates, world.nActions);
  history.length = 0;
  bestSteps = Infinity;
  drawCurve();
  els.episodes.textContent = "0";
  els.success.textContent = "—";
  els.best.textContent = "—";
  els.epsilon.textContent = agent.epsilon.toFixed(3);
  els.greedy.checked = false;
  els.train.disabled = false;
  els.train.textContent = "Train (6000 episodes)";
  els.status.textContent = "Untrained — wandering blindly: it almost never finds the exit. Hit Train.";
  realLog = [{ text: "Reset — I've forgotten the maze. Train me again and watch me learn the route.", ok: null, count: 1 }];
  updateThoughts();
});
window.addEventListener("resize", drawCurve);

els.status.textContent = "Untrained — wandering blindly: it almost never finds the exit. Hit Train.";
updateThoughts();
drawCurve();
loop();
