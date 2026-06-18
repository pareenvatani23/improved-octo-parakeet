/* UI + rendering for the in-browser Robot Stairs RL Sim. */
"use strict";

const { StairClimbEnv, TabularQLearner, ACTION_NAMES } = window.RobotStairs;

// --- shared world + agent ---------------------------------------------------
const world = new StairClimbEnv();
world.seed(7);
let agent = new TabularQLearner(world.nActions);

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
  stairs: document.getElementById("stairs"),
  epsilon: document.getElementById("epsilon"),
};

// --- canvas sizing (crisp on HiDPI) ----------------------------------------
function fitCanvas(cv, c) {
  const dpr = window.devicePixelRatio || 1;
  const rect = cv.getBoundingClientRect();
  cv.width = Math.round(rect.width * dpr);
  cv.height = Math.round(rect.height * dpr);
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return rect;
}

// --- world->screen mapping --------------------------------------------------
function makeMap(rect) {
  const pad = 28;
  const worldW = world.goalX + world.stepWidth;
  const worldH = world.topHeight + 0.9;
  const sx = (rect.width - 2 * pad) / worldW;
  const sy = (rect.height - 2 * pad) / worldH;
  const s = Math.min(sx, sy);
  return {
    X: (x) => pad + x * s,
    Y: (y) => rect.height - pad - y * s,
    s,
  };
}

function drawScene(rect, env) {
  const m = makeMap(rect);
  ctx.clearRect(0, 0, rect.width, rect.height);

  // sky gradient
  const g = ctx.createLinearGradient(0, 0, 0, rect.height);
  g.addColorStop(0, "#0f1730");
  g.addColorStop(1, "#1a2444");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, rect.width, rect.height);

  // staircase
  ctx.beginPath();
  ctx.moveTo(m.X(0), m.Y(-0.5));
  for (let i = 0; i < env.numSteps; i++) {
    const h = i * env.stepHeight;
    ctx.lineTo(m.X(i * env.stepWidth), m.Y(h));
    ctx.lineTo(m.X((i + 1) * env.stepWidth), m.Y(h));
  }
  ctx.lineTo(m.X(env.goalX), m.Y(env.topHeight));
  ctx.lineTo(m.X(env.goalX + env.stepWidth), m.Y(env.topHeight));
  ctx.lineTo(m.X(env.goalX + env.stepWidth), m.Y(-0.5));
  ctx.closePath();
  const sg = ctx.createLinearGradient(0, m.Y(env.topHeight), 0, m.Y(-0.5));
  sg.addColorStop(0, "#4a5e8a");
  sg.addColorStop(1, "#2c3a5e");
  ctx.fillStyle = sg;
  ctx.fill();
  ctx.strokeStyle = "#8fa4d4";
  ctx.lineWidth = 2;
  ctx.stroke();

  // goal flag at top
  const fx = m.X(env.goalX + env.stepWidth * 0.5);
  const fy = m.Y(env.topHeight);
  ctx.strokeStyle = "#cbd5f5";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(fx, fy);
  ctx.lineTo(fx, fy - 34);
  ctx.stroke();
  ctx.fillStyle = "#3ddc84";
  ctx.beginPath();
  ctx.moveTo(fx, fy - 34);
  ctx.lineTo(fx + 22, fy - 27);
  ctx.lineTo(fx, fy - 20);
  ctx.closePath();
  ctx.fill();

  // robot
  const rx = m.X(env.x);
  const ry = m.Y(env.y);
  const r = Math.max(9, m.s * 0.16);
  const failed = banner && !banner.ok && pauseFrames > 0;  // collapsed pose

  ctx.save();
  ctx.translate(rx, ry);
  if (failed) ctx.rotate(-Math.PI / 2.2);  // tip over when it collapses
  // legs
  ctx.strokeStyle = "#ffd166";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(-r * 0.4, 0); ctx.lineTo(-r * 0.4, r * 0.9);
  ctx.moveTo(r * 0.4, 0); ctx.lineTo(r * 0.4, r * 0.9);
  ctx.stroke();
  // body
  ctx.fillStyle = failed ? "#9aa3c0" : (env.onGround ? "#ff5d5d" : "#ff8a5d");
  roundRect(ctx, -r, -r * 1.4, r * 2, r * 1.6, 5);
  ctx.fill();
  // eye (X when failed, open otherwise)
  if (failed) {
    ctx.strokeStyle = "#222";
    ctx.lineWidth = 2;
    const ex0 = r * 0.2, ey0 = -r * 0.85, d = r * 0.22;
    ctx.beginPath();
    ctx.moveTo(ex0 - d, ey0 - d); ctx.lineTo(ex0 + d, ey0 + d);
    ctx.moveTo(ex0 + d, ey0 - d); ctx.lineTo(ex0 - d, ey0 + d);
    ctx.stroke();
  } else {
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(r * 0.35, -r * 0.75, r * 0.28, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#222";
    ctx.beginPath();
    ctx.arc(r * 0.45, -r * 0.75, r * 0.13, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // energy bar above robot
  const ew = r * 2.2, eh = 5;
  const ex = rx - ew / 2, ey = ry - r * 1.4 - 12;
  ctx.fillStyle = "rgba(255,255,255,0.25)";
  roundRect(ctx, ex, ey, ew, eh, 2); ctx.fill();
  const efrac = env.energy / env.maxEnergy;
  ctx.fillStyle = efrac < 0.25 ? "#ff5d5d" : "#3ddc84";
  roundRect(ctx, ex, ey, ew * efrac, eh, 2); ctx.fill();

  // end-of-episode banner
  if (banner && pauseFrames > 0) {
    ctx.font = "bold 22px system-ui, sans-serif";
    ctx.textAlign = "center";
    const tw = ctx.measureText(banner.text).width;
    ctx.fillStyle = "rgba(8,12,26,0.78)";
    roundRect(ctx, rect.width / 2 - tw / 2 - 16, 14, tw + 32, 38, 8);
    ctx.fill();
    ctx.fillStyle = banner.color;
    ctx.fillText(banner.text, rect.width / 2, 40);
    ctx.textAlign = "left";
  }
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

// --- live demo rollout ------------------------------------------------------
let demoEnv = new StairClimbEnv();
demoEnv.seed(99);
let demoObs = demoEnv.reset();
let pauseFrames = 0;
let banner = null;   // {text, color, ok} shown when an episode ends

function tickDemo() {
  // Hold the final frame (with banner) for a beat, then reset for a new attempt.
  if (pauseFrames > 0) {
    pauseFrames--;
    if (pauseFrames === 0) { demoObs = demoEnv.reset(); banner = null; }
    return;
  }
  const stepsPerFrame = parseInt(els.speed.value, 10);
  for (let k = 0; k < stepsPerFrame; k++) {
    const a = agent.act(demoObs, els.greedy.checked);
    const res = demoEnv.step(a);
    demoObs = res.obs;
    if (res.done) {
      lastInfo = res.info;
      if (res.info.reachedGoal) {
        banner = { text: "✓ REACHED THE TOP!", color: "#3ddc84", ok: true };
      } else if (res.info.collapsed) {
        banner = { text: "✗ OUT OF ENERGY — collapsed", color: "#ff5d5d", ok: false };
      } else {
        banner = { text: "✗ TIMED OUT", color: "#ffb05d", ok: false };
      }
      pauseFrames = 45;
      break;
    }
  }
}

let lastInfo = null;
function loop() {
  const rect = fitCanvas(canvas, ctx);
  tickDemo();
  drawScene(rect, demoEnv);
  requestAnimationFrame(loop);
}

// --- training (chunked so the UI stays responsive) --------------------------
let training = false;
const history = [];

function runEpisode(env, ag, train) {
  let obs = env.reset();
  let done = false, info = null;
  while (!done) {
    const a = ag.act(obs, !train);
    const r = env.step(a);
    if (train) ag.update(obs, a, r.reward, r.obs, r.done);
    obs = r.obs; done = r.done; info = r.info;
  }
  if (train) ag.decayEpsilon();
  return info;
}

function trainChunk(remaining, total, trainEnv, recent) {
  if (!training || remaining <= 0) {
    training = false;
    els.train.disabled = false;
    els.train.textContent = "Train (5000 episodes)";
    els.status.textContent = "Trained ✓ — watch it climb!";
    els.greedy.checked = true;
    return;
  }
  // Small chunks with a frame-ish delay so training is spread over a few
  // seconds -- the live robot (which shares this agent) is visibly clumsy and
  // collapses early, then climbs cleanly as exploration decays.
  const chunk = Math.min(25, remaining);
  for (let i = 0; i < chunk; i++) {
    const info = runEpisode(trainEnv, agent, true);
    recent.push(info.reachedGoal ? 1 : 0);
    if (recent.length > 200) recent.shift();
  }
  const done = total - remaining + chunk;
  const succ = recent.reduce((a, b) => a + b, 0) / recent.length;
  history.push(succ);
  drawCurve();

  els.episodes.textContent = done;
  els.success.textContent = (succ * 100).toFixed(0) + "%";
  els.epsilon.textContent = agent.epsilon.toFixed(3);
  els.status.textContent = `Training… ${done}/${total} — watch it improve!`;

  setTimeout(() => trainChunk(remaining - chunk, total, trainEnv, recent), 16);
}

function startTraining() {
  if (training) return;
  training = true;
  history.length = 0;
  agent = new TabularQLearner(world.nActions);
  els.train.disabled = true;
  els.train.textContent = "Training…";
  els.greedy.checked = false;
  const trainEnv = new StairClimbEnv();
  trainEnv.seed(7);
  trainChunk(5000, 5000, trainEnv, []);
}

function drawCurve() {
  const rect = fitCanvas(curve, cctx);
  cctx.clearRect(0, 0, rect.width, rect.height);
  cctx.fillStyle = "rgba(255,255,255,0.04)";
  cctx.fillRect(0, 0, rect.width, rect.height);
  // axes labels
  cctx.fillStyle = "#7d8bb5";
  cctx.font = "11px system-ui, sans-serif";
  cctx.fillText("100%", 2, 12);
  cctx.fillText("0%", 2, rect.height - 3);
  cctx.fillText("success rate vs. episode", rect.width - 150, 12);
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

// --- stats from the live demo ----------------------------------------------
setInterval(() => {
  if (lastInfo) {
    els.stairs.textContent = `${lastInfo.stepIndex}/${world.numSteps}`;
  }
}, 200);

// --- wire up ----------------------------------------------------------------
els.train.addEventListener("click", startTraining);
els.reset.addEventListener("click", () => {
  training = false;
  agent = new TabularQLearner(world.nActions);
  history.length = 0;
  drawCurve();
  els.episodes.textContent = "0";
  els.success.textContent = "—";
  els.epsilon.textContent = agent.epsilon.toFixed(3);
  els.greedy.checked = false;
  els.train.disabled = false;
  els.train.textContent = "Train (5000 episodes)";
  els.status.textContent = "Untrained — acting randomly: it often runs out of energy and collapses. Hit Train to watch it learn.";
});
window.addEventListener("resize", drawCurve);

els.status.textContent = "Untrained — acting randomly: it often runs out of energy and collapses. Hit Train to watch it learn.";
drawCurve();
loop();
