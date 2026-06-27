/* 3D Maze RL Sim -- environment + tabular Q-learning agent.
 *
 * A robot must escape a multi-floor maze. Each floor is a perfect 2D maze
 * (a spanning tree of corridors); floors are linked by vertical "shafts" the
 * robot can climb up/down. The agent learns, by tabular Q-learning, the
 * shortest route from the start cell to the exit.
 *
 * Browser + Node friendly: wrapped in an IIFE that exposes window.MazeRL,
 * and also sets module.exports for headless testing.
 */
"use strict";

(function () {

// Actions
const N = 0, S = 1, W = 2, E = 3, UP = 4, DOWN = 5;
const ACTION_NAMES = ["north", "south", "west", "east", "up", "down"];

// ----------------------------------------------------------------- maze env
class MazeEnv {
  constructor(cfg = {}) {
    this.floors = cfg.floors ?? 3;
    this.rows = cfg.rows ?? 6;
    this.cols = cfg.cols ?? 6;
    this.shaftsPerPair = cfg.shaftsPerPair ?? 2;
    this.maxSteps = cfg.maxSteps ?? 400;
    this.nActions = 6;

    this._seed = (cfg.seed ?? 1) >>> 0 || 1;
    this.generate();
    this.reset();
  }

  _rand() {
    let x = this._seed;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this._seed = x >>> 0;
    return (this._seed % 1e7) / 1e7;
  }
  _shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(this._rand() * (i + 1));
      const t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  idx(f, r, c) { return (f * this.rows + r) * this.cols + c; }
  get nStates() { return this.floors * this.rows * this.cols; }
  inBounds(f, r, c) {
    return f >= 0 && f < this.floors && r >= 0 && r < this.rows && c >= 0 && c < this.cols;
  }
  edgeKey(a, b) { return a < b ? a + ":" + b : b + ":" + a; }

  generate() {
    this.open = new Set();          // open edges (corridors + shafts)
    this.shaftUp = new Set();       // cell indices with a shaft going up
    this.shaftDown = new Set();     // cell indices with a shaft going down

    // carve each floor as a perfect maze (recursive backtracker)
    for (let f = 0; f < this.floors; f++) {
      const visited = new Set();
      const stack = [[0, 0]];
      visited.add(0 * this.cols + 0);
      while (stack.length) {
        const [r, c] = stack[stack.length - 1];
        const opts = this._shuffle([[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]])
          .filter(([nr, nc]) => this.inBounds(f, nr, nc) && !visited.has(nr * this.cols + nc));
        if (!opts.length) { stack.pop(); continue; }
        const [nr, nc] = opts[0];
        this.open.add(this.edgeKey(this.idx(f, r, c), this.idx(f, nr, nc)));
        visited.add(nr * this.cols + nc);
        stack.push([nr, nc]);
      }
    }

    // link adjacent floors with a few shafts
    for (let f = 0; f < this.floors - 1; f++) {
      const cells = [];
      for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) cells.push([r, c]);
      this._shuffle(cells);
      for (let k = 0; k < this.shaftsPerPair && k < cells.length; k++) {
        const [r, c] = cells[k];
        const a = this.idx(f, r, c), b = this.idx(f + 1, r, c);
        this.open.add(this.edgeKey(a, b));
        this.shaftUp.add(a);
        this.shaftDown.add(b);
      }
    }

    this.start = { f: 0, r: 0, c: 0 };
    this.exit = { f: this.floors - 1, r: this.rows - 1, c: this.cols - 1 };
  }

  // can the robot move from (f,r,c) in the given action direction?
  target(f, r, c, action) {
    let nf = f, nr = r, nc = c;
    if (action === N) nr--; else if (action === S) nr++;
    else if (action === W) nc--; else if (action === E) nc++;
    else if (action === UP) nf++; else if (action === DOWN) nf--;
    if (!this.inBounds(nf, nr, nc)) return null;
    if (this.open.has(this.edgeKey(this.idx(f, r, c), this.idx(nf, nr, nc)))) {
      return { f: nf, r: nr, c: nc };
    }
    return null;
  }

  seed(s) { this._seed = (s >>> 0) || 1; }

  reset() {
    this.f = this.start.f; this.r = this.start.r; this.c = this.start.c;
    this.t = 0;
    return this.idx(this.f, this.r, this.c);
  }

  step(action) {
    const tgt = this.target(this.f, this.r, this.c, action);
    let moved = false;
    if (tgt) { this.f = tgt.f; this.r = tgt.r; this.c = tgt.c; moved = true; }
    this.t += 1;

    const atExit = this.f === this.exit.f && this.r === this.exit.r && this.c === this.exit.c;
    let reward = -1;          // every step costs (encourages short paths)
    let done = false;
    if (atExit) { reward = 100; done = true; }
    else if (this.t >= this.maxSteps) { done = true; }

    return {
      state: this.idx(this.f, this.r, this.c),
      reward,
      done,
      info: {
        moved, bumped: !moved, atExit,
        f: this.f, r: this.r, c: this.c, t: this.t,
        action,
      },
    };
  }
}

// ----------------------------------------------------------------- agent
class MazeQLearner {
  constructor(nStates, nActions, cfg = {}) {
    this.nStates = nStates;
    this.nActions = nActions;
    this.alpha = cfg.alpha ?? 0.2;
    this.gamma = cfg.gamma ?? 0.95;
    this.epsilon = cfg.epsilon ?? 1.0;
    this.epsilonMin = cfg.epsilonMin ?? 0.02;
    this.epsilonDecay = cfg.epsilonDecay ?? 0.999;
    this.q = new Float64Array(nStates * nActions);
    this._seed = 987654321;
  }
  _rand() {
    let x = this._seed;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this._seed = x >>> 0;
    return (this._seed % 1e7) / 1e7;
  }
  bestAction(s) {
    const base = s * this.nActions;
    let best = -Infinity, ba = 0, ties = 1, allZero = true;
    for (let a = 0; a < this.nActions; a++) {
      const v = this.q[base + a];
      if (v !== 0) allZero = false;
      if (v > best) { best = v; ba = a; ties = 1; }
      else if (v === best) { ties++; if (this._rand() < 1 / ties) ba = a; }
    }
    return { a: ba, allZero };
  }
  act(s, greedy = false) {
    if (!greedy && this._rand() < this.epsilon) return Math.floor(this._rand() * this.nActions);
    return this.bestAction(s).a;
  }
  update(s, a, reward, ns, done) {
    const i = s * this.nActions + a;
    const nb = ns * this.nActions;
    let maxN = -Infinity;
    for (let k = 0; k < this.nActions; k++) maxN = Math.max(maxN, this.q[nb + k]);
    const target = reward + (done ? 0 : this.gamma * maxN);
    this.q[i] += this.alpha * (target - this.q[i]);
  }
  decayEpsilon() { this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay); }
}

const MazeRL = { MazeEnv, MazeQLearner, ACTION_NAMES, N, S, W, E, UP, DOWN };
if (typeof window !== "undefined") window.MazeRL = MazeRL;
if (typeof module !== "undefined" && module.exports) module.exports = MazeRL;

})();
