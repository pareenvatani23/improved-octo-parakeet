/* A tiny neural network (MLP) + a DQN-style agent for the 3D maze.
 *
 * This is a *real* value network: input = a small feature vector describing the
 * robot's current cell, output = an estimated Q-value for each of the 6 moves.
 * It learns online with experience replay and a target network. The forward
 * pass exposes every neuron's activation so the UI can draw the live "brain".
 *
 * Browser + Node friendly (window.MazeNN / module.exports).
 */
"use strict";

(function () {

function mat(out, inn, fn) {
  const m = new Array(out);
  for (let i = 0; i < out; i++) { m[i] = new Float64Array(inn); for (let j = 0; j < inn; j++) m[i][j] = fn(); }
  return m;
}
function zerosMat(out, inn) { return mat(out, inn, () => 0); }

class MLP {
  constructor(sizes, lr = 0.01) {
    this.sizes = sizes;
    this.L = sizes.length - 1;
    this.lr = lr;
    this.beta1 = 0.9; this.beta2 = 0.999; this.eps = 1e-8; this.t = 0;
    this.clip = 1.0;   // Huber-style TD-error clip -> stabilises Q-learning
    this._seed = 22222;
    this.W = []; this.b = [];
    this.mW = []; this.vW = []; this.mb = []; this.vb = [];
    for (let l = 0; l < this.L; l++) {
      const inn = sizes[l], out = sizes[l + 1];
      const scale = Math.sqrt(2 / (inn + out));
      this.W[l] = mat(out, inn, () => (this._r() * 2 - 1) * scale);
      this.b[l] = new Float64Array(out);
      this.mW[l] = zerosMat(out, inn); this.vW[l] = zerosMat(out, inn);
      this.mb[l] = new Float64Array(out); this.vb[l] = new Float64Array(out);
    }
  }
  _r() { let x = this._seed; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this._seed = x >>> 0; return (this._seed % 1e7) / 1e7; }

  forward(x) {
    const as = [x]; const zs = [];
    let a = x;
    for (let l = 0; l < this.L; l++) {
      const out = this.sizes[l + 1], inn = this.sizes[l];
      const z = new Float64Array(out);
      const Wl = this.W[l], bl = this.b[l];
      for (let i = 0; i < out; i++) {
        let s = bl[i];
        const wi = Wl[i];
        for (let j = 0; j < inn; j++) s += wi[j] * a[j];
        z[i] = s;
      }
      zs.push(z);
      let an;
      if (l === this.L - 1) { an = z; }                       // linear output
      else { an = new Float64Array(out); for (let i = 0; i < out; i++) an[i] = Math.tanh(z[i]); }
      as.push(an);
      a = an;
    }
    return { as, zs, out: a };
  }

  // batch: array of {x, a (action idx), y (target scalar)}
  trainBatch(batch) {
    const gW = [], gb = [];
    for (let l = 0; l < this.L; l++) { gW[l] = zerosMat(this.sizes[l + 1], this.sizes[l]); gb[l] = new Float64Array(this.sizes[l + 1]); }

    for (const sample of batch) {
      const f = this.forward(sample.x);
      // output-layer error (Huber-clipped, on the taken action only)
      let delta = new Float64Array(this.sizes[this.L]);
      let e = f.out[sample.a] - sample.y;
      if (e > this.clip) e = this.clip; else if (e < -this.clip) e = -this.clip;
      delta[sample.a] = e;
      for (let l = this.L - 1; l >= 0; l--) {
        const aPrev = f.as[l];
        const gWl = gW[l], gbl = gb[l];
        const out = this.sizes[l + 1], inn = this.sizes[l];
        for (let i = 0; i < out; i++) {
          const di = delta[i];
          if (di === 0) continue;
          gbl[i] += di;
          const row = gWl[i];
          for (let j = 0; j < inn; j++) row[j] += di * aPrev[j];
        }
        if (l > 0) {
          const nd = new Float64Array(inn);
          const Wl = this.W[l];
          for (let j = 0; j < inn; j++) {
            let s = 0;
            for (let i = 0; i < out; i++) s += Wl[i][j] * delta[i];
            const aj = f.as[l][j];
            nd[j] = s * (1 - aj * aj);   // tanh'
          }
          delta = nd;
        }
      }
    }

    const n = batch.length || 1;
    this.t++;
    const lr = this.lr, b1 = this.beta1, b2 = this.beta2, eps = this.eps;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (let l = 0; l < this.L; l++) {
      const out = this.sizes[l + 1], inn = this.sizes[l];
      for (let i = 0; i < out; i++) {
        // bias
        let g = gb[l][i] / n;
        this.mb[l][i] = b1 * this.mb[l][i] + (1 - b1) * g;
        this.vb[l][i] = b2 * this.vb[l][i] + (1 - b2) * g * g;
        this.b[l][i] -= lr * (this.mb[l][i] / c1) / (Math.sqrt(this.vb[l][i] / c2) + eps);
        const Wi = this.W[l][i], mWi = this.mW[l][i], vWi = this.vW[l][i], gWi = gW[l][i];
        for (let j = 0; j < inn; j++) {
          g = gWi[j] / n;
          mWi[j] = b1 * mWi[j] + (1 - b1) * g;
          vWi[j] = b2 * vWi[j] + (1 - b2) * g * g;
          Wi[j] -= lr * (mWi[j] / c1) / (Math.sqrt(vWi[j] / c2) + eps);
        }
      }
    }
  }

  // regression step: batch of {x, y:Float64Array(out)} -> fit ALL outputs (MSE)
  trainRegress(batch, clip = 6) {
    const gW = [], gb = [];
    for (let l = 0; l < this.L; l++) { gW[l] = zerosMat(this.sizes[l + 1], this.sizes[l]); gb[l] = new Float64Array(this.sizes[l + 1]); }
    const OUT = this.sizes[this.L];
    for (const sample of batch) {
      const f = this.forward(sample.x);
      let delta = new Float64Array(OUT);
      for (let i = 0; i < OUT; i++) {
        let e = f.out[i] - sample.y[i];
        if (e > clip) e = clip; else if (e < -clip) e = -clip;
        delta[i] = e;
      }
      for (let l = this.L - 1; l >= 0; l--) {
        const aPrev = f.as[l]; const gWl = gW[l], gbl = gb[l];
        const out = this.sizes[l + 1], inn = this.sizes[l];
        for (let i = 0; i < out; i++) {
          const di = delta[i]; if (di === 0) continue;
          gbl[i] += di; const row = gWl[i];
          for (let j = 0; j < inn; j++) row[j] += di * aPrev[j];
        }
        if (l > 0) {
          const nd = new Float64Array(inn); const Wl = this.W[l];
          for (let j = 0; j < inn; j++) {
            let s = 0; for (let i = 0; i < out; i++) s += Wl[i][j] * delta[i];
            const aj = f.as[l][j]; nd[j] = s * (1 - aj * aj);
          }
          delta = nd;
        }
      }
    }
    this._adam(gW, gb, batch.length || 1);
  }

  _adam(gW, gb, n) {
    this.t++;
    const lr = this.lr, b1 = this.beta1, b2 = this.beta2, eps = this.eps;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (let l = 0; l < this.L; l++) {
      const out = this.sizes[l + 1], inn = this.sizes[l];
      for (let i = 0; i < out; i++) {
        let g = gb[l][i] / n;
        this.mb[l][i] = b1 * this.mb[l][i] + (1 - b1) * g;
        this.vb[l][i] = b2 * this.vb[l][i] + (1 - b2) * g * g;
        this.b[l][i] -= lr * (this.mb[l][i] / c1) / (Math.sqrt(this.vb[l][i] / c2) + eps);
        const Wi = this.W[l][i], mWi = this.mW[l][i], vWi = this.vW[l][i], gWi = gW[l][i];
        for (let j = 0; j < inn; j++) {
          g = gWi[j] / n;
          mWi[j] = b1 * mWi[j] + (1 - b1) * g;
          vWi[j] = b2 * vWi[j] + (1 - b2) * g * g;
          Wi[j] -= lr * (mWi[j] / c1) / (Math.sqrt(vWi[j] / c2) + eps);
        }
      }
    }
  }

  copyFrom(o) {
    for (let l = 0; l < this.L; l++) {
      for (let i = 0; i < this.sizes[l + 1]; i++) {
        this.b[l][i] = o.b[l][i];
        for (let j = 0; j < this.sizes[l]; j++) this.W[l][i][j] = o.W[l][i][j];
      }
    }
  }
}

// feature dimension: 3 position + 6 openness + 3 vector-to-exit
const FEATURE_LABELS = ["floor", "row", "col", "open N", "open S", "open W", "open E", "open ↑", "open ↓", "→exit f", "→exit r", "→exit c"];
const D = FEATURE_LABELS.length;

class NeuralMazeAgent {
  constructor(world, cfg = {}) {
    this.world = world;
    this.hidden = cfg.hidden ?? 16;
    this.sizes = [D, this.hidden, 6];
    this.net = new MLP(this.sizes, cfg.lr ?? 0.005);
    this.target = new MLP(this.sizes, cfg.lr ?? 0.005);
    this.target.copyFrom(this.net);
    this.gamma = cfg.gamma ?? 0.95;
    this.epsilon = 1.0; this.epsilonMin = cfg.epsilonMin ?? 0.05; this.epsilonDecay = cfg.epsilonDecay ?? 0.999;
    this.buf = []; this.bufMax = cfg.bufMax ?? 5000; this.batch = cfg.batch ?? 32;
    this.syncEvery = cfg.syncEvery ?? 250; this._c = 0;
    this.nActions = 6;
    this._seed = 13579;
    this.rows = world.rows; this.cols = world.cols; this.floors = world.floors;
  }
  _r() { let x = this._seed; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this._seed = x >>> 0; return (this._seed % 1e7) / 1e7; }

  decode(s) {
    const rc = this.rows * this.cols;
    const f = Math.floor(s / rc), rem = s % rc;
    return { f, r: Math.floor(rem / this.cols), c: rem % this.cols };
  }
  features(s) {
    const { f, r, c } = this.decode(s);
    const w = this.world;
    const op = (d) => (w.target(f, r, c, d) ? 1 : 0);
    return Float64Array.from([
      f / Math.max(1, w.floors - 1), r / Math.max(1, w.rows - 1), c / Math.max(1, w.cols - 1),
      op(0), op(1), op(2), op(3), op(4), op(5),
      (w.exit.f - f) / w.floors, (w.exit.r - r) / w.rows, (w.exit.c - c) / w.cols,
    ]);
  }
  qvalues(s) { return this.net.forward(this.features(s)).out; }
  bestAction(s) {
    const q = this.qvalues(s);
    let best = -Infinity, ba = 0, allZero = true;
    for (let a = 0; a < 6; a++) { if (q[a] !== 0) allZero = false; if (q[a] > best) { best = q[a]; ba = a; } }
    return { a: ba, allZero, q };
  }
  act(s, greedy = false) {
    if (!greedy && this._r() < this.epsilon) return Math.floor(this._r() * 6);
    return this.bestAction(s).a;
  }
  remember(s, a, r, ns, done) { this.buf.push([s, a, r, ns, done]); if (this.buf.length > this.bufMax) this.buf.shift(); }
  learn() {
    if (this.buf.length < this.batch) return;
    const batch = [];
    for (let k = 0; k < this.batch; k++) {
      const e = this.buf[Math.floor(this._r() * this.buf.length)];
      const [s, a, r, ns, done] = e;
      let y = r;
      if (!done) {
        const qn = this.target.forward(this.features(ns)).out;
        let m = -Infinity; for (let i = 0; i < 6; i++) m = Math.max(m, qn[i]);
        y = r + this.gamma * m;
      }
      batch.push({ x: this.features(s), a, y });
    }
    this.net.trainBatch(batch);
    if ((++this._c) % this.syncEvery === 0) this.target.copyFrom(this.net);
  }
  decayEpsilon() { this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay); }

  // snapshot for visualisation
  inspect(s) {
    const f = this.net.forward(this.features(s));
    return { inputs: f.as[0], hidden: f.as[1], outputs: f.out };
  }
}

const MazeNN = { MLP, NeuralMazeAgent, FEATURE_LABELS, D };
if (typeof window !== "undefined") window.MazeNN = MazeNN;
if (typeof module !== "undefined" && module.exports) module.exports = MazeNN;

})();
