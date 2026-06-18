/* Robot Stairs RL Sim -- browser port.
 *
 * A faithful JavaScript re-implementation of the Python StairClimbEnv and the
 * tabular Q-learning agent (robot_stairs/env.py, robot_stairs/agent.py), so the
 * whole thing trains and runs live in the browser with no backend.
 */

"use strict";

// ----------------------------------------------------------------- environment
const IDLE = 0, WALK = 1, JUMP = 2, JUMP_RIGHT = 3;
const ACTION_NAMES = ["IDLE", "WALK", "JUMP", "JUMP_RIGHT"];

class StairClimbEnv {
  constructor(cfg = {}) {
    // geometry
    this.numSteps = cfg.numSteps ?? 6;
    this.stepWidth = cfg.stepWidth ?? 1.0;
    this.stepHeight = cfg.stepHeight ?? 0.5;
    // physics
    this.gravity = 9.8;
    this.dt = 0.05;
    this.accel = 10.0;
    this.airControl = 0.30;
    this.maxVx = 3.0;
    this.friction = 0.82;
    this.jumpSpeed = 3.55;
    // stamina
    this.maxEnergy = 100.0;
    this.jumpCost = 7.0;
    this.moveCost = 0.25;
    // episode
    this.maxSteps = 250;
    // reward
    this.climbReward = 10.0;
    this.progressReward = 1.0;
    this.timePenalty = 0.05;
    this.goalReward = 100.0;
    this.failPenalty = 20.0;

    this.nActions = 4;
    this._seed = 1;
    this.reset();
  }

  // small deterministic PRNG so runs are reproducible-ish
  seed(s) { this._seed = (s >>> 0) || 1; }
  _rand() {
    // xorshift32
    let x = this._seed;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this._seed = x >>> 0;
    return (this._seed % 1e7) / 1e7;
  }

  get goalX() { return this.numSteps * this.stepWidth; }
  get topHeight() { return this.numSteps * this.stepHeight; }

  stepIndex(x) {
    if (x < 0) return 0;
    return Math.min(Math.floor(x / this.stepWidth), this.numSteps);
  }
  surfaceHeight(x) { return this.stepIndex(x) * this.stepHeight; }

  reset() {
    this.x = 0.05 + this._rand() * 0.2;
    this.y = this.surfaceHeight(this.x);
    this.vx = 0.0;
    this.vy = 0.0;
    this.onGround = true;
    this.energy = this.maxEnergy;
    this.t = 0;
    this._maxStepReached = this.stepIndex(this.x);
    return this._obs();
  }

  step(action) {
    const prevX = this.x;
    const prevStep = this.stepIndex(this.x);
    const pushRight = action === WALK || action === JUMP_RIGHT;
    const doJump = action === JUMP || action === JUMP_RIGHT;

    // horizontal
    if (pushRight) {
      const eff = this.onGround ? this.accel : this.accel * this.airControl;
      this.vx += eff * this.dt;
      this.energy -= this.moveCost;
    }
    if (this.onGround) this.vx *= this.friction;
    this.vx = Math.max(-this.maxVx, Math.min(this.maxVx, this.vx));

    // vertical
    if (doJump && this.onGround) {
      this.vy = this.jumpSpeed;
      this.onGround = false;
      this.energy -= this.jumpCost;
    }
    this.vy -= this.gravity * this.dt;
    this.energy = Math.max(0.0, this.energy);

    // integrate + collide
    let newX = this.x + this.vx * this.dt;
    let newY = this.y + this.vy * this.dt;

    const boundary = (prevStep + 1) * this.stepWidth;
    if (newX >= boundary && prevStep < this.numSteps) {
      const nextSurface = (prevStep + 1) * this.stepHeight;
      if (Math.max(this.y, newY) < nextSurface - 1e-6) {
        newX = boundary - 1e-4;
        this.vx = 0.0;
      }
    }
    newX = Math.max(0.0, newX);

    const ground = this.surfaceHeight(newX);
    if (newY <= ground + 1e-9) {
      newY = ground;
      this.vy = 0.0;
      this.onGround = true;
    } else {
      this.onGround = false;
    }

    this.x = newX; this.y = newY; this.t += 1;

    // reward
    const curStep = this.stepIndex(this.x);
    let reward = -this.timePenalty;
    reward += this.progressReward * Math.max(0.0, this.x - prevX);
    if (curStep > this._maxStepReached) {
      reward += this.climbReward * (curStep - this._maxStepReached);
      this._maxStepReached = curStep;
    }

    // terminate
    let done = false;
    const reachedGoal = this.x >= this.goalX;
    const collapsed = this.energy <= 0.0 && !reachedGoal;
    if (reachedGoal) { reward += this.goalReward; done = true; }
    else if (collapsed) { reward -= this.failPenalty; done = true; }
    else if (this.t >= this.maxSteps) { done = true; }

    return {
      obs: this._obs(),
      reward,
      done,
      info: { stepIndex: curStep, reachedGoal, collapsed, energy: this.energy, t: this.t },
    };
  }

  _obs() {
    const idx = this.stepIndex(this.x);
    return [
      this.x - idx * this.stepWidth,
      this.y - idx * this.stepHeight,
      this.vx,
      this.vy,
      this.onGround ? 1.0 : 0.0,
      this.energy / this.maxEnergy,
    ];
  }
}

// ----------------------------------------------------------------- agent
function linspace(a, b, n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = a + (b - a) * i / (n - 1);
  return out;
}
// np.digitize (right=False) for increasing edges
function digitize(x, edges) {
  let i = 0;
  while (i < edges.length && x >= edges[i]) i++;
  return i;
}

const BINS = [
  linspace(0.0, 1.0, 6),
  linspace(0.0, 1.2, 7),
  linspace(-1.0, 3.0, 6),
  linspace(-5.0, 5.0, 7),
  [0.5],
  linspace(0.1, 0.9, 5),
];

class TabularQLearner {
  constructor(nActions, cfg = {}) {
    this.nActions = nActions;
    this.alpha = cfg.alpha ?? 0.1;
    this.gamma = cfg.gamma ?? 0.99;
    this.epsilon = cfg.epsilon ?? 1.0;
    this.epsilonMin = cfg.epsilonMin ?? 0.02;
    this.epsilonDecay = cfg.epsilonDecay ?? 0.999;

    this.dims = BINS.map((b) => b.length + 1);
    // strides for flattening [d0,d1,...,action]
    this.sizes = this.dims.concat([nActions]);
    this.strides = new Array(this.sizes.length);
    let s = 1;
    for (let i = this.sizes.length - 1; i >= 0; i--) {
      this.strides[i] = s; s *= this.sizes[i];
    }
    this.q = new Float64Array(s);
    this._seed = 12345;
  }

  _rand() {
    let x = this._seed;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this._seed = x >>> 0;
    return (this._seed % 1e7) / 1e7;
  }

  discretize(obs) {
    const idx = new Array(BINS.length);
    for (let i = 0; i < BINS.length; i++) idx[i] = digitize(obs[i], BINS[i]);
    return idx;
  }

  _base(stateIdx) {
    let off = 0;
    for (let i = 0; i < stateIdx.length; i++) off += stateIdx[i] * this.strides[i];
    return off;
  }

  act(obs, greedy = false) {
    if (!greedy && this._rand() < this.epsilon) {
      return Math.floor(this._rand() * this.nActions);
    }
    const base = this._base(this.discretize(obs));
    let best = -Infinity, bestA = 0, ties = 1;
    for (let a = 0; a < this.nActions; a++) {
      const v = this.q[base + a];
      if (v > best) { best = v; bestA = a; ties = 1; }
      else if (v === best) { ties++; if (this._rand() < 1 / ties) bestA = a; }
    }
    return bestA;
  }

  update(obs, action, reward, nextObs, done) {
    const base = this._base(this.discretize(obs)) + action;
    const nbase = this._base(this.discretize(nextObs));
    let maxNext = -Infinity;
    for (let a = 0; a < this.nActions; a++) maxNext = Math.max(maxNext, this.q[nbase + a]);
    const target = reward + (done ? 0 : this.gamma * maxNext);
    this.q[base] += this.alpha * (target - this.q[base]);
  }

  decayEpsilon() {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }
}

// expose
const RobotStairs = { StairClimbEnv, TabularQLearner, ACTION_NAMES, IDLE, WALK, JUMP, JUMP_RIGHT };
if (typeof window !== "undefined") window.RobotStairs = RobotStairs;
if (typeof module !== "undefined" && module.exports) module.exports = RobotStairs;
