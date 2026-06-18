"""StairClimbEnv -- a 2D sagittal-plane stair-climbing environment.

The robot is modelled as a point body with position (x, y) and velocity
(vx, vy) moving in a vertical plane. The world is a staircase: a sequence of
flat steps, each ``step_width`` wide and ``step_height`` tall, leading up to a
flat goal platform at the top.

Physics (semi-implicit Euler):
    * Gravity pulls the body down every tick.
    * A horizontal force accelerates the body right (capped at ``max_vx``);
      ground friction bleeds off horizontal speed.
    * A jump sets an upward velocity, but only when the body is on the ground.
    * The body rests on whatever step surface is beneath it.
    * The vertical face (riser) of the next step is solid: to advance, the
      robot's feet must be at least as high as the next step's surface,
      otherwise it bumps into the riser and stops. This is what forces the
      agent to *jump* its way up rather than walk through the stairs.

The interface mirrors the classic Gym contract:
    obs = env.reset()
    obs, reward, done, info = env.step(action)

Actions (discrete):
    0  IDLE        do nothing
    1  WALK        push right
    2  JUMP        jump straight up
    3  JUMP_RIGHT  push right + jump  (the move that climbs a stair)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

IDLE, WALK, JUMP, JUMP_RIGHT = 0, 1, 2, 3
ACTION_NAMES = {IDLE: "IDLE", WALK: "WALK", JUMP: "JUMP", JUMP_RIGHT: "JUMP_RIGHT"}


@dataclass
class StairClimbEnv:
    # --- world geometry ---
    num_steps: int = 6
    step_width: float = 1.0
    step_height: float = 0.5

    # --- physics ---
    gravity: float = 9.8
    dt: float = 0.05
    accel: float = 10.0          # horizontal acceleration when pushing
    air_control: float = 0.30    # fraction of accel usable while airborne
    max_vx: float = 3.0
    friction: float = 0.82       # multiplicative ground friction per tick
    jump_speed: float = 3.55     # upward velocity imparted by a jump

    # --- stamina ---
    # Jumping and moving cost energy; when it runs out the robot collapses and
    # the episode fails. This is what makes the task non-trivial: flailing
    # randomly burns the budget long before the top, so the agent must learn an
    # *efficient* gait -- jump only at the risers, coast otherwise.
    max_energy: float = 100.0
    jump_cost: float = 7.0       # energy per jump
    move_cost: float = 0.25      # energy per tick while pushing right

    # --- episode ---
    max_steps: int = 250

    # --- reward shaping ---
    climb_reward: float = 10.0   # per stair gained
    progress_reward: float = 1.0 # per unit of rightward progress
    time_penalty: float = 0.05   # per tick, encourages efficiency
    goal_reward: float = 100.0   # reaching the top platform
    fail_penalty: float = 20.0   # collapsing (out of energy) before the top

    # --- runtime state (set in reset) ---
    x: float = field(default=0.0, init=False)
    y: float = field(default=0.0, init=False)
    vx: float = field(default=0.0, init=False)
    vy: float = field(default=0.0, init=False)
    on_ground: bool = field(default=True, init=False)
    energy: float = field(default=0.0, init=False)
    t: int = field(default=0, init=False)
    _max_step_reached: int = field(default=0, init=False)

    n_actions: int = field(default=4, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ helpers
    @property
    def goal_x(self) -> float:
        """Right edge of the staircase = start of the top platform."""
        return self.num_steps * self.step_width

    @property
    def top_height(self) -> float:
        return self.num_steps * self.step_height

    def step_index(self, x: float) -> int:
        """Index of the step the body is horizontally over (clamped)."""
        if x < 0:
            return 0
        return min(int(x // self.step_width), self.num_steps)

    def surface_height(self, x: float) -> float:
        """Height of the walkable surface at horizontal position ``x``."""
        return self.step_index(x) * self.step_height

    # ------------------------------------------------------------------ gym api
    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        # Start just before the first riser, on the ground, with a tiny random
        # nudge so episodes are not identical.
        self.x = float(self._rng.uniform(0.05, 0.25))
        self.y = self.surface_height(self.x)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.energy = self.max_energy
        self.t = 0
        self._max_step_reached = self.step_index(self.x)
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"invalid action {action!r}")

        prev_x = self.x
        prev_step = self.step_index(self.x)

        push_right = action in (WALK, JUMP_RIGHT)
        do_jump = action in (JUMP, JUMP_RIGHT)

        # --- horizontal dynamics ---
        if push_right:
            eff = self.accel if self.on_ground else self.accel * self.air_control
            self.vx += eff * self.dt
            self.energy -= self.move_cost
        if self.on_ground:
            self.vx *= self.friction
        self.vx = float(np.clip(self.vx, -self.max_vx, self.max_vx))

        # --- vertical dynamics ---
        if do_jump and self.on_ground:
            self.vy = self.jump_speed
            self.on_ground = False
            self.energy -= self.jump_cost
        self.vy -= self.gravity * self.dt
        self.energy = max(0.0, self.energy)

        # --- integrate, with collision resolution ---
        new_x = self.x + self.vx * self.dt
        new_y = self.y + self.vy * self.dt

        # Riser collision: block rightward motion if we'd cross into a higher
        # step whose surface is above our feet.
        boundary = (prev_step + 1) * self.step_width
        if new_x >= boundary and prev_step < self.num_steps:
            next_surface = (prev_step + 1) * self.step_height
            if max(self.y, new_y) < next_surface - 1e-6:
                new_x = boundary - 1e-4
                self.vx = 0.0

        new_x = max(0.0, new_x)

        # Floor collision: land on whatever surface is beneath the new x.
        ground = self.surface_height(new_x)
        if new_y <= ground + 1e-9:
            new_y = ground
            self.vy = 0.0
            self.on_ground = True
        else:
            self.on_ground = False

        self.x, self.y = new_x, new_y
        self.t += 1

        # --- reward ---
        cur_step = self.step_index(self.x)
        reward = -self.time_penalty
        reward += self.progress_reward * max(0.0, self.x - prev_x)
        if cur_step > self._max_step_reached:
            reward += self.climb_reward * (cur_step - self._max_step_reached)
            self._max_step_reached = cur_step

        # --- termination ---
        done = False
        reached_goal = self.x >= self.goal_x
        collapsed = self.energy <= 0.0 and not reached_goal
        if reached_goal:
            reward += self.goal_reward
            done = True
        elif collapsed:
            reward -= self.fail_penalty
            done = True
        elif self.t >= self.max_steps:
            done = True

        info = {
            "step_index": cur_step,
            "reached_goal": reached_goal,
            "collapsed": collapsed,
            "energy": self.energy,
            "t": self.t,
        }
        return self._obs(), reward, done, info

    # ------------------------------------------------------------------ obs
    def _obs(self) -> np.ndarray:
        """Observation, expressed *relative to the current step* so a learned
        policy generalises across every stair.

        [x_in_step, height_above_step, vx, vy, on_ground, energy_fraction]
        """
        idx = self.step_index(self.x)
        x_in_step = self.x - idx * self.step_width
        height_above = self.y - idx * self.step_height
        return np.array(
            [
                x_in_step,
                height_above,
                self.vx,
                self.vy,
                float(self.on_ground),
                self.energy / self.max_energy,
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ misc
    def state(self) -> dict:
        """Absolute state, handy for rendering."""
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "on_ground": self.on_ground,
            "energy": self.energy,
            "step_index": self.step_index(self.x),
            "t": self.t,
        }
