"""Tabular Q-learning agent with state discretisation.

The environment exposes a small continuous observation vector. We bucket each
dimension into bins and keep a Q-table indexed by the resulting discrete state.
Because the observation is expressed relative to the current step, a single
compact table learns a policy that works on every stair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# Bin edges for each observation dimension. Values are clipped into range and
# then digitised. Tuned to the default StairClimbEnv geometry/physics.
DEFAULT_BINS = (
    np.linspace(0.0, 1.0, 6),    # x_in_step         -> 5 buckets
    np.linspace(0.0, 1.2, 7),    # height_above_step -> 6 buckets
    np.linspace(-1.0, 3.0, 6),   # vx                -> 5 buckets
    np.linspace(-5.0, 5.0, 7),   # vy                -> 6 buckets
    np.array([0.5]),             # on_ground (bool)  -> 2 buckets
    np.linspace(0.1, 0.9, 5),    # energy_fraction   -> 6 buckets
)


@dataclass
class TabularQLearner:
    n_actions: int
    bins: Sequence[np.ndarray] = field(default=DEFAULT_BINS)
    alpha: float = 0.1            # learning rate
    gamma: float = 0.99           # discount
    epsilon: float = 1.0          # exploration (decayed externally)
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        shape = tuple(len(b) + 1 for b in self.bins) + (self.n_actions,)
        self.q = np.zeros(shape, dtype=np.float64)

    # ------------------------------------------------------------------ encode
    def discretize(self, obs: np.ndarray) -> tuple:
        return tuple(
            int(np.digitize(obs[i], self.bins[i])) for i in range(len(self.bins))
        )

    # ------------------------------------------------------------------ policy
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        s = self.discretize(obs)
        q = self.q[s]
        # random tie-break among the best actions
        best = np.flatnonzero(q == q.max())
        return int(self._rng.choice(best))

    # ------------------------------------------------------------------ learn
    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        s = self.discretize(obs)
        ns = self.discretize(next_obs)
        target = reward + (0.0 if done else self.gamma * self.q[ns].max())
        self.q[s + (action,)] += self.alpha * (target - self.q[s + (action,)])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        np.save(path, self.q)

    def load(self, path: str) -> None:
        self.q = np.load(path)
