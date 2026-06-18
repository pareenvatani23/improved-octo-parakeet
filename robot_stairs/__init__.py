"""Robot Stairs RL Sim.

A lightweight, dependency-light reinforcement-learning environment in which a
2D robot learns to climb a staircase, plus a tabular Q-learning agent that
solves it.

Public API:
    StairClimbEnv      - the gym-style environment
    TabularQLearner    - the agent
"""

from .env import StairClimbEnv
from .agent import TabularQLearner

__all__ = ["StairClimbEnv", "TabularQLearner"]
__version__ = "0.1.0"
