"""Sanity tests for the StairClimbEnv physics and the Q-learning agent."""

import numpy as np

from robot_stairs.agent import TabularQLearner
from robot_stairs.env import IDLE, JUMP_RIGHT, WALK, StairClimbEnv


def test_reset_shape_and_ground():
    env = StairClimbEnv()
    env.seed(0)
    obs = env.reset()
    assert obs.shape == (6,)
    assert env.on_ground is True
    assert env.energy == env.max_energy
    # robot starts on the first step's surface
    assert abs(env.y - env.surface_height(env.x)) < 1e-9


def test_surface_height_monotonic():
    env = StairClimbEnv(num_steps=4, step_width=1.0, step_height=0.5)
    assert env.surface_height(0.5) == 0.0
    assert env.surface_height(1.5) == 0.5
    assert env.surface_height(3.5) == 1.5
    # top platform height is clamped
    assert env.surface_height(100.0) == env.top_height


def test_gravity_pulls_down():
    env = StairClimbEnv()
    env.seed(1)
    env.reset()
    # jump, then idle: the body should leave the ground and come back
    env.step(JUMP_RIGHT)
    assert env.on_ground is False
    for _ in range(200):
        _, _, done, _ = env.step(IDLE)
        if env.on_ground:
            break
    assert env.on_ground is True


def test_riser_blocks_walking_through():
    """Walking right without jumping cannot clear the first riser."""
    env = StairClimbEnv()
    env.seed(2)
    env.reset()
    for _ in range(200):
        _, _, done, info = env.step(WALK)
        if done:
            break
    # blocked at the first riser -> never climbs past step 0
    assert info["step_index"] == 0
    assert info["reached_goal"] is False


def test_invalid_action_raises():
    env = StairClimbEnv()
    env.reset()
    try:
        env.step(99)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid action")


def test_agent_discretize_in_bounds():
    env = StairClimbEnv()
    agent = TabularQLearner(n_actions=env.n_actions, seed=0)
    obs = env.reset()
    s = agent.discretize(obs)
    assert len(s) == 6
    assert agent.q[s].shape == (env.n_actions,)


def test_short_training_improves():
    """A brief training run should learn to climb at least one stair on
    average, well above the zero a random walker achieves."""
    from robot_stairs.train import train

    agent, hist = train(episodes=1500, seed=0, save_path=None, verbose=False)
    last = hist["steps_climbed"][-300:].mean()
    assert last >= 1.0, f"expected agent to climb stairs, got {last:.2f}"
