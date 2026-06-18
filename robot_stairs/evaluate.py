"""Evaluate / visualise a trained policy.

Usage:
    python -m robot_stairs.evaluate --qtable qtable.npy --render
    python -m robot_stairs.evaluate --qtable qtable.npy --plot trajectory.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .agent import TabularQLearner
from .env import ACTION_NAMES, StairClimbEnv
from .render import ascii_frame, plot_trajectory


def rollout(env: StairClimbEnv, agent: TabularQLearner):
    """Run one greedy episode, recording the trajectory."""
    obs = env.reset()
    xs, ys, actions = [env.x], [env.y], []
    done = False
    total = 0.0
    info = {}
    while not done:
        a = agent.act(obs, greedy=True)
        obs, r, done, info = env.step(a)
        total += r
        xs.append(env.x)
        ys.append(env.y)
        actions.append(a)
    return xs, ys, actions, total, info


def evaluate(qtable: str, episodes: int = 20, seed: int = 123, verbose: bool = True):
    env = StairClimbEnv()
    env.seed(seed)
    agent = TabularQLearner(n_actions=env.n_actions, seed=seed)
    agent.load(qtable)
    agent.epsilon = 0.0

    successes, rewards, stairs = [], [], []
    for _ in range(episodes):
        _, _, _, total, info = rollout(env, agent)
        successes.append(info["reached_goal"])
        rewards.append(total)
        stairs.append(info["step_index"])

    if verbose:
        print(
            f"eval over {episodes} eps | "
            f"success {np.mean(successes) * 100:.1f}% | "
            f"avg reward {np.mean(rewards):.2f} | "
            f"avg stairs {np.mean(stairs):.2f}/{env.num_steps}"
        )
    return {"successes": successes, "rewards": rewards, "stairs": stairs}


def animate_ascii(qtable: str, seed: int = 7, fps: float = 20.0) -> None:
    env = StairClimbEnv()
    env.seed(seed)
    agent = TabularQLearner(n_actions=env.n_actions, seed=seed)
    agent.load(qtable)
    agent.epsilon = 0.0

    obs = env.reset()
    done = False
    delay = 1.0 / fps
    last_action = None
    while not done:
        a = agent.act(obs, greedy=True)
        last_action = a
        obs, r, done, info = env.step(a)
        print("\033[2J\033[H", end="")  # clear screen + home cursor
        print(ascii_frame(env))
        print(f"action: {ACTION_NAMES[last_action]}")
        time.sleep(delay)
    print("\nReached goal!" if info.get("reached_goal") else "\nDid not reach the top.")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate trained stair-climbing agent")
    p.add_argument("--qtable", type=str, default="qtable.npy")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--render", action="store_true", help="animated ASCII rollout")
    p.add_argument("--plot", type=str, default=None, help="save trajectory PNG to path")
    args = p.parse_args()

    evaluate(args.qtable, episodes=args.episodes, seed=args.seed)

    if args.render:
        animate_ascii(args.qtable, seed=args.seed)

    if args.plot:
        env = StairClimbEnv()
        env.seed(args.seed)
        agent = TabularQLearner(n_actions=env.n_actions, seed=args.seed)
        agent.load(args.qtable)
        agent.epsilon = 0.0
        xs, ys, _, total, info = rollout(env, agent)
        out = plot_trajectory(env, xs, ys,
                              path=args.plot,
                              title=f"Robot stair climb (reward {total:.1f})")
        print(f"saved trajectory -> {out}" if out else "matplotlib unavailable; skipped plot")


if __name__ == "__main__":
    main()
