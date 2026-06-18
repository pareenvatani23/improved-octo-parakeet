"""Train the tabular Q-learning agent on StairClimbEnv.

Usage:
    python -m robot_stairs.train --episodes 4000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .agent import TabularQLearner
from .env import StairClimbEnv


def run_episode(env: StairClimbEnv, agent: TabularQLearner, train: bool = True):
    obs = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.act(obs, greedy=not train)
        next_obs, reward, done, info = env.step(action)
        if train:
            agent.update(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
    if train:
        agent.decay_epsilon()
    return total_reward, info


def train(
    episodes: int = 4000,
    seed: int = 0,
    log_every: int = 200,
    save_path: str | None = "qtable.npy",
    verbose: bool = True,
):
    env = StairClimbEnv()
    env.seed(seed)
    agent = TabularQLearner(n_actions=env.n_actions, seed=seed)

    rewards = np.zeros(episodes)
    successes = np.zeros(episodes)
    steps_climbed = np.zeros(episodes)

    t0 = time.time()
    for ep in range(episodes):
        r, info = run_episode(env, agent, train=True)
        rewards[ep] = r
        successes[ep] = float(info["reached_goal"])
        steps_climbed[ep] = info["step_index"]

        if verbose and (ep + 1) % log_every == 0:
            lo = max(0, ep + 1 - log_every)
            print(
                f"ep {ep + 1:>5}/{episodes} | "
                f"reward {rewards[lo:ep + 1].mean():7.2f} | "
                f"success {successes[lo:ep + 1].mean() * 100:5.1f}% | "
                f"stairs {steps_climbed[lo:ep + 1].mean():4.2f}/{env.num_steps} | "
                f"eps {agent.epsilon:.3f}"
            )

    if verbose:
        print(f"trained {episodes} episodes in {time.time() - t0:.1f}s")

    if save_path:
        agent.save(save_path)
        if verbose:
            print(f"saved Q-table -> {save_path}")

    return agent, {
        "rewards": rewards,
        "successes": successes,
        "steps_climbed": steps_climbed,
    }


def plot_learning_curve(history: dict, path: str = "learning_curve.png") -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    def smooth(a, w=100):
        if len(a) < w:
            return a
        return np.convolve(a, np.ones(w) / w, mode="valid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(smooth(history["rewards"]), color="#1e88e5")
    ax1.set_ylabel("avg reward (smoothed)")
    ax1.set_title("Learning curve")
    ax2.plot(smooth(history["successes"]) * 100, color="#43a047")
    ax2.set_ylabel("success rate % (smoothed)")
    ax2.set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Train robot stair-climbing RL agent")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="qtable.npy")
    p.add_argument("--plot", action="store_true", help="save learning_curve.png")
    args = p.parse_args()

    agent, history = train(episodes=args.episodes, seed=args.seed, save_path=args.save)

    if args.plot:
        out = plot_learning_curve(history)
        print(f"saved learning curve -> {out}" if out else "matplotlib unavailable; skipped plot")


if __name__ == "__main__":
    main()
