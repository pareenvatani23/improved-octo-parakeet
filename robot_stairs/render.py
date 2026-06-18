"""Rendering helpers: an always-available ASCII view and an optional
matplotlib trajectory plot.
"""

from __future__ import annotations

from typing import List, Sequence

from .env import StairClimbEnv


def ascii_frame(env: StairClimbEnv, cols: int = 60, rows: int = 16) -> str:
    """Render the current env state as an ASCII grid.

    '#' is a step surface/fill, 'R' is the robot, '.' is empty space.
    """
    world_w = env.goal_x + env.step_width          # include top platform
    world_h = env.top_height + env.step_height + 0.5

    def cx(x: float) -> int:
        return min(cols - 1, max(0, int(x / world_w * cols)))

    def cy(y: float) -> int:
        # row 0 is the top of the grid
        return min(rows - 1, max(0, rows - 1 - int(y / world_h * rows)))

    grid = [["." for _ in range(cols)] for _ in range(rows)]

    # draw the staircase as solid fill beneath each surface
    for c in range(cols):
        x = (c + 0.5) / cols * world_w
        surf = env.surface_height(x)
        top_row = cy(surf)
        for r in range(top_row, rows):
            grid[r][c] = "#"

    # draw the robot
    rr, rc = cy(env.y), cx(env.x)
    grid[rr][rc] = "R"

    st = env.state()
    header = (
        f"t={st['t']:>3}  x={st['x']:.2f}  y={st['y']:.2f}  "
        f"step={st['step_index']}/{env.num_steps}  "
        f"{'ground' if st['on_ground'] else 'air'}"
    )
    return header + "\n" + "\n".join("".join(row) for row in grid)


def plot_trajectory(env: StairClimbEnv, xs: Sequence[float], ys: Sequence[float],
                    path: str = "trajectory.png", title: str = "Robot stair climb") -> str | None:
    """Save a matplotlib plot of the staircase and the robot's path.

    Returns the output path, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    # staircase outline
    step_xs: List[float] = [0.0]
    step_ys: List[float] = [0.0]
    for i in range(env.num_steps):
        x0 = i * env.step_width
        x1 = (i + 1) * env.step_width
        h = i * env.step_height
        step_xs += [x0, x1]
        step_ys += [h, h]
    # top platform
    step_xs += [env.goal_x, env.goal_x + env.step_width]
    step_ys += [env.top_height, env.top_height]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(step_xs, step_ys, -0.5, step="post", color="#cfd8dc", label="stairs")
    ax.plot(step_xs, step_ys, color="#607d8b", lw=2, drawstyle="steps-post")
    ax.plot(xs, ys, color="#e53935", lw=2, label="robot path")
    ax.scatter([xs[0]], [ys[0]], color="green", zorder=5, label="start")
    ax.scatter([xs[-1]], [ys[-1]], color="black", zorder=5, label="end")
    ax.set_xlabel("x")
    ax.set_ylabel("height")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_ylim(-0.5, env.top_height + 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
