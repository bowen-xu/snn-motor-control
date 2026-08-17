"""Render a saved dummy-arm test as an MP4 animation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from visualize_run import (
    MUSCLE_COLORS,
    MUSCLE_LABELS,
    _configure_style,
    _load_run,
    _target_position,
)


def create_arm_video(
    run_dir: Path,
    output: Path | None = None,
    target_id: int = 1,
    arm_start_ms: float = 20.0,
    fps: int = 30,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    manifest, arm, _ = _load_run(run_dir)
    selected = arm["t"] >= arm_start_ms
    if not selected.any():
        raise ValueError(
            f"No arm samples remain at or after --arm-start-ms={arm_start_ms:g}"
        )
    arm = {name: values[selected] for name, values in arm.items()}

    if output is None:
        output = run_dir / "figures" / "arm_motion.mp4"
    else:
        output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    _configure_style()
    fig = plt.figure(figsize=(8.0, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.35))
    arm_ax = fig.add_subplot(grid[:, 0])
    command_ax = fig.add_subplot(grid[0, 1])
    error_ax = fig.add_subplot(grid[1, 1])

    time_s = arm["t"] / 1000.0
    x_cm = arm["x"] * 100.0
    y_cm = arm["y"] * 100.0
    elbow_x_cm = arm["ex"] * 100.0
    elbow_y_cm = arm["ey"] * 100.0
    target_x, target_y = _target_position(target_id)
    target_x_cm = target_x * 100.0
    target_y_cm = target_y * 100.0

    all_x = np.concatenate(([0.0, target_x_cm], elbow_x_cm, x_cm))
    all_y = np.concatenate(([0.0, target_y_cm], elbow_y_cm, y_cm))
    margin = 6.0
    arm_ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    arm_ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    arm_ax.set_aspect("equal", adjustable="box")
    arm_ax.set_xlabel("horizontal position (cm)")
    arm_ax.set_ylabel("vertical position (cm)")
    arm_ax.set_title("Dummy-arm movement")
    arm_ax.add_patch(
        patches.Circle(
            (target_x_cm, target_y_cm),
            radius=4.0,
            facecolor="#F6CFCB",
            edgecolor="#B64342",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            label="4 cm target area",
        )
    )
    arm_ax.scatter(
        target_x_cm,
        target_y_cm,
        color="#B64342",
        marker="x",
        s=36,
        linewidth=1.5,
        zorder=4,
    )
    arm_ax.scatter(0.0, 0.0, color="#272727", s=18, zorder=4)
    upper_arm_line, = arm_ax.plot([], [], color="#0F4D92", linewidth=4.0)
    forearm_line, = arm_ax.plot([], [], color="#42949E", linewidth=4.0)
    joints = arm_ax.scatter([], [], s=26, color="#272727", zorder=5)
    trajectory_line, = arm_ax.plot([], [], color="#767676", linewidth=1.0, alpha=0.8)
    time_label = arm_ax.text(
        0.03, 0.96, "", transform=arm_ax.transAxes, va="top", fontweight="bold"
    )
    arm_ax.legend(loc="lower left", fontsize=6)

    command_names = ("shext", "shflex", "elext", "elflex")
    for name, label, color in zip(command_names, MUSCLE_LABELS, MUSCLE_COLORS):
        command_ax.plot(time_s, arm[name], color=color, label=label)
    command_ax.set_xlim(time_s[0], time_s[-1])
    command_ax.set_ylabel("spikes per command window")
    command_ax.set_title("Decoded muscle commands")
    command_ax.legend(ncol=2, fontsize=5.7, loc="upper right")
    command_cursor = command_ax.axvline(time_s[0], color="#272727", linewidth=1.0)

    distance_cm = arm["errxy"] * 100.0
    error_ax.plot(time_s, distance_cm, color="#B64342")
    error_ax.axhline(
        4.0,
        color="#606060",
        linewidth=0.9,
        linestyle="--",
        label="4 cm success radius",
    )
    error_ax.set_xlim(time_s[0], time_s[-1])
    error_ax.set_xlabel("test time (s)")
    error_ax.set_ylabel("distance to target (cm)")
    error_ax.set_title("Reaching error")
    error_ax.legend(fontsize=6)
    error_cursor = error_ax.axvline(time_s[0], color="#272727", linewidth=1.0)
    error_point, = error_ax.plot([], [], "o", color="#B64342", markersize=4)

    run_id = manifest.get("run_id", run_dir.name)
    fig.suptitle(f"Closed-loop test — {run_id}", fontsize=9, fontweight="bold")

    def update(frame: int):
        upper_arm_line.set_data([0.0, elbow_x_cm[frame]], [0.0, elbow_y_cm[frame]])
        forearm_line.set_data(
            [elbow_x_cm[frame], x_cm[frame]],
            [elbow_y_cm[frame], y_cm[frame]],
        )
        joints.set_offsets(
            np.array(
                [
                    [0.0, 0.0],
                    [elbow_x_cm[frame], elbow_y_cm[frame]],
                    [x_cm[frame], y_cm[frame]],
                ]
            )
        )
        trajectory_line.set_data(x_cm[: frame + 1], y_cm[: frame + 1])
        time_label.set_text(f"simulation time = {time_s[frame]:.2f} s")
        command_cursor.set_xdata([time_s[frame], time_s[frame]])
        error_cursor.set_xdata([time_s[frame], time_s[frame]])
        error_point.set_data([time_s[frame]], [distance_cm[frame]])
        return (
            upper_arm_line,
            forearm_line,
            joints,
            trajectory_line,
            time_label,
            command_cursor,
            error_cursor,
            error_point,
        )

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(time_s),
        interval=1000.0 / fps,
        blit=True,
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2500,
        extra_args=["-pix_fmt", "yuv420p"],
        metadata={"title": f"Dummy-arm motion: {run_id}"},
    )
    movie.save(output, writer=writer, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="outputs/<run-id> directory")
    parser.add_argument("--output", type=Path, help="output MP4 path")
    parser.add_argument(
        "--target-id",
        type=int,
        choices=range(8),
        default=1,
        help="center-out target used by the current main.hoc test (default: 1, left)",
    )
    parser.add_argument(
        "--arm-start-ms",
        type=float,
        default=20.0,
        help="omit earlier dummy-arm initialization records (default: 20)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="output frames per second; 30 renders the 1 s test in about 3.3 s",
    )
    args = parser.parse_args()
    output = create_arm_video(
        args.run_dir,
        output=args.output,
        target_id=args.target_id,
        arm_start_ms=args.arm_start_ms,
        fps=args.fps,
    )
    print(output)


if __name__ == "__main__":
    main()
