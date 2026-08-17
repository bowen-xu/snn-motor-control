"""Create a compact summary figure from one saved simulation run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from neuron import h


ARM_COLUMNS = (
    "t",
    "ang0",
    "ang1",
    "x",
    "y",
    "shext",
    "shflex",
    "elext",
    "elflex",
    "ex",
    "ey",
    "errxy",
    "errang",
)
SPIKE_COLUMNS = ("id", "type", "t", "mid")

ARM_LENGTHS_M = (0.4634 - 0.173, 0.7169 - 0.4634)
CENTER_OUT_TARGET_ANGLES = (
    (0.45519364, 1.16293505),
    (0.99117650, 1.57515715),
    (1.29295714, 0.20699215),
    (0.20234833, 2.19736336),
    (0.95196038, 0.47728848),
    (1.16275143, 0.92036049),
    (0.12116826, 1.78758638),
    (0.67440973, 2.09186399),
)

MUSCLE_LABELS = (
    "shoulder extensor",
    "shoulder flexor",
    "elbow extensor",
    "elbow flexor",
)
MUSCLE_COLORS = ("#0F4D92", "#5B8FD6", "#B64342", "#E9A6A1")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
        }
    )


def _load_columns(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    table = h.NQS(str(path))
    columns = {}
    for name in names:
        if int(table.fi(name, "NOERR")) == -1:
            raise ValueError(f"{path.name}: missing NQS column {name!r}")
        columns[name] = np.asarray(table.getcol(name), dtype=float)
    return columns


def _load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    manifest_path = run_dir / "run.json"
    arm_path = run_dir / "output_test-nqa.nqs"
    spike_path = run_dir / "output_test-spk.nqs"
    for path in (manifest_path, arm_path, spike_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing run artifact: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_dir = Path(__file__).resolve().parent
    previous_dir = Path.cwd()
    try:
        os.chdir(project_dir)
        if h.load_file("nqs.hoc") != 1:
            raise RuntimeError("NEURON could not load nqs.hoc")
        arm = _load_columns(arm_path, ARM_COLUMNS)
        spikes = _load_columns(spike_path, SPIKE_COLUMNS)
    finally:
        os.chdir(previous_dir)
    return manifest, arm, spikes


def _target_position(target_id: int) -> tuple[float, float]:
    shoulder, elbow = CENTER_OUT_TARGET_ANGLES[target_id]
    upper_arm, forearm = ARM_LENGTHS_M
    x = upper_arm * np.cos(shoulder) + forearm * np.cos(shoulder + elbow)
    y = upper_arm * np.sin(shoulder) + forearm * np.sin(shoulder + elbow)
    return float(x), float(y)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )


def create_summary_figure(
    run_dir: Path,
    output_dir: Path | None = None,
    target_id: int = 1,
    arm_start_ms: float = 20.0,
) -> list[Path]:
    run_dir = run_dir.expanduser().resolve()
    manifest, arm, spikes = _load_run(run_dir)
    if output_dir is None:
        output_dir = run_dir / "figures"
    else:
        output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_style()
    fig = plt.figure(figsize=(7.2, 6.1), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.0, 1.35))
    trajectory_ax = fig.add_subplot(grid[0:2, 0])
    raster_ax = fig.add_subplot(grid[0, 1])
    muscle_ax = fig.add_subplot(grid[1, 1])
    angle_ax = fig.add_subplot(grid[2, 0])
    error_ax = fig.add_subplot(grid[2, 1])

    arm_mask = arm["t"] >= arm_start_ms
    if not arm_mask.any():
        raise ValueError(
            f"No arm samples remain at or after --arm-start-ms={arm_start_ms:g}"
        )
    arm_view = {name: values[arm_mask] for name, values in arm.items()}
    time_s = arm_view["t"] / 1000.0
    x_cm = arm_view["x"] * 100.0
    y_cm = arm_view["y"] * 100.0

    trajectory_ax.plot(x_cm, y_cm, color="#0F4D92", zorder=2)
    trajectory_ax.scatter(
        x_cm[0], y_cm[0], s=28, color="#272727", marker="o", label="start", zorder=3
    )
    trajectory_ax.scatter(
        x_cm[-1], y_cm[-1], s=34, color="#42949E", marker="o", label="end", zorder=3
    )
    target_x, target_y = _target_position(target_id)
    trajectory_ax.scatter(
        target_x * 100.0,
        target_y * 100.0,
        s=48,
        color="#B64342",
        marker="x",
        linewidth=1.5,
        label=f"target {target_id}",
        zorder=4,
    )
    trajectory_ax.set_aspect("equal", adjustable="datalim")
    trajectory_ax.set_xlabel("horizontal position (cm)")
    trajectory_ax.set_ylabel("vertical position (cm)")
    trajectory_ax.set_title("Hand trajectory")
    trajectory_ax.legend(loc="best", fontsize=6)
    _panel_label(trajectory_ax, "a")

    spike_time_s = spikes["t"] / 1000.0
    mids = spikes["mid"].astype(int)
    for muscle_id, (label, color) in enumerate(zip(MUSCLE_LABELS, MUSCLE_COLORS)):
        selected = mids == muscle_id
        raster_ax.scatter(
            spike_time_s[selected],
            spikes["id"][selected],
            s=1.2,
            color=color,
            linewidths=0,
            rasterized=True,
            label=label,
        )
    raster_ax.axhline(256, color="#A8A8A8", linewidth=0.6, linestyle=":")
    raster_ax.axhline(512, color="#A8A8A8", linewidth=0.6, linestyle=":")
    raster_ax.text(1.005, 0.18, "sensory", transform=raster_ax.transAxes, fontsize=6)
    raster_ax.text(1.005, 0.55, "motor", transform=raster_ax.transAxes, fontsize=6)
    raster_ax.text(1.005, 0.88, "DP", transform=raster_ax.transAxes, fontsize=6)
    raster_ax.set_xlim(time_s[0], time_s[-1])
    raster_ax.set_ylim(-5, 708)
    raster_ax.set_ylabel("cell ID")
    raster_ax.set_title("Spike raster by muscle group")
    raster_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=5.7,
        markerscale=3,
    )
    _panel_label(raster_ax, "b")

    command_names = ("shext", "shflex", "elext", "elflex")
    for name, label, color in zip(command_names, MUSCLE_LABELS, MUSCLE_COLORS):
        muscle_ax.plot(time_s, arm_view[name], color=color, label=label)
    muscle_ax.set_xlim(time_s[0], time_s[-1])
    muscle_ax.set_ylabel("spikes per command window")
    muscle_ax.set_title("Decoded muscle commands")
    muscle_ax.legend(ncol=2, fontsize=5.7, loc="upper right")
    _panel_label(muscle_ax, "c")

    angle_ax.plot(
        time_s, np.rad2deg(arm_view["ang0"]), color="#0F4D92", label="shoulder"
    )
    angle_ax.plot(
        time_s, np.rad2deg(arm_view["ang1"]), color="#B64342", label="elbow"
    )
    angle_ax.set_xlim(time_s[0], time_s[-1])
    angle_ax.set_xlabel("test time (s)")
    angle_ax.set_ylabel("joint angle (deg)")
    angle_ax.set_title("Joint kinematics")
    angle_ax.legend(fontsize=6)
    _panel_label(angle_ax, "d")

    distance_cm = arm_view["errxy"] * 100.0
    error_ax.plot(time_s, distance_cm, color="#B64342")
    error_ax.axhline(
        4.0,
        color="#606060",
        linewidth=0.9,
        linestyle="--",
        label="4 cm success radius",
    )
    error_ax.scatter(time_s[-1], distance_cm[-1], s=18, color="#B64342", zorder=3)
    error_ax.annotate(
        f"{distance_cm[-1]:.2f} cm",
        (time_s[-1], distance_cm[-1]),
        xytext=(-5, 6),
        textcoords="offset points",
        ha="right",
        fontsize=6,
    )
    error_ax.set_xlim(time_s[0], time_s[-1])
    error_ax.set_xlabel("test time (s)")
    error_ax.set_ylabel("distance to target (cm)")
    error_ax.set_title("Reaching error")
    error_ax.legend(fontsize=6)
    _panel_label(error_ax, "e")

    run_id = manifest.get("run_id", run_dir.name)
    fig.suptitle(f"Dummy-arm test summary — {run_id}", fontsize=9, fontweight="bold")
    if arm_start_ms > 0:
        fig.text(
            0.01,
            -0.015,
            f"Arm panels omit t < {arm_start_ms:g} ms initialization records; "
            "the spike raster shows the full test interval.",
            fontsize=5.5,
            color="#606060",
        )

    base = output_dir / "summary"
    outputs = [base.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="outputs/<run-id> directory")
    parser.add_argument("--output-dir", type=Path, help="figure output directory")
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
        help="omit earlier dummy-arm initialization records from arm panels (default: 20)",
    )
    args = parser.parse_args()
    outputs = create_summary_figure(
        args.run_dir, args.output_dir, args.target_id, args.arm_start_ms
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
