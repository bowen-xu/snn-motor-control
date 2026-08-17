"""Validate the saved NQS data for one simulation run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from neuron import h


SPIKE_COLUMNS = ("col", "gid", "id", "type", "ice", "t", "mid")
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
    "phase",
    "ML0",
    "ML1",
    "ML2",
    "ML3",
    "subphase",
    "errxy",
    "errang",
)
DP_CELL_TYPE = 2


def _load_columns(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    table = h.NQS(str(path))
    columns = {}
    for name in names:
        if int(table.fi(name, "NOERR")) == -1:
            raise ValueError(f"{path.name}: missing NQS column {name!r}")
        columns[name] = np.asarray(table.getcol(name), dtype=float)
    return columns


def _require_finite_nonempty(
    table_name: str, columns: dict[str, np.ndarray]
) -> int:
    sizes = {values.size for values in columns.values()}
    if len(sizes) != 1:
        raise ValueError(f"{table_name}: columns have inconsistent lengths")
    row_count = sizes.pop()
    if row_count == 0:
        raise ValueError(f"{table_name}: table is empty")
    for name, values in columns.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{table_name}: column {name!r} contains NaN/Inf")
    return row_count


def validate_run(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "run.json"
    spike_path = run_dir / "output_test-spk.nqs"
    arm_path = run_dir / "output_test-nqa.nqs"
    for path in (manifest_path, spike_path, arm_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing run artifact: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_dir = Path(__file__).resolve().parent
    previous_dir = Path.cwd()
    try:
        os.chdir(project_dir)
        if h.load_file("nqs.hoc") != 1:
            raise RuntimeError("NEURON could not load nqs.hoc")
        spikes = _load_columns(spike_path, SPIKE_COLUMNS)
        arm = _load_columns(arm_path, ARM_COLUMNS)
    finally:
        os.chdir(previous_dir)

    spike_rows = _require_finite_nonempty("spikes", spikes)
    arm_rows = _require_finite_nonempty("arm trajectory", arm)

    ids = spikes["id"]
    mids = spikes["mid"]
    if not np.equal(ids, np.floor(ids)).all():
        raise ValueError("spikes: cell IDs are not integers")
    if not np.equal(mids, np.floor(mids)).all():
        raise ValueError("spikes: muscle IDs are not integers")
    if not np.equal(mids, ids.astype(np.int64) % 4).all():
        raise ValueError("spikes: mid does not match id % 4")

    dp_mask = spikes["type"] == DP_CELL_TYPE
    if not dp_mask.any():
        raise ValueError("spikes: no proprioceptive DP spikes were saved")
    dp_groups = sorted(np.unique(mids[dp_mask]).astype(int).tolist())
    if dp_groups != [0, 1, 2, 3]:
        raise ValueError(
            f"spikes: incomplete DP muscle groups; found {dp_groups}, expected [0, 1, 2, 3]"
        )

    if np.any(np.diff(arm["t"]) < 0):
        raise ValueError("arm trajectory: time is not monotonically increasing")

    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "entrypoint": manifest.get("entrypoint"),
        "spike_rows": spike_rows,
        "dp_spike_rows": int(dp_mask.sum()),
        "dp_muscle_groups": dp_groups,
        "arm_rows": arm_rows,
        "arm_time_ms": [float(arm["t"][0]), float(arm["t"][-1])],
        "checked_arm_columns": list(ARM_COLUMNS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="outputs/<run-id> directory")
    args = parser.parse_args()
    summary = validate_run(args.run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
