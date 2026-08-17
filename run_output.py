"""Create an isolated output directory for one simulation run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RECORDED_ENVIRONMENT = (
    "SNN_LOAD_ANALYSIS",
    "SNN_OUTPUT_ROOT",
    "SNN_RUN_ID",
    "SNN_SMOKE_MS",
    "SNN_TEST_MS",
    "SNN_TRAIN_MS",
    "SNN_USE_NEURON_GUI",
)


@dataclass(frozen=True)
class RunOutput:
    run_id: str
    run_dir: Path
    output_stem: str


def _generated_run_id(entrypoint: str, created_at: datetime) -> str:
    entrypoint_name = Path(entrypoint).stem
    return f"{entrypoint_name}-{created_at:%Y%m%d-%H%M%S-%f}"


def _validate_run_id(run_id: str) -> None:
    if not _SAFE_RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(
            "SNN_RUN_ID must be one path-safe component containing only "
            "letters, digits, '.', '_' or '-'"
        )


def prepare_run_output(
    entrypoint: str,
    *,
    output_root: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
) -> RunOutput:
    """Create ``outputs/<run-id>`` and return its NQS output stem.

    An explicitly named run is never reused. This deliberately turns accidental
    reruns into an early error instead of allowing NQS to overwrite old results.
    """

    created_at = datetime.now(timezone.utc)
    if run_id is None:
        run_id = os.environ.get("SNN_RUN_ID")
    if run_id is None:
        run_id = _generated_run_id(entrypoint, created_at)
    _validate_run_id(run_id)

    if output_root is None:
        output_root = os.environ.get("SNN_OUTPUT_ROOT", "outputs")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    run_dir = root / run_id
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Run output directory already exists: {run_dir}. "
            "Choose a different SNN_RUN_ID."
        ) from exc

    output_stem = str(run_dir / "output")
    manifest = {
        "run_id": run_id,
        "entrypoint": entrypoint,
        "created_at_utc": created_at.isoformat(),
        "output_stem": output_stem,
        "python_version": platform.python_version(),
        "argv": sys.argv,
        "environment": {
            name: os.environ[name]
            for name in _RECORDED_ENVIRONMENT
            if name in os.environ
        },
    }
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Run output directory: {run_dir}")
    return RunOutput(run_id=run_id, run_dir=run_dir, output_stem=output_stem)
