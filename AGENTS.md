# Project-wide agent context

## Reproduction objective

- The primary goal is to reproduce the experiments and results in `paper.pdf`:
  "Cortical Spiking Network Interfaced with Virtual Musculoskeletal Arm and
  Robotic Arm" (Dura-Bernal et al., 2015).
- Treat `paper.pdf`, `README.md`, and the checked-in model code as the local
  sources of truth. Do not modify or replace `paper.pdf`.
- This repository is scoped to the 2015 paper reproduction workflow.
- A dummy-arm run is an intermediate integration test. Full paper reproduction
  ultimately requires the realistic musculoskeletal arm; the WAM robot hardware
  portion may be documented or simulated when the physical robot is unavailable.

## Required environment

- Always use the existing Conda environment named `snn` for this project.
- Current verified environment:
  - macOS on Apple Silicon (`arm64`)
  - Python 3.12.13
  - NumPy 1.26.4
  - SciPy 1.12.0
  - Matplotlib 3.8.4
  - NEURON 8.2.6 (`8.2.6-1-gb6e6a5fad+`)
  - Setuptools 80.9.0
- Prefer non-interactive commands of the form `conda run -n snn <command>` so
  the environment is explicit and reproducible.
- Do not use the Miniforge `base` environment or Homebrew Python 3.14 for model
  commands.
- Do not upgrade Python, NumPy, or NEURON without explicit user approval. The
  pinned NEURON 8.2.x line is intentional because the legacy MOD files contain
  extensive C-oriented `VERBATIM` code that may require migration for NEURON 9.
- Xcode and its command-line build tools are available.

## Model architecture

- Python is the launch, analysis, plotting, and arm-interface layer.
- HOC defines the network, experiment orchestration, stimulation, learning, and
  closed-loop arm controller.
- NMODL files under `mod/` implement neuron, synapse, stimulation, statistics,
  and utility mechanisms and must be compiled before simulation.
- Primary entry point:
  - `sim.py` -> `main.hoc`: 2015-paper training and testing workflow.
- Important implementation files:
  - `network.hoc`: populations, sizes, connectivity probabilities, and weights.
  - `params.hoc`: neuron and synapse parameters.
  - `basestdp.hoc` and `mod/intf6.mod`: reward-modulated STDP and eligibility
    traces.
  - `msarm.hoc`: motor decoding, proprioceptive feedback, reinforcement signal,
    targets, and periodic arm updates.
  - `arminterface_pipe.py`: dummy-arm and external musculoskeletal-arm bridge.
  - `msarm/`: OpenSim model, partial C++ sources, launcher, and legacy Linux
    libraries.

## Paper-derived reproduction targets

- Neural populations:
  - proprioceptive P/DP: 192 units, divided across four muscle groups;
  - sensory S: 192 excitatory, 44 fast-spiking inhibitory, and 20
    low-threshold inhibitory cells;
  - motor M: 192 excitatory, 44 fast-spiking inhibitory, and 20 low-threshold
    inhibitory cells;
  - total: 704 model units.
- Four decoded muscle groups: shoulder extensor, shoulder flexor, elbow
  extensor, and elbow flexor.
- Closed-loop arm exchange interval: 10 ms of simulated time. The realistic arm
  internally advances at 1 ms.
- Starting posture: shoulder 0.62 rad (35 degrees), elbow 1.53 rad (88 degrees).
- Paper targets: left and bottom, each 15 cm from the starting hand position;
  successful reach means entering a target area of radius 4 cm.
- Paper training duration: 360 s of simulated time for each trained network.
- Test trials last 1 s because the model has no explicit stopping mechanism.
- Primary validation evidence:
  - proprioceptive and motor-population encoding comparable to Figure 5;
  - hand trajectories and velocity profiles comparable to Figure 6;
  - reach success and target occupancy statistics;
  - jerk and dimensionless jerk comparisons comparable to Figure 7;
  - realistic-arm muscle force patterns comparable to Figure 8.
- Preserve deterministic input and wiring seeds when comparing results.

## Current compatibility constraints

- Keep Setuptools pinned to 80.9.0: NEURON 8.2.6's `nrniv` and `nrnivmodl`
  launchers still import `pkg_resources`, which newer Setuptools releases have
  removed. The deprecation warning from Setuptools 80.9.0 is expected.
- On this NEURON 8.2.6 wheel, repeatedly running `nrnivmodl` against an
  existing architecture directory can wrap `special` twice. For a reliable
  rebuild, move or remove the ignored generated directory (`arm64/` on this
  machine) first, then run `conda run -n snn nrnivmodl mod` once.
- The core Python 3.12 dummy-arm path (`sim.py`) now loads, trains, tests, and
  saves through NEURON. The primary plotting and analysis chain
  (`hocinterface.py`, `neuroplot.py`, `analysis.py`, `analyse_funcs.py`, and
  `armGraphs.py`) has been migrated to Python 3 and is covered by a short
  analysis-enabled smoke test. The unrelated legacy electrophysiology helpers
  `load.py` and `vector.py` were removed; they were not referenced by the 2015
  simulation or analysis paths.
- Use `SNN_USE_NEURON_GUI=0` for headless runs. Set `SNN_SMOKE_MS=<milliseconds>`
  to shorten both training and testing for compatibility checks. Legacy
  plotting/analysis helpers are skipped unless `SNN_LOAD_ANALYSIS=1` is set.
  Keep Matplotlib's default backend and configuration directory; do not set
  `MPLBACKEND` or `MPLCONFIGDIR` in project commands.
- The test phase defaults to the paper's 1000 ms duration. Set
  `SNN_TEST_MS=<milliseconds>` to use a different positive test duration without
  changing training. `sim.py` training defaults to 360000 ms (twelve 30000 ms
  epochs). Set `SNN_TRAIN_MS=<milliseconds>` to override total training time.
  Custom training is split into 30000 ms epochs plus a final shorter epoch so
  the established arm-reset boundaries are preserved. Explicit train/test
  overrides take precedence over `SNN_SMOKE_MS`; smoke mode fills in unspecified
  durations and disables animation.
- On local macOS, NEURON GUI runs do not require XQuartz. Launch them with
  `SNN_USE_NEURON_GUI=1 python -i sim.py`; `-i` is required so the Python
  process and native GUI remain alive after the script completes. `sim.py`
  automatically calls `plotTraj` after testing; call `h.plotTraj(h.tstop, 1)` at
  the Python prompt to recreate the final trajectory window when needed.
- Simulation results are written under `outputs/<run-id>/`, with a `run.json`
  manifest and `output_*.nqs` data files. By default the launcher generates a
  unique timestamped run ID. Set `SNN_RUN_ID=<name>` for a reproducible label;
  an existing name is rejected rather than overwritten. Set `SNN_OUTPUT_ROOT`
  only when results need to be stored outside the project `outputs/` directory.
  Validate a completed run with
  `conda run -n snn python validate_run.py outputs/<run-id>`.
  Generate the standard trajectory/raster/muscle/error summary with
  `conda run -n snn python visualize_run.py outputs/<run-id>`;
  figures are saved under that run's `figures/` subdirectory.
  Render the saved dummy-arm motion without rerunning the simulation with
  `conda run -n snn python animate_run.py outputs/<run-id>`;
  the MP4 is saved as `figures/arm_motion.mp4`.
- `dummyArm = 1` is currently the default in `arminterface_pipe.py`.
- The realistic-arm launcher expects `msarm/msarm`, which is absent.
- `msarm/Makefile` references the absent `LSODAIntegrator2.cpp`, and bundled
  shared libraries are Linux x86-64 ELF binaries, not native macOS libraries.
- `paper.pdf` is intentionally untracked user-owned input. Preserve it and do
  not stage or commit it unless the user explicitly requests that action.

## Recommended execution order

1. Compile all `mod/*.mod` mechanisms with the NEURON 8.2.6 `nrnivmodl`.
2. Keep the validated Python 3 and HOC core path passing with a very short
   headless dummy-arm smoke test.
3. Run a short `sim.py` dummy-arm workflow with duration overrides and validate
   spike, arm-feedback, learning, and output data paths.
4. Run the paper-configured 360 s training and 1 s evaluation while preserving
   seeds and parameters; initially this may still use the dummy arm as a neural
   pipeline check.
5. Restore or replace the realistic musculoskeletal-arm executable on macOS,
   then repeat the paper-configured runs with `dummyArm = 0`.
6. Compare trajectories, success metrics, velocities, jerk, and muscle forces
   against the paper before declaring reproduction complete.

## Development and verification rules

- Keep fixes minimal and behavior-preserving until a baseline reproduction is
  obtained. Record any deliberate deviation from paper parameters.
- Use short simulation durations for smoke tests; never begin a 360 s training
  run merely to test whether imports or initialization work.
- Prefer headless checks while debugging. Enable GUI/animation only for visual
  validation after the simulation path is stable.
- Run both the NEURON process and any local client or probe outside the sandbox
  and in the same network namespace when local inter-process communication is
  involved.
- Generated mechanism binaries, temporary outputs, and plots should not be
  committed unless they are explicitly selected as reproducibility artifacts.
- Preserve user-owned Git changes. Do not stage or commit changes unless the
  user explicitly requests it.
