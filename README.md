# RQ_ab Toolkit

Code for the numerical experiments and figures of `Robust Queueing for Single-Server Queues with Abandonment`: robust-queueing (RQ) approximations of the steady-state workload in `GI/GI/1+GI` queues (and a tandem variant), compared against Monte-Carlo simulation and the Ward–Glynn / hazard-rate-scaling / Huang–Gurvich benchmark approximations.

## Layout

| Path            | What it is |
|-----------------|------------|
| `reproduce.py`  | One command per paper figure (start here) |
| `rqab/`         | Python package: models, RQ solvers, table interpolators, plotting |
| `scripts/`      | Thin CLIs: `run_grid.py`, `plot_ratio_panels.py`, `plot_c_heatmap.py`, `plot_idw_effective.py`, `plot_w_tables.py`, `plot_w_tripanel.py`, `plot_b_overlay.py`, `generate_grid.py` |
| `src/`, `include/` | C++: `w_{c,k}(t)` PDE solver, `b(c)` calibration, MC simulators |
| `configs/`      | Model configs (`workload_*.json`, `effective_idw_*.json`) and the 391-tuple grid |
| `results/`      | All generated tables, CSVs, and figures |
| `tests/`        | C++ and Python test suites |

## Prerequisites

- CMake >= 3.20, a C++20 compiler
- Python >= 3.9 with `matplotlib` and `numpy` (avoid matplotlib 3.11.0 for LaTeX-rendered figures: its usetex PDF output drops minus signs; the plotting code detects this and falls back to mathtext with a warning)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

C++ binaries produced: `wck` (one `w_{c,k}(t)` curve), `wck_sweep` (matrix table over a c-grid), `wck_calibrate_b` (`b(c)` table), `workload_mc` (Monte-Carlo workload simulator; auto-detects single-station vs tandem configs), `idw_sim` (effective-IDW simulator). Every binary supports `--help`.

## Reproducing the paper figures

```bash
python3 reproduce.py --list        # targets + cost estimates
python3 reproduce.py all           # five paper figure groups (22 PDFs)
python3 reproduce.py fig:MM1_GI    # or by alias: mm1-gi; other figures: var-approx, mgi1-gi, gigi1-gi, qis
python3 reproduce.py tables        # w/b tables for k = 1, 2, 3
python3 reproduce.py aux           # every non-paper artifact in results/
```

`reproduce.py` handles the dependency chain automatically (build → w/b tables → workload MC grid → refined/first-RQ grids → figures) and reuses existing outputs; add `--force` to regenerate. Effective-IDW simulations are reused when their curve CSVs are complete, while their lightweight PDF plots are recreated whenever the target is selected. A provenance warning is printed when an existing CSV disagrees with its config (e.g. different replication count or seed).

**Smoke test** (minutes; writes to `results_quick/` — reads the production w/b tables but never writes to `results/`):

```bash
python3 reproduce.py --quick all
```

Useful flags: `--dry-run` (print the plan), `--jobs N` (parallel analytic/plot steps), `--threads N` (thread cap for the `workload_mc` grid runs).

### Memory requirements

A full production run (`reproduce.py all`, `fig:Var_approx`, or `aux`) peaks at about 87 GB of RAM (≈81 GiB). The bottleneck is the effective-IDW estimation stage (`idw_sim`), which stores a long path: it bins the accepted work of the entire sample window into one array of `sample_time / tau_shift` doubles per tau shift and holds all `n_tau_shifts` arrays simultaneously. The estimator aggregates each array in place and frees it as soon as its shift is done, so the resident arrays are the whole story — peak bytes per `idw_sim` run are approximately

```
8 * (sample_time / tau) * S,   S = sum_{j<n_tau_shifts} 2^(-j/(n_tau_shifts-1))
```

(`S = 1` when `n_tau_shifts` is 1; `S ≈ 3.64` for the shipped value 5.)

Everything else is memory-trivial (`workload_mc` streams its statistics and never stores paths; the table solvers use 1-D grids), and `reproduce.py` runs heavy steps strictly one at a time, so peaks never stack.

To trade fidelity for memory, edit the `simulation` block of `configs/effective_idw_h2m1m.json` and `configs/effective_idw_h2m1e2.json` (for `aux`, also `effective_idw_mm1m.json` and `effective_idw_mm1e2.json`, which ship the same settings and hit the same peak) — `sample_time`, `tau`, and `n_tau_shifts` have no CLI override. The knobs are independent and multiply:

| Adjustment (from shipped values) | Peak | Accuracy cost |
|---|---|---|
| none (`sample_time` 3e8, `tau` 0.1, `n_tau_shifts` 5) | ~87 GB | — |
| `sample_time`: 3e8 → 1e8 | ~29 GB | ~1.7× wider CI per point; the largest horizons end up near the `min_windows_per_t` floor (horizons start being dropped only below ~8e7) |
| `n_tau_shifts`: 5 → 1 | ~24 GB | ~4× fewer distinct points along the t-axis (dyadic horizons only); per-point quality unchanged |
| `tau`: 0.1 → 0.2 | ~44 GB | smallest resolvable horizon doubles (curve loses its small-t end) |
| `sample_time` 1e8 **and** `n_tau_shifts` 1 | ~8 GB | both costs above |

Notes:

- `simulation.threads` in the `effective_idw_*.json` configs (and the `idw_sim --threads` override) parallelizes the per-shift estimation inside the already-allocated arrays, so it trades wall time only — peak memory and results are identical for any thread count. `reproduce.py --threads` reaches only the `workload_mc` grid runs, never `idw_sim`.
- `--quick` rewrites `sample_time` to 1e5 (and shrinks `warmup_time` and `max_level`), so the smoke test peaks at ~30 MB.
- The other figure targets (`fig:MM1_GI`, `fig:MGI1_GI`, `fig:GIGI1_GI`, `fig:QIS`, `tables`) are memory-trivial: `workload_mc` and the table solvers peak in the tens of MB, plus ordinary Python/matplotlib overhead.
- If the per-alpha `model0_*_curve.csv` files from an earlier run are already in `results/`, the `idw_sim` step is skipped entirely (unless `--force`).

### Figure → experiment map

| TeX figure | Output PDFs | Models |
|---|---|---|
| `fig:Var_approx` | `idw_effective_{h2m1m,h2m1e2}.pdf` | effective-IDW sims, `H2(4)/M/1+{M,E2}` |
| `fig:MM1_GI` | `approx_ratio_{tripanel,twopanel}_{mm1m,mm1e2,mm1h2_4}.pdf` | `M/M/1+{M,E2,H2(4)}` |
| `fig:MGI1_GI` | `approx_ratio_{tripanel,twopanel}_{mln1_41h2_4,mln1_41e2}.pdf` | `M/LN(1,4)/1+{H2(4),E2}` |
| `fig:GIGI1_GI` | `approx_ratio_{tripanel,twopanel}_{e2ln1_21e2,h2_4ln1_21h2_4,h2_4ln1_21e2}.pdf` | `{E2,H2(4)}/LN(1,2)/1+{E2,H2(4)}` |
| `fig:QIS` | `approx_ratio_{tripanel,twopanel}_tandem_*.pdf` | tandem `GI/GI/1 -> ./M/1+GI` |

## Running individual experiments

One grid experiment (391 `(lambda, alpha)` tuples) at a time:

```bash
# Monte-Carlo ground truth -> results/workload_grid_aggregate_<alias>.csv
python3 scripts/run_grid.py --method workload --model-config configs/workload_mm1m.json

# Refined RQ + WG/Hazard/HG columns -> results/refined_rq_grid_<alias>.csv
python3 scripts/run_grid.py --method refined --model-config configs/workload_mm1m.json

# First (crude) RQ with standardized b_k(q) -> results/first_rq_grid_<alias>.csv
python3 scripts/run_grid.py --method first --model-config configs/workload_mm1m.json
```

Tandem configs (`configs/workload_tandem_*.json`) work unchanged — the method scripts and the `workload_mc` binary detect the `model.queue1` block.

Heatmap figures from the CSVs (missing inputs are generated automatically):

```bash
python3 scripts/plot_ratio_panels.py --panels tripanel --model-config configs/workload_mm1m.json --no-show
python3 scripts/plot_ratio_panels.py --panels twopanel --model-config configs/workload_mm1m.json --no-show
python3 scripts/plot_c_heatmap.py --model-config configs/workload_mm1m.json --no-show   # refined-RQ c on the same grid
```

Effective-IDW overlay and table diagnostics:

```bash
python3 scripts/plot_idw_effective.py --config configs/effective_idw_h2m1m.json --no-show
python3 scripts/plot_w_tables.py overlay --k 2 --no-show
python3 scripts/plot_w_tables.py b --k 2 --no-show
python3 scripts/plot_w_tripanel.py --no-show   # w_{c,k}(t) tripanel, k = 1,2,3 -> results/w_tripanel.pdf
python3 scripts/plot_b_overlay.py --no-show    # calibrated b(c) overlay, k = 1,2,3 -> results/b_overlay.pdf
```

Single simulator runs (bypassing the grid drivers):

```bash
./build/workload_mc --config configs/workload_mm1m.json --lambda 0.99 --alpha 0.125 --summary-json /tmp/summary.json
./build/idw_sim --config configs/effective_idw_h2m1m.json --out-dir results
```

## Reproducibility

- Every simulation seed lives in the config (`simulation.seed`, default 123456789); per-replication seeds are derived deterministically from (seed, model name, lambda, alpha, replication index), so results are independent of `--threads` and reproducible run-to-run.
- The analytic (refined/first RQ, WG/Hazard/HG) grids are deterministic.
- `results/w_table_matrix_k*.csv` and `results/b_table_k*.csv` are inputs to the refined RQ; regenerate them with `python3 reproduce.py tables --force`.

## Testing

```bash
ctest --test-dir build --output-on-failure          # C++
python3 -m unittest discover -s tests -p 'test_*.py' # Python
```

## Formulas

The implementations map to `Robust Queueing for Single-Server Queues with Abandonment` as follows: first RQ fixed point (eq:RQ_ab_1) and its standardized `b_k(q)` calibration—with an explicit `b=0` fallback when exact matching is infeasible—in `rqab/first.py`; refined RQ (eq:RQ_ab_2) with the `w_{c,k}` scaling (Lemma var_expression) in `rqab/refined.py` + `rqab/effective_idw.py`; the effective-IDW approximation (eq:V) in `rqab/effective_idw.py`; WG (eq:ROU_expectation), hazard-rate scaling, and HG benchmarks in `rqab/secondary.py`; the tandem departure-IDC blend (eq:IDC_dep_app, eq:wstar_app) in `rqab/idc.py`; the `w_{c,k}(t)` PDE solver and `b(c)` calibration in `src/wck_table/`, `src/rq_calibration/`.
