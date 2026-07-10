#!/usr/bin/env python3
"""Reproduce the numerical figures of RQ_ab.tex.

Targets (aliases in parentheses):
  fig:Var_approx  (var-approx)  Effective-IDW overlay, H2(4)/M/1+M and +E2
  fig:MM1_GI      (mm1-gi)      Tripanel + RQ comparison: M/M/1+{M,E2,H2(4)}
  fig:MGI1_GI     (mgi1-gi)     Tripanel + RQ comparison: M/LN(1,4)/1+{H2(4),E2}
  fig:GIGI1_GI    (gigi1-gi)    Tripanel + RQ comparison:
                                {E2,H2(4)}/LN(1,2)/1+{E2,H2(4)}
  fig:QIS         (qis)         Tripanel + RQ comparison: two tandem systems
  tables                        w_{c,k}(t) matrix tables + b(c) tables, k=1,2,3
  all                           the five paper figure groups (22 PDFs)
  aux                           complete plot set for every model, all IDW
                                overlays, w-table overlays and b diagnostics

Existing outputs are reused (with a provenance check against the config);
use --force to regenerate everything a target needs.  --quick runs the whole
pipeline with drastically reduced simulation effort into results_quick/ as an
end-to-end smoke test (minutes, not hours); paper outputs are untouched.

Examples:
  python3 reproduce.py --list
  python3 reproduce.py mm1-gi
  python3 reproduce.py all
  python3 reproduce.py --quick all
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from rqab.util import CONFIGS_DIR, RESULTS_DIR, BUILD_DIR  # noqa: E402

PY = sys.executable
SCRIPTS = REPO / "scripts"

# ---------------------------------------------------------------------------
# Figure registry: exact TeX figure labels -> model configs / artifacts.
# ---------------------------------------------------------------------------

TRIPANEL_FIGURES: dict[str, dict] = {
    "fig:MM1_GI": {
        "alias": "mm1-gi",
        "models": ["mm1m", "mm1e2", "mm1h2_4"],
        "desc": "M/M/1+{M, E2, H2(4)} tripanel heatmaps",
    },
    "fig:MGI1_GI": {
        "alias": "mgi1-gi",
        "models": ["mln1_41h2_4", "mln1_41e2"],
        "desc": "M/LN(1,4)/1+{H2(4), E2} tripanel heatmaps",
    },
    "fig:GIGI1_GI": {
        "alias": "gigi1-gi",
        "models": ["e2ln1_21e2", "h2_4ln1_21h2_4", "h2_4ln1_21e2"],
        "desc": "{E2, H2(4)}/LN(1,2)/1+{E2, H2(4)} tripanel heatmaps",
    },
    "fig:QIS": {
        "alias": "qis",
        "models": ["tandem_h2_4e2_to_m1h2_4", "tandem_e2h2_4_to_m1e2"],
        "desc": "Tandem-system tripanel heatmaps",
    },
}

IDW_FIGURE = {
    "label": "fig:Var_approx",
    "alias": "var-approx",
    "configs": ["effective_idw_h2m1m", "effective_idw_h2m1e2"],
    "desc": "Effective-IDW simulation vs approximation overlays",
}

ALL_IDW_CONFIGS = [
    "effective_idw_h2m1m",
    "effective_idw_h2m1e2",
    "effective_idw_e2m1h2_4",
    "effective_idw_mm1e2",
    "effective_idw_mm1m",
    "effective_idw_h2ln2e2",
]

ALL_MODELS = [
    "mm1m",
    "mm1e2",
    "mm1h2_4",
    "mln1_41e2",
    "mln1_41h2_4",
    "mln1_21e2",
    "mln1_21h2_4",
    "mh2_41e2",
    "mh2_41h2_4",
    "e2ln1_21e2",
    "e2ln1_21h2_4",
    "h2_4ln1_21e2",
    "h2_4ln1_21h2_4",
    "tandem_h2_4e2_to_m1h2_4",
    "tandem_e2h2_4_to_m1e2",
]

TABLE_KS = [1, 2, 3]

QUICK_SIM_OVERRIDES = {
    "warmup_time": 2000.0,
    "sample_time": 20000.0,
    "replications": 20,
}
QUICK_IDW_OVERRIDES = {
    "warmup_time": 5000.0,
    "sample_time": 100000.0,
    "max_level": 14,
}


# ---------------------------------------------------------------------------
# Step engine.
# ---------------------------------------------------------------------------


@dataclass
class Step:
    key: str
    cmd: list[str]
    outputs: list[Path]
    heavy: bool = False  # heavy steps (MC sims) run serially: each saturates cores
    deps: tuple[str, ...] = ()
    always: bool = False  # run whenever the step is part of the plan
    note: str = ""


@dataclass
class Plan:
    steps: dict[str, Step] = field(default_factory=dict)

    def add(self, step: Step) -> str:
        existing = self.steps.get(step.key)
        if existing is None:
            self.steps[step.key] = step
        return step.key


def outputs_exist(step: Step) -> bool:
    if step.always or not step.outputs:
        return False
    return all(p.exists() for p in step.outputs)


def run_plan(plan: Plan, force: bool, jobs: int, dry_run: bool) -> int:
    done: set[str] = set()
    failed: set[str] = set()
    skipped: list[str] = []
    executed: list[str] = []
    pending = dict(plan.steps)

    def runnable(step: Step) -> bool:
        return all(d in done for d in step.deps if d in plan.steps)

    def blocked(step: Step) -> bool:
        return any(d in failed for d in step.deps if d in plan.steps)

    def execute(step: Step) -> tuple[str, int]:
        print(f"[run ] {step.key}: {' '.join(str(c) for c in step.cmd)}", flush=True)
        proc = subprocess.run(step.cmd, cwd=REPO)
        return step.key, proc.returncode

    while pending:
        # Cascade failures: steps depending on a failed step cannot run.
        newly_blocked = [s for s in pending.values() if blocked(s)]
        for s in newly_blocked:
            failed.add(s.key)
            del pending[s.key]
            print(f"[fail] {s.key} (blocked by failed prerequisite)", file=sys.stderr)
        if newly_blocked:
            continue

        ready = [s for s in pending.values() if runnable(s)]
        if not ready:
            print("error: dependency cycle in plan", file=sys.stderr)
            return 1

        batch_skip = [s for s in ready if not force and outputs_exist(s)]
        for s in batch_skip:
            done.add(s.key)
            skipped.append(s.key)
            del pending[s.key]
            print(f"[skip] {s.key} (outputs exist)")
        if batch_skip:
            continue

        if dry_run:
            for s in ready:
                print(f"[plan] {s.key}{' (heavy)' if s.heavy else ''}: "
                      f"{' '.join(str(c) for c in s.cmd)}")
                done.add(s.key)
                del pending[s.key]
            continue

        heavy = [s for s in ready if s.heavy]
        light = [s for s in ready if not s.heavy]

        if light:
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                for key, rc in pool.map(execute, light):
                    if rc != 0:
                        failed.add(key)
                        print(f"[fail] {key} (exit {rc})", file=sys.stderr)
                    else:
                        done.add(key)
                        executed.append(key)
                    del pending[key]
        elif heavy:
            s = heavy[0]
            key, rc = execute(s)
            if rc != 0:
                failed.add(key)
                print(f"[fail] {key} (exit {rc})", file=sys.stderr)
            else:
                done.add(key)
                executed.append(key)
            del pending[key]

    print(f"\nplan complete: {len(executed)} step(s) executed, {len(skipped)} reused, "
          f"{len(failed)} failed.")
    if failed:
        print(f"failed steps: {', '.join(sorted(failed))}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------


def check_workload_provenance(csv_path: Path, config_path: Path) -> None:
    """Warn when an existing aggregate CSV disagrees with its config or grid."""
    if not csv_path.exists() or not config_path.exists():
        return
    try:
        import csv as _csv

        with config_path.open() as fh:
            sim = json.load(fh)["simulation"]
        with csv_path.open() as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            return
        row = rows[0]
        mismatches = []
        if int(row["n_reps"]) != int(sim["replications"]):
            mismatches.append(f"n_reps {row['n_reps']} != config replications {sim['replications']}")
        if int(row["seed"]) != int(sim["seed"]):
            mismatches.append(f"seed {row['seed']} != config seed {sim['seed']}")
        expected_rows = _grid_tuple_count()
        if expected_rows and len(rows) != expected_rows:
            mismatches.append(f"row count {len(rows)} != grid tuples {expected_rows} (truncated?)")
        if mismatches:
            print(
                f"[provenance] {csv_path.name}: {'; '.join(mismatches)} "
                "(reusing anyway; --force to regenerate)",
                file=sys.stderr,
            )
    except Exception:
        pass


_GRID_COUNT_CACHE: list[int] = []


def _grid_tuple_count() -> int:
    if not _GRID_COUNT_CACHE:
        try:
            with (CONFIGS_DIR / "workload_lambda_alpha_grid_391.json").open() as fh:
                _GRID_COUNT_CACHE.append(len(json.load(fh)["tuples"]))
        except Exception:
            _GRID_COUNT_CACHE.append(0)
    return _GRID_COUNT_CACHE[0]


# ---------------------------------------------------------------------------
# Quick mode: derived configs with reduced simulation effort.
# ---------------------------------------------------------------------------


def make_quick_config(src: Path, dst: Path, overrides: dict) -> None:
    with src.open() as fh:
        cfg = json.load(fh)
    for key, value in overrides.items():
        if key in cfg.get("simulation", {}):
            cfg["simulation"][key] = value
        else:
            cfg["simulation"][key] = value
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")


class Context:
    """Paths for normal vs --quick runs."""

    def __init__(self, quick: bool, threads: int | None, force: bool = False):
        self.quick = quick
        self.threads = threads
        self.force = force
        if quick:
            self.results = REPO / "results_quick"
            self.configs = self.results / "configs"
            self.results.mkdir(exist_ok=True)
            self.configs.mkdir(exist_ok=True)
        else:
            self.results = RESULTS_DIR
            self.configs = CONFIGS_DIR

    def model_config(self, model: str) -> Path:
        src = CONFIGS_DIR / f"workload_{model}.json"
        if not self.quick:
            return src
        dst = self.configs / f"workload_{model}.json"
        if not dst.exists():
            make_quick_config(src, dst, QUICK_SIM_OVERRIDES)
        return dst

    def idw_config(self, name: str) -> Path:
        src = CONFIGS_DIR / f"{name}.json"
        if not self.quick:
            return src
        dst = self.configs / f"{name}.json"
        if not dst.exists():
            # Beyond the reduced sim settings, embedded config-dir-relative
            # paths must be rewritten: they would otherwise resolve under
            # results_quick/configs/.. instead of the intended locations.
            with src.open() as fh:
                cfg = json.load(fh)
            cfg["simulation"].update(QUICK_IDW_OVERRIDES)
            for model in cfg.get("models", []):
                if "w_table" in model:
                    rel = (src.parent / model["w_table"]).resolve()
                    # w/b tables are deterministic inputs: use the production ones.
                    model["w_table"] = str(rel)
            if "output" in cfg and "path" in cfg["output"]:
                out_name = Path(cfg["output"]["path"]).name
                cfg["output"]["path"] = str(self.results / out_name)
            if "simulation_overlay" in cfg and "results_dir" in cfg["simulation_overlay"]:
                cfg["simulation_overlay"]["results_dir"] = str(self.results)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("w") as fh:
                json.dump(cfg, fh, indent=2)
                fh.write("\n")
        return dst


# ---------------------------------------------------------------------------
# Step builders.
# ---------------------------------------------------------------------------


def add_build_step(plan: Plan) -> str:
    cmake = REPO / ".venv" / "bin" / "cmake"
    cmake_cmd = str(cmake) if cmake.exists() else "cmake"
    script = (
        f'"{cmake_cmd}" -S "{REPO}" -B "{REPO}/build" -DCMAKE_BUILD_TYPE=Release '
        f'&& "{cmake_cmd}" --build "{REPO}/build" -j'
    )
    return plan.add(
        Step(
            key="build",
            cmd=["/bin/sh", "-c", script],
            outputs=[
                BUILD_DIR / "wck",
                BUILD_DIR / "wck_sweep",
                BUILD_DIR / "wck_calibrate_b",
                BUILD_DIR / "workload_mc",
                BUILD_DIR / "idw_sim",
            ],
        )
    )


def add_table_steps(plan: Plan, ctx: Context, k: int) -> tuple[Path, Path, tuple[str, ...]]:
    """Ensure w/b tables for k exist; return (w_path, b_path, dep step keys).

    Production runs generate into results/.  --quick runs read existing
    production tables (deterministic inputs) but never write to results/:
    missing tables are generated into results_quick/ instead.
    """
    build = add_build_step(plan)
    prod_w = RESULTS_DIR / f"w_table_matrix_k{k}.csv"
    prod_b = RESULTS_DIR / f"b_table_k{k}.csv"
    if ctx.quick:
        if prod_w.exists() and prod_b.exists() and not ctx.force:
            return prod_w, prod_b, ()
        w_csv = ctx.results / f"w_table_matrix_k{k}.csv"
        b_csv = ctx.results / f"b_table_k{k}.csv"
    else:
        w_csv, b_csv = prod_w, prod_b

    w_key = plan.add(
        Step(
            key=f"w-table:k{k}",
            cmd=[str(BUILD_DIR / "wck_sweep"), "--k", str(k), "--out", str(w_csv)],
            outputs=[w_csv],
            heavy=True,
            deps=(build,),
        )
    )
    b_key = plan.add(
        Step(
            key=f"b-table:k{k}",
            cmd=[
                str(BUILD_DIR / "wck_calibrate_b"),
                "--k",
                str(k),
                "--w-table",
                str(w_csv),
                "--out",
                str(b_csv),
            ],
            outputs=[b_csv],
            heavy=True,
            deps=(w_key,),
        )
    )
    return w_csv, b_csv, (b_key,)


def model_k(model: str) -> int:
    """Patience index k for a workload model config (Erlang-k patience -> k)."""
    with (CONFIGS_DIR / f"workload_{model}.json").open() as fh:
        cfg = json.load(fh)["model"]
    patience = cfg["queue2"]["patience"] if "queue2" in cfg else cfg["patience"]
    dist = patience["distribution"]
    if dist["family"] in ("erlang_k", "erlang"):
        return int(dist["params"]["k"])
    return 1


def add_workload_step(plan: Plan, ctx: Context, model: str) -> str:
    build = add_build_step(plan)
    config = ctx.model_config(model)
    out_csv = ctx.results / f"workload_grid_aggregate_{model}.csv"
    check_workload_provenance(out_csv, config)
    cmd = [
        PY,
        str(SCRIPTS / "run_grid.py"),
        "--method",
        "workload",
        "--model-config",
        str(config),
        "--out-csv",
        str(out_csv),
    ]
    if ctx.threads is not None:
        cmd += ["--threads", str(ctx.threads)]
    if ctx.force:
        cmd.append("--force-rerun")
    return plan.add(
        Step(key=f"workload:{model}", cmd=cmd, outputs=[out_csv], heavy=True, deps=(build,))
    )


def add_analytic_step(plan: Plan, ctx: Context, model: str, method: str) -> str:
    config = ctx.model_config(model)
    k = model_k(model)
    out_name = {"refined": f"refined_rq_grid_{model}.csv", "first": f"first_rq_grid_{model}.csv"}[method]
    out_csv = ctx.results / out_name
    deps: tuple[str, ...] = ()
    cmd = [
        PY,
        str(SCRIPTS / "run_grid.py"),
        "--method",
        method,
        "--model-config",
        str(config),
        "--out-csv",
        str(out_csv),
    ]
    if method == "refined":
        w_csv, b_csv, deps = add_table_steps(plan, ctx, k)
        cmd += ["--w-table", str(w_csv), "--b-table", str(b_csv)]
    if ctx.force:
        cmd.append("--force-rerun")
    return plan.add(
        Step(key=f"{method}:{model}", cmd=cmd, outputs=[out_csv], deps=deps)
    )


def add_tripanel_step(plan: Plan, ctx: Context, model: str) -> str:
    workload = add_workload_step(plan, ctx, model)
    refined = add_analytic_step(plan, ctx, model, "refined")
    out_pdf = ctx.results / f"approx_ratio_tripanel_{model}.pdf"
    cmd = [
        PY,
        str(SCRIPTS / "plot_ratio_panels.py"),
        "--panels",
        "tripanel",
        "--model-config",
        str(ctx.model_config(model)),
        "--workload-csv",
        str(ctx.results / f"workload_grid_aggregate_{model}.csv"),
        "--refined-csv",
        str(ctx.results / f"refined_rq_grid_{model}.csv"),
        "--save",
        str(out_pdf),
        "--no-show",
    ]
    return plan.add(
        Step(key=f"tripanel:{model}", cmd=cmd, outputs=[out_pdf], deps=(workload, refined))
    )


def add_twopanel_step(plan: Plan, ctx: Context, model: str) -> str:
    workload = add_workload_step(plan, ctx, model)
    refined = add_analytic_step(plan, ctx, model, "refined")
    first = add_analytic_step(plan, ctx, model, "first")
    out_pdf = ctx.results / f"approx_ratio_twopanel_{model}.pdf"
    cmd = [
        PY,
        str(SCRIPTS / "plot_ratio_panels.py"),
        "--panels",
        "twopanel",
        "--model-config",
        str(ctx.model_config(model)),
        "--workload-csv",
        str(ctx.results / f"workload_grid_aggregate_{model}.csv"),
        "--refined-csv",
        str(ctx.results / f"refined_rq_grid_{model}.csv"),
        "--first-rq-csv",
        str(ctx.results / f"first_rq_grid_{model}.csv"),
        "--save",
        str(out_pdf),
        "--no-show",
    ]
    return plan.add(
        Step(
            key=f"twopanel:{model}",
            cmd=cmd,
            outputs=[out_pdf],
            deps=(workload, refined, first),
        )
    )


def add_heatmap_steps(plan: Plan, ctx: Context, model: str) -> tuple[str, str]:
    """Add benchmark and first-vs-refined RQ heatmaps for one model."""
    return (
        add_tripanel_step(plan, ctx, model),
        add_twopanel_step(plan, ctx, model),
    )


def add_ratio_step(plan: Plan, ctx: Context, model: str) -> str:
    workload = add_workload_step(plan, ctx, model)
    refined = add_analytic_step(plan, ctx, model, "refined")
    out_pdf = ctx.results / f"refined_rq_ratio_{model}.pdf"
    cmd = [
        PY,
        str(SCRIPTS / "plot_ratio_panels.py"),
        "--panels",
        "ratio",
        "--model-config",
        str(ctx.model_config(model)),
        "--workload-csv",
        str(ctx.results / f"workload_grid_aggregate_{model}.csv"),
        "--refined-csv",
        str(ctx.results / f"refined_rq_grid_{model}.csv"),
        "--save",
        str(out_pdf),
        "--no-show",
    ]
    return plan.add(
        Step(key=f"ratio:{model}", cmd=cmd, outputs=[out_pdf], deps=(workload, refined))
    )


def idw_curves_exist(results: Path, cfg: dict, alpha_indices: list[int]) -> bool:
    """Exact per-model curve-CSV existence check (same naming as idw_sim)."""
    from rqab.plotting.idw_curves import _simulation_curve_path

    for m, model in enumerate(cfg["models"]):
        for i in alpha_indices:
            if not _simulation_curve_path(results, m, str(model["name"]), i).exists():
                return False
    return True


def add_idw_steps(plan: Plan, ctx: Context, config_name: str) -> str:
    build = add_build_step(plan)
    config = ctx.idw_config(config_name)
    with config.open() as fh:
        cfg = json.load(fh)
    alpha_indices = [int(i) for i in cfg["alpha"]["indices"]]
    alias = config_name.replace("effective_idw_", "")
    out_pdf = ctx.results / f"idw_effective_{alias}.pdf"

    sim_outputs_exist = idw_curves_exist(ctx.results, cfg, alpha_indices)
    sim_key = f"idw-sim:{config_name}"
    if ctx.force or not sim_outputs_exist:
        plan.add(
            Step(
                key=sim_key,
                cmd=[
                    str(BUILD_DIR / "idw_sim"),
                    "--config",
                    str(config),
                    "--out-dir",
                    str(ctx.results),
                ],
                outputs=[],
                heavy=True,
                deps=(build,),
                always=True,
            )
        )
        sim_deps: tuple[str, ...] = (sim_key,)
    else:
        sim_deps = ()

    # The overlay needs the w tables referenced by the config (k per model).
    table_deps: list[str] = []
    for model in cfg["models"]:
        k = int(model.get("scaling", {}).get("k", 1))
        _, _, deps = add_table_steps(plan, ctx, k)
        table_deps.extend(deps)

    plot_cmd = [
        PY,
        str(SCRIPTS / "plot_idw_effective.py"),
        "--config",
        str(config),
        "--curves-dir",
        str(ctx.results),
        "--save",
        str(out_pdf),
        "--no-show",
    ]
    return plan.add(
        Step(
            key=f"idw-plot:{config_name}",
            cmd=plot_cmd,
            outputs=[out_pdf],
            deps=tuple([*sim_deps, *table_deps]),
            always=True,
        )
    )


def add_diagnostics_steps(plan: Plan, ctx: Context) -> list[str]:
    keys = []
    for k in TABLE_KS:
        w_csv, b_csv, deps = add_table_steps(plan, ctx, k)
        overlay_pdf = ctx.results / f"w_overlay_k{k}.png"
        keys.append(
            plan.add(
                Step(
                    key=f"w-overlay:k{k}",
                    cmd=[
                        PY,
                        str(SCRIPTS / "plot_w_tables.py"),
                        "overlay",
                        "--k",
                        str(k),
                        "--table",
                        str(w_csv),
                        "--save",
                        str(overlay_pdf),
                        "--no-show",
                    ],
                    outputs=[overlay_pdf],
                    deps=deps,
                )
            )
        )
        bcal_png = ctx.results / f"b_calibration_k{k}.png"
        keys.append(
            plan.add(
                Step(
                    key=f"b-calibration:k{k}",
                    cmd=[
                        PY,
                        str(SCRIPTS / "plot_w_tables.py"),
                        "b",
                        "--k",
                        str(k),
                        "--table",
                        str(b_csv),
                        "--save",
                        str(bcal_png),
                        "--no-show",
                    ],
                    outputs=[bcal_png],
                    deps=deps,
                )
            )
        )
    return keys


# ---------------------------------------------------------------------------
# Target expansion.
# ---------------------------------------------------------------------------


def expand_target(plan: Plan, ctx: Context, target: str) -> bool:
    t = target.lower().lstrip()
    if t.startswith("fig:"):
        t = t[4:].lower()

    for label, spec in TRIPANEL_FIGURES.items():
        if t in (label.lower(), label.split(":")[1].lower(), spec["alias"]):
            for model in spec["models"]:
                add_heatmap_steps(plan, ctx, model)
            return True

    if t in (IDW_FIGURE["label"].lower(), "var_approx", IDW_FIGURE["alias"]):
        for name in IDW_FIGURE["configs"]:
            add_idw_steps(plan, ctx, name)
        return True

    if t == "tables":
        for k in TABLE_KS:
            add_table_steps(plan, ctx, k)
        return True

    if t == "all":
        for name in IDW_FIGURE["configs"]:
            add_idw_steps(plan, ctx, name)
        for spec in TRIPANEL_FIGURES.values():
            for model in spec["models"]:
                add_heatmap_steps(plan, ctx, model)
        return True

    if t == "aux":
        for model in ALL_MODELS:
            add_heatmap_steps(plan, ctx, model)
            add_ratio_step(plan, ctx, model)
        for name in ALL_IDW_CONFIGS:
            add_idw_steps(plan, ctx, name)
        add_diagnostics_steps(plan, ctx)
        return True

    return False


def print_target_list() -> None:
    print(__doc__.split("Examples:")[0])
    print("Approximate full-run costs on a ~16-core machine (empty results/):")
    print("  tables         ~minutes per k (PDE sweep + calibration)")
    print("  fig:MM1_GI     ~20 min MC for mm1m + ~2 min/model analytics + plots")
    print("  other tripanel ~5-10 min MC per model + analytics + plots")
    print("  fig:Var_approx hours per config if the idw sims must rerun; seconds if")
    print("                 the committed model0_* curve CSVs are reused")
    print("  aux            everything above for all 16 models")
    print("With committed results/ present, plots re-render in seconds ([skip]/reuse).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the figures of RQ_ab.tex.",
        usage="reproduce.py [options] target [target ...]",
    )
    parser.add_argument("targets", nargs="*", help="fig:<label>, alias, tables, all, aux")
    parser.add_argument("--list", action="store_true", help="List targets and costs.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if outputs exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, run nothing.")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel light steps (default 4).")
    parser.add_argument("--threads", type=int, default=None, help="Thread cap for MC binaries.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Reduced-effort smoke run into results_quick/ "
            "(reads production w/b tables, never writes to results/)."
        ),
    )
    args = parser.parse_args()

    if args.list or not args.targets:
        print_target_list()
        return 0

    ctx = Context(quick=args.quick, threads=args.threads, force=args.force)
    plan = Plan()
    for target in args.targets:
        if not expand_target(plan, ctx, target):
            print(f"error: unknown target '{target}' (see --list)", file=sys.stderr)
            return 2

    if args.quick:
        # Quick runs read existing production w/b tables (deterministic inputs)
        # but write everything, including missing tables, to results_quick/.
        print(f"[quick] outputs -> {ctx.results}")

    t0 = time.time()
    rc = run_plan(plan, force=args.force, jobs=args.jobs, dry_run=args.dry_run)
    print(f"total wall time: {time.time() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
