"""v07-A — λ_aux sweep launcher (subprocess orchestrator over v06 drivers).

(한글 요약)
plan ``v07-01_loss_and_budget_sweeps.md`` §1 (v07-A) — `λ_aux ∈ {0.05, 0.1, 0.2}`
× 6 cells × 3 seeds = **54 new runs**. v06 결과(`λ=0`, `λ=0.3`) 는 재사용,
launcher 가 이미 작성된 v06 driver 두 개 (`01_centralised.py`, `02_fl_dynamics.py`)
를 `--output_namespace v07_loss_budget_sweeps` 로 호출하여 결과를 v07 namespace
로 리다이렉트.

각 호출은 ``--seed S`` per invocation (memory: feedback_argparse_per_seed).
Launcher 자체는 outer loop만 담당하고, 내부 학습 코드는 v06 drivers 가 그대로
처리한다 — v07 가 별도 train code 를 가지지 않으므로 v06 method 가 frozen 임을
구조적으로 보장.

Resume-friendly: ``result.json`` 가 이미 있는 cell 은 자동 skip.

사용법
------

    uv run python experiments/v07_loss_budget_sweeps/01_run_aux_sweep.py \\
        --seeds 42 123 7 --lambdas 0.05 0.1 0.2 \\
        --algorithms fedavg fedprox fedrep ditto fedproto

부분 실행 (예: seed 42 + FedAvg + λ=0.1 만):

    uv run python experiments/v07_loss_budget_sweeps/01_run_aux_sweep.py \\
        --seeds 42 --lambdas 0.1 --algorithms fedavg --skip_centralised
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import OUTPUT_DIR  # noqa: E402

V06_EXP_DIR = REPO_ROOT / "experiments" / "v06_round_dynamics"
V07_NAMESPACE = "v07_loss_budget_sweeps"

# Cell-name suffix logic mirrors v06.{01,02}_*.py:_aux_suffix exactly.
# We import the actual functions so any future suffix change in v06 propagates
# automatically.
sys.path.insert(0, str(V06_EXP_DIR))
import importlib.util


def _import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_centralised_mod = _import_module_from_path(
    "v06_centralised", V06_EXP_DIR / "01_centralised.py"
)
_fl_mod = _import_module_from_path(
    "v06_fl_dynamics", V06_EXP_DIR / "02_fl_dynamics.py"
)


def _result_exists(seed: int, cell_name: str) -> bool:
    """Resume guard — skip if a v07 result.json is already on disk."""
    result_path = (
        OUTPUT_DIR / V07_NAMESPACE / f"seed{seed}" / cell_name / "result.json"
    )
    return result_path.exists()


def _run(cmd: list[str], dry_run: bool) -> int:
    """Run a subprocess; returns its exit code (0 on dry-run)."""
    print(f"[v07-A] $ {' '.join(cmd)}")
    if dry_run:
        return 0
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "v07-A λ_aux sweep launcher. Calls the v06 drivers with "
            "--output_namespace v07_loss_budget_sweeps so v06 results stay frozen."
        )
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.05, 0.1, 0.2],
                    help="λ_aux values to sweep over (default: 0.05, 0.1, 0.2 — "
                         "v07-A new cells; λ=0 / 0.3 are already in v06).")
    ap.add_argument("--algorithms", type=str, nargs="+",
                    default=["fedavg", "fedprox", "fedrep", "ditto", "fedproto"],
                    choices=["fedavg", "fedprox", "fedrep", "ditto", "fedproto"])
    ap.add_argument("--skip_centralised", action="store_true",
                    help="Skip the centralised pooled-SGD upper-bound cell.")
    ap.add_argument("--skip_fl", action="store_true",
                    help="Skip all FL cells (run centralised only).")
    ap.add_argument("--epochs", type=int, default=40,
                    help="Centralised pooled epochs (passed as --epochs).")
    ap.add_argument("--local_epochs", type=int, default=40,
                    help="FL local epochs (passed as --local_epochs; "
                         "matches v06 0502 protocol).")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python interpreter to call (default: sys.executable).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the planned commands and exit without executing.")
    ap.add_argument("--no_resume", action="store_true",
                    help="Always re-run, even if result.json already exists.")
    args = ap.parse_args()

    py = args.python
    centralised_driver = str(V06_EXP_DIR / "01_centralised.py")
    fl_driver = str(V06_EXP_DIR / "02_fl_dynamics.py")

    planned: list[tuple[str, list[str]]] = []
    skipped: list[str] = []
    failed: list[str] = []

    for seed in args.seeds:
        for lam in args.lambdas:
            # 1) centralised cell.
            if not args.skip_centralised:
                cell = _centralised_mod._build_cell_name(lam)
                cmd = [
                    py, centralised_driver,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--aux_lambda", str(lam),
                    "--output_namespace", V07_NAMESPACE,
                ]
                planned.append((cell, cmd))

            # 2) FL cells.
            if not args.skip_fl:
                for algo in args.algorithms:
                    cell = _fl_mod._build_cell_name(algo, lam)
                    cmd = [
                        py, fl_driver,
                        "--algorithm", algo,
                        "--seed", str(seed),
                        "--local_epochs", str(args.local_epochs),
                        "--aux_lambda", str(lam),
                        "--output_namespace", V07_NAMESPACE,
                    ]
                    planned.append((cell, cmd))

    print(f"[v07-A] {len(planned)} runs planned over "
          f"seeds={args.seeds} lambdas={args.lambdas} algorithms={args.algorithms}")

    t0 = time.time()
    for i, (cell, cmd) in enumerate(planned, start=1):
        seed_str = cmd[cmd.index("--seed") + 1]
        seed = int(seed_str)
        if (not args.no_resume) and _result_exists(seed, cell):
            print(f"[v07-A] [{i:>2d}/{len(planned)}] SKIP {cell} (seed={seed}) "
                  f"-- result.json already exists.")
            skipped.append(f"seed{seed}/{cell}")
            continue
        print(f"[v07-A] [{i:>2d}/{len(planned)}] RUN  {cell} (seed={seed})")
        rc = _run(cmd, dry_run=args.dry_run)
        if rc != 0:
            failed.append(f"seed{seed}/{cell} (rc={rc})")
            print(f"[v07-A] [{i:>2d}/{len(planned)}] FAIL {cell} -- rc={rc}")

    elapsed = time.time() - t0
    print()
    print(f"[v07-A] launcher done.  elapsed={elapsed/60.0:.1f} min  "
          f"planned={len(planned)}  skipped={len(skipped)}  failed={len(failed)}")
    if failed:
        print("[v07-A] FAILED:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
