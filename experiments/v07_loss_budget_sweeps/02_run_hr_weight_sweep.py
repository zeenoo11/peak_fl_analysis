"""v07-A2 — hr_weight sweep launcher (subprocess orchestrator over v06 drivers).

(한글 요약)
v07-A 후속 ablation. v07-A 가 `λ_aux ∈ {0.05, 0.1, 0.2}` 차원을 풀었다면,
v07-A2 는 *peak-aux 의 내부 가중치* `hr_weight` 차원을 푼다.

L_combined = MAE(ŷ, y) + λ_aux · (peak_amp_MSE + hr_weight · peak_hour_CE)

centralised optimum λ_aux=0.1 에 고정한 채 `hr_weight ∈ {0.05, 0.5, 1.0}` 을
sweep — *default hr_weight=0.1* 의 결과는 v07-A 의 (--aux_lambda 0.1) 결과를
그대로 재사용. 따라서 새 runs = 3 hr_weights × 6 cells × 3 seeds = **54**.

Cell name suffix:

    --aux_lambda 0.1 --hr_weight 0.1   → V6-Dyn-{...}-aux0.1            (default; v07-A 와 동일)
    --aux_lambda 0.1 --hr_weight 0.05  → V6-Dyn-{...}-aux0.1-hr0.05     (new v07-A2)
    --aux_lambda 0.1 --hr_weight 0.5   → V6-Dyn-{...}-aux0.1-hr0.5      (new v07-A2)
    --aux_lambda 0.1 --hr_weight 1.0   → V6-Dyn-{...}-aux0.1-hr1.0      (new v07-A2)

Resume-friendly: result.json 가 이미 있는 cell 은 자동 skip.

사용법
------

    uv run python experiments/v07_loss_budget_sweeps/02_run_hr_weight_sweep.py \\
        --seeds 42 123 7 --hr_weights 0.05 0.5 1.0
"""

from __future__ import annotations

import argparse
import importlib.util
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


def _import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_centralised_mod = _import_module_from_path(
    "v06_centralised_hr", V06_EXP_DIR / "01_centralised.py"
)
_fl_mod = _import_module_from_path(
    "v06_fl_dynamics_hr", V06_EXP_DIR / "02_fl_dynamics.py"
)


def _result_exists(seed: int, cell_name: str) -> bool:
    result_path = (
        OUTPUT_DIR / V07_NAMESPACE / f"seed{seed}" / cell_name / "result.json"
    )
    return result_path.exists()


def _run(cmd: list[str], dry_run: bool) -> int:
    print(f"[v07-A2] $ {' '.join(cmd)}")
    if dry_run:
        return 0
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description="v07-A2 hr_weight sweep at fixed λ_aux=0.1 (centralised optimum)."
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--hr_weights", type=float, nargs="+", default=[0.05, 0.5, 1.0],
                    help="hr_weight values to sweep. default 0.1 is excluded "
                         "(reuse v07-A `--aux_lambda 0.1` result).")
    ap.add_argument("--aux_lambda", type=float, default=0.1,
                    help="Fixed λ_aux (default 0.1 = centralised v07-A optimum).")
    ap.add_argument("--algorithms", type=str, nargs="+",
                    default=["fedavg", "fedprox", "fedrep", "ditto", "fedproto"],
                    choices=["fedavg", "fedprox", "fedrep", "ditto", "fedproto"])
    ap.add_argument("--skip_centralised", action="store_true")
    ap.add_argument("--skip_fl", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--local_epochs", type=int, default=40)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    py = args.python
    centralised_driver = str(V06_EXP_DIR / "01_centralised.py")
    fl_driver = str(V06_EXP_DIR / "02_fl_dynamics.py")

    planned: list[tuple[str, list[str]]] = []
    skipped: list[str] = []
    failed: list[str] = []

    for seed in args.seeds:
        for hr in args.hr_weights:
            if not args.skip_centralised:
                cell = _centralised_mod._build_cell_name(args.aux_lambda, hr)
                cmd = [
                    py, centralised_driver,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--aux_lambda", str(args.aux_lambda),
                    "--hr_weight", str(hr),
                    "--output_namespace", V07_NAMESPACE,
                ]
                planned.append((cell, cmd))
            if not args.skip_fl:
                for algo in args.algorithms:
                    cell = _fl_mod._build_cell_name(algo, args.aux_lambda, hr)
                    cmd = [
                        py, fl_driver,
                        "--algorithm", algo,
                        "--seed", str(seed),
                        "--local_epochs", str(args.local_epochs),
                        "--aux_lambda", str(args.aux_lambda),
                        "--hr_weight", str(hr),
                        "--output_namespace", V07_NAMESPACE,
                    ]
                    planned.append((cell, cmd))

    print(f"[v07-A2] {len(planned)} runs planned over "
          f"seeds={args.seeds} aux_lambda={args.aux_lambda} hr_weights={args.hr_weights} "
          f"algorithms={args.algorithms}")

    t0 = time.time()
    for i, (cell, cmd) in enumerate(planned, start=1):
        seed = int(cmd[cmd.index("--seed") + 1])
        if (not args.no_resume) and _result_exists(seed, cell):
            print(f"[v07-A2] [{i:>2d}/{len(planned)}] SKIP {cell} (seed={seed})")
            skipped.append(f"seed{seed}/{cell}")
            continue
        print(f"[v07-A2] [{i:>2d}/{len(planned)}] RUN  {cell} (seed={seed})")
        rc = _run(cmd, dry_run=args.dry_run)
        if rc != 0:
            failed.append(f"seed{seed}/{cell} (rc={rc})")
            print(f"[v07-A2] [{i:>2d}/{len(planned)}] FAIL {cell} -- rc={rc}")

    elapsed = time.time() - t0
    print()
    print(f"[v07-A2] launcher done.  elapsed={elapsed/60.0:.1f} min  "
          f"planned={len(planned)}  skipped={len(skipped)}  failed={len(failed)}")
    if failed:
        print("[v07-A2] FAILED:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
