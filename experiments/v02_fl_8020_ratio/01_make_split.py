"""Generate the v02 stratified 80:20 train/cold split for one seed.

Per-seed invocation (matches v01 conventions):
    uv run python experiments/v02_fl_8020_ratio/01_make_split.py --seed 42
    uv run python experiments/v02_fl_8020_ratio/01_make_split.py --seed 123
    uv run python experiments/v02_fl_8020_ratio/01_make_split.py --seed 7

Each invocation:
    1. Calls dataloader.splits.make_v02_split(seed) — 4-feature StandardScaler
       -> KMeans(k=2, random_state=seed) -> per-cluster proportional alternating
       extraction -> KL gate (retry once with seed+1 if KL > 0.5).
    2. Writes outputs/v02_fl_8020_ratio/splits/v02_8020_seed{seed}.yaml.
    3. Idempotently regenerates split_summary.json by scanning for all
       v02_8020_seed*.yaml files currently on disk; reports per-seed KL,
       cluster sizes, and pairwise cold-set Jaccard overlap.

The KMeans random_state is intentionally bound to --seed so that running with
{42, 123, 7} produces three distinct stratified cold-20 sets without any
explicit fold-rotation logic (open question 1 in plans/v02-01_fl_8020_ratio.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml

from config import RANDOM_SEED
from dataloader.splits import V02_SPLITS_DIR, make_v02_split, v02_yaml_path


def _save_yaml(path: Path, train: list[str], cold: list[str], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train": list(train), "cold": list(cold), "metadata": meta}
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(payload, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _refresh_summary(splits_dir: Path) -> dict:
    """Scan splits_dir for v02_8020_seed*.yaml and rebuild split_summary.json."""
    yamls = sorted(splits_dir.glob("v02_8020_seed*.yaml"))
    per_seed = {}
    cold_sets: dict[int, set[str]] = {}
    for path in yamls:
        raw = _read_yaml(path)
        meta = raw.get("metadata", {})
        seed = int(meta.get("seed"))
        cold_sets[seed] = set(raw["cold"])
        per_seed[seed] = {
            "yaml": path.name,
            "n_train": int(meta.get("n_train", len(raw["train"]))),
            "n_cold": int(meta.get("n_cold", len(raw["cold"]))),
            "kl_divergence": float(meta.get("kl_divergence")),
            "kl_threshold": float(meta.get("kl_threshold", 0.5)),
            "retry_seed": meta.get("retry_seed"),
            "cluster_sizes": meta.get("cluster_sizes"),
            "n_train_from_cluster_0": meta.get("n_train_from_cluster_0"),
            "n_train_from_cluster_1": meta.get("n_train_from_cluster_1"),
        }
    seeds_sorted = sorted(per_seed.keys())
    overlap = {
        f"{a}_vs_{b}": {
            "jaccard": _jaccard(cold_sets[a], cold_sets[b]),
            "intersection_size": len(cold_sets[a] & cold_sets[b]),
            "union_size": len(cold_sets[a] | cold_sets[b]),
        }
        for i, a in enumerate(seeds_sorted)
        for b in seeds_sorted[i + 1 :]
    }
    summary = {
        "split_version": "v02",
        "n_train": 80,
        "n_cold": 20,
        "seeds_present": seeds_sorted,
        "per_seed": {str(s): per_seed[s] for s in seeds_sorted},
        "cold_overlap_pairwise": overlap,
        "all_kl_within_threshold": all(
            ps["kl_divergence"] <= ps["kl_threshold"] for ps in per_seed.values()
        ),
    }
    out = splits_dir / "split_summary.json"
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate v02 stratified 80:20 split for one seed.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED, help="KMeans / shuffle seed (default %(default)s).")
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--n_cold", type=int, default=20)
    ap.add_argument("--year", type=str, default="2016")
    ap.add_argument("--min_hours", type=int, default=7000)
    ap.add_argument("--kl_threshold", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="Regenerate even if YAML already exists.")
    args = ap.parse_args()

    out_path = v02_yaml_path(args.seed)
    if out_path.exists() and not args.force:
        print(f"[v02 split] {out_path.name} already exists; use --force to regenerate.")
    else:
        print(f"[v02 split] generating split for seed={args.seed} (n_train={args.n_train}, n_cold={args.n_cold})...")
        train, cold, meta = make_v02_split(
            seed=args.seed,
            n_train=args.n_train,
            n_cold=args.n_cold,
            year=args.year,
            min_hours=args.min_hours,
            kl_threshold=args.kl_threshold,
        )
        _save_yaml(out_path, train, cold, meta)
        kl = meta["kl_divergence"]
        retry = meta["retry_seed"]
        retry_str = f" (retried with seed={retry})" if retry is not None else ""
        print(f"  KL = {kl:.4f}{retry_str}")
        print(f"  cluster_sizes = {meta['cluster_sizes']}")
        print(f"  train_per_cluster = ({meta['n_train_from_cluster_0']}, {meta['n_train_from_cluster_1']})")
        print(f"  cold preview: {cold[:5]}{' ...' if len(cold) > 5 else ''}")
        print(f"  saved -> {out_path}")

    summary = _refresh_summary(V02_SPLITS_DIR)
    print(f"\n[v02 split] split_summary.json refreshed (seeds present: {summary['seeds_present']})")
    if not summary["all_kl_within_threshold"]:
        print("  [WARN] some seeds exceed KL threshold even after retry.")
    if summary["cold_overlap_pairwise"]:
        print("  pairwise cold-set Jaccard:")
        for pair, info in summary["cold_overlap_pairwise"].items():
            print(f"    {pair}: J={info['jaccard']:.3f} (|∩|={info['intersection_size']}/{info['union_size']})")


if __name__ == "__main__":
    main()
