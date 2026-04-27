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

------------------------------------------------------------------------------
한글 요약 (v02 stratified 80:20 split 생성 스크립트)

- 한 번 실행 = 한 seed (--seed S) → v02_8020_seed{S}.yaml 한 개 생성.
  멀티시드 루프는 스크립트 안에 두지 않고 외부 launcher가 {42, 123, 7} 각각을
  따로 호출한다 (프로젝트 컨벤션: feedback_argparse_per_seed).

- stratification 절차 (src/dataloader/splits.py:make_v02_split):
    1) 100개 valid 아파트 → 4-d profile feature
       (mean / std / daily_peak_mean / weekday_ratio) 추출.
    2) StandardScaler 후 KMeans(k=2, random_state=seed)로 cluster 분리.
       random_state가 곧 --seed이므로 시드를 바꾸면 cluster 경계 자체가
       달라지고, 결과적으로 시드별로 서로 다른 cold-20 집합이 만들어진다.
    3) 각 cluster에서 80:20 비율로 train/cold를 나눠 합친다.
    4) KL(cold || train)을 'mean' feature 위 10-bin 히스토그램으로 계산.
       기본 임계값 0.5를 넘으면 seed+1로 한 번 retry.

- 한 seed YAML 저장 후 _refresh_summary로 splits_dir 안에 존재하는 모든
  seed YAML을 스캔해 split_summary.json을 idempotent하게 다시 만든다
  (특정 seed만 새로 돌려도 전체 요약이 항상 최신 상태로 유지됨).

- "3-fold cold rotation" (plan Open question 1)은 채택하지 않는다.
  대신 시드별 KMeans random_state 차이로 사실상 3개의 다른 cold-20을 얻는다.
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
    """seed별 split YAML 저장.

    payload 구조:
        train:    [Apt..., ...]   # 80개
        cold:     [Apt..., ...]   # 20개
        metadata: {seed, kl_divergence, retry_seed, cluster_sizes, ...}
    sort_keys=False는 train / cold / metadata 순서를 사람이 읽기 쉬운
    원래 순서로 보존하기 위함.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train": list(train), "cold": list(cold), "metadata": meta}
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(payload, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _read_yaml(path: Path) -> dict:
    """기존에 저장된 seed별 split YAML을 dict로 읽어온다.

    summary 재생성 시 디스크의 모든 seed YAML을 다시 모으는 데 쓰임.
    """
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _jaccard(a: set, b: set) -> float:
    """두 cold-set 사이의 Jaccard overlap 계산.

    시드별로 cold-20이 얼마나 다른지를 한 번에 보여주는 진단 지표.
    값이 0.0에 가까울수록 시드 간 cold gucha가 거의 겹치지 않는다는 뜻 →
    multi-seed 평균이 의미 있는 분산을 갖게 된다. 둘 다 빈 집합이면
    convention상 1.0 (해석상 의미는 없고 division-by-zero 방어용).
    """
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _refresh_summary(splits_dir: Path) -> dict:
    """Scan splits_dir for v02_8020_seed*.yaml and rebuild split_summary.json.

    한글 설명:
        디스크에 있는 모든 v02_8020_seed*.yaml을 글롭으로 모아서
        split_summary.json을 처음부터 다시 만든다 (idempotent).
        → 어느 seed 하나만 재생성해도 summary는 항상 "현재 디스크에 존재하는
          전체 seed 집합"을 정확히 반영한다.

        포함하는 정보:
          - seeds_present:        존재하는 seed 목록 (정렬된 int)
          - per_seed:             각 seed의 KL, retry, cluster_sizes 등
          - cold_overlap_pairwise: 시드 쌍별 cold-set Jaccard
                                   (cold-20이 얼마나 분리됐는지 확인용)
          - all_kl_within_threshold: 모든 seed가 KL ≤ threshold인지 한 줄 요약
    """
    # splits_dir의 모든 seed YAML을 모은다 (정렬은 안정성 위해 알파벳순).
    # 가정: 동일 splits_dir에 동시에 다른 launcher가 쓰지 않음 (race 없음).
    yamls = sorted(splits_dir.glob("v02_8020_seed*.yaml"))
    per_seed = {}
    cold_sets: dict[int, set[str]] = {}
    for path in yamls:
        raw = _read_yaml(path)
        meta = raw.get("metadata", {})
        seed = int(meta.get("seed"))
        # cold_sets는 뒤에서 pairwise Jaccard 계산할 때 사용.
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
    # 시드 쌍 (a < b)별로 cold-set Jaccard를 계산한다. 3 seed면 3쌍 (42-123, 42-7, 7-123 ...).
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
        # 모든 seed가 KL gate를 통과했는지 (retry까지 포함한 최종 KL 기준).
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
    """단일 seed에 대한 v02 80:20 split 생성 + summary 재갱신.

    실행 흐름:
        1. --seed S 한 개에 대해 split YAML이 이미 존재하면 skip (단, --force면 재생성).
        2. 존재하지 않으면 make_v02_split을 호출 → KL gate retry까지 거친
           최종 train/cold list와 metadata를 받아 YAML로 저장.
        3. 마지막에는 항상 디스크 전체를 스캔해 split_summary.json을 갱신.
           (이 단계는 어떤 seed가 새로 추가됐든 무조건 동작 → idempotent.)
    """
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
    # 이미 같은 seed로 만들어둔 YAML이 있으면 기본적으로 건너뛴다 (실수 덮어쓰기 방지).
    # --force를 줘야만 같은 seed를 다시 생성한다.
    if out_path.exists() and not args.force:
        print(f"[v02 split] {out_path.name} already exists; use --force to regenerate.")
    else:
        print(f"[v02 split] generating split for seed={args.seed} (n_train={args.n_train}, n_cold={args.n_cold})...")
        # make_v02_split 내부에서 KL > kl_threshold면 seed+1로 한 번 retry함.
        # retry된 경우 meta['retry_seed']에 새 시드가 기록된다 (None이면 retry 없음).
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
        # retry가 발생했음을 콘솔에서 명시적으로 보여주는 것이 디버깅에 중요.
        retry_str = f" (retried with seed={retry})" if retry is not None else ""
        print(f"  KL = {kl:.4f}{retry_str}")
        print(f"  cluster_sizes = {meta['cluster_sizes']}")
        print(f"  train_per_cluster = ({meta['n_train_from_cluster_0']}, {meta['n_train_from_cluster_1']})")
        print(f"  cold preview: {cold[:5]}{' ...' if len(cold) > 5 else ''}")
        print(f"  saved -> {out_path}")

    # 새 YAML 생성 여부와 무관하게 summary는 매번 다시 만든다.
    # → 다른 seed가 따로 추가됐을 때도 자동으로 반영됨.
    summary = _refresh_summary(V02_SPLITS_DIR)
    print(f"\n[v02 split] split_summary.json refreshed (seeds present: {summary['seeds_present']})")
    # KL retry까지 거치고도 임계값을 못 넘은 seed가 있으면 경고.
    # (그 자체로 실패는 아니지만, 그 seed의 cold pool은 train과 분포가 꽤 떨어진 셈.)
    if not summary["all_kl_within_threshold"]:
        print("  [WARN] some seeds exceed KL threshold even after retry.")
    # 시드별 cold-set이 얼마나 다른지 한눈에 보여준다 (작을수록 시드 간 다양성 확보).
    # 첫 seed만 실행된 직후엔 pair가 비어 있어서 print를 건너뛴다 (단일 seed에서는 의미 없음).
    if summary["cold_overlap_pairwise"]:
        print("  pairwise cold-set Jaccard:")
        for pair, info in summary["cold_overlap_pairwise"].items():
            print(f"    {pair}: J={info['jaccard']:.3f} (|∩|={info['intersection_size']}/{info['union_size']})")


if __name__ == "__main__":
    main()
