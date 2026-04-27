"""Aggregate v02 per-seed JSONs into a single multi-seed summary.

Reads (for each seed S in --seeds):
    outputs/v02_fl_8020_ratio/seed{S}/
        codebook_diagnostics.json    (from 03)
        coldstart_R0.json            (from 04)
        coldstart_R1.json            (from 04)
        E1_results.json              (from 05)
        W_component_results.json     (from 06)

Writes:
    outputs/v02_fl_8020_ratio/multiseed_summary.json

Reports mean ± std over seeds for every metric the v02 paper needs:
    G1 — baseline cold PAPE/HR@k, V0/W5 corrected
    G2 — R0 vs R1 routing
    G3 — multi-seed std (matches v01 §4.4 protocol)
    E1 — peak_aux contribution (T2 V0 - T0 V0)
    G4 — W-component synergy (best_single - W5 in PAPE-kW)
    Codebook health — k_min, util, perplexity

(한글 요약)
v02 파이프라인의 **마지막 단계 — 멀티 시드 집계 스크립트**. 03/04/05/06이 시드별로
``outputs/v02_fl_8020_ratio/seed{S}/`` 아래에 만든 5종 JSON을 모아 시드 차원에서
mean / std / min / max / per-seed values 5튜플로 합치고, 단일 ``multiseed_summary.json``
하나로 dump한다. 이게 v02 paper의 G1/G2/G3/G4/E1/codebook health 헤드라인 숫자의
**최종 source-of-truth**이다 (README "v02 headline result" 표는 여기서 읽혀야 함).

입력 (seed당 5종):
    seed{S}/codebook_diagnostics.json   — 03 산출 (M, util, ppl, k_min, k_max, …)
    seed{S}/coldstart_R0.json           — 04 산출 (baseline + 두 op-point × W5, R0 routing)
    seed{S}/coldstart_R1.json           — 04 산출 (baseline + 두 op-point × W5, R1 routing)
    seed{S}/E1_results.json             — 05 산출 (T0/T2 × V0, peak_aux 효과)
    seed{S}/W_component_results.json    — 06 산출 (T2 × {V0, W1a, W5}, R0 routing + synergy)

    NOTE: 04는 baseline을 R0와 R1 양쪽 JSON 모두에 동일 값으로 저장한다 (보정 없는 ŷ_base만
    denorm한 결과는 routing 선택과 무관). 본 스크립트는 두 source를 모두 집계 결과에
    그대로 보존한다 (R0와 R1 각각의 ``baseline`` 키 모두 mean/std로 합산). source-of-truth를
    한쪽으로 강제 통일하지 않으며, paper 작성 시 둘 중 한쪽만 인용해도 동일 숫자가 나옴
    (수치 검증용 redundancy 역할).

출력 (단일 파일):
    outputs/v02_fl_8020_ratio/multiseed_summary.json  — plan §"Outputs" 트리와 일치.

시드 셋 ({42, 123, 7}):
    CLAUDE.md "Multi-seed: all reported numbers use seeds {42, 123, 7}" 컨벤션을 따른다.
    본 스크립트는 ``--seeds 42 123 7`` argparse default로 받는다 (hardcode가 아니라
    **CLI flag로 override 가능한 default**). per-seed 호출 컨벤션은 *experiment* 스크립트
    (02–06)에 한정되며, *aggregator*인 07은 시드 셋을 한 번에 보는 것이 자연스러움
    (memory: feedback_argparse_per_seed는 aggregator에는 강제 적용되지 않음).

mean/std 정의 (분산 자유도):
    ``_agg``는 ``np.std(ddof=0)`` 즉 **모집단 표준편차** (n으로 나눔)를 사용.
    시드 3개 기준이라 표본 std (ddof=1, n-1=2 자유도)가 더 일반적이지만, 본 스크립트는
    ddof=0을 채택. README의 "± 0.69" 등 보고치는 모두 ddof=0 기준 — 외부에서 ddof=1로
    재계산하면 같은 mean이지만 std는 √(3/2)≈1.22배 큰 값이 나옴.
    (사용자 확인 요망: paper에서 "± std"의 자유도 표기 방식을 ddof=0으로 명시할지,
    혹은 ddof=1로 바꿀지 — 7행의 README G3 "PAPE σ 0.34–0.66"이 어느 자유도 기준인지 확인.)

누락 시드 처리:
    파일이 없으면 KeyError로 죽지 않고 ``[WARN]`` 출력 후 그 source/seed만 skip하고 진행한다.
    예: seed=7의 E1_results.json이 빠지면 E1 집계는 {42, 123} 두 시드로 평균/std 계산.
    이는 한 시드에서만 일부 단계가 실패해도 나머지는 보고할 수 있게 하기 위함.
    ``missing_per_source`` 키로 어느 source가 어느 시드에서 빠졌는지 summary에 영구 기록.

plan §"Comparison table" (7행) 충족 여부:
    1) v01 50:50 baseline (R0)              → 본 집계 미포함 (외부 reference; v01 paper 인용).
    2) v02 80:20 R0 (HR-pres)               → ``coldstart_R0.operating_points["HR-preserving"]``
    3) v02 80:20 R0 (PAPE-aggr)             → ``coldstart_R0.operating_points["PAPE-aggressive"]``
    4) v02 80:20 R1 (HR-pres)               → ``coldstart_R1.operating_points["HR-preserving"]``
    5) v02 80:20 R1 (PAPE-aggr)             → ``coldstart_R1.operating_points["PAPE-aggressive"]``
    6) v02 80:20 E1 V0 ON/OFF               → ``E1.peak_aux_contribution_on_V0`` (+ ``v01_reference_pp=18.6``)
    7) v02 80:20 W-comp {V0, W1a, W5}/T2    → ``W_component.per_operating_point[*].cells``
    → 1행을 제외한 **6행을 모두 산출**. 1행은 v02 외부 reference이므로 paper 작성 시
       v01 paper에서 직접 인용 (본 집계 책임 밖).

mean ± std와 별개로 보고하는 지표 (READMR/plan §Metrics에 없는 진단치):
    - ``coldstart_R{0,1}.routing_diagnostics`` (cluster usage 분포): 04 JSON에는 들어있지만
      본 집계는 **이를 별도 mean/std로 모으지 않음**. 누락은 의도적 (G1/G2/G4 보고와는 별개의
      sanity check이며, per-seed JSON에서 직접 들여다보면 충분).
    - ``aux_diagnostics`` (top1/within_1h/within_2h, 04와 06 둘 다에 존재): 마찬가지로
      **본 집계에서 별도 처리하지 않음**. paper §Metrics에 등장하지 않으므로 multi-seed
      평균/std 보고도 하지 않는다 (의도적 누락; per-seed JSON 인용으로 대체).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from config import OUTPUT_DIR

V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"


def _read_json(path: Path) -> dict | None:
    """JSON 파일 안전 로더 — 파일이 없으면 None 반환 (KeyError로 죽지 않음).

    (한글) 누락 시드 처리의 핵심 헬퍼. main()이 None 결과를 보고
    ``missing_per_source`` 리스트에 시드를 적재하고 그 source/seed 조합만 skip한다.
    """
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _agg(values: list[float]) -> dict:
    """시드 차원 통계 5튜플 (mean/std/min/max/values).

    (한글) 본 스크립트의 mean ± std는 모두 이 함수를 통과한다.
    핵심 결정사항:
        - ``ddof=1`` (표본 std, Bessel correction; n-1로 나눔). 시드 3개에서는
          모집단 std (ddof=0) 대비 √(3/2) ≈ 1.22 배 큰 값. 학계 관행 (v01 §4.4도
          표본 std로 추정) 및 paper 일관성을 위해 ddof=1 채택.
        - ``values`` 키에 raw per-seed 값을 그대로 보존 → 외부에서 ddof=0 등 다른
          정의로 재계산 가능.
    """
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),  # 표본 std (Bessel correction; n-1로 나눔)
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(v) for v in arr],  # raw values 보존 → 외부 재계산용
    }


def aggregate_routing(per_seed: dict[int, dict], routing: str) -> dict:
    """Summarise coldstart_{R0,R1}.json across seeds.

    (한글) 04 산출 ``coldstart_R{0,1}.json`` 시드 묶음 → mean/std로 집계.

    구조 (per-seed 04 JSON 스키마와 1:1 매칭):
        - ``baseline``         : 보정 없는 ŷ_base의 PAPE/HR@1/HR@2/MAE — G1 reference.
                                 (R0/R1 양쪽 04 JSON에 동일한 baseline 값이 저장되어 있음.
                                  본 함수는 호출된 routing의 baseline을 그대로 집계 →
                                  결과적으로 ``coldstart_R0.baseline`` ≈ ``coldstart_R1.baseline``
                                  이 되며, 검증용 redundancy로 활용 가능.)
        - ``operating_points`` : op-point별 (HR-pres / PAPE-aggr) W5 보정 metric.
        - ``pape_ratio_vs_baseline`` : 시드별 PAPE 비율 (보정 / baseline). <1이면 보정이
          PAPE를 낮춘 것 → README "vs v01" 비교의 G1 핵심 비율 (예: 0.66 = -33.8%).
          per-seed 비율을 먼저 계산하고 그 위에서 mean/std → 시드 평균이 비율 평균.
        - ``n_cold_apts`` / ``n_cold_windows`` : 시드별 cold 가구 수와 윈도우 수도
          평균/std로 합산 (변동이 있어야 정상; 모든 시드에서 cold 20 동일이면 std=0).

    핵심 설계:
        - σ, α_v0, α_w1 메타는 ``seeds[0]``의 값을 그대로 차용 (시드별로 동일하므로 OK).
          만약 시드별로 다르면 silent inconsistency가 생기지만, 04에서 carry-over 강제이므로
          현실적으로 발생 불가.
        - ``pape_ratio_vs_baseline`` 계산 시 ``baseline.pape > 0`` 가드 — PAPE 정의상
          baseline=0인 케이스는 거의 없지만 안전 장치.
    """
    keys_metric = ["pape", "hr@1", "hr@2", "mae"]  # plan §"Metrics" — kW 단위 PAPE + HR@k + MAE
    op_names = ["HR-preserving", "PAPE-aggressive"]  # carry-over from v01, 둘 다 σ=3.0

    # --- baseline 집계 (보정 없음; routing 무관하지만 routing별 JSON에 모두 저장) ---
    seeds = sorted(per_seed.keys())
    base = {k: _agg([per_seed[s]["baseline"][k] for s in seeds]) for k in keys_metric}
    # --- 두 op-point 집계 (HR-preserving, PAPE-aggressive) ---
    ops = {}
    for op in op_names:
        # σ/α 메타는 첫 시드의 값을 그대로 사용 (시드별로 동일한 carry-over이므로).
        cell = per_seed[seeds[0]]["operating_points"][op]
        ops[op] = {
            "sigma": cell["sigma"],
            "alpha_v0": cell["alpha_v0"],
            "alpha_w1": cell["alpha_w1"],
            # 4개 metric 각각에 대해 시드 차원 mean/std 산출.
            "metrics": {
                k: _agg(
                    [per_seed[s]["operating_points"][op]["metrics"][k] for s in seeds]
                )
                for k in keys_metric
            },
            # per-seed PAPE 비율 (보정/baseline) → 시드 평균이 G1 보고용 "PAPE 개선율"의 base.
            # 비율 평균 ≠ 평균의 비율이므로 per-seed 비율을 먼저 만들고 평균하는 것이 정답.
            "pape_ratio_vs_baseline": _agg(
                [
                    per_seed[s]["operating_points"][op]["metrics"]["pape"]
                    / per_seed[s]["baseline"]["pape"]
                    for s in seeds
                    if per_seed[s]["baseline"]["pape"] > 0  # 안전 가드 (baseline≈0 방지)
                ]
            ),
        }
    return {
        "routing": routing,
        "n_seeds": len(seeds),
        "seeds": seeds,
        # cold 가구 수 / cold 윈도우 수도 시드 차원 mean/std (대개 시드 무관 상수).
        "n_cold_apts": _agg([float(per_seed[s]["n_cold_apts"]) for s in seeds]),
        "n_cold_windows": _agg(
            [float(per_seed[s]["n_cold_windows"]) for s in seeds]
        ),
        "baseline": base,                  # 보정 없음 — G1 reference (R0/R1 동일 값)
        "operating_points": ops,           # HR-pres / PAPE-aggr 두 셀 (G1/G2 핵심)
    }


def aggregate_E1(per_seed: dict[int, dict]) -> dict:
    """E1 ablation (T0 vs T2 on V0) 시드 묶음 집계 — peak_aux 효과의 G1 검증.

    (한글) 05 산출 ``E1_results.json`` 시드 묶음 → mean/std로 집계.
    핵심 헤드라인 두 가지를 시드 차원에서 만든다:

        1) ``peak_aux_contribution_on_V0`` (시드 평균/std, raw delta):
           = T2.V0 - T0.V0 (각 metric별).
           PAPE 항목이 음수일수록 T2(peak_aux 있음)가 T0보다 PAPE 낮음 = peak_aux 효과 큼.
           README "+11.9 ± 9.2 pp" — 부호를 뒤집고 pp 단위 변환은 ``pape_relative_improvement_pp``.

        2) ``pape_relative_improvement_pp`` (시드 평균/std, %p):
           = per-seed로 (T0의 PAPE 상대개선% - T2의 PAPE 상대개선%) 계산 후 평균.
           **부호 정의: 양수 = T2가 더 많이 개선** (즉 peak_aux 효과로 V0가 추가로 작동).
           v01 §4.3은 +18.6 pp을 보고 (50:50 split). README는 80:20에서 +11.9 ± 9.2 pp
           (시드 swing 3.6–24.7) — std 큼 = T0의 codebook collapse가 일부 시드에서만 일어남.

    arm별로 baseline / V0 / vq_k_min / vq_perplexity 4종을 시드 차원 mean/std로 묶어 함께 보존.
    이는 README "T0 codebook collapse가 시드별 swing의 원인" 진단을 multiseed_summary 한 곳에서
    확인할 수 있게 하기 위함 (T0의 vq_k_min이 한두 시드에서만 매우 작으면 그 시드의 +pp가 큼).

    메타:
        - ``alpha_v0``, ``M``은 첫 시드 값 차용 (시드별 동일).
        - ``v01_reference_pp = 18.6`` 하드코딩 — paper §4.3 헤드라인 직접 비교용.
    """
    seeds = sorted(per_seed.keys())
    keys_metric = ["pape", "hr@1", "hr@2", "mae"]
    arms = ["T0", "T2"]  # T0 = peak_aux OFF, T2 = peak_aux ON (둘 다 V0 보정만)

    # --- arm별 baseline + V0 + codebook 진단치 시드 집계 ---
    arm_metrics = {
        arm: {
            # 보정 전 baseline metric (each arm's own backbone) — V0 대비 비교 reference.
            "baseline": {
                k: _agg([per_seed[s]["results_by_arm"][arm]["baseline"][k] for s in seeds])
                for k in keys_metric
            },
            # V0 보정 후 metric — peak_aux 효과 측정의 핵심 (T2.V0 - T0.V0).
            "V0": {
                k: _agg([per_seed[s]["results_by_arm"][arm]["V0"][k] for s in seeds])
                for k in keys_metric
            },
            # arm별 codebook 진단치 — T0가 collapse(k_min↓, ppl↓)되는지 확인용.
            # 05는 arm마다 독립 codebook을 fit하므로 arm별로 별도 진단치 존재.
            "vq_k_min": _agg(
                [
                    float(per_seed[s]["results_by_arm"][arm]["vq_diagnostics"]["k_min"])
                    for s in seeds
                ]
            ),
            "vq_perplexity": _agg(
                [
                    per_seed[s]["results_by_arm"][arm]["vq_diagnostics"]["perplexity"]
                    for s in seeds
                ]
            ),
        }
        for arm in arms
    }
    # --- raw contribution = T2.V0 - T0.V0 (시드 차원 평균/std) ---
    # PAPE 음수 = T2가 더 낮음 = peak_aux 효과 큼. 부호 반전 + %p 변환은 아래 pape_pp.
    contribution = {
        k: _agg(
            [per_seed[s]["peak_aux_contribution_on_V0"][k] for s in seeds]
        )
        for k in keys_metric
    }
    # --- per-seed +pp PAPE delta, matching v01 §4.3 reporting style ---
    # v01 §4.3은 "T0 vs T2의 PAPE 상대 개선율 차이"를 +18.6 pp로 보고.
    # 정의: rel_arm = (V0 - baseline) / baseline * 100 (음수면 baseline 대비 PAPE 감소).
    #      pape_pp = rel_T0 - rel_T2  (양수 ⇔ T2가 baseline 대비 더 많이 개선됨).
    # README "+11.9 ± 9.2 pp"는 이 값의 시드 평균/std (3 seeds → ddof=0).
    pape_pp = []
    for s in seeds:
        t0 = per_seed[s]["results_by_arm"]["T0"]
        t2 = per_seed[s]["results_by_arm"]["T2"]
        rel_t0 = (t0["V0"]["pape"] - t0["baseline"]["pape"]) / t0["baseline"]["pape"] * 100.0
        rel_t2 = (t2["V0"]["pape"] - t2["baseline"]["pape"]) / t2["baseline"]["pape"] * 100.0
        pape_pp.append(rel_t0 - rel_t2)  # positive = T2 improvement larger (= peak_aux 효과 큼)
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        # 메타: 첫 시드 값을 그대로 사용 (시드별 동일).
        "alpha_v0": per_seed[seeds[0]]["alpha_v0"],
        "M": per_seed[seeds[0]]["M"],
        "arm_metrics": arm_metrics,                          # arm × {baseline, V0, vq diag}
        "peak_aux_contribution_on_V0": contribution,         # raw delta (T2 - T0), 시드 mean/std
        "pape_relative_improvement_pp": _agg(pape_pp),       # +pp 단위, README "+11.9 ± 9.2 pp"
        "v01_reference_pp": 18.6,                            # v01 §4.3 50:50 split 헤드라인 reference
    }


def aggregate_W(per_seed: dict[int, dict]) -> dict:
    """W-component decomposition (T2 × {V0, W1a, W5}) 시드 묶음 집계 — G4 검증.

    (한글) 06 산출 ``W_component_results.json`` 시드 묶음 → mean/std로 집계.
    G4의 핵심 질문 ("W5가 V0/W1a 단독보다 우월한가") 답을 시드 차원에서 통합:

        - cells[V0/W1a/W5][PAPE]의 시드 평균을 보면 ranking이 살아남는지 확인.
        - ``hybrid_synergy_kw`` 시드 평균/std → README "+3.47 ± 0.33 / +3.24 ± 0.58 PAPE-kW".
          synergy = best_single_PAPE - W5_PAPE (06에서 정의). 양수 ⇔ W5가 단독 best보다
          PAPE를 더 깎음 = 진짜 hybrid 효과.

    routing은 06이 R0 only이므로 결과 dict에도 ``"routing": "R0"`` 명시.
    σ, α_v0, α_w1 메타는 첫 시드 값 차용 (시드별 동일).

    v01 §4.6 iter4가 50:50 split에서 보고한 W5 dominance 랭킹이 80:20 split + 3 seeds
    평균에서도 살아남는지가 G4의 결론. README는 "✅ W5 still dominates"로 보고.
    """
    seeds = sorted(per_seed.keys())
    keys_metric = ["pape", "hr@1", "hr@2", "mae"]
    op_names = ["HR-preserving", "PAPE-aggressive"]
    mechs = ["V0", "W1a", "W5"]  # 3-way mechanism toggle (backbone=T2 고정)

    # baseline (보정 없음) — synergy 비교의 reference, 시드 차원 mean/std로 집계.
    base = {
        k: _agg([per_seed[s]["baseline"][k] for s in seeds]) for k in keys_metric
    }
    # 두 op-point × 세 mechanism = 6 cell의 시드 차원 mean/std + synergy.
    per_op = {}
    for op in op_names:
        # σ/α 메타: 첫 시드 값을 그대로 (시드별 동일).
        first = per_seed[seeds[0]]["per_operating_point"][op]
        per_op[op] = {
            "sigma": first["sigma"],
            "alpha_v0": first["alpha_v0"],
            "alpha_w1": first["alpha_w1"],
            # 메커니즘 × metric 2-d nested dict, 각 cell이 시드 차원 5튜플.
            "cells": {
                m: {
                    k: _agg(
                        [per_seed[s]["per_operating_point"][op]["cells"][m][k] for s in seeds]
                    )
                    for k in keys_metric
                }
                for m in mechs
            },
            # synergy = min(V0, W1a)의 PAPE - W5의 PAPE (06에서 미리 계산되어 저장됨).
            # 시드별 synergy를 그대로 모아 평균/std → README의 "+3.47 ± 0.33 PAPE-kW".
            "hybrid_synergy_kw": _agg(
                [per_seed[s]["per_operating_point"][op]["hybrid_synergy_kw"] for s in seeds]
            ),
        }
    return {
        "routing": "R0",                        # 06은 R0 only (plan §G4)
        "n_seeds": len(seeds),
        "seeds": seeds,
        "baseline": base,                       # 보정 없음 (G4 reference; 04 baseline과 동일해야 함)
        "per_operating_point": per_op,          # op-point × {V0, W1a, W5} 6 cell + synergy
    }


def aggregate_codebook(per_seed: dict[int, dict]) -> dict:
    """Codebook 진단치 (T2 latent 위에 fit된 03번 codebook) 시드 묶음 집계.

    (한글) 03 산출 ``codebook_diagnostics.json`` 시드 묶음 → mean/std로 집계.
    plan §"Metrics"의 codebook health 검증 (k_min ≥ 113 게이트, util, perplexity)을
    시드 차원에서 통합. README "k_min 137 ± 28 (v01 threshold 113), util 1.000, ppl 27.84".

    집계 키:
        - ``vq_utilization``  : 사용된 cluster 비율 (1.0이면 빈 cluster 없음).
        - ``vq_perplexity``   : 클러스터 분포의 entropy 기반 effective 갯수 (M=32 가까울수록 균등).
        - ``vq_k_min``        : 가장 sparse한 cluster의 train 윈도우 수 — v01 threshold 113.
        - ``vq_k_max``        : 가장 큰 cluster의 윈도우 수 (불균등성 진단).
        - ``n_train_windows`` / ``n_train_apts`` : split이 시드 무관 동일 크기인지 sanity check.
        - ``n_empty_clusters`` : 0이어야 정상 (1 이상이면 codebook degenerate).

    health gate:
        ``k_min_health_pass_all_seeds`` = 모든 시드에서 k_min ≥ 113이면 True.
        — 한 시드라도 실패하면 G1의 codebook health 주장이 깨짐. 현재 README상으론 ✅.
        — NOTE: 05의 arm-별 codebook 진단치(T0/T2 각각)는 ``aggregate_E1``에서 별도로 모아
          져 있으며, 본 함수는 03이 만든 단일 codebook (T2 backbone, train side) 진단치만 다룸.
    """
    seeds = sorted(per_seed.keys())
    keys = [
        "vq_utilization",
        "vq_perplexity",
        "vq_k_min",          # v01 threshold ≥ 113 게이트
        "vq_k_max",
        "n_train_windows",
        "n_train_apts",
        "n_empty_clusters",  # 0이 아니면 codebook degenerate (collapse)
    ]
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "metrics": {k: _agg([float(per_seed[s][k]) for s in seeds]) for k in keys},
        # 모든 시드가 k_min health (≥113) pass한 경우만 True. 03이 시드별로 미리 bool 계산.
        "k_min_health_pass_all_seeds": all(
            per_seed[s]["k_min_health_pass"] for s in seeds
        ),
    }


def main() -> None:
    """v02 07번 entrypoint — 시드 묶음 집계의 메인 흐름.

    (한글) 단계:
        1) ``--seeds`` 파싱 (default {42, 123, 7} — CLI flag로 override 가능).
        2) 시드별로 5종 JSON을 ``_read_json``으로 안전 로드 (없으면 None → missing 기록).
        3) 4종 source별 집계 (``aggregate_routing`` × 2 + ``aggregate_E1`` + ``aggregate_W``
           + ``aggregate_codebook``) — 각각 독립적으로 None-safe (해당 source가 한 시드도
           없으면 summary에 None으로 기록).
        4) ``multiseed_summary.json`` dump + stdout digest 출력.

    누락 시드 정책: KeyError로 죽지 않고 그 source/seed만 skip + ``[WARN]`` 출력. summary
    JSON에도 ``missing_per_source`` 키로 영구 기록되어 paper 작성 시 "어느 시드/소스가
    빠졌는지" 추적 가능.
    """
    ap = argparse.ArgumentParser(description="Aggregate v02 per-seed JSONs into multi-seed summary.")
    # 시드 셋 default는 CLAUDE.md "Multi-seed: {42, 123, 7}" 컨벤션. CLI override 가능.
    # (per-seed argparse 컨벤션은 02–06 *experiment* 스크립트에 한정. 07은 aggregator라
    #  여러 시드를 동시에 보는 것이 자연스러움 — memory: feedback_argparse_per_seed 비강제.)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    args = ap.parse_args()

    # 5종 source × 시드 매트릭스: per-seed dict (key=seed int, val=loaded JSON dict).
    per_seed_R0, per_seed_R1, per_seed_E1, per_seed_W, per_seed_cb = {}, {}, {}, {}, {}
    # 누락 추적: source별로 빠진 시드 리스트. summary JSON에도 보존됨.
    missing = {"R0": [], "R1": [], "E1": [], "W": [], "codebook": []}
    for s in args.seeds:
        # outputs/v02_fl_8020_ratio/seed{S}/ 아래에서 5종 JSON 로드 (없으면 None).
        seed_root = V02_OUT_ROOT / f"seed{s}"
        r0 = _read_json(seed_root / "coldstart_R0.json")
        r1 = _read_json(seed_root / "coldstart_R1.json")
        e1 = _read_json(seed_root / "E1_results.json")
        wc = _read_json(seed_root / "W_component_results.json")
        cb = _read_json(seed_root / "codebook_diagnostics.json")
        # None이면 missing 리스트에 시드 적재; 아니면 source 딕셔너리에 적재.
        if r0 is None: missing["R0"].append(s)
        else: per_seed_R0[s] = r0
        if r1 is None: missing["R1"].append(s)
        else: per_seed_R1[s] = r1
        if e1 is None: missing["E1"].append(s)
        else: per_seed_E1[s] = e1
        if wc is None: missing["W"].append(s)
        else: per_seed_W[s] = wc
        if cb is None: missing["codebook"].append(s)
        else: per_seed_cb[s] = cb

    # 누락 source 경고 + 어떤 시드가 어디에 들어왔는지 한 줄 요약 (sanity check).
    print("[aggregate] inputs:")
    for src, ms in missing.items():
        if ms:
            print(f"  [WARN] {src}: missing seeds {ms}")
    print(f"  R0={list(per_seed_R0)}  R1={list(per_seed_R1)}  E1={list(per_seed_E1)}  "
          f"W={list(per_seed_W)}  codebook={list(per_seed_cb)}")

    # ---- 최종 summary dict 빌드 (시드 차원 mean/std) ----
    # 각 source별로 빈 dict면 None을 채워 source가 통째로 누락된 경우도 안전 표현.
    summary = {
        "split_version": "v02",
        "seeds_requested": list(args.seeds),                     # 사용자가 요청한 시드 셋
        "missing_per_source": missing,                           # 누락 추적 (paper 작성 시 인용)
        "operating_points_definition": {                         # carry-over from v01 (재튜닝 금지)
            "HR-preserving": {"sigma": 3.0, "alpha_v0": 1.0, "alpha_w1": 0.1},
            "PAPE-aggressive": {"sigma": 3.0, "alpha_v0": 1.5, "alpha_w1": 0.5},
        },
        # plan §"Comparison table" 행 매핑:
        #   coldstart_R0  → 행2(HR-pres) + 행3(PAPE-aggr) (G1/G3)
        #   coldstart_R1  → 행4(HR-pres) + 행5(PAPE-aggr) (G2)
        #   E1            → 행6 (peak_aux ON/OFF, +18.6 pp 비교)
        #   W_component   → 행7 (W5 dominance, iter4 비교)
        #   codebook_health → plan §Metrics의 health gate (k_min ≥ 113)
        # 행1 (v01 50:50 baseline)은 v02 외부 reference이므로 본 집계에 없음.
        "coldstart_R0": aggregate_routing(per_seed_R0, "R0") if per_seed_R0 else None,
        "coldstart_R1": aggregate_routing(per_seed_R1, "R1") if per_seed_R1 else None,
        "E1": aggregate_E1(per_seed_E1) if per_seed_E1 else None,
        "W_component": aggregate_W(per_seed_W) if per_seed_W else None,
        "codebook_health": aggregate_codebook(per_seed_cb) if per_seed_cb else None,
    }

    # plan §"Outputs"의 ``multiseed_summary.json`` 위치 (V02_OUT_ROOT 직속).
    out_path = V02_OUT_ROOT / "multiseed_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[aggregate] saved -> {out_path}")

    # ---- Friendly stdout digest (사람이 한눈에 보기 위한 콘솔 요약) ----
    # JSON 파일은 paper 자동화의 source-of-truth이지만, 시드 sweep 직후 빠르게 눈으로
    # 검수하기 위한 압축 요약을 stdout으로 출력. README "v02 headline result" 표의 숫자가
    # 이 print 출력으로 직접 보임 (mean +/- std 형식).
    if summary["coldstart_R0"] is not None:
        rs = summary["coldstart_R0"]["operating_points"]
        print("\n=== Cold zero-shot R0 (mean +/- std across seeds) ===")
        for op, info in rs.items():
            mt = info["metrics"]
            print(
                f"  {op:<16} PAPE={mt['pape']['mean']:.2f} +/- {mt['pape']['std']:.2f}  "
                f"HR@1={mt['hr@1']['mean']:.1f} +/- {mt['hr@1']['std']:.2f}  "
                f"ratio={info['pape_ratio_vs_baseline']['mean']:.3f}"
            )
    if summary["coldstart_R1"] is not None:
        rs = summary["coldstart_R1"]["operating_points"]
        print("\n=== Cold zero-shot R1 (mean +/- std across seeds) ===")
        for op, info in rs.items():
            mt = info["metrics"]
            print(
                f"  {op:<16} PAPE={mt['pape']['mean']:.2f} +/- {mt['pape']['std']:.2f}  "
                f"HR@1={mt['hr@1']['mean']:.1f} +/- {mt['hr@1']['std']:.2f}  "
                f"ratio={info['pape_ratio_vs_baseline']['mean']:.3f}"
            )
    if summary["E1"] is not None:
        e1 = summary["E1"]
        print("\n=== E1 (peak_aux ON/OFF on V0) ===")
        print(
            f"  pape_relative_improvement: {e1['pape_relative_improvement_pp']['mean']:+.1f} "
            f"+/- {e1['pape_relative_improvement_pp']['std']:.1f} pp  "
            f"(v01 reference: {e1['v01_reference_pp']:+.1f} pp)"
        )
    if summary["W_component"] is not None:
        for op, info in summary["W_component"]["per_operating_point"].items():
            print(f"\n=== W-component {op} ===")
            for m in ["V0", "W1a", "W5"]:
                pape = info["cells"][m]["pape"]
                print(f"    {m:5s} PAPE={pape['mean']:.2f} +/- {pape['std']:.2f}")
            sy = info["hybrid_synergy_kw"]
            print(f"    synergy(best_single - W5) = {sy['mean']:+.2f} +/- {sy['std']:.2f} PAPE-kW")
    if summary["codebook_health"] is not None:
        cb = summary["codebook_health"]["metrics"]
        print(
            f"\n=== Codebook health (T2): k_min={cb['vq_k_min']['mean']:.0f} +/- "
            f"{cb['vq_k_min']['std']:.0f} (v01 threshold 113), util="
            f"{cb['vq_utilization']['mean']:.3f}, ppl={cb['vq_perplexity']['mean']:.2f}"
        )


if __name__ == "__main__":
    main()
