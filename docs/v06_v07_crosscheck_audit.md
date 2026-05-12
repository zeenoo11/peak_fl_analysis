# v06 / v07 교차검증 감사 (Cross-check Audit)

## 메타데이터

- 일자: 2026-05-12
- 팀: `v06-v07-crosscheck` (Claude Code experimental agent teams)
- 리뷰어: sonnet × 6 (v06-α, v06-β, v06-γ, v07-α, v07-β, v07-γ)
- 라운드: 2 (각 6명이 독립적으로 이슈 3건씩 × 2회 = 36 raw issue reports)
- 감사 대상: `plans/v{06,07}-*.md`, `experiments/v{06,07}_*/`, `outputs/v{06,07}_*/`, `papers/v{06,07}_draft/`
- 감사 축:
  1. experiments 스크립트 산출물 경로 ↔ `outputs/` 구조
  2. `outputs/` 수치/figure ↔ paper draft 본문/표/그림
  3. `plans/` 설계 ↔ experiments 코드 구현

## Cross-version systemic finding

### S1. `local_epochs` 실행값 vs 명시값 불일치 (high)

- **plan/paper 명시**: `local_epochs = 2`
- **실제 실행**: `local_epochs = 40` — 모든 FL `result.json`, 모든 launcher 명령어
- 적발: v06-α (R2), v06-γ (R2), v07-β (R1), v07-γ (R1) — **두 버전에 걸쳐 4명 독립 검출**
- 영향: 두 버전 모두 "v06 protocol 고정" / "conference Phase A와 bit-equivalent" invariant 무너짐

Evidence:
- `outputs/v06_round_dynamics/seed42/V6-Dyn-B-FedAvg/result.json:7` (`"local_epochs": 40`)
- `outputs/v07_loss_budget_sweeps/seed42/V6-Dyn-B-FedAvg-aux0.05/result.json`
- `experiments/v07_loss_budget_sweeps/01_run_aux_sweep.py:108` (`default=40`)
- `experiments/v06_round_dynamics/Readme.md:16-32, 81-99` (`--local_epochs 40`)
- `plans/v06-01_round_dynamics.md:86, 102, 175, 180` (E=2)
- `plans/v07-01_loss_and_budget_sweeps.md:93` (E=2)
- `papers/v06_draft/v06_round_dynamics.md:77`, `papers/v07_draft/v07_loss_weight_sensitivity.md:55`

**결정 필요**: 실제 실행값(40)이 의도였는지(→ plan/paper 정정), 아니면 v06-conference 프로토콜 위반인지(→ E=2로 재실행).

## v06 이슈 (round_dynamics)

### Round 1 — high-consensus (3/3 reviewers)

#### v06-1. 아파트 수: plan 100 vs paper/code 114 (high)
- `plans/v06-01_round_dynamics.md:158` ("100 apartments")
- `papers/v06_draft/v06_round_dynamics.md:53, 66, 80` ("114")
- `src/dataloader/per_client_split.py:6, 26, 98` (docstring "100" 잔존 — v06-β R2 추가)
- 영향: plan §3 schema `n_clients=100`, §4 wall-clock 추정, G1/G2/G3 등 100 기준으로 적힘

#### v06-2. `K_local`: plan 4 vs code/paper 2 (high)
- `plans/v06-01_round_dynamics.md:30, 110, 237, 335`
- `experiments/v06_round_dynamics/08_codebook_stacking.py:28, 295`
- `papers/v06_draft/v06_round_dynamics.md:122, 361, 382, 500`
- plan의 Phase 2 codebook 설계가 dead letter — paper §5.5 sweep도 K=2를 baseline

#### v06-3. Figure 경로 분기 / plan figure 미카운트 (med)
- plan: `outputs/v06_round_dynamics/figures/` F1–F3 (3개)
- 실제: `papers/v06_draft/figures/` 10개 (F1, F1b, F1c, F2~F8)
- 적발: v06-α, v06-β

### Round 2 — 추가 발견

#### v06-4. paper §4.2 vs §5.3 BEFORE 컬럼이 동일 MAEonly 셀에서 다름 (high)
- §4.2 V6-Dyn-A-MAEonly = 48.91 ± 0.70, §5.3 BEFORE = 48.90 ± 0.68
- FedRep: 49.08 ± 0.50 vs 49.07 ± 0.50
- FedProx: 48.51 ± 0.03 vs 48.50 ± 0.02
- 동일 backbone에서 다른 result.json 출처 의심 → reproducibility 우려
- 적발: v06-β
- Evidence: `papers/v06_draft/v06_round_dynamics.md:194,288,197,291,196,290`

#### v06-5. FedRep `drift_l2` 참조점 비대칭 (med)
- FedAvg/FedProx/Ditto/FedProto: `server_state_pre = round_start_state` (단일 글로벌)
- FedRep: `server_state_pre = mean_head_pre` (per-client round-start full state 평균)
- F3 (drift_vs_round)의 FedRep drift_l2=2.20은 비교 불가능한 metric
- 적발: v06-γ
- Evidence: `src/fl/round_aux.py:541-548 (FedRep), 184/305/690 (others)`

#### v06-6. `epoch_equivalent` 로깅 버그 (med)
- 의도: `round × local_epochs` (centralised ↔ FL 공통 x축)
- 실제: `round_idx` (FL drivers가 `epoch_equivalent`를 logger에 전달 X → `round_logger.py:264` fallback)
- 결과: centralised(40 epoch) vs FL(20 round) 의 x축 정렬이 silently broken
- 적발: v06-α

#### v06-7. Paper §4.1 SI MB vs F2 figure binary MiB (med)
- §4.1 표: FedAvg 641 MB (= 640853280 / 1e6, SI)
- F2 figure x축: ~611 MiB (= 640853280 / 1024²)
- `experiments/v06_round_dynamics/07_make_figures.py:148`이 binary 단위 사용
- 약 5% 차이, 단위 표기 없음
- 적발: v06-α

#### v06-8. `log_terminal`이 jsonl에 `round=-1` 2회 append (med)
- plan(line 215): 1행 명시
- 실제: drivers가 val + test 두 번 호출 → 2행
- 현재 aggregator는 result.json 사용 → OK이나, 향후 jsonl 파서 오동작 위험
- 적발: v06-β
- Evidence: `src/fl/round_logger.py:287-329`, `01_centralised.py:154-155`, `02_fl_dynamics.py:187-188`

#### v06-9. `per_client_split.py` docstring "100-apt pool" 잔존 (med)
- v06-1의 코드 내 잔존 manifestation
- 적발: v06-β

#### v06-10. centralised `result.json`에 `rounds` 필드 누락 (low)
- plan §3 스키마는 `rounds` 공통 필드
- `01_centralised.py:159-177`은 `epochs`만 출력
- 적발: v06-γ

#### v06-11. `08_codebook_stacking.py` docstring "12종 cell × 36" → 실제 30 (low)
- lambda suffix 추가로 30개로 늘었으나 docstring 미갱신
- 적발: v06-γ

## v07 이슈 (loss_budget_sweeps)

### Round 1 — high-consensus

#### v07-1. `local_epochs` mismatch — S1 cross-version finding (위 참조)

#### v07-2. plan §5 driver 파일명이 실제와 다름 (med)
- plan(290-298): `01_centralised.py, 02_fl_dynamics.py, 03_fedsgd.py, 04_codebook_trajectory.py, 06_aggregate_budget.py, 07_aggregate_traj.py`
- 실제: `01_run_aux_sweep.py, 02_run_hr_weight_sweep.py, 05_*, 08_*`
- 적발: v07-β, v07-γ, v07-α

#### v07-3. hr suffix "-hr1.0" (paper) vs "-hr1" (disk) (low)
- `_hr_suffix(1.0) → "1"` (g format)
- paper §5.2:198, §8 표기 `-hr1.0`
- 적발: v07-β, v07-γ

#### v07-4. plan §5에 v07-B/C driver 파일이 deferred 표기 없이 나열됨 (low)
- 적발: v07-γ

### Round 2 — 추가 발견

#### v07-5. paper §3.4/§5.4/§7 "+1.5 to +3.5 PAPE" 하한값 오류 (high)
- 실제 λ=0 → λ=0.3 PAPE 차이: FedAvg=+2.94, FedProx=+2.89, **FedRep=+2.28 (최소)**, FedProto=+3.01, Ditto=+3.51
- paper "+1.5" 하한 ≈ 35% 과소표기, §3.4 / §5.4 / §7 세 곳 반복
- 적발: v07-β, v07-γ 독립 합의
- Evidence: `outputs/v07_loss_budget_sweeps/aux_sweep_summary.json`, `papers/v07_draft/v07_loss_weight_sensitivity.md:115, 251, 347`
- 비고: plan §1.35는 이미 "+2.3-3.5 PAPE"로 올바르게 적혀 있음 — paper만 오류

#### v07-6. paper §3.2 vs §4.2 같은 cell 다른 JSON 출처 (med)
- §3.2 centralised λ=0: 48.91 ± 0.70 (test_terminal in `result.json`)
- §4.2 before: 48.90 ± 0.68 (test_before in `codebook_lift.json`)
- 같은 패턴 λ=0.1: §3.2 48.47 vs §4.2 48.46
- 동일 backbone, 동일 test split이지만 두 JSON 출처의 PAPE 계산 경로 차이
- 적발: v07-α, v07-γ

#### v07-7. CLAUDE.md sweep grid 표기 (med)
- CLAUDE.md(line 21): `λ_aux × hr_weight × 6 algos × 3 seeds` (cross-product 5×4×6×3=360 처럼 보임)
- 실제: 2축 coordinate sweep (lambda at fixed hr=0.1, hr at fixed λ=0.1) ≈ 107 unique
- plan/paper는 정확, CLAUDE.md만 오해 유발
- 적발: v07-α

#### v07-8. paper §8 reproducibility가 bash for-loop 사용 (med)
- 실제 환경: Windows PowerShell (CLAUDE.md, `pyproject.toml` `tool.uv.environments`)
- `experiments/Readme.md`는 PowerShell `foreach` 사용
- §8 step 2 (v07-A1 codebook stacking) 그대로 실행 불가
- 적발: v07-β
- Evidence: `papers/v07_draft/v07_loss_weight_sensitivity.md:379-385`

#### v07-9. test 파일 부재 + README "42 passed" 클레임의 coverage 갭 (low)
- 존재: `tests/test_v07_aux_sweep.py` (10 cases, aux suffix만 검증)
- 부재 (plan §5 명시): `test_v07_budget_argparse.py, test_v07_fedsgd_step.py, test_v07_checkpoint_roundtrip.py`
- `_hr_suffix` / `_PAT_AUX01_HR` 미검증
- 적발: v07-β, v07-γ

#### v07-10. plan에 v07-A2 (hr_weight sweep) 섹션 자체가 없음 (med)
- 54개 run + paper §5 한 챕터 분량이 plan에서 설계 근거 부재
- 적발: v07-α

#### v07-11. paper §4.3 "within 0.5 PAPE" claim self-contradiction (low)
- §4.2 표: 44.41 / 44.53 / 44.92 → range 0.51 PAPE
- §4.3 클레임: "within 0.5 PAPE"
- 적발: v07-α

## 권장 조치 우선순위

### Tier 1 — 즉시 (high, 수치/클레임 오류)
1. **S1 `local_epochs`**: plan/paper의 "E=2" 표기를 실제 실행값(40)으로 정정 OR E=2로 재실행. paper §2 표 정정.
2. **v06-4** §4.2 vs §5.3 BEFORE: 동일 `result.json` 출처에서 재계산.
3. **v07-5** "+1.5 PAPE" 하한 → "+2.3" 또는 "+2.28" 로 §3.4 / §5.4 / §7 세 곳 정정.

### Tier 2 — med
- v06-2 `K_local`=2로 plan 정정
- v06-1 / v06-9: 100→114로 plan/per_client_split.py docstring 갱신, wall-clock 재추정
- v06-5 FedRep drift 참조 통일 또는 비대칭 paper에 명시
- v06-6 / v06-7: `epoch_equivalent` × `local_epochs` 적용, MB 단위 통일(SI 또는 binary 둘 중 하나로 paper+figure 정렬)
- v07-2 driver 파일명 plan §5 갱신
- v07-7 CLAUDE.md sweep grid 표기 명확화
- v07-8 §8 reproducibility PowerShell화
- v07-10 plan에 v07-A2 섹션 추가

### Tier 3 — low / cosmetic
- v06-3 figure 경로 plan 갱신
- v06-8 `log_terminal` 1행 호출로 통일
- v06-10 centralised `result.json`에 `rounds` 추가
- v06-11 docstring 30으로 정정
- v07-3 hr suffix 표기 통일
- v07-4 plan §5 deferred 마커
- v07-6 §3.2 vs §4.2 출처 차이를 각주로 명시
- v07-9 hr_weight test 추가
- v07-11 "within 0.5 PAPE" → "within 0.55"

## 팀 운영 관찰

- **Swarm 효과 입증**: S1 (`local_epochs`) 가 4명에게 독립 적발 — single reviewer로는 systemic임을 못 봤을 가능성.
- **이슈 적발 패턴**: round 1은 plan-level 표면 일관성(수치/파일명), round 2는 코드 내부 로직(drift 참조, epoch_equivalent, dual-JSON 출처) 같은 더 깊은 층.
- **v07-α 흐름 미스**: round 1을 건너뛰고 round 2 메시지에서야 첫 응답. team task-claim flow에 한 곳 디테일이 어긋남.
- **신호 대비 노이즈**: 6명 × 2라운드 = 36 raw → 22개 unique high-value, 14건은 cross-validation 신뢰도 증거.
