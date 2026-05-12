# v06-01 — Conference 비교 실험 위에 round-level 학습 동역학 평가

> Successor to `v05-01_fedcb_codebook.md` and the conference pipeline
> (`papers/conference_draft/presentation.md`, `experiments/conference/`).
> v06은 **conference의 비교 실험 (백본 / 알고리즘 / hyperparameter / 비교군) 을
> 그대로 가져오되, 평가 protocol 만 *cold zero-shot* → *per-round across-client
> validation* 으로 교체**한다. v01–v05의 cold 구조를 따르지 않으며, 114가구
> 전체가 federated 학습에 참여한다. 분류축 background는
> `docs/fl_methodologies_fedsgd_vs_fedavg.md` (FedSGD vs FedAvg 두 representative
> methodology). v06 본 실험은 **FedAvg family** 한 쪽만 다루고 (conference 비교
> 그대로), FedSGD 는 옵셔널 limit reference 로만 깔아둔다.

> **Status (2026-05-01).** Plan only. **Phase 1 (FL 5종 + centralised
> pooled reference, no codebook) ready for implementation; Phase 2
> (codebook terminal-stacking + optional NF/FM/FedSGD) deferred until
> Phase 1 trajectory is verified.**

## Phase 분리

User 지시 (2026-05-01) 에 따라 v06 을 두 phase 로 나눈다:

- **Phase 1 — FL 5종 trajectory 만 (이번 implementation 대상).**
  V6-Dyn-A (centralised pooled SGD upper bound) + V6-Dyn-B-{FedAvg,
  FedProx, FedRep, Ditto, FedProto} 5종 = **6 cells × 3 seeds = 18 runs**.
  Codebook 코드 / NF baseline / FM zero-shot / FedSGD limit 모두 Phase 1
  에 *없음*.
- **Phase 2 — codebook terminal-stacking + 옵션 reference (Phase 1 끝나면
  scope 확정).** Phase 1 의 5 FL cell × 3 seed = 15 final state_dict 위
  에 v05 hierarchical federated codebook (`src/fl/codebook_fl.py`,
  K_local=2, M=32, α=1.0) 을 한 번씩 fit + cluster-mean correction →
  각 cell 의 `result.json` 에 `with_codebook_cmo` 블록 추가 (= **Option
  α terminal-stacking**). conference Proposed 는 자연스럽게 V6-Dyn-B-
  FedAvg 의 `with_codebook_cmo` 행으로 reproduce 되며, 다른 4 cell
  에도 같은 codebook 을 stack 함으로써 *codebook 의 알고리즘-orthogonality*
  도 자동 측정. NF / FM / FedSGD reference 는 Phase 2 결정 시 재논의.

---

## 동기

Conference 발표 (`papers/conference_draft/presentation.md`) 의 핵심 비교 표
세 개는 모두 **80가구 학습 → 20가구 cold zero-shot PAPE** 위에서 측정된 단일
종점 snapshot 이다 (line 197–204 표가 대표):

```
| FedRep    | 57.18 ± 1.52 |
| Ditto     | 56.38 ± 1.63 |
| FedProto  | 56.37 ± 1.44 |
| FedAvg    | 56.34 ± 1.41 |
| FedProx   | 56.30 ± 1.55 |
| Proposed  | 50.17 ± 0.97 |
```

이 표는 *cold personalisation* 의 정확도를 측정한 것이지 *federated learning
자체의 학습 trajectory* 를 측정한 것이 아니다. 두 reviewer 질문은 conference
artefact 만으로는 답변 불가:

1. 5종 FL 알고리즘이 cold PAPE 56–57 % 좁은 군집을 보이는 것은 algorithm
   간 본질적 차이가 작아서인가, 아니면 **20 라운드 budget 내에서 모두
   동일한 plateau 에 막혀** 그런 것인가? 라운드별 trajectory 가 있어야
   분리된다.
2. Proposed 의 codebook 효과 (PAPE 57.32 → 50.17, ablation 표 line 213–216)
   는 codebook 자체의 기여인가, 아니면 codebook 이 적용되는 *terminal
   backbone state* 가 단지 "더 잘 학습된 시점" 이어서인가? 이것도
   trajectory 로만 답변된다.

v06 은 **conference 의 비교군 / 백본 / hyperparameter 를 모두 그대로** 가져온
뒤 평가만 교체한다:

- 114가구 전부 federated 학습에 참여 (cold 분리 없음).
- 각 가구의 internal `train(70%) / val(10%) / test(20%)` split (CLAUDE.md
  `src/config.py` 의 `TRAIN_RATIO=0.7`, `VAL_RATIO=0.1`).
- 라운드 종료 직후 114 클라이언트 각자의 **자기 val 윈도우 위에서** PAPE/HR/MAE 계산
  → 서버에서 across-client 평균.
- 학습 종료 후 같은 방식으로 test split 위에서 한 번 더 측정 (terminal row).

비교군 / backbone / hyperparameter 는 모두 conference 와 동일:

- 백본: **NBEATSxAux** (`latent_source='h_generic'`, peak-aux head 부착,
  L = MAE + 0.3 · peak_aux, hr_weight = 0.1).
- FL 알고리즘 5종: FedAvg, FedProx, FedRep, Ditto, FedProto
  (`src/fl/{fedavg, fedprox, fedrep, ditto, fedproto}.py` 의 round-loop
  helpers — 모두 이미 v04 에서 추출된 상태).
- Proposed: NBEATSxAux backbone + post-training v05 federated codebook +
  cluster-mean correction (`src/fl/codebook_fl.py`).
- Hyperparameter: rounds = 20, local_epochs = 40, batch = 512, lr = 1e-3,
  wd = 1e-5, λ = 0.3, hr_weight = 0.1. Note: plan originally specified
  local_epochs=2 (conference Phase A design); actual execution used 40
  (see audit S1). batch/lr/λ/hr_weight remain bit-equivalent to conference.

v01 method body — NBEATSx + peak-aux + W5 — 는 그대로 frozen. v06 은
*평가 protocol 만* 바꾼다.

## Goals

**G1.** **Centralised pooled SGD upper bound.** NBEATSxAux 를 114가구
training 부분 합쳐서 pooled 로 학습. 라운드별 (= epoch 별) val mean across
clients 기록. v06 의 **상한선** 이며 동시에 round logger / per-client val
hook 이 정상 동작하는지 검증하는 Gate 1.

**G2.** **Conference 5종 FL 알고리즘의 라운드별 trajectory.** FedAvg,
FedProx, FedRep, Ditto, FedProto 각각을 114가구 위에서 conference hyperparameter
(rounds=20, local_epochs=40, batch=512) 로 학습. 라운드별 across-client val
PAPE/HR/MAE + cumulative comm bytes + drift L2 기록. **Conference 표 (line
197–204) 의 cold 종점 PAPE 56–57 % 군집** 이 라운드별 trajectory 위에서 어떻게
형성되는지 (군집이 *학습 초기부터* 인지 *후반에 plateau 형성* 인지) 가
제 1 deliverable.

**G3.** **Proposed (FedAvg + post-hoc federated codebook + correction)
의 trajectory.** Conference Proposed 와 동일한 recipe: FedAvg-NBEATSxAux 학습
+ 학습 종료 후 v05 hierarchical federated codebook (K_local=2, M=32) fit +
cluster-mean correction (α=1.0). 라운드별 trajectory 는 *backbone 부분만*
(codebook 은 post-hoc 1-shot 이므로 라운드별 update 없음 — CLAUDE.md "post-hoc
1-shot" 불변), terminal 에서 codebook fit + corrected metric 한 번 보고. 이
구조에서 Proposed 의 *backbone trajectory 자체* 가 다른 4종 FL 보다 다른지가
관찰 포인트.

**G4.** **Centralised neural forecasting baselines (NHITS / Crossformer /
DLinear) terminal reference.** Conference Table (line 170–175) 와 일관되도록
3종 NF 를 같은 114가구 pooled training 위에서 한 번 학습하고 terminal val/test
metric 만 보고. trajectory 는 v06 main figure 에 *terminal 점* 으로 표기 (각
방법은 학습 곡선이 있지만 v06 의 비교축은 FL family 라운드 동역학이므로
NF/FM 은 reference 점 으로 충분). 시간이 부족하면 Gate 4 cell 통째로 skip
가능 (필수 아님).

**G5.** **Foundation model zero-shot terminal reference.** Conference Table
(line 184–188) 의 Chronos-T5 tiny / TimesFM / Chronos-Bolt small 을 같은
114가구 val/test 위에서 한 번 zero-shot inference. trajectory 없음. **Gate
5 cell skip 가능** — conference 결과를 reuse 해도 평가 데이터가 cold 20가구
→ 114가구 internal val 로 바뀌었으므로 재측정 필요.

**G6.** **(옵션) FedSGD limit reference.** docs 분석의 두 representative
methodology 중 다른 한 쪽. 114가구 위에서 1 SGD step / round, R ≈ 60–80
rounds (B=512 기준 가구당 1 step/epoch ≈ 60 epoch budget 채우려면 60 rounds).
라운드별 logger 동일. drift L2 = 0 이 자명한 reference; FedAvg 5종의 drift
가 의미 있는 크기인지 sanity check 용. **Gate 6 cell 은 plan 채택 후 결정**.

## Non-goals

- **Cold-client zero-shot 평가** (= v01–v05 protocol). 의도적으로 제거.
- **80:20 split YAML / v10 households YAML reuse**. 사용 안 함; 114가구
  전부 internal split.
- **E (local_epochs) sweep**. `local_epochs=40` (actual execution value;
  see audit S1) 로 고정. v06 의 비교축은 *알고리즘 종류* (FL 5종 + Proposed)
  이지 E 가 아니다.
- **Codebook 라운드별 re-fit**. CLAUDE.md "post-hoc 1-shot" 그대로 — terminal
  에서 한 번만 fit.
- **Personalised FL on round-level dynamics, async, partial participation,
  drift-corrected variants beyond FedProx, second dataset, method redesign**
  → 모두 v07+.

---

## Method

### 1. 클라이언트 모집단 (전 cell 공통)

- 모든 valid UMass 2016 apartments: `dataloader.umass.list_available_apartments
  ('2016')` → `filter_valid_apartments(min_hours=7000)` → **114 apartments**.
- **클라이언트 분리 없음**. 114가구 모두 학습 + 평가에 참여.
- 가구별 internal split (CLAUDE.md `src/config.py`):
  - `TRAIN_RATIO=0.7`, `VAL_RATIO=0.1`, test = 0.2.
  - 슬라이딩 윈도우 `INPUT_SIZE=96, HORIZON=24, stride=24`.
  - z-norm 은 train 부분에서만 fit.
- 가구당 대략적 윈도우 수 (≈7000 시간 기준):
  - train ≈ 199, val ≈ 24, test ≈ 53.
  → train pool ≈ 22,700 windows, val pool ≈ 2,700, test pool ≈ 6,000.

### 2. 백본 / hyperparameter (전 FL cell 공통, conference 그대로)

- `NBEATSxAux(latent_source='h_generic')`.
- 손실 `L = MAE(ŷ, y) + λ · peak_aux(ŷ, y)` with **λ=0.3, hr_weight=0.1**.
- AdamW, **lr = 1e-3, weight_decay = 1e-5**.
- **batch_size = 512** (conference Phase A 와 bit-equivalent — v04
  09_fix_rerun 도 동일).
- **rounds = 20, local_epochs = 40** (actual execution value; plan
  originally specified 2 matching conference Phase A, but all drivers ran
  40 — see audit S1. T = 20 × 40 = 800 epoch-equiv per client).
- 풀 참여 (`C = 1.0`, 114/114 가구 매 라운드).
- Seed `{42, 123, 7}`.

> **B=512 의 의미.** 가구당 ≈ 199 train windows / B=512 → 가구당 1 mini-batch /
> epoch. 즉 local_epochs=40 → 가구당 라운드당 40 SGD step. 20 라운드 × 40 step
> = 800 SGD step / 가구 (T = 800 epoch-equiv per client). conference Phase A 는
> local_epochs=2 (T=40) 로 실행되었으나 v06 실제 실행값은 40 (audit S1).
> v06 은 이 setting 에서 in-train val trajectory 를 관찰하는 것이 목적.
> (`docs/fl_methodologies_fedsgd_vs_fedavg.md` §6 의 "B=64 sweep" 노트는
> v06 본 실험 외의 v07+ 변형으로 보류.)

### 3. 라운드별 logging contract (전 FL cell 공통)

매 round, 서버 aggregation 직후 (broadcast 직전) 한 줄을 `round_log.jsonl`
에 기록:

```json
{
  "round": 7,
  "epoch_equivalent": 14.0,
  "val":   {"pape_mean": ..., "pape_std_across_clients": ...,
             "hr@1_mean": ..., "hr@2_mean": ..., "mae_mean": ...,
             "mse_kw2_mean": ...,
             "n_clients": 114, "n_windows_total": ...},
  "train": {"loss_mean_last_epoch": ..., "n_steps_round": ...},
  "comm":  {"upload_bytes_round": ..., "upload_bytes_cum": ...,
             "broadcast_bytes_round": ..., "broadcast_bytes_cum": ...},
  "drift_l2": ...,
  "wall_seconds_round": ...
}
```

- 가구별 val PAPE/HR/MAE/MSE(kW²) 는 가구 내부 ≈ 24 윈도우 위에서 계산
  (z-space → kW 역정규화 후), 114 가구 평균이 `*_mean`, across-client
  표준편차가 `*_std_across_clients`. MSE 단위 kW² 는 conference Table 과
  같은 정의 (`experiments/conference/ablation/codebook_module_effect.py`
  의 `metrics_z_to_kw` + 직접 kW²-MSE 계산 패턴).
- `drift_l2 = mean_i ‖θ_i^{end-of-local} − θ_global^{round-start}‖₂`,
  aggregation 직전 (각 클라이언트의 local 학습 직후) 측정. FedAvg-style
  알고리즘에 자연스럽게 정의되며, FedSGD 는 항상 0.
- terminal-test 행은 같은 jsonl 에 `round = -1` 로 한 번 append.

### 4. Cell 정의

**Phase 1 cells (이번 implementation 대상):**

| Cell | Algorithm | Round-loop | 역할 |
|---|---|---|---|
| **V6-Dyn-A** | Centralised pooled SGD | n/a, T=40 epochs | Gate 1 reference, 상한선. |
| **V6-Dyn-B-FedAvg** | FedAvg | `fl/fedavg_aux.py` | Conference 비교군. |
| **V6-Dyn-B-FedProx** | FedProx | `fl/fedprox.py` | Conference 비교군. |
| **V6-Dyn-B-FedRep** | FedRep | `fl/fedrep.py` | Conference 비교군. |
| **V6-Dyn-B-Ditto** | Ditto | `fl/ditto.py` | Conference 비교군. |
| **V6-Dyn-B-FedProto** | FedProto | `fl/fedproto.py` | Conference 비교군. |

3 seed × 6 cell = **18 runs**. Phase 1 에서는 codebook 적용 / NF 학습 /
FM zero-shot / FedSGD limit 코드 모두 *작성하지 않음*.

**Phase 2 add-ons (Phase 1 trajectory 확인 후 결정):**

| Add-on | 처리 방식 | 비고 |
|---|---|---|
| Codebook terminal-stacking | Phase 1 5 FL cell × 3 seed = 15 final state_dict 각각에 v05 hierarchical federated codebook (K_local=2, M=32, α=1.0) 한 번 fit + cluster-mean correction → `with_codebook_cmo` block 을 해당 cell 의 `result.json` 에 추가. **별도 cell 정의하지 않음**. | Option α — 본 plan §"Phase 분리". V6-Dyn-B-FedAvg + codebook = conference Proposed. |
| V6-Dyn-D-NF (옵션) | NHITS / Crossformer / DLinear, 114가구 pooled centralised, terminal only. | conference NF reference. |
| V6-Dyn-E-FM (옵션) | Chronos-T5 tiny / TimesFM / Chronos-Bolt small, zero-shot. | conference FM reference. trajectory 없음. |
| V6-Dyn-F-FedSGD (옵션) | FedSGD, 1 step / round, R≈40. | docs 두 representative methodology 중 한 쪽 limit reference. |

**Wall-clock 추산** (5070 Ti, NBEATSxAux ≈ 65 K params, 114가구, B=512, E=40):
- 라운드당 step ≈ 114 클라이언트 × 40 step × ~5 ms ≈ 22.8 s/round compute +
  ~0.5 s aggregation + ~2.7 s val eval (114 클라이언트 × 24 윈도우 forward).
- V6-Dyn-B-{FedAvg, FedProx, FedRep, Ditto, FedProto}: 20 라운드 × ~26 s
  ≈ ~8.7 분 / cell / seed.
- V6-Dyn-A centralised: 40 epoch × 114 가구 × 1 batch × ~5 ms + 40 × ~2.7 s
  eval ≈ ~3.1 분 / seed.
- Phase 1: 18 runs × ~7 분 ≈ ~126 분 직렬.

---

## Go/No-go gates

| Gate | After | Pass | Fail action |
|---|---|---|---|
| **Gate 1** | V6-Dyn-A (3 seeds) | (a) 종료 무에러; (b) 라운드(=epoch)별 val PAPE 단조 감소 (첫 10 epoch); (c) terminal test PAPE finite. | 학습 loop / dataloader / round logger regression. 멈추고 디버그. |
| **Gate 2** | V6-Dyn-B 전 5종 (3 seeds each) | (a) 모든 cell 의 라운드 20 끝 val PAPE 가 라운드 1 보다 낮음; (b) 5종 FL 의 라운드 20 종점 val PAPE 가 1 pp 이내 군집을 형성하거나 *반대로* 분리되거나 — 어느 쪽이든 *일관된 trajectory 형태* 가 보임 (NaN / divergence 없음); (c) `drift_l2` non-zero, finite. | aggregator 깨짐 / 발산. 디버그. |
| **Gate 3** | 라운드별 trajectory 분석 | conference cold-PAPE 군집 (56–57 %) 이 라운드별 val-PAPE 군집과 *defensible 한 관계* 를 보임 (예: 둘 다 라운드 5 이후 plateau, 또는 둘 다 같은 ranking). 즉 conference 종점 결과가 v06 trajectory 의 자연스러운 endpoint 임이 그림에서 읽힘. | conference 종점과 v06 trajectory 가 *모순* 되면 (예: 라운드별 ranking 이 cold 종점 ranking 과 정반대), framing 위협 → 추가 분석 (val pool size, 가구 heterogeneity) 후 결정. |

---

## Build order

| Step | Module | 설명 | Verify |
|---|---|---|---|
| **1** | `src/dataloader/per_client_split.py` (new) | `build_per_client_splits(seed) -> dict[apt -> {train_idx, val_idx, test_idx, mean, std}]`. 114 가구 전부, internal 70/10/20 sliding-window split. train z-norm only. seed 마다 deterministic. `outputs/v06_round_dynamics/seed{S}/per_client_split.pkl` 캐시. | pytest: 114 apts, 비율 + 비중복 verify. |
| **2** | `src/fl/round_logger.py` (new) | `RoundLogger`: open `round_log.jsonl` append-only; `log_round(round_idx, model, server_state_pre, client_states, val_loaders, comm_stats, wall)` 호출 시 모델로 114 클라이언트 val 윈도우 forward → per-client PAPE/HR/MAE/MSE(kW²) → across-client mean+std → 한 줄 write. resumable. | pytest: 2 라운드 dummy, jsonl 파싱, resume 동작. |
| **3** | 5종 FL helper 의 *round-callback hook* 패치 | `fl/fedavg_aux.py`, `fl/fedprox.py`, `fl/fedrep.py`, `fl/ditto.py`, `fl/fedproto.py` 의 `*_round_loop()` 함수에 `on_round_end: Optional[Callable]` 인자 추가. 매 라운드 aggregation 직후 (broadcast 직전) 호출되어 logger 가 server state + client states + drift + wall 을 받음. signature 통일. 기존 호출자 (conference Phase A 등) 와 backward compatible (`on_round_end=None` default). | 기존 conference Phase A smoke 가 그대로 통과; 새 hook 으로 logger 가 V6-Dyn-B-FedAvg 1-round trace 생성. |
| **4** | `src/fl/centralised_pooled.py` (new) | V6-Dyn-A driver helper. 114 가구 train 윈도우 합친 단일 DataLoader, NBEATSxAux 학습. epoch 단위로 `on_round_end` 호출하여 logger 가 round-equivalent 행 작성. | pytest: 2 epoch 동기 작동. |
| **5** | `src/fl/fedsgd.py` (new, **옵션**) | V6-Dyn-F-FedSGD limit. 가구당 1 mini-batch grad → 서버 평균 → 1 SGD step. logger hook 동일. | pytest: 5 라운드, weight delta = −lr · mean_grads 검증. |
| **6** | `experiments/v06_round_dynamics/01_centralised.py` | V6-Dyn-A 드라이버. argparse: `--seed S --epochs 40 --batch 512`. | smoke `--seed 42 --epochs 2`. |
| **7** | `experiments/v06_round_dynamics/02_fl_dynamics.py` | V6-Dyn-B-{FedAvg, FedProx, FedRep, Ditto, FedProto} 통합 드라이버. argparse: `--seed S --algorithm {fedavg,fedprox,fedrep,ditto,fedproto} --rounds 20 --local_epochs 40 --batch 512`. **codebook 호출 없음** — Phase 1 은 backbone trajectory 만. | smoke `--seed 42 --algorithm fedavg --rounds 2`. |
| **8** | `experiments/v06_round_dynamics/06_aggregate.py` | Phase 1 모든 `round_log.jsonl` + terminal 행 → `multiseed_summary.json` (terminal numbers; conference Table 과 같은 schema 의 PAPE / HR@1 / HR@2 / MSE) + `trajectories.npz` (라운드별 array). | one shot. |
| **9** | `experiments/v06_round_dynamics/07_make_figures.py` (Phase 1), `experiments/v06_round_dynamics/08_codebook_stacking.py` (Phase 2) | Phase 1: 7 figures — F1_round_vs_val_pape, F1b_round_vs_test_pape, F1c_round_vs_train_loss, F2_bytes_vs_val_pape, F3_drift_vs_round, F4_round_vs_test_pape_MAEonly, F5_round_vs_train_loss_MAEonly (5종 FL + V6-Dyn-A reference; 3-seed mean ± std band). Phase 2 (codebook stacking): F6_codebook_lift, F7_alpha_pareto, F8_klocal_sweep. 총 10 figures. | one shot each. |
| **10** | 3-seed sweep | Phase 1 = 18 runs. | summary + figures. |
| **11** | `papers/v06_draft/v06_round_dynamics.md` (new) | conference Table 과 v06 trajectory 양쪽이 어떻게 같은 ranking 으로 수렴하는지 보이는 short paper. (Phase 1 분량으로 일단 작성, Phase 2 결과는 추후 추가.) | reviewer pass. |
| **(Phase 2)** | `experiments/v06_round_dynamics/08_codebook_stacking.py` (Phase 2) | Phase 1 의 5 FL cell × 3 seed 의 `final_state_dict.pt` 각각을 v05 federated codebook 으로 한 번 fit + correction → 해당 `result.json` 에 `with_codebook_cmo` block append. | Phase 1 통과 후 dispatch. |

---

## Outputs

```
outputs/v06_round_dynamics/
├── seed{42,123,7}/
│   ├── per_client_split.pkl
│   ├── V6-Dyn-A_centralised/{round_log.jsonl, final_state_dict.pt, result.json}
│   ├── V6-Dyn-B-FedAvg/...
│   ├── V6-Dyn-B-FedProx/...
│   ├── V6-Dyn-B-FedRep/...
│   ├── V6-Dyn-B-Ditto/...
│   └── V6-Dyn-B-FedProto/...
│   # Phase 2 add-ons (Phase 1 trajectory 확인 후):
│   # - 위 5 FL cell 의 result.json 에 `with_codebook_cmo` block 추가
│   # - V6-Dyn-D-NF/, V6-Dyn-E-FM/, V6-Dyn-F-FedSGD/ (옵션)
├── trajectories.npz
├── multiseed_summary.json
└── figures/                          ← 런타임 산출물 (gitignored)
    ├── F1_round_vs_val_pape.png      # Phase 1
    ├── F1b_round_vs_test_pape.png    # Phase 1
    ├── F1c_round_vs_train_loss.png   # Phase 1
    ├── F2_bytes_vs_val_pape.png      # Phase 1
    ├── F3_drift_vs_round.png         # Phase 1
    ├── F4_round_vs_test_pape_MAEonly.png   # Phase 1
    ├── F5_round_vs_train_loss_MAEonly.png  # Phase 1
    ├── F6_codebook_lift.png          # Phase 2
    ├── F7_alpha_pareto.png           # Phase 2
    └── F8_klocal_sweep.png           # Phase 2
# 총 10 figures; 커밋 사본은 papers/v06_draft/figures/ 에 보관
```

`round_log.jsonl` schema 는 §3. `result.json` (terminal-only, conference
Table 과 schema 호환):

```json
{
  "cell": "V6-Dyn-B-FedAvg",
  "algorithm": "fedavg_aux",
  "seed": 42,
  "n_clients": 114,
  "rounds": 20,
  "local_epochs": 40,
  "batch": 512,
  "C": 1.0,
  "val_terminal":  {"pape_mean": ..., "hr@1_mean": ..., "hr@2_mean": ...,
                    "mae_mean": ..., "mse_kw2_mean": ...,
                    "pape_std_across_clients": ...},
  "test_terminal": {"pape_mean": ..., "hr@1_mean": ..., "hr@2_mean": ...,
                    "mae_mean": ..., "mse_kw2_mean": ...,
                    "pape_std_across_clients": ...},
  "comm_total_bytes": {"upload_cum": ..., "broadcast_cum": ...},
  "drift_l2_mean_over_rounds": ...,
  "elapsed_seconds": ...
}
```

V6-Dyn-C-Proposed 의 `result.json` 에는 추가로 `with_codebook_cmo`
하위 블록 (terminal val/test 위에서 codebook 보정 후 metric):

```json
"with_codebook_cmo": {
  "K_local": 2, "M": 32, "alpha": 1.0,
  "val":  {"pape_mean": ..., ...},
  "test": {"pape_mean": ..., ...}
}
```

---

## Dependencies

- `dataloader.umass.list_available_apartments`, `filter_valid_apartments`,
  `load_apartment_hourly` — already exist; v06 calls directly (no v01/v02
  /v05 split YAML).
- `src/fl/{fedavg, fedavg_aux, fedprox, fedrep, ditto, fedproto}.py`
  round-loop helpers — exist; **step 3 patches a single new keyword arg
  `on_round_end` into each**.
- `src/fl/codebook_fl.py` (`local_codebook_step`, `merge_local_codebooks`,
  `federated_residual_offsets`) — v05 helpers, used by V6-Dyn-C-Proposed
  terminal step only.
- `src/utils/metrics.py` (PAPE, HR@k, seven_axis_metrics).
- `src/models/nbeatsx_aux.py` (`NBEATSxAux(latent_source='h_generic')`).
- New: `src/dataloader/per_client_split.py`, `src/fl/round_logger.py`,
  `src/fl/centralised_pooled.py`, (옵션) `src/fl/fedsgd.py`.

**No reuse** of: `outputs/v02_fl_8020_ratio/splits/`, `Peak_Analysis/configs/
v10_households.yaml`, any v01/v04/v05/conference backbone state_dict, any
codebook artefact (codebook 은 V6-Dyn-C terminal 단계에서 새로 fit).

---

## Open questions

A. **B = 512 의 epoch 해상도 한계.** 가구당 ≈ 199 train win / B=512 → 가구당 1 mini-batch / epoch. local_epochs=40 → 가구당 라운드당 40 SGD step. 따라서 *라운드 내부* 의 SGD 동역학은 관찰되지 않음 (라운드 단위 점만 찍힘). 만약 라운드 내부 dynamics 가 필요하면 B=64 로 가야 함. 본 plan 은 현행 실행값(E=40) 기준, 라운드-내부 dynamics 는 v07+ 로 보류.

B. **Per-client val ≈ 24 windows 의 noise.** across-client mean 은 안정적이지만 per-client trajectory 는 noisy. "best vs worst client" 같은 figure 가 필요해지면 val 비율 0.10 → 0.15 로 늘리는 것을 v07+ 에서 검토 (TRAIN_RATIO 도 비례 축소 필요).

C. **Drift proxy 정의.** `mean_i ‖θ_i − θ_global‖₂` 한 가지 scalar / round / cell. 알고리즘별 비교가 더 흥미로우면 cosine 변형 추가 가능.

D. **NF / FM cell skip 여부.** Conference 종점 비교를 v06 trajectory 옆에 두려면 같은 114가구 internal val/test 위에서 재측정 필요 — conference 의 cold 20가구 결과를 그대로 옮길 수 없음. wall-clock 부담이 크면 D / E cell 은 v06 에서 빼고 paper draft 에서만 conference 결과를 인용.

E. **FedSGD cell 의 R 결정.** B=512 기준 가구당 1 batch / epoch 이므로 1 epoch = 1 step / 가구 = 1 round. 40 epoch budget = 40 round 가 자연스러움. 하지만 FedSGD limit 의 "통신 폭발" 효과를 보려면 더 작은 B 가 필요 — A 와 같은 trade-off. 옵션이라 본 plan 결정 보류.

F. **Codebook 라운드별 trajectory.** 의도적으로 out of scope (CLAUDE.md "post-hoc 1-shot" 불변). v07+ 에서 필요해지면 v05 helper 를 라운드 logger 에 넣어 추가 trajectory line 을 얻을 수 있지만 두 dynamics (backbone + codebook) 가 entangle 되어 v06 의 깨끗한 비교가 흐려짐.

---

## Conventions

- **Per-seed argparse.** 모든 v06 driver 가 `--seed S` 받음 (memory:
  `feedback_argparse_per_seed`).
- **No MLflow.** `result.json` + `round_log.jsonl` + `print` (repo 컨벤션).
- **Output namespacing.** `outputs/v06_round_dynamics/seed{S}/{cell}/`.
- **Conference invariant.** 백본 / 알고리즘 / batch / lr / λ / hr_weight /
  비교군 5종 은 conference 와 동일. 단, local_epochs 는 conference Phase A=2
  에서 실제 실행값 40 으로 달라짐 (audit S1); v06 result 를 conference
  cold-PAPE 와 직접 비교할 때 이 점 명시 필요. 평가 (cold zero-shot →
  per-round across-client val) 만 변경.
- **Method frozen.** encoder, aux head, peak descriptor, λ, hr_weight 모두
  v01 design 그대로.
- **Backbone fresh-init.** 모든 cell × seed 가 새로 학습 (학습 trajectory
  자체가 종속변수).

---

## What is NOT in scope

- Cold-client zero-shot 평가 (v01–v05 protocol).
- 80:20 split YAML / v10 household YAML 사용.
- E (local_epochs) sweep, B sweep.
- 라운드별 codebook re-fit.
- pFL / Async / FedBuff / drift-corrected FedAvg variants beyond FedProx /
  partial participation.
- 두 번째 데이터셋.
- 백본 / method redesign.
