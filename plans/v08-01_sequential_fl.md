# v08-01 — Sequential / Cyclic FL (CWT) trajectory on the v06 protocol

> Successor to `v06-01_round_dynamics.md` and `v07-01_loss_and_budget_sweeps.md`.
> v06/v07은 모두 **parallel FedAvg-family** (각 round 마다 N 클라이언트가 *병렬*
> 로 local 학습 → 서버 weighted-mean) 위에서 round-level trajectory를 측정했다.
> v08은 **같은 protocol (114 가구, internal 70/10/20, NBEATSxAux, T=800
> epoch-equiv) 위에서 aggregation pattern 만 *sequential / cyclic* 으로
> 교체** 하여 trajectory를 재측정한다. 즉:
>
> ```
> Round t = 1, ..., R:
>     θ ← global state at start of round t
>     For client i in order π_t[1] → π_t[2] → ... → π_t[N]:
>         θ ← θ - η · ∇L_i(θ)   over E local epochs
>         # No averaging — model state is *handed off* to next client
>     θ_{t+1} ← θ (state at last client of round)
>     # Server: snapshot + per-client val eval (no aggregation step)
> ```
>
> 학계 명칭은 **Cyclic Weight Transfer (CWT)** [Chang et al. 2018, JAMIA;
> Sheller et al. 2020, Sci Rep], 한국어 문헌에서는 "순차 FL / 라운드-로빈 FL"
> 로도 부른다. Yuan et al. NeurIPS 2024 가 SFL (Sequential FL) 의 convergence
> 를 분석하고 non-IID에서 parallel FedAvg 와 다른 lower bound를 보였다.
>
> v08은 *no new method* — backbone / 손실 / split / metric / logging schema 는
> 모두 v06과 bit-equivalent. 비교축은 단 하나: *aggregation pattern* (parallel
> mean vs sequential hand-off).

> **Status (2026-05-13).** Plan only. v06 paper draft + v07 sweep results
> available; v08 implementation begins after this plan is signed off.

---

## §0 Motivation — v06/v07이 답하지 못한 한 가지 축

v06 결론: **FedAvg / FedProx / FedRep / Ditto / FedProto** 5종 + centralised
pooled reference 의 round-level trajectory는 broadly 일치 (terminal val PAPE 가
1–2 pp 군집). v07 결론: peak-aux loss λ_aux 의 FL-side optimum 은 strict
boundary (λ=0) — peak-aux 가 *FL 어떤 알고리즘과도 incompatible*. 두 결과 모두
**parallel weighted-mean aggregation 위에서 측정**되었다.

남는 자연스러운 질문 한 줄:

> *parallel mean aggregation 이 만들어내는 "drift-then-average" 자체*가
> 위 결과의 원인인가, 아니면 federation 자체의 본질적 한계인가?

이 질문에 대답하려면 *averaging 단계가 없는* federation pattern 위에서 같은
실험을 한 번 더 돌려봐야 한다. SFL/CWT 가 정확히 그 setting:

- **No weight-mean** — model state는 client→client 로 직렬 hand-off.
- **No drift-then-average** — 매 client 가 *직전 client 가 막 학습을 끝낸 모델*
  위에서 출발하므로, "self-local optimum 방향으로 drift 한 N 모델을 평균
  내서 흐릿한 global을 만들어내는" parallel FedAvg 의 핵심 mechanism이 없다.
- **Catastrophic-forgetting risk** — 대신 last-client-overwrite (한 라운드의
  마지막 client 가 본 자료만 dominant 한 모델을 만들 가능성) 라는 *다른 형태의*
  heterogeneity 취약성이 생긴다.

두 pattern 의 PAPE / HR / drift / bytes trajectory 를 같은 axes 위에 겹쳐
그려서, v06/v07 의 결론 중 어느 것이 *aggregation-pattern artifact* 이고 어느
것이 *federation-intrinsic* 인지 분리하는 것이 v08의 단일 contribution.

---

## §1 Method

v06의 §1, §2 와 동일. 차이가 있는 항목만 명시.

### 1.1 클라이언트 모집단 (v06과 동일)

- `dataloader.umass.list_available_apartments('2016')` →
  `filter_valid_apartments(min_hours=7000)` → **114 apartments**.
- 가구별 internal split: `TRAIN_RATIO=0.7`, `VAL_RATIO=0.1`, test=0.2
  (CLAUDE.md `src/config.py`). 슬라이딩 윈도우 `INPUT_SIZE=96, HORIZON=24,
  stride=24`. z-norm 은 train 부분에서만 fit.
- v06의 `outputs/v06_round_dynamics/seed{S}/per_client_split.pkl` 을 **재사용**
  (재계산 없음). v08 driver 가 같은 seed로 같은 split을 받도록 `build_per_client_splits`
  를 호출하면 deterministic 하게 일치한다.

### 1.2 백본 / hyperparameter (v06과 동일)

- `NBEATSxAux(latent_source='h_generic')`.
- 손실 `L = MAE(ŷ, y) + λ_aux · peak_aux(ŷ, y)`, λ_aux=0.3, hr_weight=0.1.
- AdamW, lr=1e-3, weight_decay=1e-5, batch_size=512.
- **rounds=20, local_epochs=40** — 한 client 가 한 라운드 안에 받아서 40 epoch
  로컬 학습 후 다음 client 로 hand-off. T = 20 × 40 = 800 epoch-equiv per
  client (v06과 동일).
- Seed `{42, 123, 7}`.

### 1.3 Aggregation pattern (유일한 v06 대비 차이)

매 라운드 t = 1, …, 20:

1. **순서 결정.** `order ∈ {"fixed", "perm"}` 에 따라:
   - `fixed`: apt 이름의 lexicographic order, 매 라운드 동일.
   - `perm`: numpy `np.random.RandomState(seed + 1000 + t).permutation(N)`
     — seed-deterministic 하면서 라운드마다 다름.
2. **Hand-off 루프.** `θ ← θ_{t-1, last_client}` (t=1 이면 fresh-init).
   ```
   for i in π_t[1], π_t[2], ..., π_t[N]:
       apply_state_dict(model, θ)
       optimizer = AdamW(model.parameters(), lr, wd)  # fresh optimizer per client
       run E=40 epochs of local SGD on client i
       θ ← model.state_dict()
   ```
3. **End of round.** `θ_{t, last_client} = θ_t` ← logger 가 snapshot.
4. **No weighted average.** server step = "찍어두기" 만 함.

**Optimizer state 처리.** v06 의 5종 FL helper 와 동일하게 매 client 시작 시
`Adam` 을 새로 init (momentum 비유지). client-persistent optimizer state는
*personalised* SFL variant이고, v08의 단일 비교축 (pattern only) 을 흐리므로
v08 본 plan 에서는 fresh-init 고정. Persistent variant 는 §8 open question 으로.

**Drift 정의.** v06 의 `drift_l2 = mean_i ‖θ_i^{end-of-local} - θ_global^{round-start}‖₂`
는 SFL 에서는 ill-defined (round-start state 가 *각 client 마다 다름*). v08은
두 가지 변형을 모두 기록:

- `drift_consecutive`: `mean_i ‖θ_{π_t[i+1]} - θ_{π_t[i]}‖₂` — 한 client 가
  hand-off 받기 *직전 / 직후* 의 거리. CWT 의 "round 내 누적 drift" proxy.
- `drift_intra_round`: `‖θ_{t, last} - θ_{t-1, last}‖₂` — round 시작/끝 거리.
  parallel FedAvg 의 round-level drift 와 직접 비교 가능.

### 1.4 라운드별 logging contract (v06 schema 그대로)

`round_log.jsonl` 한 줄 / round, schema 는 v06 §3 (`src/fl/round_logger.py`)
와 비트-동일. drift 키만 추가:

```json
{
  "round": 7,
  "epoch_equivalent": 14.0,
  "val":  {...},
  "test": {...},
  "train": {"loss_mean_last_epoch": ..., "n_steps_round": ...},
  "comm":  {"upload_bytes_round": ..., "upload_bytes_cum": ...,
            "broadcast_bytes_round": ..., "broadcast_bytes_cum": ...},
  "drift_l2": ...,                            // = drift_intra_round (v06 호환 키)
  "drift_consecutive_mean": ...,              // v08 추가
  "wall_seconds_round": ...
}
```

**Comm bytes 정의 차이.**

- v06 parallel FedAvg: 라운드당 `N · |θ|` upload + `N · |θ|` broadcast = `2N|θ|`.
- v08 CWT: 라운드당 `N · |θ|` relay (서버 broker 가정; client-to-client
  direct relay 면 `N|θ|`). v08 에서는 `upload_bytes = N|θ|`, `broadcast_bytes
  = 0` 로 기록 — Σ comm = `N|θ|`, parallel FedAvg 의 **절반**.

**Intra-round per-client snapshot (옵션, --intra_round_log).** F4 의 catastrophic-
forgetting figure 를 위해, 각 client hand-off 직후 val PAPE 를 한 번 더 기록.
N=114 × R=20 = 2,280 행 — 별도 jsonl (`intra_round_log.jsonl`) 에 분리. default
는 *비활성* (jsonl bloat 방지); F4 를 그릴 cell × seed 에서만 on.

### 1.5 Cell 정의

**Phase 1 (이번 implementation 대상):**

| Cell | Order | Reference 비교 | Runs |
|---|---|---|---|
| **V8-Seq-A** | n/a | v06 V6-Dyn-A centralised pooled — **reuse, no re-run** | 0 |
| **V8-Seq-B-CWT-Fixed** | lexicographic, 매 라운드 동일 | 새 학습 | 3 seeds |
| **V8-Seq-B-CWT-Perm** | per-round random permutation (seed+1000+t) | 새 학습 | 3 seeds |
| **V8-Seq-C-Reference-FedAvg** | n/a | v06 V6-Dyn-B-FedAvg — **reuse, no re-run** | 0 |

Phase 1 = **2 new cells × 3 seeds = 6 new runs**. v06 의 A 와 V6-Dyn-B-FedAvg
cell 은 overlay 용으로 그대로 가져옴 (재학습 금지 — bit-equivalent reproducibility
를 깨면 v08 의 단일 contribution 인 "같은 protocol 비교" 가 흐려진다).

**Phase 2 (Phase 1 trajectory 확인 후 결정):**

| Add-on | 처리 방식 | 비고 |
|---|---|---|
| Codebook terminal-stacking | Phase 1 의 V8-Seq-B-CWT-{Fixed, Perm} 2 cells × 3 seed = 6 final state_dict 에 v05 hierarchical federated codebook (K_local=2, M=32, α=1.0) 한 번 fit + cluster-mean correction → 해당 cell `result.json` 에 `with_codebook_cmo` block append. | v06 Phase 2 와 동일 mechanism. |
| λ_aux sweep (v07 mirror) | CWT-Perm 위에서 λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3} sweep. v07이 모든 FL cell 에서 strict boundary λ=0 을 발견했는데, *aggregation pattern* 이 바뀌면 interior optimum 이 돌아오는지 검증. | 5 λ × 3 seeds = 15 runs. |
| hr_weight sweep | 동일하게 CWT-Perm 위에서 hr_weight ∈ {0.05, 0.1, 0.5, 1.0} sweep. | 4 × 3 = 12 runs. |
| V8-Seq-D — Reverse order (sanity check) | π_t 를 lexicographic 의 *역순* 으로 매 라운드 동일하게 — Fixed 와 비교하여 last-client overwrite 효과의 방향성 확인. | 3 seeds. 옵션. |
| V8-Seq-E — Persistent optimizer | Adam state 를 client-persistent 로 유지 (서버가 같이 relay) — pattern 변형 ablation. | 3 seeds. 옵션. |

---

## §2 Goals

**G1.** **CWT round-level trajectory.** V8-Seq-B-CWT-{Fixed, Perm} 의 라운드별
val/test PAPE/HR/MAE/MSE trajectory. v06 의 V6-Dyn-A reference + V6-Dyn-B-FedAvg
와 같은 axes 위에 겹쳐서 *aggregation pattern 의 trajectory 차이* 가 보이는지.

**G2.** **Order sensitivity.** Fixed (매 라운드 동일 순서) vs Perm (라운드별
permutation) 의 trajectory 차이. Permutation 이 catastrophic-forgetting 을
완화하는지, Fixed 의 last-client overwrite 가 terminal PAPE 에 보이는지.

**G3.** **Comm vs PAPE frontier.** v06 FedAvg 와 같은 round budget (R=20, T=800)
에서 v08 CWT 가 절반 comm 으로 같은 PAPE 에 도달하면, "동일 protocol 위에서 SFL
이 communication-Pareto-우위" 라는 단순한 결론이 그려진다.

**G4.** **Intra-round catastrophic-forgetting (옵션, --intra_round_log).** 한 라운드
내 client hand-off 마다 val PAPE 를 찍어서, "마지막 N/10 client 직전까지는 PAPE
가 떨어졌다가 마지막에 다시 튀는" 형태가 있는지. SFL 에서 자주 보고되는 현상.

**G5.** **(Phase 2) Codebook stacking on CWT-trained backbone.** v06 의 5 FL cell
위에서는 codebook lift 가 cell-orthogonal 이었음. CWT-trained backbone 위에서도
같은 lift 가 보이면 codebook 이 *aggregation pattern* 에도 orthogonal 함이 확정.

**G6.** **(Phase 2) λ_aux sweep on CWT.** v07이 5종 parallel FL cell 전부에서
발견한 strict boundary λ=0 이, CWT 위에서도 boundary 인지 / interior 로 이동
하는지. 이동하면 "peak-aux 의 FL 비호환성" 은 *averaging artifact* 였다는 강한
증거 — v07의 결론을 정정하는 사후 발견이 된다.

## §3 Non-goals

- **v06과 다른 split / 데이터셋 / backbone / hyperparameter.** v08은 *pattern
  단 한 axis* 만 바꿈.
- **Asynchronous FL, FedBuff, partial participation.** v07 §6 open question 과
  동일하게 v09+.
- **Per-client personalisation (Ditto/FedRep/FedProto style on CWT).** v08은
  "깨끗한 federation pattern" 만 다룸 — personalisation 은 또 다른 axis.
- **Codebook 라운드별 re-fit.** CLAUDE.md "post-hoc 1-shot" 불변.
- **두 번째 데이터셋, 새 metric.**

---

## §4 Go/No-go gates

| Gate | After | Pass | Fail action |
|---|---|---|---|
| **G1** | V8-Seq-B-CWT-Fixed (3 seeds) | (a) 종료 무에러; (b) round-20 val PAPE 가 round-1 보다 낮음; (c) `drift_consecutive_mean` non-zero, finite, monotonic-ish across rounds; (d) terminal test PAPE finite. | hand-off loop / state_dict serialise 회귀. 멈추고 디버그. |
| **G2** | V8-Seq-B-CWT-Perm (3 seeds) | (a)–(d) G1과 동일; (e) Fixed 대비 trajectory 가 *유의미하게 다르게* 그려지거나 *동일* 하거나 — 어느 쪽이든 "permutation seed 마다 변동이 explosive 하지 않음" (3 seed std < 5 pp). | permutation seeding 버그. 디버그. |
| **G3** | Trajectory 분석 | v06 V6-Dyn-B-FedAvg 와 같은 axes 위에 겹쳤을 때 *defensible 한 관계* — 둘 다 plateau, 또는 한쪽이 명백히 빠르거나 느림, 또는 같은 terminal 에 도달. NaN / divergence 없음. | CWT 가 발산하면 (round 5 이전 PAPE > round 1) 학습률 / batch / E 재검토 — 본 plan 의 v06 hyperparameter 그대로 채택의 타당성 재검증. |

---

## §5 Build order

| Step | Module | 설명 | Verify |
|---|---|---|---|
| **1** | `src/fl/cyclic_fl.py` (new) | `cwt_round_loop(train_apts, *, rounds, local_epochs, lr, batch_size, weight_decay, seed, use_amp, aux_lambda, hr_weight, order='fixed' \| 'perm', on_round_end, on_client_handoff=None)`. 내부적으로 `fl.fedavg_aux._local_step_aux` 재사용 — local step contract 가 동일하므로 코드 중복 회피. `clone_state_dict / apply_state_dict` 는 `fl.base` 재사용. `order='perm'` 일 때 `np.random.RandomState(seed + 1000 + round_idx).permutation(N)` 으로 deterministic permutation. signature 가 `on_round_end` 를 v06 RoundLogger 의 callback 시그니처 그대로 받음. | pytest: 2 라운드 dummy run on 3 가구, hand-off 후 모델 state 가 *직전 client 의 학습 결과* 와 일치 (no averaging) 검증; permutation seed 가 fix 되면 reproducibility 검증. |
| **2** | `src/fl/round_logger.py` patch | 기존 `RoundLogger` 에 `drift_consecutive` 키를 받을 수 있도록 한 줄 추가 (signature 확장, 기본 None — back-compat). 별도 `intra_round_log.jsonl` 을 쓰는 두 번째 logger class `IntraRoundLogger` 를 같은 파일에 추가 (옵션, --intra_round_log on 일 때만 driver 가 사용). | pytest: drift_consecutive 가 jsonl 에 forward; IntraRoundLogger 가 N hand-off / round 마다 1 행 write. v06 의 기존 callsite 회귀 없음. |
| **3** | `experiments/v08_sequential_fl/01_cwt_dynamics.py` (new) | argparse: `--seed S --order {fixed,perm} --rounds 20 --local_epochs 40 --batch 512 --aux_lambda 0.3 --hr_weight 0.1 [--intra_round_log]`. v06 의 `02_fl_dynamics.py` 와 동일한 구조 (split build → logger 구성 → round_loop 호출 → result.json/final_state_dict.pt save). cell 이름 = `V8-Seq-B-CWT-Fixed` 또는 `V8-Seq-B-CWT-Perm`. **codebook 호출 없음** (Phase 2). | smoke `--seed 42 --order fixed --rounds 2 --local_epochs 2`. |
| **4** | `experiments/v08_sequential_fl/06_aggregate.py` (new) | Phase 1 모든 `round_log.jsonl` + terminal 행 → `multiseed_summary.json` (v06과 같은 schema; CWT cell 들 + v06 reference cell 들을 같은 dict 에 합침) + `trajectories.npz` (라운드별 array). v06 의 06_aggregate 출력을 *읽어서* reference 행으로 포함. | one shot. |
| **5** | `experiments/v08_sequential_fl/07_make_figures.py` (new) | Phase 1: 5 figures — F1_round_vs_val_pape, F1b_round_vs_test_pape, F2_bytes_vs_val_pape, F3_drift_vs_round, F4_intra_round_pape (옵션, --intra_round_log 가 있을 때만), F5_order_sensitivity (Fixed vs Perm side-by-side). 모두 v06 V6-Dyn-A reference + V6-Dyn-B-FedAvg 를 회색 underlay 로 항상 그림. | one shot. |
| **6** | 3-seed sweep | Phase 1 = 2 cells × 3 seeds = 6 runs. 약 8.5 min × 6 ≈ ~51 min 직렬. | summary + figures. |
| **7** | `papers/v08_draft/v08_sequential_fl.md` (new) | v06 / v07 표 옆에 CWT trajectory 한 줄을 추가하는 short paper. § contributions = "aggregation pattern artifact 분리". 표지 그림 = F1 overlay. Phase 1 분량으로 일단 작성, Phase 2 결과는 추후 추가. | reviewer pass. |
| **(Phase 2-A)** | `experiments/v08_sequential_fl/08_codebook_stacking.py` (new) | Phase 1 의 2 CWT cell × 3 seed = 6 `final_state_dict.pt` 각각에 v05 federated codebook 한 번 fit + correction → `result.json` 에 `with_codebook_cmo` block append. v06의 08 driver 를 그대로 차용 (cell list 만 다름). | Phase 1 통과 후. |
| **(Phase 2-B)** | λ_aux sweep launcher | `experiments/v08_sequential_fl/02_lambda_sweep.py` — 01 driver 를 5 λ × 3 seed 로 외부 launch. v07 의 launcher 와 동일한 패턴, --aux_lambda 만 sweep. | Phase 2-A 통과 후. |
| **(Phase 2-C)** | hr_weight sweep launcher | 동일 패턴. | Phase 2-B 통과 후. |

---

## §6 Outputs

```
outputs/v08_sequential_fl/
├── seed{42,123,7}/
│   ├── V8-Seq-B-CWT-Fixed/
│   │   ├── round_log.jsonl
│   │   ├── intra_round_log.jsonl       (옵션)
│   │   ├── final_state_dict.pt
│   │   └── result.json
│   └── V8-Seq-B-CWT-Perm/
│       └── ...
├── trajectories.npz
├── multiseed_summary.json
└── figures/                              ← gitignored runtime artefact
    ├── F1_round_vs_val_pape.png          # Phase 1
    ├── F1b_round_vs_test_pape.png        # Phase 1
    ├── F2_bytes_vs_val_pape.png          # Phase 1
    ├── F3_drift_vs_round.png             # Phase 1
    ├── F4_intra_round_pape.png           # Phase 1 (옵션)
    ├── F5_order_sensitivity.png          # Phase 1
    ├── F6_codebook_lift.png              # Phase 2-A
    ├── F7_lambda_sweep.png               # Phase 2-B
    └── F8_hr_weight_sweep.png            # Phase 2-C
# 커밋 사본은 papers/v08_draft/figures/
```

`result.json` schema 는 v06과 동일 + `algorithm = "cwt"`, `order ∈ {fixed,
perm}`. `comm_total_bytes.broadcast_cum = 0`, `comm_total_bytes.upload_cum =
R · N · |θ|·4` (fp32).

`outputs/v06_round_dynamics/seed{S}/{V6-Dyn-A_centralised, V6-Dyn-B-FedAvg}/`
는 **수정 금지** — overlay 시 read-only 로 import.

---

## §7 Dependencies

- `dataloader.umass.list_available_apartments`, `filter_valid_apartments`,
  `load_apartment_hourly` — 기존.
- `src/dataloader/per_client_split.py` — v06에서 추가됨; v08 driver 가 동일 seed
  로 호출하면 같은 split.
- `src/fl/base.py` (`apply_state_dict`, `clone_state_dict`, `build_clients`,
  `client_loader`).
- `src/fl/fedavg_aux.py` (`init_backbone_aux`, `_local_step_aux`) — **재사용**;
  cyclic helper 가 같은 local step contract 를 호출.
- `src/fl/round_logger.py` — v06 와 동일, step 2 에서 drift_consecutive 키 한 줄
  추가 + IntraRoundLogger 추가.
- `src/fl/codebook_fl.py` — Phase 2-A 에서만 사용 (v05 helper 그대로).
- `src/utils/metrics.py` (PAPE, HR@k, seven_axis_metrics).
- `src/models/nbeatsx_aux.py`.
- **New**: `src/fl/cyclic_fl.py`.

**No reuse** of: `outputs/v02_fl_8020_ratio/splits/`, v10 households YAML,
v01–v05 backbone checkpoints, any codebook artefact.

---

## §8 Open questions

A. **Server snapshot vs last-client state.** v08 plan은 "round end = last
   client 의 state" 로 정의했는데, 일부 SFL 논문은 "라운드 끝에 last-K client
   의 평균" (mini-FedAvg over tail) 을 server snapshot 으로 쓴다. 이건 본질적
   으로 hybrid (sequential within prefix, parallel over tail) 이고 v08 의 단일
   비교축을 흐리므로 v09+ 로.

B. **Optimizer state relay.** v08 plan은 매 client Adam fresh-init. Yuan et al.
   NeurIPS 2024 는 momentum/Adam state 를 같이 relay 해야 SFL 의 이론적 convergence
   bound 가 깨끗하게 나옴을 보였다. 본 plan 은 v06 의 fresh-init convention 을
   그대로 가져와 비교 가능성을 우선. *Persistent variant* 는 Phase 2 의 V8-Seq-E
   옵션 cell 로 보류.

C. **Permutation seed entanglement.** `order='perm'` 일 때 round t 의 permutation
   이 `seed + 1000 + t` 로 생성되므로 seed 3개 → 3개 permutation sequence. seed
   를 늘려도 *permutation 의 분포 자체* 가 좁게 sample 되므로 "permutation seed"
   를 분리축으로 두고 싶으면 별도 seed 그룹 필요. 본 plan 은 v06 의 `{42, 123, 7}`
   을 그대로 차용.

D. **Per-round eval 의 시점.** v06 은 round end = aggregation 직후 eval. v08은
   round end = last client of round 의 state — 따라서 첫 client 가 막 학습한
   직후 평가하면 last-client bias 가 있다. 평가 자체는 train 가구의 *자기 val
   윈도우* 위에서 이뤄지므로 (cold 가 아님), bias 의 방향은 round t 의 client
   순서에 의존. permutation cell 에서는 round-average bias 가 자동으로 mix-out
   됨. Fixed cell 에서는 last-N 가구 쪽으로 약간 bias.

E. **NF / FM reference.** v06 §G4/G5 에서 NF/FM 은 옵션이었음. v08 도 동일하게
   skip 가능; 필요해지면 같은 114가구 internal val/test 위에서 한 번만 재측정
   하여 *terminal 점* 으로 figure 에 표기.

F. **wall-clock 추산.** CWT 라운드당 compute ≈ 114 클라이언트 × 40 epoch × ~5 ms
   ≈ 22.8s + val eval ~2.7s ≈ ~25.5s/round. 20 라운드 × ~25.5s ≈ ~8.5 분 /
   cell / seed. Phase 1 = 2 cells × 3 seeds × ~8.5 min ≈ ~51 분 직렬 — v06
   Phase 1 (18 runs, ~126 분) 대비 훨씬 가볍다. Phase 2 도 합쳐서 ~2 시간 내.

---

## §9 Conventions

- **Per-seed argparse.** 모든 v08 driver 가 `--seed S` 받음 (memory
  `feedback_argparse_per_seed`).
- **No MLflow.** `result.json` + `round_log.jsonl` + `print`.
- **Output namespacing.** `outputs/v08_sequential_fl/seed{S}/{cell}/`. v06 의
  output 은 read-only.
- **v06 invariant.** backbone / 손실 / λ_aux / hr_weight / batch / lr / wd /
  split / metric / logging schema 는 v06과 bit-equivalent. 단일 차이는
  *aggregation pattern*.
- **Method frozen.** encoder, aux head, peak descriptor 모두 v01 design 그대로.
- **Backbone fresh-init.** 모든 cell × seed 가 새로 학습.
- **No reuse of cold-side artefacts.** v01–v05 의 cold split / checkpoint 사용
  금지.

---

## §10 References

[1] Chang, K. et al. *Distributed deep learning networks among institutions for
    medical imaging*. JAMIA 2018. — CIIL / CWT 원조.

[2] Sheller, M. J. et al. *Federated learning in medicine: facilitating
    multi-institutional collaborations without sharing patient data*. Scientific
    Reports 2020. — CWT vs FedAvg head-to-head 비교 (의료영상).

[3] Yuan, Y., Liang, B., Ma, S. *Convergence analysis of sequential federated
    learning on heterogeneous data*. NeurIPS 2024. — SFL 의 non-IID convergence
    bound 가 parallel FedAvg 와 다름을 증명.

[4] McMahan, H. B. et al. *Communication-Efficient Learning of Deep Networks
    from Decentralized Data*. AISTATS 2017. — parallel FedAvg 원조 (v06/v07
    reference).

[5] `docs/fl_methodologies_fedsgd_vs_fedavg.md` — 본 repo 의 v06 design 문서.
    v08은 같은 framework 위에서 *세 번째 reference point* 를 추가하는 셈.

---

*Last updated: 2026-05-13.*
