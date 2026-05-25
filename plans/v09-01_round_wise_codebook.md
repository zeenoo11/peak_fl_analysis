# v09-01 — Round-wise federated codebook learning (FedVQ)

> Successor to `v08_round_dynamics_long` (`experiments/v08_round_dynamics_long/`,
> `outputs/v08_round_dynamics_long/`). v08은 v06과 동일한 **post-hoc 1-shot
> codebook**을 (E=5, R=150) backbone 위에 stacking했다. v09는 codebook을
> **라운드 단위로 federated하게 학습**하는 새 mechanism으로 교체한다 —
> backbone과 codebook이 매 라운드 co-train 한다.

> **Status (2026-05-18).** Plan only. v08 결과는 baseline으로 보존
> (`outputs/v08_round_dynamics_long/`). v09는 동일 setup (E=5, R=150,
> NBEATSxAux, 114 가구 per-client 70/10/20 split, seeds {42, 123, 7}) 위에
> 새 codebook component를 추가한다.

---

## 동기

v08 결과 (3-seed, codebook stacking 적용):

| Cell | BEFORE PAPE | AFTER PAPE | ΔPAPE |
|---|---|---|---|
| Centralised | 49.43 | 44.92 | −4.51 |
| FedAvg | 52.36 | 46.02 | −6.33 |
| FedProx | 52.25 | 46.01 | −6.24 |
| FedRep | 50.54 | 45.69 | −4.85 |
| Ditto | 53.13 | 46.05 | **−7.08** |
| FedProto | 52.46 | 46.02 | −6.44 |

핵심 비판 (2026-05-18 토론):

1. **Codebook이 실제로 학습되지 않음** — KMeans cluster center + cluster mean
   training residual. Classical statistics, no gradient flow.
2. **Backbone과 codebook이 분리됨** — backbone freeze 후 post-hoc stacking이라
   backbone이 codebook-friendly h_g 구조로 학습되지 않음.
3. **"왜 v09인가" reviewer 질문 가능** — v08과 단순히 (E, R) 변경 후 같은
   codebook이면 incremental rerun으로 reject 위험.

가설: **codebook을 federated하게 학습**하면 backbone이 cluster-friendly 한
h_g 구조로 co-train 되어 final PAPE가 개선된다. Deep clustering 문헌
(arxiv:1704.06327, arxiv:2210.04142)의 합의: codebook-task 결합도가 강한
경우 joint training > post-hoc.

예상 효과:
- v08 codebook (post-hoc): test PAPE 45.7–46.0
- **v09 codebook (round-wise): test PAPE 44.0–45.0 (ΔPAPE −1 ~ −2)**
- MAE는 유사 (cluster mean residual mechanism 동일)

---

## Goals

G1. **Round-wise federated codebook 학습**을 v08 (E=5, R=150) backbone 위에
    구현. matched: NBEATSxAux 본체 + 모든 v06 invariant (λ_aux=0.3,
    hr_weight=0.1, INPUT_SIZE=96, HORIZON=24, M=32, K_local=2).
G2. **5 FL 알고리즘 + centralised** 동일하게 적용. 단일 codebook 학습
    mechanism이 모든 backbone에 직교적임 검증.
G3. **TAR mitigation** (mass-weighted aggregation + EMA blending + 명시적
    limitation 문단)으로 학회 short paper 수준 reviewer 방어.
G4. **v08 baseline 보존** + v09와 직접 비교 figure 작성. paper의 핵심 ablation
    matrix.

Non-goal:
- Hungarian alignment / per-cluster permutation 처리 (broadcast-init으로
  무력화). 향후 cell이 round 후반 심각하게 drift 시 future work.
- DP noise / secure aggregation. limitation 문단 명시, future work.
- Codebook size M sweep. M=32 fixed (v06 invariant).

---

## Method

### 1. VQ codebook module (`src/fl/vq_codebook.py`)

```python
class VQCodebook(nn.Module):
    """VQ-VAE EMA codebook (van den Oord 2017 §3.3) for federated training.

    Buffers (state_dict 포함, FedAvg 대상 아님 — server aggregation 별도):
        embedding ∈ R^(M, D)      — current codebook entries
        cluster_size ∈ R^M        — EMA cluster usage (sum of counts)
        ema_w ∈ R^(M, D)          — EMA cluster sum (Σ h_g per cluster)

    Forward (frozen during forward, EMA updates explicit):
        h_g → c_star = argmin_c ||h_g - embedding[c]||²
        q_h_g = embedding[c_star]  (straight-through gradient through h_g)

    Commitment loss (returned, added to backbone training loss):
        L_commit = β · ||h_g - sg(q_h_g)||²    (β=0.25, van den Oord 표준)

    EMA codebook update (local, called explicitly after forward):
        cluster_size = γ · cluster_size + (1-γ) · count_in_batch
        ema_w        = γ · ema_w        + (1-γ) · sum_h_g_per_cluster
        embedding[c] = ema_w[c] / max(cluster_size[c], 1e-5)
        γ = 0.95 (TAR mitigation)

    Dead-code respawn (every K=10 rounds at round-end):
        if cluster_size[c] < N_min=5:
            embedding[c] = random_sample_from(active_h_g)
```

### 2. Modified backbone wrapper (`src/models/nbeatsx_vq.py`)

```python
class NBEATSxAuxVQ(nn.Module):
    """NBEATSxAux + VQCodebook on h_generic.

    forward(x) -> dict:
        y_hat:    (B, H)      — forecast (unchanged from NBEATSxAux)
        h_generic:(B, D=64)   — generic stack hidden
        c_star:   (B,)        — codebook entry indices (NEW)
        q_h_g:    (B, D)      — quantized h_g (NEW)
        L_commit: scalar      — commitment loss (NEW)
        aux:      (a_hat, h_hat) — peak-aux outputs (unchanged)
    """
```

### 3. Federated round structure (`src/fl/round_aux_vq.py`)

```
매 라운드 r ∈ {1..150}:

  [Server → all clients] broadcast:
      backbone_state, codebook_embedding, cluster_size, ema_w

  [Each client i, local training 5 epochs]:
      For each batch:
          forward → y_hat, h_g, c_star, q_h_g, L_commit
          L_total = MAE(y_hat, y) + 0.3·L_aux + L_commit
          backward + opt.step()  (backbone only — codebook is buffer)

          # EMA codebook update (local, explicit, no gradient)
          codebook.ema_update(h_g, c_star)

      Round-end client → server upload:
          backbone_delta
          codebook_delta = (embedding_local_end - embedding_round_start)
          cluster_size_local
          ema_w_local

  [Server aggregation]:
      backbone_global = FedAvg(backbone_local)
      For c ∈ [0, M):
          total_mass = Σ_i cluster_size_local_i[c]
          if total_mass > 0:
              embedding[c]      = 0.95·embedding[c]_prev + 0.05·(Σ_i cluster_size_i[c] · embedding_i[c] / total_mass)
              cluster_size[c]   = Σ_i cluster_size_i[c]
              ema_w[c]          = Σ_i ema_w_i[c]
          else:
              # dead cluster — keep previous
              pass

      Every K=10 rounds:
          dead_idx = {c : cluster_size[c] < N_min=5}
          if dead_idx:
              sample h_g from random active client's last batch
              embedding[dead_idx] = sampled h_g
              cluster_size[dead_idx] = 1.0
              ema_w[dead_idx] = embedding[dead_idx]
```

### 4. Test-time correction (unchanged from v06/v08)

```
ŷ_corr = ŷ_base + α_v0 · offsets[c*]
```

`offsets[c]`는 학습이 끝난 시점에 마지막으로 cluster-mass weighted aggregation으로
계산된 cluster-mean training residual. v08의 federated_residual_offsets() 함수
재사용 가능. α_v0=1.0 (v06/v08 invariant).

---

## Experimental setup

| Item | Value | 비고 |
|---|---|---|
| Backbone | NBEATSxAux | v06/v08 동일 |
| λ_aux, hr_weight | 0.3, 0.1 | v06 default 유지 |
| INPUT_SIZE / HORIZON | 96 / 24 | invariant |
| M, K_local | 32, 2 | v06 invariant |
| **Commitment β** | **0.25** | VQ-VAE 표준 (van den Oord 2017) |
| **EMA γ** | **0.95** | TAR mitigation (v06 v05 K_local 선택 logic 참고) |
| **Dead-code N_min** | **5** | v08 utilization 1.0 baseline |
| **Respawn period** | **10 rounds** | round-by-round 잡음 회피 |
| **Aggregation EMA blend** | **0.95 (prev) + 0.05 (new)** | TAR mitigation |
| Rounds | 150 | v08 invariant |
| Local epochs | 5 | v08 invariant |
| Batch | 512 | invariant |
| LR / weight_decay | 1e-3 / 1e-5 | invariant |
| FL 5종 | FedAvg, FedProx, FedRep, Ditto, FedProto | v06 invariant |
| Centralised | epochs=40 | v06 invariant (codebook은 centralised에서도 동일 학습) |
| Seeds | {42, 123, 7} | invariant |

Output namespace: `outputs/v09_fedvq/seed{S}/{cell}/`

---

## Phase 분리

### Phase 1 — main FedVQ training (priority)

18 runs = 6 cells × 3 seeds:

| Cell | algo |
|---|---|
| V9-FedVQ-A_centralised | centralised pooled SGD with VQ |
| V9-FedVQ-B-FedAvg | FedAvg backbone + FedVQ codebook |
| V9-FedVQ-B-FedProx | FedProx backbone + FedVQ codebook |
| V9-FedVQ-B-FedRep | FedRep backbone + FedVQ codebook |
| V9-FedVQ-B-Ditto | Ditto backbone + FedVQ codebook |
| V9-FedVQ-B-FedProto | FedProto backbone + FedVQ codebook |

산출: `result.json` (terminal PAPE/HR/MAE/MSE), `round_log.jsonl` (per-round
trajectory + codebook diagnostics: utilization, perplexity, dead clusters),
`final_state_dict.pt`, `final_codebook.pt` (embedding + ema_w + cluster_size).

### Phase 2 — ablation grid (선택, page budget 봐서)

| Cell suffix | 의미 | priority |
|---|---|---|
| `-noVQ` | commitment loss β=0 (codebook은 학습되지만 backbone에 영향 X = Option C 의 naive 버전) | high — user hypothesis 검증 |
| `-noRespawn` | dead-code respawn 끄기 | mid — mechanism component ablation |
| `-naiveAvg` | cluster-mass weighting 없이 단순 FedAvg | mid |
| `-postHoc` | v08 그대로 (이미 있음, 비교용) | reuse |

---

## Comparison axes (paper의 핵심 표)

| Axis | v08 (post-hoc) | v09 (FedVQ) | 가설 |
|---|---|---|---|
| Final PAPE | 45.7–46.0 | **44.0–45.0** | ΔPAPE −1 ~ −2 stat. significant |
| h_g utilization | 1.0 (post-hoc) | 1.0 (학습 중 respawn) | 둘 다 full |
| h_g perplexity | 26.2 (post-hoc) | **>27 예상** | commitment loss가 더 균등 분포 유도 |
| Codebook trajectory | 없음 (single-shot) | **per-round** | round-wise plot 가능 (paper figure F9) |
| Wall-clock | Phase1 ~5h + Phase2 ~3min | **~5.5h** | codebook overhead ~10% |
| Communication | normal + Phase2 0.08% | normal + ~0.1% | codebook 8KB/round + cluster_size 256B |
| TAR surface | 매우 작음 (single-shot) | **moderate (round-wise broadcast)** | mass-weighting + EMA로 mitigation |

---

## TAR mitigation 명시 (paper limitation 문단 초안)

> "v09는 매 라운드 codebook을 federated하게 학습한다. 이 mechanism은 v05/v06/v08의
> single-shot federated KMeans 대비 TAR (Training-time Attack on Representations,
> arxiv:2511.07073) 공격면이 늘어난다. 본 연구는 세 가지 mitigation으로 위험을
> 부분 완화한다: (i) cluster-mass weighted aggregation으로 client당 entry-level
> contribution을 sparse하게 노출, (ii) EMA blending (γ=0.95)으로 매 라운드 delta를
> 작게 유지, (iii) raw h_g는 가구를 벗어나지 않으며 (centroid sum/count만 upload)
> 학습 단계에서 sum/count가 raw representation을 직접 노출하지 않는다. 다만 강한
> adversarial model 하에서 secure aggregation 또는 DP noise 추가가 권장되며 이는
> future work이다."

---

## Build order

1. **`src/fl/vq_codebook.py`** — VQCodebook nn.Module + EMA + respawn (pytest 포함)
2. **`src/models/nbeatsx_vq.py`** — NBEATSxAux wrapper, h_generic에 codebook 결합
3. **`src/fl/round_aux_vq.py`** — 기존 round_aux.py 확장, codebook broadcast/aggregation hook
4. **`experiments/v09_fedvq/`** scaffolding (v08 mirror):
   - `01_centralised.py` (centralised with VQ codebook)
   - `02_fl_vq_dynamics.py` (5 FL algos with VQ codebook)
   - `06_aggregate.py` (Phase 1 multi-seed)
   - `07_make_figures.py` (F1/F1b/F1c/F2/F3 + F9 codebook trajectory)
   - `08_apply_correction.py` (test-time CMO correction, post-training)
5. **Smoke run** — FedRep seed 42 (~25분), trajectory + codebook diagnostics 확인
6. **Full Phase 1 sweep** — 18 runs, 병렬화로 ~5-6h
7. **Aggregate + figures + paper draft 작성**

---

## Success criteria

- Phase 1 18 runs 모두 정상 종료 (no NaN, no codebook collapse)
- v09 codebook utilization ≥ 0.95, perplexity ≥ 25, dead clusters ≤ 2
- 적어도 3개 FL cell에서 v08 대비 ΔPAPE < −0.5 (3-seed mean) — user hypothesis 반증
- 모든 cell에서 v08 대비 PAPE 악화 없음 (≥ same)
- Wall-clock < 7h (v08 대비 ≤ +40% 허용)

Success criteria 미달 시: codebook component를 단계적으로 disable해서 어떤
component가 문제인지 isolate (β=0.25 → 0.1, γ=0.95 → 0.99, no commitment loss
fallback 등).

---

## Reproducibility

```bash
# Phase 1 (single seed × single cell example)
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" `
    "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v09_fedvq/02_fl_vq_dynamics.py" `
    --algorithm fedavg --seed 42 --local_epochs 5 --rounds 150

# 18 runs full sweep — multi-seed launcher 는 experiments/v09_fedvq/Readme.md
```

Seeds: {42, 123, 7}. Per-seed argparse (CLAUDE.md feedback_argparse_per_seed).

---

## Cross-references

- v06 plan: `plans/v06-01_round_dynamics.md`
- v06 paper: `papers/v06_draft/v06_round_dynamics.md` (Phase 1 + Phase 2 baseline)
- v08 results: `outputs/v08_round_dynamics_long/multiseed_summary.json`
                + `codebook_lift_summary.json` (post-hoc baseline)
- TAR attack reference: arxiv:2511.07073
- VQ-VAE: arxiv:1711.00937 (EMA + commitment loss)
- FedProto: arxiv:2105.00243 (cluster-mass weighted prototype aggregation)
- Dead-code respawn: arxiv:2411.16550
- Deep clustering (joint > post-hoc): arxiv:1704.06327, arxiv:2012.03740,
  arxiv:2210.04142 (survey)
