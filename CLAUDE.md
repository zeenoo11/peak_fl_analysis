# CLAUDE.md

이 파일은 두 부분으로 구성된다.

- **Part 1** — 일반 LLM 코딩 실수를 줄이기 위한 행동 가이드라인 (프로젝트 무관).
- **Part 2** — 이 저장소에 한정된 프로젝트 운영 정보 (환경 / 데이터 / method invariants / active work / conventions / agent workflow).

충돌 시 Part 1의 행동 원칙이 우선한다.

---

# Part 1 — Behavioral guidelines

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Part 2 — Project orchestration

Orchestration guidance for Claude Code working in this repository. Detail lives in pointed-to files; keep this lean.

## Project context

Peak-aware residential load forecasting on UMass Smart* 2016 hourly data. NBEATSx + peak-aux head + post-hoc Peak-VQ + W5 hybrid Gaussian correction. **Method is frozen at v01**; only protocol/framing changes across versions.

- v01–v05 are past versions, **reference only**. See [`experiments/README.md`](experiments/README.md) (themes + frozen method spec + code conventions) and [`papers/README.md`](papers/README.md) (paper status manifest).
- **v09 (RoundCB) is the current focus**; v06 / v07 / v08 are supporting/reference lines feeding it. See *Active work* below.

## Environment & commands

Python 3.11, `uv`-managed (`pyproject.toml` + `uv.lock`). Editable install of `peak-proto` exposing `src/` modules. PyTorch pinned to a CUDA 12.8 nightly index (Windows-only per `tool.uv.environments`); CPU machines fall back automatically. No tests, no linter, no build step. `pyproject.toml` declares `pythonpath = ["src"]` for pytest, but `tests/` does not exist yet.

Target hardware: RTX 5070 Ti (16 GB VRAM) + 64 GB system RAM. Batch sizes, federated client fan-out, and any in-memory window caching should be sized against the 16 GB VRAM ceiling — assume single-GPU, no model parallelism.

```bash
uv sync
uv run python experiments/v06_round_dynamics/01_<step>.py --seed 42
uv run python -c "from models.nbeatsx import MinimalNBEATSx"   # smoke import
```

## Data

- Raw CSVs: `data/raw/Umass/2016/Apt{N}_2016.csv`  

## Method invariants (load-bearing — do not drift)

For the full method spec see [`experiments/README.md`](experiments/README.md). Hard constraints that apply every session:

- `src/config.py` constants (`INPUT_SIZE=96`, `HORIZON=24`, `D_MODEL=64`, `TRAIN_RATIO=0.7`, `VAL_RATIO=0.1`, `RANDOM_SEED=42`) are hard-coded. Per-apartment z-norm computed on training portion only.

## Active work

**현재 목표: v09 (RoundCB) 실험을 논문화.** 최종 산출물은 컨퍼런스 제출 원고
[`papers/conference_draft/presentation_final.md`](papers/conference_draft/presentation_final.md)
— v09 실험 결과로 이 원고를 완성하는 것이 작업의 종착점이다.

**RoundCB** = Round-wise federated Codebook. backbone hidden `h_g ∈ ℝ⁶⁴`(NBEATSx
generic stack)를 입력으로, codebook을 **R(Representation) / A(Aggregation) /
C(Correction)** 세 축으로 분석하며 v09는 **A·C에 집중**한다 (R은 `h_g`를 그대로 사용,
commitment loss 없음). A = 2-stage hierarchical federated KMeans (client K_local=2
→ server count-weighted M=32, raw `h_g` 미전송), C = cluster-mean offset (CMO)로
추론 시점 1-NN routing 보정 (`ŷ_corr = ŷ_base + α·offset[c*]`, α 기본 1.0).
backbone-agnostic → 5개 FL 알고리즘 전부 −5.7~−6.5 PAPE lift.

진행 중인 버전 디렉토리:

- **v09 (메인)** — `experiments/v09_round_vq_codebook/` · `outputs/v09_round_vq_codebook/` · `plans/v09-01_round_wise_codebook.md` · `papers/v09_draft/` · 방법론 초안 `papers/v09_roundcb_methodology_draft.md`. R=20, 114가구, seeds {42,123,7}. **미해결 TODO (원고 §4-1/§5)**: ① v09 centralised cell 미실행(상한 49.4는 v06 참조값), ② TimesFM 등 TSFM baseline 수치 미확보.
- **v08** — `experiments/v08_round_dynamics_long/` · `outputs/v08_round_dynamics_long/` — v06 mirror를 (E=5, R=150) long-rounds로 옮긴 버전(R/E≈30, FL 표준 영역). post-hoc 1-shot codebook stacking, 3-seed 완료 (F1–F6). v09의 직전 baseline.
- **v06** — `experiments/v06_round_dynamics/` · `outputs/v06_round_dynamics/` · `plans/v06-01_round_dynamics.md` · `papers/v06_draft/v06_round_dynamics.md` (완성, F1–F8). 5개 FL 프로토콜 round-level 동역학 비교 + post-hoc codebook stacking. centralised 상한·K_local elbow 등 v09가 참조하는 ablation 출처.
- **v07** — `experiments/v07_loss_budget_sweeps/` · `outputs/v07_loss_budget_sweeps/` · `plans/v07-01_loss_and_budget_sweeps.md` · `papers/v07_draft/v07_loss_weight_sensitivity.md` (완성, v07-B/C 미포함) — v06 프로토콜 상 λ_aux × hr_weight 민감도. v09 aux-head ablation(§4-2)의 근거.

## Conventions

- Multi-seed: all reported numbers use seeds `{42, 123, 7}`. Aggregate as **mean ± std across seeds**.
- **Per-seed CLI**: scripts take `--seed S` per invocation. Never put the seed loop inside the script.
- Output paths are version-namespaced: `outputs/v{NN}_<theme>/seed{S}/...` (e.g., `outputs/v06_round_dynamics/seed42/...`). Never write to a flat `outputs/`.
- Numbered scripts (`01_*.py`, `02_*.py`, ...) run in order; `sys.path.insert(0, 'src')` at script top is intentional.
