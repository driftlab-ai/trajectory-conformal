# trajectory-conformal

Trajectory-level conformal calibration for multi-step LLM agents: split-conformal and level-schedule calibrators, closed-form coverage bounds, drift diagnostics, and a ReAct trajectory-generation and scoring pipeline.

## Install

Python 3.11 or 3.12.

```
pip install -r requirements.txt
```

## Run

```
export HF_TOKEN=...                             # Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct and GAIA are gated on the Hugging Face Hub
python main.py bounds                           # closed-form bounds
python main.py synthetic                        # calibrators on two-state kernel trajectories
python main.py run-all -c configs/hotpotqa.yaml
python main.py run-all -c configs/musique.yaml
python main.py run-all -c configs/financebench.yaml
python main.py run-all -c configs/gaia.yaml
python main.py run-all -c configs/workarena.yaml
python main.py horizon -c configs/horizon_hotpotqa.yaml
python main.py mechanism -c configs/mechanism_gaia.yaml
python main.py run-all -c configs/hotpotqa.yaml -c configs/prm_qwen_judge.yaml --model llama8b
python main.py run-all -c configs/hotpotqa.yaml -c configs/llama70b.yaml
python main.py run-all -c configs/musique.yaml -c configs/llama70b.yaml
python main.py run-all -c configs/financebench.yaml -c configs/llama70b.yaml
python main.py run-all -c configs/gaia.yaml -c configs/llama70b.yaml
python main.py run-all -c configs/workarena.yaml -c configs/llama70b.yaml
```

## Layout

```
main.py         entry point (bounds, synthetic, experiment stages)
wrapper/        calibrators, bounds, drift diagnostics
agents/         LM wrapper, ReAct agent, environments, data, scores, PRM
evaluation/     metrics and experiment stages
configs/        one YAML per experiment plus base
prompts/        ReAct prompt templates
```

| File | Purpose |
|---|---|
| `wrapper/conformal.py` | Split-conformal thresholds, level schedules (polynomial, LIL), calibrators (per-step, Bonferroni, CRC, ACI, TC-Wrapper, TC-Wrapper-DC), trajectory exceedance events |
| `wrapper/theory.py` | Closed-form bounds, two-state kernels, continuous score lift |
| `wrapper/drift.py` | Binned TV drift, conditional drift, permutation and i.i.d. nulls, jackknife, split-bank tau, initial-state TV |
| `agents/agent.py` | Hugging Face LM wrapper, ReAct loop with per-step scores and context reset, LLM judge |
| `agents/envs.py` | BM25 retrieval environment, chained horizon environment, randomized-tool wrapper, GAIA, WorkArena |
| `agents/data.py` | Benchmark loaders, pool splits, with-replacement draws |
| `agents/scores.py` | Semantic-entropy clustering, self-evaluation score |
| `agents/prm.py` | Frozen-encoder embeddings, logistic PRM head, judge labelling |
| `evaluation/metrics.py` | Miscoverage, certified prefix, bootstrap, Holm, Wilson, Pearson, ECE, density |
| `evaluation/experiments.py` | Config loading and experiment stages |

## Configuration

| Key | Values |
|---|---|
| `eps` | `0.10` (nominal trajectory miscoverage) |
| `T`, `max_steps` | evaluation horizon and agent step cap, set per benchmark |
| `n_cal`, `n_test` | `1000` calibration scores per seed, `500` test trajectories |
| `n_seeds` | `10` calibration seeds (`3` in `llama70b.yaml` and `horizon_hotpotqa.yaml`) |
| `agent.max_new_tokens`, `agent.max_prompt_tokens` | `128`, `4096` (a longer prompt is left-truncated; the share of truncated steps is reported) |
| `semant.n_samples`, `semant.temperature` | `5` samples at `0.7` per step |
| `semant.entail_threshold`, `semant.nli_max_length` | `0.5`, `512` |
| `gamma_aci` | `0.05` (ACI step size) |
| `bootstrap_resamples` | `10000` |
| `m_drift`, `n_prm_bank` | `200` drift-bank and `1000` PRM-bank trajectories |
| `gate_eps_list` | `[0.05, 0.10, 0.20]` (levels of the calibratability gate) |
| `perturbation_factors` | `[0.5, 1.0, 1.5]` (drift multipliers in the drift-corrected schedule) |
| `pool_limit` | `0` (whole pool) or `235` (WorkArena) |
| `models[].revision` | `null` (default Hugging Face revision) |
| `device_map` | `auto` in `llama70b.yaml` (weights spread over the visible GPUs); unset elsewhere |
| `model_average_se` | `mean_of_model_ses` or `pooled` (SE of the two-model mean) |
