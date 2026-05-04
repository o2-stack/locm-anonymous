# locm-anonymous

Anonymized implementation for coordination-aware reward reformulation in multi-agent reinforcement learning.

## Repository structure

```text
src/config.py                    Global constants and runtime configuration
src/envs/power_grid_env.py        Distribution-network simulation environment
src/agents/replay_buffer.py       Replay buffer
src/agents/sac.py                 SAC networks and agent
src/locm/trajectory.py            Trajectory recording and phase summarization
src/locm/knowledge.py             Basin knowledge base
src/locm/llm.py                   LLM client and semantic analyzer
src/locm/reward.py                LOCM reward engine and phase adapter
src/locm/retriever.py             Historical phase retriever
src/outputs/logging_and_plots.py  Logging, serialization, and visualization
src/train.py                      Main training entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Optional LLM configuration

The code never hard-codes API keys. Set the key as an environment variable if the LLM module is used.

Windows:

```bat
set OPENAI_API_KEY=your_api_key_here
```

Linux/macOS:

```bash
export OPENAI_API_KEY=your_api_key_here
```

If no API key is provided, the analyzer falls back to rule-based analysis where applicable.

## Training

```bash
python -m src.train
```

or on Windows:

```bat
scripts\run_train.bat
```

## Reproducibility notes

Due to the stochastic nature of multi-agent reinforcement learning and the numerical behavior of power-flow solvers, exact point-wise reproduction of training curves is not expected across different machines. The scripts are intended to reproduce the main trends and relative performance conclusions reported in the paper.

## Anonymity note

This repository is an anonymized research artifact. It intentionally omits author names, affiliations, local absolute paths, and private credentials.
