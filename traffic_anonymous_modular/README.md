# locm-traffic-anonymous

Anonymized modular implementation for discrete-action traffic signal control with coordination-aware reward reformulation.

## Structure

```text
src/traffic/config.py          Global constants and runtime configuration
src/traffic/utils.py           Utility functions, CSV logging, and physical acceptability check
src/traffic/env.py             Multi-intersection traffic signal environment
src/traffic/discrete_sac.py    Discrete Soft Actor-Critic implementation
src/traffic/trajectory.py      Episode trajectory recording and phase summarization
src/traffic/llm.py             LLM client and semantic phase analyzer
src/traffic/reward.py          LOCM reward engine, phase adapter, and action bias
src/traffic/retriever.py       Historical phase retrieval
src/traffic/outputs.py         Result saving and plotting utilities
src/traffic/train.py           Training entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Optional LLM configuration

The code reads the API key from environment variables and does not hard-code private credentials.

Windows:

```bat
set OPENAI_API_KEY=your_api_key_here
```

Linux/macOS:

```bash
export OPENAI_API_KEY=your_api_key_here
```

If a proxy is required, set `HTTP_PROXY` and `HTTPS_PROXY` in the shell before running.

## Optional real traffic data

The environment attempts to read `Metro_Interstate_Traffic_Volume.csv` from the working directory. If this file is unavailable, it automatically falls back to synthetic traffic profiles.

## Training

```bash
python -m src.traffic.train
```

or on Windows:

```bat
scripts\run_traffic.bat
```

## Reproducibility note

Due to stochastic multi-agent reinforcement learning, random traffic perturbations, and dependency differences, exact point-wise reproduction of every training curve is not expected. The code is intended to reproduce the main experimental trends and relative performance conclusions.
