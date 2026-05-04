import os
import time
from dataclasses import dataclass

# Keep numerical libraries stable on some local machines.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Optional proxy configuration. Set these environment variables outside the code if needed.
# Example on Windows: set HTTP_PROXY=http://127.0.0.1:7897

import torch

# ==============================================================================
# Global configuration
# ==============================================================================
ROUND_EPISODES = [1000, 1000]
MAX_STEPS = 96
SEED = 5
SMOOTH = 10
PLOT_INTERVAL = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-agent power-grid configuration
N_AGENTS = 3
AGENT_BUSES = (17, 19, 21)
A_SCALE = 4.0
SOC_CAP = 11.11
SOC_LIMIT = 0.5

# SAC hyperparameters
SAC_ALPHA_INIT = 0.2
SAC_AUTO_ENTROPY = True
SAC_TARGET_ENTROPY_SCALE = 1.0

# LOCM phase configuration
PHASE_EPISODES = 100
NUM_ROUNDS = 2

# LLM configuration. Never hard-code API keys in source code.
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
ROOT_DIR = os.path.join("results", RUN_ID)
GLOBAL_MEMORY_DIR = os.path.join("results", "global_memory")
GLOBAL_KNOWLEDGE_PATH = os.path.join(GLOBAL_MEMORY_DIR, "global_knowledge.json")
GLOBAL_PHASE_ARCHIVE_PATH = os.path.join(GLOBAL_MEMORY_DIR, "global_phase_archive.json")


def ensure_output_dirs() -> None:
    os.makedirs(ROOT_DIR, exist_ok=True)
    os.makedirs(GLOBAL_MEMORY_DIR, exist_ok=True)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 500_000
    warmup_steps: int = 5000
    updates_per_step: int = 1
    target_update_interval: int = 1
    auto_entropy: bool = True
    alpha_init: float = 0.2


CFG = SACConfig()
