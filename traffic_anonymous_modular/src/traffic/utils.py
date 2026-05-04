import csv
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smooth(arr, w=10):
    a = np.asarray(arr, np.float32)
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode="valid")


def plot_curve(arr, title, fname, w=10):
    if len(arr) == 0:
        return
    a = np.asarray(arr, np.float32)
    plt.figure(figsize=(8, 6))
    if len(a) >= w:
        plt.plot(smooth(a, w), label=f"{title} (smoothed {w})")
    plt.plot(a, alpha=0.35, label=f"{title} (raw)")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_team_and_agents(team_series, agents_series, title, ylabel, fname, w=10):
    plt.figure(figsize=(8, 6))
    for k in range(len(agents_series)):
        arr = np.asarray(agents_series[k], np.float32)
        y = smooth(arr, w) if len(arr) >= w else arr
        plt.plot(y, linewidth=1.0, alpha=0.85, label=f"agent{k}")
    arrT = np.asarray(team_series, np.float32)
    yT = smooth(arrT, w) if len(arrT) >= w else arrT
    plt.plot(yT, linewidth=2.8, label="team", zorder=5)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


class SafeCSV:
    def __init__(self, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(header)
        self.f.flush()
        os.fsync(self.f.fileno())

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
        except Exception:
            pass
        self.f.close()


def safe_json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_human_feedback(path="human_feedback.txt.txt", clear_after_read=True):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if clear_after_read and text:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
        return text
    except Exception as e:
        print(f"[Human Feedback] Failed to read {path}: {e}")
        return ""


def basin_is_acceptable(summary, baseline_ref: dict,
                        delay_tol=1.05, overflow_tol=1.10):
    if baseline_ref is None:
        return True
    o = summary.overview if summary and summary.overview else {}
    if not o:
        return True
    cur_delay = float(o.get("avg_delay", 0.0))
    cur_overflow = float(o.get("avg_queue_overflow", 0.0))
    base_delay = float(baseline_ref.get("avg_delay", cur_delay))
    base_overflow = float(baseline_ref.get("avg_queue_overflow",
                                           cur_overflow if cur_overflow is not None else 0.0))
    ok_delay = (cur_delay <= base_delay * delay_tol) if base_delay > 0 else True
    ok_overflow = (cur_overflow <= base_overflow * overflow_tol) if base_overflow > 0 else True
    return bool(ok_delay and ok_overflow)
