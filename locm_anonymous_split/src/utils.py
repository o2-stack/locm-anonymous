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
    """
    从文本文件读取人工反馈。
    如果文件不存在，返回空字符串。
    如果 clear_after_read=True，读取后清空文件内容。
    """
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

# ==================== 替换为这段新代码 ====================
def basin_is_acceptable(summary, baseline_ref: dict,
                        vbias_tol=1.05, ploss_tol=1.10):
    """
    判断当前phase是否物理上可接受：
    - (方案B专用) 移除了 reward 检查，因为引导期完全优化纯行为奖励或拉格朗日惩罚
    - 仅对比纯物理指标
    """
    if baseline_ref is None:
        return True

    o = summary.overview if summary and summary.overview else {}
    if not o:
        return True

    cur_vbias = float(o.get("avg_vbias", 0.0))
    cur_ploss = float(o.get("avg_p_loss", 0.0)) if "avg_p_loss" in o else None

    base_vbias = float(baseline_ref.get("avg_vbias", cur_vbias))
    base_ploss = float(baseline_ref.get("avg_p_loss", cur_ploss if cur_ploss is not None else 0.0))

    ok_vbias = (cur_vbias <= base_vbias * vbias_tol) if base_vbias > 0 else True

    if cur_ploss is not None and base_ploss > 0:
        ok_ploss = (cur_ploss <= base_ploss * ploss_tol)
    else:
        ok_ploss = True

    return bool(ok_vbias and ok_ploss)
