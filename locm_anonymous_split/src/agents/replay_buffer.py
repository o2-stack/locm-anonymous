import numpy as np
import torch

from src.config import device

class ReplayBuffer:
    def __init__(self, s_actor: int, s_critic: int, a: int, capacity: int = 500_000):
        self.max = capacity
        self.ptr = 0
        self.size = 0
        self.SA = np.zeros((capacity, s_actor), np.float32)
        self.SC = np.zeros((capacity, s_critic), np.float32)
        self.A = np.zeros((capacity, a), np.float32)
        self.R = np.zeros((capacity, 1), np.float32)
        self.NA = np.zeros((capacity, s_actor), np.float32)
        self.NC = np.zeros((capacity, s_critic), np.float32)
        self.D = np.zeros((capacity, 1), np.float32)

    def add(self, sa, sc, a, r, na, nc, d):
        i = self.ptr
        self.SA[i] = sa
        self.SC[i] = sc
        self.A[i] = a
        self.R[i] = r
        self.NA[i] = na
        self.NC[i] = nc
        self.D[i] = d
        self.ptr = (i + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.SA[idx]).to(device),
            torch.FloatTensor(self.SC[idx]).to(device),
            torch.FloatTensor(self.A[idx]).to(device),
            torch.FloatTensor(self.R[idx]).to(device),
            torch.FloatTensor(self.NA[idx]).to(device),
            torch.FloatTensor(self.NC[idx]).to(device),
            torch.FloatTensor(self.D[idx]).to(device)
        )
