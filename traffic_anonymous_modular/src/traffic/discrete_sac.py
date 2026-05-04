import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.traffic.config import CFG, SAC_TARGET_ENTROPY_SCALE, device

class ReplayBuffer:
    """
    离散 SAC 专用 Buffer
    动作存储为 int64 标量
    """
    def __init__(self, s_actor: int, s_critic: int, capacity: int = 500_000):
        self.max = capacity
        self.ptr = 0
        self.size = 0
        self.SA  = np.zeros((capacity, s_actor),  np.float32)
        self.SC  = np.zeros((capacity, s_critic), np.float32)
        self.A   = np.zeros(capacity,             np.int64)
        self.R   = np.zeros(capacity,             np.float32)
        self.NA  = np.zeros((capacity, s_actor),  np.float32)
        self.NC  = np.zeros((capacity, s_critic), np.float32)
        self.D   = np.zeros(capacity,             np.float32)

    def add(self, sa, sc, a, r, na, nc, d):
        i = self.ptr
        self.SA[i]  = sa
        self.SC[i]  = sc
        self.A[i]   = int(a)
        self.R[i]   = float(r)
        self.NA[i]  = na
        self.NC[i]  = nc
        self.D[i]   = float(d)
        self.ptr = (i + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.SA[idx]).to(device),
            torch.FloatTensor(self.SC[idx]).to(device),
            torch.LongTensor(self.A[idx]).to(device),
            torch.FloatTensor(self.R[idx]).to(device),
            torch.FloatTensor(self.NA[idx]).to(device),
            torch.FloatTensor(self.NC[idx]).to(device),
            torch.FloatTensor(self.D[idx]).to(device),
        )


# ==============================================================================
# Discrete SAC Networks
# ==============================================================================
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class DiscreteSACActor(nn.Module):
    """
    离散 SAC 的 Actor：输出每个离散动作的概率分布
    输出 logits → softmax → Categorical 采样
    """
    def __init__(self, s_dim, n_actions, hidden_dim=256):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logits_head = nn.Linear(hidden_dim, n_actions)
        self.apply(weights_init_)
        nn.init.uniform_(self.logits_head.weight, -3e-3, 3e-3)
        nn.init.constant_(self.logits_head.bias, 0)

    def forward(self, s):
        x = self.net(s)
        logits = self.logits_head(x)
        return logits

    def get_action_probs(self, s):
        logits = self.forward(s)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        return probs, log_probs

    def sample(self, s):
        probs, log_probs = self.get_action_probs(s)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1))
        return action, action_log_prob, probs

    def deterministic(self, s):
        logits = self.forward(s)
        return logits.argmax(dim=-1)


class DiscreteQNetwork(nn.Module):
    """
    离散 SAC 的 Q 网络：Q(s) → R^{n_actions}
    只输入 state，输出所有动作的 Q 值
    保持 Double-Q 结构
    """
    def __init__(self, s_dim, n_actions, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.apply(weights_init_)

    def forward(self, s):
        return self.q1(s), self.q2(s)

class SACAgent:
    """
    离散 SAC Agent（Christodoulou 2019）

    核心区别（vs 连续 SAC）：
    1. Actor 输出 Categorical 分布
    2. Critic 输出所有动作的 Q 值（不需要动作作为输入）
    3. Critic/Actor loss 用"期望形式"（对所有动作求期望）
    4. 目标熵用 -log(1/n_actions) * scale
    """

    def __init__(self, s_actor_dim, s_critic_dim, n_actions):
        self.n_actions = n_actions

        self.actor = DiscreteSACActor(s_actor_dim, n_actions).to(device)
        self.critic = DiscreteQNetwork(s_critic_dim, n_actions).to(device)
        self.critic_target = DiscreteQNetwork(s_critic_dim, n_actions).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=CFG.lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=CFG.lr_critic)

        self.auto_entropy = CFG.auto_entropy
        if self.auto_entropy:
            self.target_entropy = -np.log(1.0 / n_actions) * SAC_TARGET_ENTROPY_SCALE
            self.log_alpha = torch.tensor(
                np.log(CFG.alpha_init), requires_grad=True, device=device
            )
            self.alpha_optim = optim.Adam([self.log_alpha], lr=CFG.lr_alpha)
        else:
            self.log_alpha = torch.tensor(np.log(CFG.alpha_init), device=device)

        self.buffer = ReplayBuffer(s_actor_dim, s_critic_dim, CFG.buffer_size)
        self.train_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, s_actor, deterministic=False):
        s = torch.FloatTensor(s_actor).unsqueeze(0).to(device)
        with torch.no_grad():
            if deterministic:
                return int(self.actor.deterministic(s).item())
            else:
                action, _, _ = self.actor.sample(s)
                return int(action.item())

    def store(self, s_actor, s_critic, action, reward, next_s_actor, next_s_critic, done):
        self.buffer.add(s_actor, s_critic, int(action), float(reward),
                        next_s_actor, next_s_critic, float(done))

    def train(self):
        if self.buffer.size < CFG.warmup_steps:
            return {}

        metrics = {}
        for _ in range(CFG.updates_per_step):
            s_actor, s_critic, action, reward, next_s_actor, next_s_critic, done = \
                self.buffer.sample(CFG.batch_size)

            # ---- Critic Loss ----
            with torch.no_grad():
                next_probs, next_log_probs = self.actor.get_action_probs(next_s_actor)
                q1_next, q2_next = self.critic_target(next_s_critic)
                q_next = torch.min(q1_next, q2_next)
                # V(s') = Σ_a π(a|s') * [Q(s',a) - α * log π(a|s')]
                v_next = (next_probs * (q_next - self.alpha.detach() * next_log_probs)).sum(dim=-1)
                q_target = reward + (1 - done) * CFG.gamma * v_next

            q1, q2 = self.critic(s_critic)
            q1_a = q1.gather(1, action.unsqueeze(-1)).squeeze(-1)
            q2_a = q2.gather(1, action.unsqueeze(-1)).squeeze(-1)
            critic_loss = F.mse_loss(q1_a, q_target) + F.mse_loss(q2_a, q_target)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optim.step()

            # ---- Actor Loss ----
            probs, log_probs = self.actor.get_action_probs(s_actor)

            with torch.no_grad():
                q1_curr, q2_curr = self.critic(s_critic)
                q_curr = torch.min(q1_curr, q2_curr)

            # L = Σ_a π(a|s) * [α * log π(a|s) - Q(s,a)]
            actor_loss = (probs * (self.alpha.detach() * log_probs - q_curr)).sum(dim=-1).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optim.step()

            # ---- Alpha Loss ----
            if self.auto_entropy:
                entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
                alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                with torch.no_grad():
                    # 限制 log_alpha 最大不超过 1.0 (即 Alpha 约 2.7)
                    # 最小不低于 -5.0 (即 Alpha 约 0.006)
                    self.log_alpha.clamp_(min=-5.0, max=1.0)
                metrics["alpha_loss"] = alpha_loss.item()
                metrics["entropy"] = entropy.mean().item()

            # ---- Soft Update ----
            self.train_steps += 1
            if self.train_steps % CFG.target_update_interval == 0:
                with torch.no_grad():
                    for p, tp in zip(self.critic.parameters(),
                                     self.critic_target.parameters()):
                        tp.data.copy_(CFG.tau * p.data + (1 - CFG.tau) * tp.data)

        metrics.update({
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "q_mean": q_curr.mean().item(),
        })
        return metrics

    def save(self, path):
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "train_steps": self.train_steps,
        }
        if self.auto_entropy:
            state["log_alpha"] = self.log_alpha.detach().cpu()
            state["alpha_optim"] = self.alpha_optim.state_dict()
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        self.train_steps = state.get("train_steps", 0)
        if self.auto_entropy and "log_alpha" in state:
            self.log_alpha.data.copy_(state["log_alpha"].to(device))
        if self.auto_entropy and "alpha_optim" in state:
            self.alpha_optim.load_state_dict(state["alpha_optim"])
