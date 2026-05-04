import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.agents.replay_buffer import ReplayBuffer
from src.config import CFG, SAC_TARGET_ENTROPY_SCALE, device

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class GaussianActor(nn.Module):
    def __init__(self, s_dim, a_dim, max_action, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, a_dim)
        self.log_std = nn.Linear(hidden_dim, a_dim)
        self.max_action = max_action

        self.apply(weights_init_)
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def sample(self, s):
        mean, log_std = self(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.max_action

        log_prob = normal.log_prob(z) - torch.log(1 - (action / self.max_action).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def deterministic(self, s):
        mean, _ = self(s)
        return torch.tanh(mean) * self.max_action


class QNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim=256):
        super().__init__()
        self.q1_fc1 = nn.Linear(s_dim + a_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        self.q2_fc1 = nn.Linear(s_dim + a_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)

        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = F.relu(self.q1_fc3(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = F.relu(self.q2_fc3(q2))
        q2 = self.q2_out(q2)

        return q1, q2

class SACAgent:
    def __init__(self, s_actor_dim, s_critic_dim, a_dim, max_action):
        self.a_dim = a_dim
        self.max_action = max_action

        self.actor = GaussianActor(s_actor_dim, a_dim, max_action).to(device)
        self.critic = QNetwork(s_critic_dim, a_dim).to(device)
        self.critic_target = QNetwork(s_critic_dim, a_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=CFG.lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=CFG.lr_critic)

        self.auto_entropy = CFG.auto_entropy
        if self.auto_entropy:
            self.target_entropy = -float(a_dim) * SAC_TARGET_ENTROPY_SCALE
            self.log_alpha = torch.tensor(np.log(CFG.alpha_init), requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=CFG.lr_alpha)
        else:
            self.log_alpha = torch.tensor(np.log(CFG.alpha_init), device=device)

        self.buffer = ReplayBuffer(s_actor_dim, s_critic_dim, a_dim, CFG.buffer_size)
        self.train_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, s_actor, deterministic=False):
        s = torch.FloatTensor(s_actor).unsqueeze(0).to(device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(s)
            else:
                action, _, _ = self.actor.sample(s)
        return action.cpu().numpy().flatten()

    def store(self, s_actor, s_critic, action, reward, next_s_actor, next_s_critic, done):
        self.buffer.add(s_actor, s_critic, action, reward, next_s_actor, next_s_critic, done)

    def train(self):
        if self.buffer.size < CFG.warmup_steps:
            return {}

        metrics = {}

        for _ in range(CFG.updates_per_step):
            s_actor, s_critic, action, reward, next_s_actor, next_s_critic, done = self.buffer.sample(CFG.batch_size)

            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_s_actor)
                q1_next, q2_next = self.critic_target(next_s_critic, next_action)
                q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
                q_target = reward + (1 - done) * CFG.gamma * q_next

            q1, q2 = self.critic(s_critic, action)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optim.step()

            new_action, log_prob, _ = self.actor.sample(s_actor)
            q1_new, q2_new = self.critic(s_critic, new_action)
            q_new = torch.min(q1_new, q2_new)

            actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optim.step()

            if self.auto_entropy:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                metrics["alpha_loss"] = alpha_loss.item()

            self.train_steps += 1
            if self.train_steps % CFG.target_update_interval == 0:
                with torch.no_grad():
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(CFG.tau * param.data + (1 - CFG.tau) * target_param.data)

        metrics.update({
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "q_mean": q_new.mean().item(),
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
