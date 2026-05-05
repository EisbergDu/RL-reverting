import argparse
import math
import multiprocessing as mp
import random
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEVICE = torch.device("cpu")
pd.set_option("display.max_columns", 80)
plt.rcParams["figure.dpi"] = 120


@dataclass
class MarketConfig:
    kappa: float = 0.2
    sigma: float = 0.1
    lam: float = 0.01
    gamma: float = 0.95
    a_max: float = 10.0
    T: int = 80
    log_s_clip: Optional[float] = None


@dataclass
class Budget:
    hidden: int
    n_eval: int
    sequential_path_len: int
    ppo_steps: int
    ppo_rollout: int
    ppo_epochs: int
    td3_steps: int
    sac_steps: int
    warmup_steps: int
    batch_size: int


BASE_CFG = MarketConfig()

# High-budget setup close to notebook FULL mode.
MAIN_BUDGET = Budget(
    hidden=64,
    n_eval=1000,
    sequential_path_len=100_000,
    ppo_steps=60_000,
    ppo_rollout=2048,
    ppo_epochs=10,
    td3_steps=60_000,
    sac_steps=60_000,
    warmup_steps=2000,
    batch_size=256,
)

PARAMETER_SET_NAMES = (
    "baseline",
    "slow_reversion",
    "high_volatility",
    "strong_regularization",
)
SEEDS = (0, 1)
REWARD_TYPES = ("R1", "R2")
SETTINGS = ("oracle", "sequential")
ALGOS = ("PPO", "TD3", "SAC")

PLOT_CHECKPOINT_TASKS = {
    ("baseline", 0, "R1", "oracle", "PPO"),
    ("baseline", 0, "R1", "oracle", "TD3"),
    ("baseline", 0, "R1", "oracle", "SAC"),
    ("baseline", 0, "R2", "oracle", "PPO"),
}
PLOT_LOG_TASKS = {
    ("baseline", 0, "R1", "oracle", "PPO"),
    ("baseline", 0, "R1", "oracle", "TD3"),
    ("baseline", 0, "R1", "oracle", "SAC"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _task_stem(param_label: str, reward_type: str, setting: str, algo: str, seed: int) -> str:
    return f"{param_label}__{reward_type}__{setting}__{algo}__seed{seed}"


class MeanRevertingStockEnv:
    """Oracle environment for the assignment's mean-reverting stock model."""

    def __init__(self, cfg: MarketConfig, reward_type: str = "R1", seed: int = 0):
        assert reward_type in {"R1", "R2"}
        self.cfg = cfg
        self.reward_type = reward_type
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def state_dim(self) -> int:
        return 2 if self.reward_type == "R1" else 3

    def _state(self) -> np.ndarray:
        state = [self.log_s, self.L]
        if self.reward_type == "R2":
            state.append(self.a_prev)
        return np.array(state, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.t = 0
        self.log_s = 0.0
        self.S = 1.0
        self.L = 0.0
        self.a_prev = 0.0
        return self._state()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        cfg = self.cfg
        A = float(np.clip(action, -cfg.a_max, cfg.a_max))

        L_next = (1.0 - cfg.kappa) * self.L + cfg.sigma * self.rng.normal()
        log_s_next = self.log_s + L_next
        if cfg.log_s_clip is not None:
            log_s_next = float(np.clip(log_s_next, -cfg.log_s_clip, cfg.log_s_clip))
            L_next = log_s_next - self.log_s
        S_next = float(np.exp(log_s_next))

        pnl = A * (S_next - self.S)
        if self.reward_type == "R1":
            reward = pnl - cfg.lam * A**2
        else:
            reward = pnl - cfg.lam * (A - self.a_prev) ** 2 * self.S

        info = {
            "S_t": self.S,
            "S_next": S_next,
            "L_next": L_next,
            "action": A,
            "pnl": pnl,
            "a_prev": self.a_prev,
        }
        self.log_s = log_s_next
        self.S = S_next
        self.L = L_next
        self.a_prev = A
        self.t += 1
        done = self.t >= cfg.T
        return self._state(), float(reward), done, info


def simulate_price_path(cfg: MarketConfig, length: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_s = np.zeros(length + 1, dtype=np.float64)
    L = np.zeros(length + 1, dtype=np.float64)
    S = np.ones(length + 1, dtype=np.float64)
    for t in range(length):
        L[t + 1] = (1.0 - cfg.kappa) * L[t] + cfg.sigma * rng.normal()
        log_s[t + 1] = log_s[t] + L[t + 1]
        if cfg.log_s_clip is not None:
            log_s[t + 1] = np.clip(log_s[t + 1], -cfg.log_s_clip, cfg.log_s_clip)
            L[t + 1] = log_s[t + 1] - log_s[t]
        S[t + 1] = np.exp(log_s[t + 1])
    return pd.DataFrame({"t": np.arange(length + 1), "log_s": log_s, "L": L, "S": S})


class SequentialTradingEnv:
    """Trading environment over a fixed pre-generated price path."""

    def __init__(self, cfg: MarketConfig, path: pd.DataFrame, reward_type: str = "R1"):
        assert reward_type in {"R1", "R2"}
        self.cfg = cfg
        self.path = path.reset_index(drop=True)
        self.reward_type = reward_type
        self.max_t = len(self.path) - 1
        self.reset()

    @property
    def state_dim(self) -> int:
        return 2 if self.reward_type == "R1" else 3

    def _state(self) -> np.ndarray:
        row = self.path.iloc[self.idx]
        state = [float(row.log_s), float(row.L)]
        if self.reward_type == "R2":
            state.append(self.a_prev)
        return np.array(state, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.idx = 0
        self.a_prev = 0.0
        return self._state()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        cfg = self.cfg
        A = float(np.clip(action, -cfg.a_max, cfg.a_max))
        row = self.path.iloc[self.idx]
        row_next = self.path.iloc[self.idx + 1]
        S_t = float(row.S)
        S_next = float(row_next.S)

        pnl = A * (S_next - S_t)
        if self.reward_type == "R1":
            reward = pnl - cfg.lam * A**2
        else:
            reward = pnl - cfg.lam * (A - self.a_prev) ** 2 * S_t

        info = {"S_t": S_t, "S_next": S_next, "action": A, "a_prev": self.a_prev}
        self.a_prev = A
        self.idx += 1
        done = self.idx >= min(cfg.T, self.max_t)
        return self._state(), float(reward), done, info


def behavior_policy(state: np.ndarray, cfg: MarketConfig, rng: np.random.Generator) -> float:
    L_t = float(state[1])
    action = 2.0 * L_t + rng.normal(0.0, 1.0)
    return float(np.clip(action, -cfg.a_max, cfg.a_max))


def random_policy_factory(cfg: MarketConfig, seed: int = 0) -> Callable[[np.ndarray], float]:
    rng = np.random.default_rng(seed)
    return lambda state: float(rng.uniform(-cfg.a_max, cfg.a_max))


def rollout_dataframe(env, policy_fn: Callable[[np.ndarray], float]) -> pd.DataFrame:
    state = env.reset()
    rows = []
    done = False
    while not done:
        action = policy_fn(state)
        next_state, reward, done, info = env.step(action)
        rows.append(
            {
                "t": len(rows),
                "log_s": float(state[0]),
                "L": float(state[1]),
                "action": float(info["action"]),
                "reward": reward,
                "S_t": info["S_t"],
                "S_next": info["S_next"],
            }
        )
        state = next_state
    return pd.DataFrame(rows)


def analytic_policy_r1(state: np.ndarray, cfg: MarketConfig) -> float:
    log_s, L_t = float(state[0]), float(state[1])
    S_t = math.exp(log_s)
    expected_delta_s = S_t * (math.exp((1.0 - cfg.kappa) * L_t + 0.5 * cfg.sigma**2) - 1.0)
    action = expected_delta_s / (2.0 * cfg.lam)
    return float(np.clip(action, -cfg.a_max, cfg.a_max))


def evaluate_policy_mc(
    cfg: MarketConfig,
    reward_type: str,
    policy_fn: Callable[[np.ndarray], float],
    n_eval: int,
    seed: int = 0,
) -> Dict[str, float]:
    returns, turnovers, abs_positions = [], [], []
    for i in range(n_eval):
        env = MeanRevertingStockEnv(cfg, reward_type=reward_type, seed=seed + 10_000 * i)
        state = env.reset()
        total = 0.0
        actions = []
        for t in range(cfg.T):
            action = float(np.clip(policy_fn(state), -cfg.a_max, cfg.a_max))
            state, reward, done, _ = env.step(action)
            total += (cfg.gamma**t) * reward
            actions.append(action)
            if done:
                break
        arr_actions = np.array(actions, dtype=np.float64)
        prev = np.r_[0.0, arr_actions[:-1]]
        returns.append(total)
        turnovers.append(float(np.mean(np.abs(arr_actions - prev))))
        abs_positions.append(float(np.mean(np.abs(arr_actions))))
    arr = np.array(returns, dtype=np.float64)
    return {
        "mean_return": float(arr.mean()),
        "std_error": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "return_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "turnover": float(np.mean(turnovers)),
        "avg_abs_position": float(np.mean(abs_positions)),
        "num_eval_paths": int(n_eval),
        "T_eval": int(cfg.T),
    }


def policy_mse_reward1(cfg: MarketConfig, policy_fn: Callable[[np.ndarray], float]) -> float:
    L_grid = np.linspace(-0.5, 0.5, 101)
    sq = []
    for L_t in L_grid:
        state = np.array([0.0, L_t], dtype=np.float32)
        sq.append((policy_fn(state) - analytic_policy_r1(state, cfg)) ** 2)
    return float(np.mean(sq))


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.storage = []
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        item = (
            np.asarray(state, dtype=np.float32),
            np.asarray([action], dtype=np.float32),
            np.asarray([reward], dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            np.asarray([done], dtype=np.float32),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in idx]
        s, a, r, ns, d = zip(*batch)
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.array(a), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.array(r), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.array(ns), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.array(d), dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.storage)


def mlp(input_dim: int, output_dim: int, hidden: int, activation=nn.ReLU) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        activation(),
        nn.Linear(hidden, hidden),
        activation(),
        nn.Linear(hidden, output_dim),
    )


class PPOActor(nn.Module):
    def __init__(self, state_dim: int, hidden: int, a_max: float):
        super().__init__()
        self.a_max = a_max
        self.net = mlp(state_dim, 1, hidden, activation=nn.Tanh)
        self.log_std = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))

    def distribution(self, states: torch.Tensor) -> Normal:
        mean = torch.tanh(self.net(states)) * self.a_max
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, state: np.ndarray, deterministic: bool = False) -> float:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            dist = self.distribution(s)
            action = dist.mean if deterministic else dist.sample()
        return float(torch.clamp(action, -self.a_max, self.a_max).cpu().item())


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(state_dim, 1, hidden, activation=nn.Tanh)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, hidden: int, a_max: float):
        super().__init__()
        self.a_max = a_max
        self.net = mlp(state_dim, 1, hidden, activation=nn.ReLU)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(states)) * self.a_max

    def act(self, state: np.ndarray) -> float:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return float(self.forward(s).cpu().item())


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(state_dim + 1, 1, hidden, activation=nn.ReLU)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1))


class SACActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim: int, hidden: int, a_max: float):
        super().__init__()
        self.a_max = a_max
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)

    def _params(self, states: torch.Tensor):
        h = self.trunk(states)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, states: torch.Tensor):
        mean, log_std = self._params(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        tanh_z = torch.tanh(z)
        action = tanh_z * self.a_max
        log_prob = normal.log_prob(z) - torch.log(self.a_max * (1.0 - tanh_z.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def act(self, state: np.ndarray, deterministic: bool = True) -> float:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            if deterministic:
                mean, _ = self._params(s)
                action = torch.tanh(mean) * self.a_max
            else:
                action, _ = self.sample(s)
        return float(action.cpu().item())


def soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float = 0.95,
):
    adv = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        adv[t] = gae
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def make_env_for_setting(cfg: MarketConfig, reward_type: str, setting: str, seed: int):
    if setting == "oracle":
        return MeanRevertingStockEnv(cfg, reward_type=reward_type, seed=seed)
    path = simulate_price_path(cfg, MAIN_BUDGET.sequential_path_len, seed=seed)
    return SequentialTradingEnv(cfg, path, reward_type=reward_type)


def train_ppo(
    cfg: MarketConfig,
    reward_type: str,
    setting: str,
    seed: int,
    lr: float = 3e-4,
    total_steps: Optional[int] = None,
):
    set_seed(seed)
    total_steps = MAIN_BUDGET.ppo_steps if total_steps is None else total_steps
    env = make_env_for_setting(cfg, reward_type, setting, seed=seed + 11)
    actor = PPOActor(env.state_dim, MAIN_BUDGET.hidden, cfg.a_max).to(DEVICE)
    critic = ValueNet(env.state_dim, MAIN_BUDGET.hidden).to(DEVICE)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    state = env.reset()
    ep_return = 0.0
    recent = deque(maxlen=10)
    log_rows = []
    steps_done = 0

    while steps_done < total_steps:
        s_buf, raw_a_buf, logp_buf, r_buf, d_buf, v_buf = [], [], [], [], [], []
        for _ in range(MAIN_BUDGET.ppo_rollout):
            st = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                dist = actor.distribution(st)
                raw_action = dist.sample()
                logp = dist.log_prob(raw_action).sum(dim=-1)
                value = critic(st)
            clipped_action = float(torch.clamp(raw_action, -cfg.a_max, cfg.a_max).cpu().item())
            next_state, reward, done, _ = env.step(clipped_action)

            s_buf.append(state.copy())
            raw_a_buf.append([float(raw_action.cpu().item())])
            logp_buf.append(float(logp.cpu().item()))
            r_buf.append(float(reward))
            d_buf.append(bool(done))
            v_buf.append(float(value.cpu().item()))

            ep_return += reward
            state = next_state
            steps_done += 1
            if done:
                recent.append(ep_return)
                ep_return = 0.0
                state = env.reset()
            if steps_done >= total_steps:
                break

        with torch.no_grad():
            last_value = float(
                critic(torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)).cpu().item()
            )
        adv, ret = compute_gae(r_buf, v_buf, d_buf, last_value, cfg.gamma)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        s_t = torch.as_tensor(np.asarray(s_buf), dtype=torch.float32, device=DEVICE)
        a_t = torch.as_tensor(np.asarray(raw_a_buf), dtype=torch.float32, device=DEVICE)
        old_logp_t = torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=DEVICE)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=DEVICE)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=DEVICE)

        n = len(s_buf)
        for _ in range(MAIN_BUDGET.ppo_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, MAIN_BUDGET.batch_size):
                idx = perm[start : start + MAIN_BUDGET.batch_size]
                dist = actor.distribution(s_t[idx])
                new_logp = dist.log_prob(a_t[idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_logp_t[idx])
                unclipped = ratio * adv_t[idx]
                clipped = torch.clamp(ratio, 0.8, 1.2) * adv_t[idx]
                actor_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(critic(s_t[idx]), ret_t[idx])
                entropy_bonus = dist.entropy().sum(dim=-1).mean()
                loss = actor_loss + 0.5 * value_loss - 0.001 * entropy_bonus

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step()

        if recent:
            log_rows.append({"steps": steps_done, "recent_train_return": float(np.mean(recent))})

    return {"algo": "PPO", "actor": actor, "critic": critic}, pd.DataFrame(log_rows)


def build_behavior_buffer(cfg: MarketConfig, reward_type: str, seed: int, n_steps: int) -> ReplayBuffer:
    rng = np.random.default_rng(seed)
    path = simulate_price_path(cfg, max(n_steps + 2, cfg.T + 2), seed=seed + 101)
    env = SequentialTradingEnv(cfg, path, reward_type=reward_type)
    buffer = ReplayBuffer(capacity=n_steps + 10)
    state = env.reset()
    for _ in range(n_steps):
        action = behavior_policy(state, cfg, rng)
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
    return buffer


def train_td3(
    cfg: MarketConfig,
    reward_type: str,
    setting: str,
    seed: int,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    total_steps: Optional[int] = None,
):
    set_seed(seed)
    total_steps = MAIN_BUDGET.td3_steps if total_steps is None else total_steps
    state_dim = 2 if reward_type == "R1" else 3
    actor = DeterministicActor(state_dim, MAIN_BUDGET.hidden, cfg.a_max).to(DEVICE)
    actor_tgt = DeterministicActor(state_dim, MAIN_BUDGET.hidden, cfg.a_max).to(DEVICE)
    q1, q2 = QNet(state_dim, MAIN_BUDGET.hidden).to(DEVICE), QNet(state_dim, MAIN_BUDGET.hidden).to(DEVICE)
    q1_tgt, q2_tgt = QNet(state_dim, MAIN_BUDGET.hidden).to(DEVICE), QNet(state_dim, MAIN_BUDGET.hidden).to(DEVICE)
    actor_tgt.load_state_dict(actor.state_dict())
    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=critic_lr)

    if setting == "sequential":
        buffer = build_behavior_buffer(cfg, reward_type, seed=seed + 33, n_steps=max(total_steps, 500))
        env = None
        state = None
    else:
        buffer = ReplayBuffer(capacity=200_000)
        env = MeanRevertingStockEnv(cfg, reward_type=reward_type, seed=seed + 33)
        state = env.reset()

    log_rows = []
    ep_return = 0.0
    recent = deque(maxlen=10)
    tau = 0.005
    policy_delay = 2
    target_noise = 0.2 * cfg.a_max
    noise_clip = 0.5 * cfg.a_max
    exploration_noise = 0.15 * cfg.a_max

    for step in range(1, total_steps + 1):
        if setting == "oracle":
            if step <= MAIN_BUDGET.warmup_steps:
                action = np.random.uniform(-cfg.a_max, cfg.a_max)
            else:
                action = actor.act(state) + np.random.normal(0.0, exploration_noise)
            action = float(np.clip(action, -cfg.a_max, cfg.a_max))
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_return += reward
            if done:
                recent.append(ep_return)
                ep_return = 0.0
                state = env.reset()

        if len(buffer) >= MAIN_BUDGET.batch_size:
            s, a, r, ns, d = buffer.sample(MAIN_BUDGET.batch_size)
            with torch.no_grad():
                noise = torch.randn_like(a) * target_noise
                noise = torch.clamp(noise, -noise_clip, noise_clip)
                next_action = torch.clamp(actor_tgt(ns) + noise, -cfg.a_max, cfg.a_max)
                q_target = torch.min(q1_tgt(ns, next_action), q2_tgt(ns, next_action))
                y = r + cfg.gamma * (1.0 - d) * q_target
            critic_loss = F.mse_loss(q1(s, a), y) + F.mse_loss(q2(s, a), y)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            if step % policy_delay == 0:
                actor_loss = -q1(s, actor(s)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                soft_update(actor, actor_tgt, tau)
                soft_update(q1, q1_tgt, tau)
                soft_update(q2, q2_tgt, tau)

        if step % max(100, total_steps // 3) == 0:
            log_rows.append(
                {
                    "steps": step,
                    "recent_train_return": float(np.mean(recent)) if recent else np.nan,
                    "buffer_size": len(buffer),
                }
            )

    return {"algo": "TD3", "actor": actor}, pd.DataFrame(log_rows)


def train_sac(
    cfg: MarketConfig,
    reward_type: str,
    setting: str,
    seed: int,
    lr: float = 3e-4,
    total_steps: Optional[int] = None,
):
    set_seed(seed)
    total_steps = MAIN_BUDGET.sac_steps if total_steps is None else total_steps
    state_dim = 2 if reward_type == "R1" else 3
    actor = SACActor(state_dim, MAIN_BUDGET.hidden, cfg.a_max).to(DEVICE)
    q1, q2 = QNet(state_dim, MAIN_BUDGET.hidden).to(DEVICE), QNet(state_dim, MAIN_BUDGET.hidden).to(DEVICE)
    q1_tgt, q2_tgt = QNet(state_dim, MAIN_BUDGET.hidden).to(DEVICE), QNet(state_dim, MAIN_BUDGET.hidden).to(DEVICE)
    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, requires_grad=True, device=DEVICE)
    alpha_opt = torch.optim.Adam([log_alpha], lr=lr)
    target_entropy = -1.0

    if setting == "sequential":
        buffer = build_behavior_buffer(cfg, reward_type, seed=seed + 44, n_steps=max(total_steps, 500))
        env = None
        state = None
    else:
        buffer = ReplayBuffer(capacity=200_000)
        env = MeanRevertingStockEnv(cfg, reward_type=reward_type, seed=seed + 44)
        state = env.reset()

    log_rows = []
    ep_return = 0.0
    recent = deque(maxlen=10)
    tau = 0.005

    for step in range(1, total_steps + 1):
        if setting == "oracle":
            if step <= MAIN_BUDGET.warmup_steps:
                action = np.random.uniform(-cfg.a_max, cfg.a_max)
            else:
                action = actor.act(state, deterministic=False)
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_return += reward
            if done:
                recent.append(ep_return)
                ep_return = 0.0
                state = env.reset()

        if len(buffer) >= MAIN_BUDGET.batch_size:
            s, a, r, ns, d = buffer.sample(MAIN_BUDGET.batch_size)
            alpha = log_alpha.exp().detach()
            with torch.no_grad():
                next_a, next_logp = actor.sample(ns)
                next_q = torch.min(q1_tgt(ns, next_a), q2_tgt(ns, next_a)) - alpha * next_logp
                y = r + cfg.gamma * (1.0 - d) * next_q
            q_loss = F.mse_loss(q1(s, a), y) + F.mse_loss(q2(s, a), y)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            new_a, logp = actor.sample(s)
            q_new = torch.min(q1(s, new_a), q2(s, new_a))
            actor_loss = (log_alpha.exp().detach() * logp - q_new).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            soft_update(q1, q1_tgt, tau)
            soft_update(q2, q2_tgt, tau)

        if step % max(100, total_steps // 3) == 0:
            log_rows.append(
                {
                    "steps": step,
                    "recent_train_return": float(np.mean(recent)) if recent else np.nan,
                    "buffer_size": len(buffer),
                    "alpha": float(log_alpha.exp().detach().cpu().item()),
                }
            )

    return {"algo": "SAC", "actor": actor}, pd.DataFrame(log_rows)


def policy_from_agent(agent: Dict) -> Callable[[np.ndarray], float]:
    algo = agent["algo"]
    actor = agent["actor"]
    if algo == "PPO":
        return lambda state: actor.act(state, deterministic=True)
    if algo == "TD3":
        return lambda state: actor.act(state)
    if algo == "SAC":
        return lambda state: actor.act(state, deterministic=True)
    raise ValueError(algo)


def train_agent(algo: str, cfg: MarketConfig, reward_type: str, setting: str, seed: int, **kwargs):
    if algo == "PPO":
        return train_ppo(cfg, reward_type, setting, seed, **kwargs)
    if algo == "TD3":
        return train_td3(cfg, reward_type, setting, seed, **kwargs)
    if algo == "SAC":
        return train_sac(cfg, reward_type, setting, seed, **kwargs)
    raise ValueError(algo)


def parameter_sets() -> List[Tuple[str, MarketConfig]]:
    return [
        ("baseline", BASE_CFG),
        ("slow_reversion", replace(BASE_CFG, kappa=0.05)),
        ("high_volatility", replace(BASE_CFG, sigma=0.2)),
        ("strong_regularization", replace(BASE_CFG, lam=0.05)),
    ]


def _save_plot_artifacts(
    output_dir: Path,
    task_key: Tuple[str, int, str, str, str],
    cfg_exp: MarketConfig,
    agent: Dict,
    log_df: pd.DataFrame,
) -> None:
    param_label, seed, reward_type, setting, algo = task_key
    stem = _task_stem(param_label, reward_type, setting, algo, seed)

    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if task_key in PLOT_CHECKPOINT_TASKS:
        ckpt = {
            "algo": algo,
            "param_label": param_label,
            "seed": seed,
            "reward_type": reward_type,
            "setting": setting,
            "state_dim": 2 if reward_type == "R1" else 3,
            "hidden": MAIN_BUDGET.hidden,
            "a_max": cfg_exp.a_max,
            "actor_state_dict": agent["actor"].state_dict(),
        }
        torch.save(ckpt, models_dir / f"{stem}.pt")

    if task_key in PLOT_LOG_TASKS and len(log_df):
        log_df.to_csv(logs_dir / f"{stem}.csv", index=False)


def _train_and_evaluate(
    param_label: str,
    cfg_exp: MarketConfig,
    analytic_metrics: Dict[str, float],
    seed: int,
    reward_type: str,
    setting: str,
    algo: str,
    output_dir_str: str,
) -> Dict[str, float]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    set_seed(seed)

    output_dir = Path(output_dir_str)
    task_key = (param_label, seed, reward_type, setting, algo)
    print(f"Training {algo:>3} | {reward_type} | {setting:10s} | seed={seed} | param={param_label}")
    agent, log_df = train_agent(algo, cfg_exp, reward_type, setting, seed=seed)
    policy_fn = policy_from_agent(agent)
    metrics = evaluate_policy_mc(
        cfg_exp,
        reward_type,
        policy_fn,
        n_eval=MAIN_BUDGET.n_eval,
        seed=30_000 + 1000 * seed,
    )
    row = {
        "parameter_set": param_label,
        "reward_type": reward_type,
        "method": algo,
        "train_setting": setting,
        "seed": seed,
        **metrics,
        "relative_to_analytic_pct": np.nan,
        "regret": np.nan,
        "policy_mse_to_analytic": np.nan,
    }
    if reward_type == "R1":
        row["relative_to_analytic_pct"] = 100.0 * metrics["mean_return"] / analytic_metrics["mean_return"]
        row["regret"] = analytic_metrics["mean_return"] - metrics["mean_return"]
        row["policy_mse_to_analytic"] = policy_mse_reward1(cfg_exp, policy_fn)

    _save_plot_artifacts(output_dir, task_key, cfg_exp, agent, log_df)
    return row


def _build_actor_for_checkpoint(algo: str, state_dim: int, hidden: int, a_max: float) -> nn.Module:
    if algo == "PPO":
        return PPOActor(state_dim, hidden, a_max)
    if algo == "TD3":
        return DeterministicActor(state_dim, hidden, a_max)
    if algo == "SAC":
        return SACActor(state_dim, hidden, a_max)
    raise ValueError(algo)


def _load_policy_from_checkpoint(ckpt_path: Path) -> Optional[Callable[[np.ndarray], float]]:
    if not ckpt_path.exists():
        return None
    payload = torch.load(ckpt_path, map_location=DEVICE)
    actor = _build_actor_for_checkpoint(
        payload["algo"],
        payload["state_dim"],
        payload["hidden"],
        payload["a_max"],
    ).to(DEVICE)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    return policy_from_agent({"algo": payload["algo"], "actor": actor})


def generate_sanity_plot(cfg: MarketConfig, output_dir: Path) -> None:
    env_check = MeanRevertingStockEnv(cfg, "R2", seed=7)
    df_check = rollout_dataframe(env_check, random_policy_factory(cfg, seed=8))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(df_check["t"], df_check["S_t"])
    axes[0].set(title="Sample price path", xlabel="t", ylabel="S_t")
    axes[1].plot(df_check["t"], df_check["L"])
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set(title="Mean-reverting L_t", xlabel="t", ylabel="L_t")
    axes[2].plot(df_check["t"], df_check["action"])
    axes[2].set(title="Random actions", xlabel="t", ylabel="A_t")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_sanity_trajectory.png", bbox_inches="tight")
    plt.close(fig)


def generate_training_curves_plot(output_dir: Path) -> None:
    logs_dir = output_dir / "logs"
    targets = [
        ("PPO", "baseline__R1__oracle__PPO__seed0.csv"),
        ("TD3", "baseline__R1__oracle__TD3__seed0.csv"),
        ("SAC", "baseline__R1__oracle__SAC__seed0.csv"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for label, file_name in targets:
        p = logs_dir / file_name
        if not p.exists():
            continue
        log_df = pd.read_csv(p)
        if len(log_df) == 0:
            continue
        ax.plot(log_df["steps"], log_df["recent_train_return"], label=f"{label} oracle")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set(title="Training curves (R1 oracle)", xlabel="training steps", ylabel="recent train return")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_training_curves.png", bbox_inches="tight")
    plt.close(fig)


def generate_policy_plots(output_dir: Path) -> None:
    models_dir = output_dir / "models"
    r1_ckpts = {
        "PPO": models_dir / "baseline__R1__oracle__PPO__seed0.pt",
        "TD3": models_dir / "baseline__R1__oracle__TD3__seed0.pt",
        "SAC": models_dir / "baseline__R1__oracle__SAC__seed0.pt",
    }
    ppo_r2_ckpt = models_dir / "baseline__R2__oracle__PPO__seed0.pt"

    policies_r1 = {algo: _load_policy_from_checkpoint(path) for algo, path in r1_ckpts.items()}
    pol_r2 = _load_policy_from_checkpoint(ppo_r2_ckpt)

    L_grid = np.linspace(-0.5, 0.5, 201)
    states_r1 = [np.array([0.0, L], dtype=np.float32) for L in L_grid]

    fig = plt.figure(figsize=(8, 4))
    plt.plot(L_grid, [analytic_policy_r1(s, BASE_CFG) for s in states_r1], "k-", lw=2, label="Analytic R1")
    styles = {"PPO": "--", "TD3": ":", "SAC": "-."}
    for algo in ALGOS:
        pol = policies_r1.get(algo)
        if pol is None:
            continue
        plt.plot(L_grid, [pol(s) for s in states_r1], styles[algo], lw=1.5, label=f"{algo} oracle")
    plt.axhline(0, color="black", lw=0.6)
    plt.axvline(0, color="black", lw=0.6)
    plt.xlabel("$L_t$")
    plt.ylabel("$A_t$")
    plt.title("Reward (1): learned policies vs analytic policy at $S_t=1$")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_reward1_policy_comparison.png", bbox_inches="tight")
    plt.close(fig)

    ppo_r1 = policies_r1.get("PPO")
    if ppo_r1 is not None:
        L_vals = np.linspace(-0.4, 0.4, 50)
        log_s_vals = np.linspace(-0.5, 0.5, 50)
        LL, SS = np.meshgrid(L_vals, log_s_vals)

        def grid_eval_r1(policy_fn):
            out = np.zeros_like(LL)
            for i in range(LL.shape[0]):
                for j in range(LL.shape[1]):
                    out[i, j] = policy_fn(np.array([SS[i, j], LL[i, j]], dtype=np.float32))
            return out

        A_analytic = grid_eval_r1(lambda s: analytic_policy_r1(s, BASE_CFG))
        A_ppo = grid_eval_r1(ppo_r1)
        vmax = max(np.abs(A_analytic).max(), np.abs(A_ppo).max())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, data, title in zip(axes, [A_analytic, A_ppo], ["Analytic", "PPO oracle"]):
            im = ax.pcolormesh(L_vals, log_s_vals, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            fig.colorbar(im, ax=ax, label="$A_t$")
            ax.set(xlabel="$L_t$", ylabel="$\\log S_t$", title=title)
        plt.tight_layout()
        plt.savefig(output_dir / "fig_reward1_policy_heatmap.png", bbox_inches="tight")
        plt.close(fig)

    if pol_r2 is not None:
        L_vals = np.linspace(-0.4, 0.4, 50)
        a_prev_vals = np.linspace(-BASE_CFG.a_max, BASE_CFG.a_max, 50)
        LL, AA = np.meshgrid(L_vals, a_prev_vals)
        grid = np.zeros_like(LL)
        for i in range(LL.shape[0]):
            for j in range(LL.shape[1]):
                grid[i, j] = pol_r2(np.array([0.0, LL[i, j], AA[i, j]], dtype=np.float32))
        fig = plt.figure(figsize=(6, 4.5))
        im = plt.pcolormesh(L_vals, a_prev_vals, grid, cmap="RdBu_r", shading="auto")
        plt.colorbar(im, label="$A_t$")
        plt.xlabel("$L_t$")
        plt.ylabel("$A_{t-1}$")
        plt.title("Reward (2): PPO oracle policy heatmap at $S_t=1$")
        plt.tight_layout()
        plt.savefig(output_dir / "fig_reward2_policy_heatmap.png", bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_output_dir = str(Path(__file__).resolve().parent / "output_highbudget")
    parser = argparse.ArgumentParser(description="RL trading main experiment (CPU multiprocessing only).")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of multiprocessing workers. Default: cpu_count - 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help=f"Output directory for csv/figures/checkpoints. Default: {default_output_dir}",
    )
    parser.add_argument(
        "--plots",
        type=str,
        choices=("on", "off"),
        default="on",
        help="Generate figures from saved models/logs.",
    )
    return parser.parse_args()


def run_main_matrix(workers: int, output_dir: Path) -> pd.DataFrame:
    result_rows: List[Dict] = []
    tasks = []
    for param_label, cfg_exp in parameter_sets():
        print(f"\n=== Parameter set: {param_label} ===")
        analytic_metrics = evaluate_policy_mc(
            cfg_exp,
            "R1",
            lambda s, c=cfg_exp: analytic_policy_r1(s, c),
            n_eval=MAIN_BUDGET.n_eval,
            seed=90_000,
        )
        result_rows.append(
            {
                "parameter_set": param_label,
                "reward_type": "R1",
                "method": "Analytic",
                "train_setting": "none",
                "seed": -1,
                **analytic_metrics,
                "relative_to_analytic_pct": 100.0,
                "regret": 0.0,
                "policy_mse_to_analytic": 0.0,
            }
        )
        for seed in SEEDS:
            for reward_type in REWARD_TYPES:
                for setting in SETTINGS:
                    for algo in ALGOS:
                        tasks.append(
                            (
                                param_label,
                                cfg_exp,
                                analytic_metrics,
                                seed,
                                reward_type,
                                setting,
                                algo,
                                str(output_dir),
                            )
                        )

    expected_task_count = len(PARAMETER_SET_NAMES) * len(SEEDS) * len(REWARD_TYPES) * len(SETTINGS) * len(ALGOS)
    if len(tasks) != expected_task_count:
        raise RuntimeError(f"Task count mismatch: expected {expected_task_count}, got {len(tasks)}")

    print(f"\nTotal train tasks: {len(tasks)} (expected 96)")
    print(f"Workers: {workers}")

    with mp.Pool(processes=workers) as pool:
        rows = pool.starmap(_train_and_evaluate, tasks)
    result_rows.extend(rows)
    return pd.DataFrame(result_rows)


def main() -> None:
    args = parse_args()
    set_seed(0)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Base config:", BASE_CFG)
    print("Main budget:", MAIN_BUDGET)
    print("Parameter sets:", PARAMETER_SET_NAMES)
    print("Seeds:", SEEDS)

    mp.set_start_method("spawn", force=True)
    results_main = run_main_matrix(max(1, args.workers), output_dir)
    results_csv = output_dir / "results_main.csv"
    results_main.to_csv(results_csv, index=False)

    summary_cols = [
        "parameter_set",
        "reward_type",
        "method",
        "train_setting",
        "mean_return",
        "std_error",
        "relative_to_analytic_pct",
        "regret",
        "turnover",
        "policy_mse_to_analytic",
    ]
    summary = results_main[summary_cols].sort_values(["parameter_set", "reward_type", "method", "train_setting"])
    print("\nSaved:", results_csv)
    print("Result rows:", len(results_main))
    print(summary.head(12).to_string(index=False))

    if args.plots == "on":
        generate_sanity_plot(BASE_CFG, output_dir)
        generate_training_curves_plot(output_dir)
        generate_policy_plots(output_dir)

    print("\nExpected output artifacts:")
    print(" - results_main.csv")
    print(" - fig_sanity_trajectory.png")
    print(" - fig_training_curves.png")
    print(" - fig_reward1_policy_comparison.png")
    print(" - fig_reward1_policy_heatmap.png")
    print(" - fig_reward2_policy_heatmap.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
