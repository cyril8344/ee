"""
rl_agent.py
===========
RL agent wrapper — Stable-Baselines3 PPO with a PyTorch DQN fallback.

Priority:
  1. Stable-Baselines3 PPO  (pip install stable-baselines3)
  2. Custom PyTorch DQN     (pip install torch)
  3. RandomAgent            (always works, baseline)

The wrapper exposes a unified interface regardless of backend:
    agent = RLAgent.load("model.zip")   # or RLAgent(env)
    action = agent.predict(obs)
    agent.learn(total_timesteps=100_000)
    agent.save("model.zip")
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Try importing backends
# ─────────────────────────────────────────────────────────────────────────────
_SB3_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    _SB3_AVAILABLE = True
    logger.info("rl_agent: Stable-Baselines3 available → using PPO")
except ImportError:
    logger.warning("rl_agent: stable-baselines3 not installed — trying PyTorch DQN")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque
    _TORCH_AVAILABLE = True
    if not _SB3_AVAILABLE:
        logger.info("rl_agent: PyTorch available → using custom DQN")
except ImportError:
    logger.warning("rl_agent: PyTorch not installed — falling back to RandomAgent")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Stable-Baselines3 PPO agent
# ─────────────────────────────────────────────────────────────────────────────
class PPOAgent:
    """
    Proximal Policy Optimisation via Stable-Baselines3.
    PPO is the gold standard for trading RL: stable, sample-efficient,
    handles continuous observations with discrete actions.
    """

    def __init__(self, env, model_path: Optional[str] = None):
        self.env = env
        if model_path and os.path.exists(model_path + ".zip"):
            self.model = PPO.load(model_path, env=env)
            logger.info("PPOAgent: loaded from %s", model_path)
        else:
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,       # encourage exploration
                policy_kwargs=dict(
                    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
                ),
            )
            logger.info("PPOAgent: created new model")

    def learn(self, total_timesteps: int = 100_000,
              eval_env=None, save_path: Optional[str] = None) -> None:
        callbacks = []
        if eval_env and save_path:
            reward_threshold = StopTrainingOnRewardThreshold(
                reward_threshold=500, verbose=1
            )
            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=os.path.dirname(save_path) or ".",
                log_path=os.path.dirname(save_path) or ".",
                eval_freq=5000,
                callback_on_new_best=reward_threshold,
                verbose=0,
            )
            callbacks.append(eval_cb)

        logger.info("PPOAgent: training for %d timesteps…", total_timesteps)
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks or None)
        logger.info("PPOAgent: training complete")

        if save_path:
            self.model.save(save_path)
            logger.info("PPOAgent: saved to %s", save_path)

    def predict(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env) -> "PPOAgent":
        agent = cls.__new__(cls)
        agent.env = env
        agent.model = PPO.load(path, env=env)
        return agent


# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom PyTorch DQN (fallback)
# ─────────────────────────────────────────────────────────────────────────────
if _TORCH_AVAILABLE:
    class _QNetwork(nn.Module):
        def __init__(self, obs_dim: int, n_actions: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, 256),     nn.ReLU(),
                nn.Linear(256, n_actions),
            )

        def forward(self, x):
            return self.net(x)

    class DQNAgent:
        """
        Double DQN with experience replay and ε-greedy exploration.
        Simpler than PPO but works well for discrete action spaces.
        """

        def __init__(self, env, model_path: Optional[str] = None,
                     gamma: float = 0.99, lr: float = 1e-3,
                     buffer_size: int = 50_000, batch_size: int = 64):
            self.env = env
            obs_dim   = env.observation_space.shape[0]
            n_actions = env.action_space.n
            self.n_actions  = n_actions
            self.gamma      = gamma
            self.batch_size = batch_size
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.q_net      = _QNetwork(obs_dim, n_actions).to(self.device)
            self.target_net = _QNetwork(obs_dim, n_actions).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)

            self.replay_buffer: deque = deque(maxlen=buffer_size)
            self.epsilon = 1.0
            self.epsilon_min = 0.05
            self.epsilon_decay = 0.995
            self._steps = 0
            self._target_update_freq = 500

            if model_path and os.path.exists(model_path):
                state = torch.load(model_path, map_location=self.device)
                self.q_net.load_state_dict(state)
                self.target_net.load_state_dict(state)
                self.epsilon = self.epsilon_min
                logger.info("DQNAgent: loaded from %s", model_path)

        def predict(self, obs: np.ndarray) -> int:
            if random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q = self.q_net(t)
            return int(q.argmax().item())

        def remember(self, obs, action, reward, next_obs, done):
            self.replay_buffer.append((obs, action, reward, next_obs, done))

        def _replay(self):
            if len(self.replay_buffer) < self.batch_size:
                return
            batch = random.sample(self.replay_buffer, self.batch_size)
            obs_b, act_b, rew_b, nobs_b, done_b = zip(*batch)

            obs_t  = torch.FloatTensor(np.array(obs_b)).to(self.device)
            nobs_t = torch.FloatTensor(np.array(nobs_b)).to(self.device)
            act_t  = torch.LongTensor(act_b).to(self.device)
            rew_t  = torch.FloatTensor(rew_b).to(self.device)
            done_t = torch.FloatTensor(done_b).to(self.device)

            q_vals = self.q_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = self.target_net(nobs_t).max(1)[0]
                target = rew_t + self.gamma * next_q * (1 - done_t)

            loss = nn.MSELoss()(q_vals, target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

        def learn(self, total_timesteps: int = 100_000,
                  save_path: Optional[str] = None, **kwargs) -> None:
            obs, _ = self.env.reset()
            logger.info("DQNAgent: training for %d timesteps…", total_timesteps)

            for step in range(total_timesteps):
                action = self.predict(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.remember(obs, action, reward, next_obs, done)
                self._replay()
                obs = next_obs if not done else self.env.reset()[0]

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                self._steps += 1

                if self._steps % self._target_update_freq == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                if step % 10_000 == 0:
                    logger.info("DQNAgent: step %d / %d | ε=%.3f",
                                step, total_timesteps, self.epsilon)

            logger.info("DQNAgent: training complete")
            if save_path:
                torch.save(self.q_net.state_dict(), save_path)
                logger.info("DQNAgent: saved to %s", save_path)

        def save(self, path: str) -> None:
            torch.save(self.q_net.state_dict(), path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random baseline (always works)
# ─────────────────────────────────────────────────────────────────────────────
class RandomAgent:
    """Uniform random action — useful as a performance baseline."""

    def __init__(self, env, **kwargs):
        self.env = env
        self.n_actions = env.action_space.n

    def predict(self, obs: np.ndarray) -> int:
        return random.randint(0, self.n_actions - 1)

    def learn(self, total_timesteps: int = 0, **kwargs) -> None:
        logger.warning("RandomAgent: install stable-baselines3 or torch for real training")

    def save(self, path: str) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
class RLAgent:
    """
    Unified factory — picks the best available backend automatically.

    Usage:
        agent = RLAgent(env)                    # create new
        agent = RLAgent(env, model_path="...")  # load existing
        agent.learn(total_timesteps=200_000)
        action = agent.predict(obs)
        agent.save("models/rl_xauusd")
    """

    def __new__(cls, env, model_path: Optional[str] = None, **kwargs):
        if _SB3_AVAILABLE:
            return PPOAgent(env, model_path=model_path)
        if _TORCH_AVAILABLE:
            return DQNAgent(env, model_path=model_path, **kwargs)
        return RandomAgent(env, **kwargs)

    @staticmethod
    def backend() -> str:
        if _SB3_AVAILABLE:
            return "PPO (stable-baselines3)"
        if _TORCH_AVAILABLE:
            return "DQN (PyTorch)"
        return "Random (no ML library)"
