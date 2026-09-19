"""从零实现的 PPO：MLP Actor(高斯)/Critic，GAE + clipped surrogate + 熵正则。"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import config as C


class RunningNorm(nn.Module):
    """冻结于一个rollout及PPO更新内；更新后供下一rollout使用。"""
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))

    def forward(self, values):
        return (values-self.mean) / torch.sqrt(self.var.clamp_min(1e-4))

    @torch.no_grad()
    def update(self, values):
        batch = values.reshape(-1, self.mean.numel())
        n = len(batch)
        mean, var = batch.mean(0), batch.var(0, unbiased=False)
        total = self.count+n
        delta = mean-self.mean
        new_var = (self.var*self.count+var*n+delta.square()*self.count*n/total)/total
        self.mean.add_(delta*n/total)
        self.var.copy_(new_var)
        self.count.copy_(total)


def _mlp(sizes):
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if j < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = _mlp([obs_dim] + hidden + [act_dim])
        self.norm = RunningNorm(obs_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
        # 扩大可行角范围时保持此前0.5rad尺度下的初始物理探索幅度。
        with torch.no_grad():
            joint_mask = torch.arange(act_dim) % 3 != 2
            self.log_std[joint_mask] += np.log(.5 / C.JOINT_ACTION_SCALE)

    def forward(self, obs):
        # 标准高斯参数化；执行端限幅，避免均值/标准差硬约束冻结梯度。
        mean = self.net(self.norm(obs))
        std = torch.exp(self.log_std)
        return mean, std


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = _mlp([obs_dim] + hidden + [1])
        self.norm = RunningNorm(obs_dim)

    def forward(self, obs):
        return self.net(self.norm(obs)).squeeze(-1)


class Estimator(nn.Module):
    """二维特权状态：归一化身体速度、四足高度、四足接触概率。"""
    def __init__(self, obs_dim):
        super().__init__()
        self.net = _mlp([obs_dim] + C.ESTIMATOR_HIDDEN + [9])
        self.norm = RunningNorm(obs_dim)

    def features(self, obs):
        features = obs.clone()
        features[..., 0] = 0.0
        features[..., C.CONTACT_OBS_INDICES] = 0.0
        features[..., C.HEIGHT_OBS_INDICES] = 0.0
        return features

    def forward(self, obs):
        output = self.net(self.norm(self.features(obs)))
        return torch.cat((output[..., :1], output[..., 1:5],
                          output[..., 5:9].sigmoid()), dim=-1)


class PPO:
    def __init__(self, obs_dim, act_dim, hidden=None, lr=None, seed=None):
        hidden = hidden or C.HIDDEN
        lr = lr or C.LR
        self.seed = seed if seed is not None else C.SEED
        torch.manual_seed(seed if seed is not None else C.SEED)
        np.random.seed(seed if seed is not None else C.SEED)
        self.actor = Actor(obs_dim, act_dim, hidden)
        self.critic = Critic(obs_dim, hidden)
        self.estimator = Estimator(obs_dim)
        # 论文未公开初始化：采用正交初始化和小Actor输出头，避免初始目标撞限。
        for model in (self.actor,self.critic,self.estimator):
            for layer in model.net:
                if isinstance(layer,nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(model.net[-1].weight,.01 if model is self.actor else 1.)
        self.training_variant = "full"
        self.iteration = -1
        # 三网络各自优化/裁剪，防止大数值价值损失压制Actor梯度。
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.estimator_opt = torch.optim.Adam(self.estimator.parameters(), lr=lr)

    def actor_observation(self, obs):
        estimate = self.estimator(obs)
        actor_obs = obs.clone()
        actor_obs[..., 0] = estimate[..., 0].detach()
        actor_obs[..., C.HEIGHT_OBS_INDICES] = estimate[..., 1:5].detach()
        actor_obs[..., C.CONTACT_OBS_INDICES] = estimate[..., 5:9].detach()
        return actor_obs, estimate

    def act(self, obs, deterministic=False, privileged=None):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32)
            actor_obs, estimate = self.actor_observation(o)
            self.last_contact = estimate[5:9].numpy().copy()
            mean, std = self.actor(actor_obs)
            dist = Normal(mean, std)
            a = mean if deterministic else dist.sample()
            # 保存高斯原始动作及其概率；环境裁剪只发生在执行时。
            # PPO 更新仍使用原始动作，避免错误计算裁剪边界的密度。
            logp = dist.log_prob(a).sum(-1)
            critic_obs = torch.as_tensor(privileged, dtype=torch.float32) if privileged is not None else o
            val = self.critic(critic_obs)
        return a.numpy(), float(logp), float(val)

    def value(self, obs):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32)
            return float(self.critic(o))

    def act_batch(self, obs, privileged, with_values=True):
        with torch.no_grad():
            observations = torch.as_tensor(obs, dtype=torch.float32)
            actor_obs, estimate = self.actor_observation(observations)
            mean, std = self.actor(actor_obs)
            distribution = Normal(mean, std)
            actions = distribution.sample()
            values = (self.critic(torch.as_tensor(privileged, dtype=torch.float32)) if with_values
                      else torch.zeros(len(observations)))
            return (actions.numpy(), distribution.log_prob(actions).sum(-1).numpy(),
                    values.numpy(), estimate[..., 5:9].numpy())

    def value_batch(self, privileged):
        with torch.no_grad():
            return self.critic(torch.as_tensor(privileged, dtype=torch.float32)).numpy()

    def update(self, obs_np, act_np, logp_np, ret_np, adv_np, privileged_np=None):
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        act = torch.as_tensor(act_np, dtype=torch.float32)
        logp_old = torch.as_tensor(logp_np, dtype=torch.float32)
        ret = torch.as_tensor(ret_np, dtype=torch.float32)
        adv = torch.as_tensor(adv_np, dtype=torch.float32)
        privileged = torch.as_tensor(privileged_np, dtype=torch.float32) if privileged_np is not None else obs

        n = obs.shape[0]
        idxs = np.arange(n)
        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_fraction": 0.0, "estimator_loss": 0.0}
        stats.update(actor_grad_norm=0.0, critic_grad_norm=0.0)
        updates = 0
        stop_actor = False
        with torch.no_grad():
            actor_inputs, estimates_all = self.actor_observation(obs)

        for _ in range(C.EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, n, C.MINIBATCH):
                idx = idxs[start:start + C.MINIBATCH]
                o, a, lp_old, rt, ad = obs[idx], act[idx], logp_old[idx], ret[idx], adv[idx]

                actor_obs, estimate = actor_inputs[idx], estimates_all[idx]
                mean, std = self.actor(actor_obs)
                dist = Normal(mean, std)
                logp = dist.log_prob(a).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                logratio = logp - lp_old
                ratio = torch.exp(logratio)
                approx_kl = ((ratio - 1) - logratio).mean()
                if approx_kl.item() > C.TARGET_KL:
                    stop_actor = True
                surr1 = ratio * ad
                surr2 = torch.clamp(ratio, 1.0 - C.CLIP, 1.0 + C.CLIP) * ad
                actor_loss = -torch.min(surr1, surr2).mean() - C.ENT_COEF * entropy

                truth = privileged[idx]
                val = self.critic(truth)
                critic_loss = F.mse_loss(val, rt)
                estimator_loss = F.mse_loss(estimate[..., 0], truth[..., 0])
                estimator_loss += F.mse_loss(estimate[..., 1:5], truth[..., C.HEIGHT_OBS_INDICES])
                estimator_loss += F.binary_cross_entropy(estimate[..., 5:9].clamp(1e-6, 1-1e-6),
                                                         truth[..., C.CONTACT_OBS_INDICES])

                self.opt.zero_grad()
                if not stop_actor:
                    actor_loss.backward()
                    stats["actor_grad_norm"] += float(nn.utils.clip_grad_norm_(self.actor.parameters(), C.MAX_GRAD_NORM))
                    self.opt.step()
                self.critic_opt.zero_grad()
                (.5 * critic_loss).backward()
                stats["critic_grad_norm"] += float(nn.utils.clip_grad_norm_(self.critic.parameters(), C.MAX_GRAD_NORM))
                self.critic_opt.step()

                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy.item()
                stats["approx_kl"] += approx_kl.item()
                stats["clip_fraction"] += ((ratio - 1).abs() > C.CLIP).float().mean().item()
                stats["estimator_loss"] += estimator_loss.item()
                updates += 1

        # Estimator在Actor全部更新之后训练，避免旧logp的输入定义在更新期间变化。
        for _ in range(C.EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0,n,C.MINIBATCH):
                idx = idxs[start:start+C.MINIBATCH]
                estimate, truth = self.estimator(obs[idx]), privileged[idx]
                loss = F.mse_loss(estimate[...,0],truth[...,0])
                loss += F.mse_loss(estimate[...,1:5],truth[...,C.HEIGHT_OBS_INDICES])
                loss += F.binary_cross_entropy(estimate[...,5:9].clamp(1e-6,1-1e-6),truth[...,C.CONTACT_OBS_INDICES])
                self.estimator_opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.estimator.parameters(),C.MAX_GRAD_NORM)
                self.estimator_opt.step()
        with torch.no_grad():
            actor_obs, _ = self.actor_observation(obs)
            self.actor.norm.update(actor_obs)
            self.critic.norm.update(privileged)
            self.estimator.norm.update(self.estimator.features(obs))
        for key in stats:
            stats[key] /= max(1, updates)
        return stats

    def save(self, path):
        config = {key: value for key, value in vars(C).items() if key.isupper()}
        torch.save({"model_version": C.MODEL_VERSION, "config": config, "seed": self.seed,
                    "iteration": self.iteration, "training_variant": self.training_variant,
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "estimator": self.estimator.state_dict(),
                    "optimizer": self.opt.state_dict(),
                    "critic_optimizer": self.critic_opt.state_dict(),
                    "estimator_optimizer": self.estimator_opt.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu")
        if ck.get("model_version") != C.MODEL_VERSION:
            raise ValueError(f"checkpoint版本{ck.get('model_version')}与{C.MODEL_VERSION}不兼容，需要重新训练。")
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.estimator.load_state_dict(ck["estimator"])
        self.training_variant = ck.get("training_variant", "full")
        self.iteration = ck.get("iteration", -1)
