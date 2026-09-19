"""批量网络推理、独立环境 GAE；区分真正终止和时间限制截断。"""
import csv
import math
import json
import hashlib
from collections import deque
from pathlib import Path
import numpy as np
import torch
import config as C
from env import WallClimbEnv
from ppo import PPO


def compute_gae(rew, val, done, last_val, next_values=None, terminated=None):
    """delta 使用真实终止掩码；递推使用 episode 边界掩码。"""
    rew, val, done = map(np.asarray, (rew, val, done))
    if next_values is None:
        next_values = np.concatenate((val[1:], np.expand_dims(last_val, 0)))
    if terminated is None:
        terminated = done
    adv = np.zeros(rew.shape, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rew))):
        delta = rew[t] + C.GAMMA * next_values[t] * (1 - terminated[t]) - val[t]
        running = delta + C.GAMMA * C.LAMBDA * (1 - done[t]) * running
        adv[t] = running
    return adv, adv + val


def train(render=False, save_dir=C.CHECKPOINT_DIR, total_iters=C.TOTAL_ITERS, seed=C.SEED,
          variant="full"):
    if total_iters < 1:
        raise ValueError("iterations must be positive")
    torch.set_num_threads(1)
    if C.ROLLOUT_STEPS % C.NUM_ENVS:
        raise ValueError("rollout samples must be divisible by number of environments")
    envs = [WallClimbEnv(seed + i, random_commands=True, domain_randomization=True,
                        variant=variant) for i in range(C.NUM_ENVS)]
    env = envs[0]
    ppo = PPO(C.OBS_DIM, C.ACT_DIM, seed=seed)
    ppo.training_variant = variant
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "training.csv").exists():
        raise FileExistsError("Use a new save directory to preserve experiment evidence")
    metadata = {"seed": seed, "variant": variant, "iterations": total_iters,
                "config": {k:v for k,v in vars(C).items() if k.isupper()},
                "source_sha256": {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).resolve().parent.glob("*.py")}}
    (directory / "training_config.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    renderer = None
    if render:
        from render import Renderer
        renderer = Renderer(controller="PPO training")
    for instance in envs:
        instance.set_curriculum(0)
    obs = np.stack([instance.reset() for instance in envs])
    recent = deque(maxlen=50)
    fields = ["iteration", "theta_deg", "p_attach", "return", "climb",
              "survival", "velocity_rmse", "actor_loss", "critic_loss",
              "approx_kl", "clip_fraction", "estimator_loss", "entropy",
              "actor_grad_norm", "critic_grad_norm", "execution_clip_fraction"]
    try:
        with (directory / "training.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for it in range(total_iters):
                for instance in envs:
                    instance.set_curriculum(it)
                ppo.iteration = it
                obs = np.stack([instance.observe() for instance in envs])
                truth_buf = []
                obs_buf, act_buf, rewards, dones, terminals, logps, next_truth_buf = (
                    [] for _ in range(7))
                for _ in range(C.ROLLOUT_STEPS // C.NUM_ENVS):
                    if renderer:
                        quit_, force, reset = renderer.poll()
                        if quit_:
                            ppo.save(directory / "interrupted.pt")
                            return
                        if reset:
                            # 不在人为重置处接续 GAE 轨迹。
                            if dones:
                                dones[-1][0] = 1.0
                            obs[0] = env.reset()
                        if force:
                            env.disturb()
                            obs[0] = env.observe()
                    truth = np.stack([instance.privileged() for instance in envs])
                    actions, logp, _, contacts = ppo.act_batch(obs, truth, with_values=False)
                    truth_buf.append(truth)
                    obs_buf.append(obs.copy())
                    act_buf.append(actions)
                    logps.append(logp)
                    step_rewards, step_dones, step_terminals, step_truth = [], [], [], []
                    next_obs = []
                    for index, instance in enumerate(envs):
                        instance.set_contact_estimate(contacts[index])
                        observation, reward, done, info = instance.step(actions[index])
                        step_rewards.append(reward)
                        step_dones.append(float(done))
                        step_terminals.append(float(info["terminated"]))
                        step_truth.append(instance.privileged())
                        if done:
                            recent.append(dict(return_=instance.ep_reward, **info))
                            observation = instance.reset()
                        next_obs.append(observation)
                    rewards.append(step_rewards)
                    dones.append(step_dones)
                    terminals.append(step_terminals)
                    next_truth_buf.append(step_truth)
                    obs = np.stack(next_obs)
                    if renderer:
                        renderer.draw(env)
                arrays = [np.asarray(x, dtype=np.float32) for x in
                          (obs_buf, act_buf, rewards, dones, logps)]
                observations, actions, rew, done_array, old_logp = arrays
                # rollout内Critic/归一化冻结，集中推理与逐步推理数学等价。
                val = ppo.value_batch(np.asarray(truth_buf,dtype=np.float32).reshape(-1,C.OBS_DIM)).reshape(rew.shape)
                next_values = ppo.value_batch(np.asarray(next_truth_buf,dtype=np.float32).reshape(-1,C.OBS_DIM)).reshape(rew.shape)
                adv, ret = compute_gae(rew, val, done_array, 0.0,
                                       np.asarray(next_values), np.asarray(terminals))
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                stats = ppo.update(observations.reshape(-1, C.OBS_DIM),
                                   actions.reshape(-1, C.ACT_DIM), old_logp.reshape(-1),
                                   ret.reshape(-1), adv.reshape(-1),
                                   np.asarray(truth_buf, dtype=np.float32).reshape(-1, C.OBS_DIM))
                def average(key):
                    return float(np.mean([episode[key] for episode in recent])) if recent else 0.0
                row = dict(iteration=it, theta_deg=math.degrees(env.theta),
                           p_attach=env.p_attach, climb=average("climb"),
                           survival=average("survived"), velocity_rmse=average("velocity_rmse"),
                           **{"return": average("return_")},
                           **{key: stats[key] for key in
                              ("actor_loss", "critic_loss", "approx_kl", "clip_fraction", "estimator_loss",
                               "entropy", "actor_grad_norm", "critic_grad_norm")},
                           execution_clip_fraction=float(np.mean(np.abs(actions)>1)))
                writer.writerow(row)
                file.flush()
                if it % C.LOG_EVERY == 0 or it == total_iters - 1:
                    print(f"iter {it:4d} | theta {row['theta_deg']:5.1f} | "
                          f"p {env.p_attach:.3f} | climb {row['climb']:+.3f}m | "
                          f"survival {row['survival']:.1%} | RMSE {row['velocity_rmse']:.3f} | "
                          f"KL {stats['approx_kl']:.4f}")
                if it > 0 and it % C.SAVE_EVERY == 0:
                    ppo.save(directory / f"ckpt_{it}.pt")
        ppo.save(directory / "final.pt")
        print(f"saved {directory / 'final.pt'}")
    finally:
        if renderer:
            renderer.close()
