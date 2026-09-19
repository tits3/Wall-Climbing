"""固定条件独立评估；脚本与学习策略明确区分，恢复率按事件聚合。"""
import json
import csv
from pathlib import Path
import numpy as np
import config as C
from env import WallClimbEnv
from controllers import scripted_action


def evaluate(args):
    from main import make_policy
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.disturb_at is not None and not 0 <= args.disturb_at < C.EPISODE_SECONDS:
        raise ValueError("disturb-at must lie inside the evaluation horizon")
    policy = make_policy(args.checkpoint, args.seed, args.demo)
    if policy is not None:
        import torch
        torch.set_num_threads(1)
    renderer = None
    if args.render:
        from render import Renderer
        renderer = Renderer(controller="SCRIPTED BASELINE" if args.demo else "PPO evaluation")
    episodes, delays, surviving_delays = [], [], []
    event_count = 0
    source_delays = {name: [] for name in ("probabilistic", "forced_slip")}
    source_surviving_delays = {name: [] for name in source_delays}
    source_counts = {name: 0 for name in source_delays}
    trace = []
    try:
        for episode in range(args.episodes):
            env = WallClimbEnv(args.seed + episode, random_commands=True, domain_randomization=True)
            env.set_conditions(args.theta, args.p_attach, args.theta > 0)
            env.iter = policy.iteration if policy is not None else C.ITER_FAIL_END
            obs = env.reset()
            injected = False
            injection_succeeded = False
            for step in range(C.MAX_STEPS):
                if renderer:
                    quit_, _, _ = renderer.poll()
                    if quit_:
                        return None
                if (not injected and args.disturb_at is not None
                        and step * C.DT >= args.disturb_at):
                    injection_succeeded = env.disturb()
                    obs = env.observe()
                    injected = True
                action = scripted_action(env) if policy is None else policy.act(
                    obs, deterministic=not args.stochastic, privileged=env.privileged())[0]
                if policy is not None:
                    env.set_contact_estimate(policy.last_contact)
                obs, _, done, info = env.step(action)
                if episode == 0:
                    row = dict(time=info["time"], position=env.y_b, velocity=env.v_b,
                               command=env.v_desired, reward=_, n_attached=info["n_attached"])
                    for foot in range(4):
                        row[f"gap_{foot}"] = float(env.x[foot])
                        row[f"contact_{foot}"] = int(env.contact[foot])
                        row[f"magnet_{foot}"] = int(env.a[foot])
                    trace.append(row)
                if renderer:
                    renderer.draw(env)
                if done:
                    break
            episodes.append({key: info[key] for key in
                             ("survived", "fell", "stuck", "time", "climb", "velocity_rmse",
                              "retention", "failures", "attempts", "recovery_events", "recovery")})
            episodes[-1].update(command=env.v_desired, seed=args.seed + episode,
                                friction=env.friction, action_delay=env.action_delay,
                                total_reward=env.ep_reward,
                                forced_injection_succeeded=bool(injection_succeeded))
            delays.extend(env.recovery_delays)
            if not info["terminated"]:
                surviving_delays.extend(env.recovery_delays)
            event_count += info["recovery_events"]
            for source in source_counts:
                recovered = [r["delay"] for r in env.recovery_records if r["source"] == source]
                source_counts[source] += len(recovered) + sum(s == source for s in env.pending_sources)
                source_delays[source].extend(recovered)
                if not info["terminated"]:
                    source_surviving_delays[source].extend(recovered)
    finally:
        if renderer:
            renderer.close()
    summary = {"survival_rate": float(np.mean([e["survived"] for e in episodes])),
               "early_termination_rate": float(np.mean([e["fell"] or e["stuck"] for e in episodes])),
               "recovery_events": event_count,
               "recovery": {str(t): sum(d <= t for d in surviving_delays) / event_count if event_count else None
                            for t in C.RECOVERY_WINDOWS},
               "reattachment": {str(t): sum(d <= t for d in delays) / event_count if event_count else None
                                for t in C.RECOVERY_WINDOWS}}
    summary["forced_injection_success_count"] = sum(e["forced_injection_succeeded"] for e in episodes)
    summary["recovery_by_source"] = {
        source: {"events": source_counts[source],
                 "recovery": {str(t): sum(d <= t for d in source_surviving_delays[source]) / source_counts[source]
                              if source_counts[source] else None for t in C.RECOVERY_WINDOWS},
                 "reattachment": {str(t): sum(d <= t for d in source_delays[source]) / source_counts[source]
                                  if source_counts[source] else None for t in C.RECOVERY_WINDOWS}}
        for source in source_counts}
    for key in ("time", "climb", "velocity_rmse", "retention"):
        values = [e[key] for e in episodes]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    result = {"model_version": C.MODEL_VERSION,
              "training_variant": policy.training_variant if policy else None,
              "checkpoint_iteration": policy.iteration if policy else None,
              "controller": "scripted_baseline" if args.demo else "ppo",
              "action_mode": "scripted" if args.demo else "stochastic" if args.stochastic else "deterministic",
              "conditions": {"theta_deg": args.theta, "p_attach": args.p_attach,
                             "horizon_seconds": C.EPISODE_SECONDS, "episodes": args.episodes,
                             "seed": args.seed, "disturb_at": args.disturb_at,
                             "command_range": C.COMMAND_RANGE,
                             "friction_range": C.FRICTION_RANGE,
                             "action_delay_max": C.ACTION_DELAY_MAX,
                             "reward_iteration": policy.iteration if policy is not None else C.ITER_FAIL_END,
                             "metric_scope": "single_tangent_velocity_only"},
              "recovery_definition": "reattachment within window AND survival to 10s; late events kept in denominator; conservative assumption",
              "summary": summary, "episodes": episodes}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if trace:
        with path.with_suffix(".trajectory.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"controller={result['controller']} saved {path}")
    return result
