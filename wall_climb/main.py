"""二维演示、PPO 训练和固定协议评估入口。"""
import argparse
import config as C
from env import WallClimbEnv
from controllers import scripted_action


def make_policy(checkpoint, seed, demo=False):
    if demo:
        return None
    if not checkpoint:
        raise ValueError("PPO play/eval requires --checkpoint; use --demo for scripted baseline")
    from ppo import PPO
    import torch
    torch.set_num_threads(1)
    policy = PPO(C.OBS_DIM, C.ACT_DIM, seed=seed)
    policy.load(checkpoint)
    return policy


def play(args):
    from render import Renderer
    env = WallClimbEnv(args.seed)
    if args.curriculum:
        env.set_curriculum(0)
    else:
        env.set_conditions(args.theta, args.p_attach, args.theta > 0)
    obs = env.reset()
    policy = make_policy(args.checkpoint, args.seed, args.demo)
    if not args.curriculum:
        env.iter = policy.iteration if policy is not None else C.ITER_FAIL_END
    label = ("SCRIPTED BASELINE" if args.demo else
             "PPO sampled" if args.stochastic else "PPO deterministic")
    renderer = Renderer(controller=label)
    try:
        while True:
            quit_, force, reset = renderer.poll()
            if quit_:
                break
            if reset:
                if args.curriculum:
                    env.set_curriculum(0)
                obs = env.reset()
            if force:
                env.disturb()
                obs = env.observe()
            if args.curriculum:
                iteration = round(env.step_count / (C.MAX_STEPS - 1) * C.ITER_FAIL_END)
                env.set_curriculum(iteration)
                obs = env.observe()
            action = scripted_action(env) if policy is None else policy.act(
                obs, deterministic=not args.stochastic, privileged=env.privileged())[0]
            if policy is not None:
                env.set_contact_estimate(policy.last_contact)
            obs, _, done, info = env.step(action)
            renderer.draw(env)
            if done:
                print(label, {key: info[key] for key in
                              ("survived", "climb", "velocity_rmse", "retention", "recovery")})
                if args.curriculum:
                    env.set_curriculum(0)
                obs = env.reset()
    finally:
        renderer.close()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--play", action="store_true")
    mode.add_argument("--eval", action="store_true")
    mode.add_argument("--experiment", action="store_true", help="complete training and ablation suite")
    parser.add_argument("--demo", action="store_true", help="explicit scripted baseline")
    parser.add_argument("--checkpoint")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--iters", type=int, default=C.TOTAL_ITERS)
    parser.add_argument("--num-envs", type=int, default=C.NUM_ENVS,
                        help="implementation assumption; must divide rollout samples")
    parser.add_argument("--save-dir", default=C.CHECKPOINT_DIR)
    parser.add_argument("--theta", type=float, default=90.0)
    parser.add_argument("--p-attach", type=float, default=C.P_ATTACH_MIN)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--disturb-at", type=float, help="evaluation: force one attached foot off at this time, seconds")
    parser.add_argument("--stochastic", action="store_true", help="sample PPO actions during evaluation")
    parser.add_argument("--curriculum", action="store_true",
                        help="fast three-phase scripted demonstration within one episode")
    parser.add_argument("--output", default="evaluation_2d.json")
    parser.add_argument("--variant", choices=("full", "no_curriculum", "no_probabilistic", "no_modeling"),
                        default="full", help="training ablation; evaluation always uses realistic adhesion")
    args = parser.parse_args()
    if args.num_envs < 1 or C.ROLLOUT_STEPS % args.num_envs:
        parser.error("num-envs must be positive and divide ROLLOUT_STEPS")
    C.NUM_ENVS = args.num_envs
    if args.demo and args.train:
        parser.error("--demo is a baseline for --play or --eval")
    if args.curriculum and (not args.demo or args.eval or args.train):
        parser.error("--curriculum is only for the scripted --demo --play illustration")
    if args.experiment:
        from run_experiment import run
        output = C.SUITE_DIR if args.output == "evaluation_2d.json" else args.output
        run(output, args.seed, args.iters)
    elif args.eval:
        from evaluate import evaluate
        evaluate(args)
    elif args.play or args.demo:
        play(args)
    else:
        from train import train
        train(args.render, args.save_dir, args.iters, args.seed, args.variant)


if __name__ == "__main__":
    main()
