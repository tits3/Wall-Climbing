"""脚本基线检查二维环境是否支持各倾角爬行；不是学习策略成绩。"""
from env import WallClimbEnv
from controllers import scripted_action
import config as C


def run_scripted(theta_deg, probability=1.0, seed=C.SEED):
    env = WallClimbEnv(seed)
    env.set_conditions(theta_deg, probability, theta_deg > 0)
    env.reset()
    for _ in range(C.MAX_STEPS):
        _, _, done, info = env.step(scripted_action(env))
        if done:
            break
    print(f"SCRIPTED theta={theta_deg} p={probability}: "
          f"survived={info['survived']} climb={info['climb']:+.3f}m "
          f"RMSE={info['velocity_rmse']:.3f}m/s recovery={info['recovery']}")
    return info


if __name__ == "__main__":
    for angle in (0, 45, 90):
        run_scripted(angle)
    run_scripted(90, .85)
