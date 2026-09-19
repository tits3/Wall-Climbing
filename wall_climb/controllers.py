"""脚本基线仅用于环境验证与演示，不作为 PPO 学习结果。"""
import math
import numpy as np
import config as C
from kinematics import action_for_pose


def scripted_action(env):
    tangent = env.y - env.y_b - env.hip_offsets
    # 脚本控制器压向表面直至发生碰撞，避免把趋近0的浮点间隙当完整接触。
    gap = np.full(4, -C.CONTACT_GAP)
    magnet = np.ones(4)
    for i, phi in enumerate(env.phases()):
        swing = 0 < phi < math.pi / 2
        if swing:
            progress = phi / (math.pi / 2)
            tangent[i] = env.v_desired * C.GAIT_PERIOD * .375
            gap[i] = C.SWING_HEIGHT * math.sin(math.pi * progress)
            magnet[i] = 0
        else:
            tangent[i] -= env.v_desired * (1 + C.BODY_DAMPING / C.DRIVE_GAIN) / C.DRIVE_GAIN
    return action_for_pose(tangent, gap, magnet)
