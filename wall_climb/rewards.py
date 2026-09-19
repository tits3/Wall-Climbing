"""表I、Eq.8–14的二维投影：关节角rad、速度rad/s、力矩N m。"""
import math
import numpy as np
import config as C
from kinematics import joint_targets, nominal


def paper_action(action):
    """执行动作转换为物理关节目标rad与磁信号[0,1]。"""
    result = np.asarray(action, dtype=np.float64).copy()
    result = result.reshape(4, 3)
    result[:, :2] = joint_targets(action)
    result[:, 2] = (result[:, 2] + 1.0) / 2
    result = result.ravel()
    return result


def paper_reward(iteration, phases, contact, foot_height, foot_normal_velocity,
                 foot_tangent_velocity, velocity, desired_velocity,
                 action, previous_action, older_action, magnet, phase1=None,
                 joint_position=None, joint_velocity=None, joint_acceleration=None, torque=None):
    contact = np.asarray(contact, dtype=bool)
    swing = (phases > 0) & (phases < math.pi / 2)
    desired_height = np.where(swing, 0.08, 0.0)
    kappa = 0.99975 ** max(iteration - 1200, 0)
    tracking_scale = 1.5 - 0.5 * kappa
    penalty_scale = 0.5 + 0.5 * kappa
    standing = desired_velocity == 0.0
    if phase1 is None:
        phase1 = iteration <= 1200
    smooth_enabled = not (phase1 and iteration < 1000)
    joint_position = nominal() if joint_position is None else np.asarray(joint_position)
    joint_velocity = np.zeros((4,2)) if joint_velocity is None else np.asarray(joint_velocity)
    joint_acceleration = np.zeros((4,2)) if joint_acceleration is None else np.asarray(joint_acceleration)
    torque = np.zeros((4,2)) if torque is None else np.asarray(torque)
    terms = {
        "Rlv": tracking_scale * 3 * math.exp(-5 * (desired_velocity - velocity) ** 2),
        # 二维模型无侧向和偏航，自身及指令角速度均固定为0。
        "Rav": tracking_scale * 3,
        "Rsc": 0.5 * float(np.where(contact & standing, 1, -1).sum()),
        "Rg": 0.5 * float(np.where(contact == ~swing, 1, -1).sum()),
        "Rfh": 0.5 * math.exp(-float((swing * (desired_height - foot_height) ** 2).sum())),
        "Rfs": penalty_scale * 0.5 * float((contact * foot_tangent_velocity ** 2).sum()),
        "Rfc": 140 * float(((~contact) * (desired_height - foot_height) ** 2 *
                           np.abs(foot_normal_velocity) ** 0.5).sum()),
        "Ro": 0.0,
        "Rtau": penalty_scale * 0.003 * float(np.square(torque).sum()),
        "Rjp": (3.0 if standing else .75) * float(np.square(joint_position-nominal()).sum()),
        "Rjs": .003 * float(np.square(joint_velocity).sum()),
        "Rja": .003 * float(np.square(joint_acceleration).sum()),
        "Ras1": 2.5 * float(((action - previous_action) ** 2).sum()) if smooth_enabled else 0.0,
        "Ras2": 1.2 * float(((action - 2 * previous_action + older_action) ** 2).sum())
                if smooth_enabled else 0.0,
        # 按表I字面公式：固定身体姿态、零法向速度时 Rbm=3。
        "Rbm": 3.0,
        "Ram": 0.15 * float(((contact.astype(float) - magnet) ** 2).sum()),
    }
    positive = sum(terms[k] for k in ("Rlv", "Rav", "Rg", "Rfh", "Rsc"))
    penalties = sum(terms[k] for k in
                    ("Rfs", "Rfc", "Ro", "Rtau", "Rjp", "Rjs", "Rja",
                     "Ras1", "Ras2", "Rbm", "Ram"))
    return positive * math.exp(-0.2 * penalties), terms
