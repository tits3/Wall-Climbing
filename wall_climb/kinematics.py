"""二维二连杆运动学；URDF未公开，几何和惯量选择见实验假设。"""
import numpy as np
import config as C


def forward(q):
    l1, l2 = C.LINK_LENGTHS
    a, b = q[..., 0], q[..., 0] + q[..., 1]
    tangent = l1 * np.sin(a) + l2 * np.sin(b)
    normal = l1 * np.cos(a) + l2 * np.cos(b)
    return tangent, C.BODY_NORMAL_HEIGHT - normal


def inverse(tangent, gap, reference=None):
    l1, l2 = C.LINK_LENGTHS
    normal = C.BODY_NORMAL_HEIGHT - np.asarray(gap)
    tangent = np.asarray(tangent)
    cosine = np.clip((tangent**2 + normal**2 - l1*l1 - l2*l2) / (2*l1*l2), -1, 1)
    knee = -np.arccos(cosine)
    hip = np.arctan2(tangent, normal) - np.arctan2(l2*np.sin(knee), l1+l2*np.cos(knee))
    if reference is not None:
        previous = np.asarray(reference)[..., 0]
        hip = previous + np.arctan2(np.sin(hip-previous), np.cos(hip-previous))
        # 完全折叠时足点不能确定髋角；沿用连续关节坐标。
        singular = tangent**2+normal**2 <= np.finfo(float).eps*(l1+l2)**2
        hip = np.where(singular, previous, hip)
    return np.stack((hip, knee), axis=-1)


def nominal():
    return inverse(np.zeros(4), np.zeros(4))


def joint_targets(action):
    targets = nominal() + C.JOINT_ACTION_SCALE * np.asarray(action).reshape(4, 3)[:, :2]
    # 所选负膝角IK分支的机械范围；没有作者URDF，明确为二维假设。
    targets[:,1] = np.clip(targets[:,1], -np.pi+1e-6, -1e-6)
    return targets


def action_for_pose(tangent, gap, magnet):
    result = np.zeros((4, 3))
    result[:, :2] = (inverse(tangent, gap) - nominal()) / C.JOINT_ACTION_SCALE
    result[:, 2] = 2*np.asarray(magnet)-1
    return np.clip(result.ravel(), -1, 1).astype(np.float32)
