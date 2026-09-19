"""论文核心机制的二维近似；除注明外，参数是演示设置而非论文原值。"""
import math

MODEL_VERSION = "paper-planar-joints-v3.8"
DT = 0.02
G = 9.81
MASS = 8.0                 # 论文机器人质量，kg
FOOT_COUNT = 4
FOOT_NAMES = ("RR", "FR", "RL", "FL")
HIP_OFFSETS = (-0.12, 0.12, -0.12, 0.12)
V_FOOT_MAX = 2.0           # 25%摆动下，满足 |vx|max=.5 的最低世界足速 4*.5
LEG_REACH = 0.35           # 相对髋部沿壁可达距离，m
GAIT_PERIOD = 1.2          # 论文步态周期，s；25% 摆动
SWING_HEIGHT = 0.08        # 论文摆动足高，m
FOOT_NORMAL_SPEED = 1.0    # m/s，二维足抬起/落下速度
CONTACT_GAP = 0.002        # 接触识别容差，m
ALIGN_GAP = 1e-9           # Eq.4 完整接触的浮点容差；不是允许 1mm 气隙
ATTACH_RETRY_TIME = 0.12   # 失败后的重试间隔，s（二维假设）
FILTER_ALPHA = 0.35        # 论文观测低通系数
FRICTION = 0.4
MAGNET_FORCE = 697.0       # 论文 EPM 最大法向保持力，N；只在完整对齐时施加
DRIVE_GAIN = 18.0          # 支撑腿速度伺服增益，1/s
BODY_DAMPING = 1.0         # 速度阻尼，1/s
SUPPORT_FEET = 2
LINK_LENGTHS = (0.175, 0.175)  # 未公开URDF：沿用原可达范围0.35m，等长二连杆假设
BODY_NORMAL_HEIGHT = 0.20     # 二维固定身体离墙距离假设，不是论文值
def _joint_target_scale():
    # 覆盖论文命令|vx|<=.5、75%支撑期和0.08m摆动高；不是拟合实验曲线。
    l1, l2 = LINK_LENGTHS
    nominal_knee = -math.acos((BODY_NORMAL_HEIGHT**2-l1*l1-l2*l2)/(2*l1*l2))
    nominal_hip = -math.atan2(l2*math.sin(nominal_knee),l1+l2*math.cos(nominal_knee))
    angles = []
    for tangent in (-.5*GAIT_PERIOD*.375, .5*GAIT_PERIOD*.375):
        for gap in (-CONTACT_GAP, SWING_HEIGHT):
            height = BODY_NORMAL_HEIGHT-gap
            knee = -math.acos((tangent*tangent+height*height-l1*l1-l2*l2)/(2*l1*l2))
            hip = math.atan2(tangent,height)-math.atan2(l2*math.sin(knee),l1+l2*math.cos(knee))
            angles.extend((abs(hip-nominal_hip),abs(knee-nominal_knee)))
    return max(angles)


JOINT_ACTION_SCALE = _joint_target_scale()  # rad；以上述可行域计算得到
JOINT_INERTIA = 0.002         # kg m^2；二维独立关节惯量假设，不是作者参数
JOINT_P_RANGE = (0.4, 0.6)
JOINT_D_RANGE = (0.12, 0.18)
BODY_INIT_Y = 6.0          # 10s * 最大反向指令 .5m/s + 1m，避免边界冒充坠落
MIN_HEIGHT = 0.05
WALL_HEIGHT = 3.0          # 仅渲染窗口尺寸，物理表面向上无限延伸
EPISODE_SECONDS = 10.0
MAX_STEPS = round(EPISODE_SECONDS / DT)
STUCK_STEPS = round(5.0 / DT)
V_DESIRED = 0.12           # 单方向速度跟踪指令，m/s
COMMAND_RANGE = (-0.5, 0.5)  # 论文 vx 指令范围；二维省略 vy/偏航
FRICTION_RANGE = (0.3, 0.5)
ACTION_DELAY_MAX = 0.008
RECOVERY_WINDOWS = (1.2, 2.4, 3.6)

# 论文 Eq.5–7 的原始迭代节点；每迭代样本预算另列为假设。
ITER_FLAT_END = 1200
ITER_TILT_END = 21200
ITER_FAIL_END = 35000
P_ATTACH_MIN = 0.85
TOTAL_ITERS = 35001

# 6全局 + 4*7(q2,qd2,切向位置,法向高度,接触) + 8时钟 + 两步12维目标。
OBS_DIM = 58
ACT_DIM = 12                # 二维8关节位置目标 + 4磁信号；原论文为12+4
HIDDEN = [256, 128, 64]
ESTIMATOR_HIDDEN = [256, 128]
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
EPOCHS = 4
MINIBATCH = 64              # 论文未公布，CPU 单环境实验假设
ENT_COEF = 0.001
MAX_GRAD_NORM = 0.5
LOG_EVERY = 1000
SAVE_EVERY = 5000
TARGET_KL = 0.02
ROLLOUT_STEPS = 64          # 35001 次完整课程，共 2240064 环境步
NUM_ENVS = 2               # 同64样本预算：2环境x32步=每环境.64s，覆盖两段摆动时长
SEED = 0
CHECKPOINT_DIR = "experiments/paper_2d/revision_v3/train"
SUITE_DIR = "experiments/paper_2d/revision_v3/final_v38"
CONTACT_OBS_INDICES = [12, 19, 26, 33]
HEIGHT_OBS_INDICES = [11, 18, 25, 32]


def theta_schedule(iteration):
    fraction = min(max((iteration - ITER_FLAT_END) /
                       (ITER_TILT_END - ITER_FLAT_END), 0.0), 1.0)
    return math.pi / 2 * fraction


def p_attach_schedule(iteration):
    fraction = min(max((iteration - ITER_TILT_END) /
                       (ITER_FAIL_END - ITER_TILT_END), 0.0), 1.0)
    return 1.0 - (1.0 - P_ATTACH_MIN) * fraction
