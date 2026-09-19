# Wall-Climbing RL — 论文机制二维复现

本项目复现论文 **“Reinforcement Learning-based Robust Wall Climbing Locomotion Controller in Ferromagnetic Environment”**（Yong Um 等，arXiv:2510.20174）。论文地址：[arXiv:2510.20174](https://arxiv.org/abs/2510.20174)。

这是一个便于检查和演示的 **二维 pygame 近似实现**，保留了论文的主要机制：磁足接触与附着门控、概率附着失败、三阶段课程学习、状态估计器、PPO、GAE、论文奖励结构以及消融实验。它不是作者三维 RaiSim、真实机器人或硬件实验的完整复现。

## 当前复现程度

当前版本为 `paper-planar-joints-v3.8`。四种配置各训练 35,001 次 PPO 更新，每种模型约 224 万环境步，并使用固定末检查点评估。42 项核心回归检查通过。

实现已对齐或显式投影的内容包括：

- 四只磁足、接触置信度阈值和独立磁铁命令；
- 最大磁力 697 N、8 kg 质量、1.2 s 步态周期和 0.08 m 摆足高度；
- 平地爬行、重力旋转、概率附着失败三个课程阶段；
- Actor、Critic、状态估计器及最近两步关节目标历史；
- PPO clipped surrogate、GAE、观测归一化及论文奖励项；
- Full、No Curriculum、No Probabilistic Failure、No Modeling 四组训练。

最终 PPO 策略仍退化为接近静止的策略，尚未复现论文中的有效速度跟踪和鲁棒连续爬行。项目保留这个负结果，没有为匹配论文曲线而调整无依据的参数。

## 目录结构

```text
.
├── README.md
├── .gitignore
├── run.ps1                 # Windows PowerShell 启动器
├── launch_2d.py            # Python 入口包装与路径设置
├── wall_climb/
│   ├── config.py           # 物理参数、课程、网络和 PPO 配置
│   ├── env.py              # 二维环境、接触、磁附着和随机化
│   ├── kinematics.py       # 二连杆 FK/IK
│   ├── rewards.py          # 论文奖励的二维投影
│   ├── ppo.py              # Actor、Critic、Estimator、GAE、PPO
│   ├── train.py            # 训练循环
│   ├── evaluate.py         # 固定协议评估
│   ├── run_experiment.py   # 完整模型和三项消融
│   ├── render.py           # pygame 可视化
│   ├── controllers.py      # 脚本测试控制器
│   ├── test_core.py        # 42 项回归检查
│   ├── diagnose_policy.py  # 响应、饱和及动作裁剪诊断
│   ├── diag_gait.py        # 脚本步态诊断
│   ├── requirements.txt
│   ├── requirements.lock.txt
│   └── checkpoints/full_v38.pt
└── results/
    ├── README.md
    ├── comparison.csv
    ├── training.png
    └── representative_trace.png
```

## 环境与依赖

本次运行环境：Python 3.12、NumPy 2.5.2、PyTorch 2.13.0+cpu、pygame 2.6.1。

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r wall_climb\requirements.txt
```

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r wall_climb/requirements.txt
```

PyTorch 的版本和安装命令可能随平台而异；如普通 `pip` 无法安装，请使用 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。

## 运行

Windows 上从项目根目录执行：

```powershell
# 查看最终 PPO 策略；F 人为失附，R 重置，Esc 退出
.\run.ps1 --play --checkpoint checkpoints\full_v38.pt

# 脚本测试控制器（不是 PPO 成绩）
.\run.ps1 --demo --play
.\run.ps1 --demo --play --curriculum

# 重新评估 100 回合
.\run.ps1 --eval --checkpoint checkpoints\full_v38.pt --episodes 100 --output evaluation_v38.json

# 单模型从头训练；使用新目录
.\run.ps1 --train --seed 0 --save-dir experiments\new_train

# 四模型训练与完整评估
.\run.ps1 --experiment --output experiments\new_suite
```

其他平台可直接使用：

```bash
python launch_2d.py --play --checkpoint checkpoints/full_v38.pt
python launch_2d.py --demo --play
python launch_2d.py --eval --checkpoint checkpoints/full_v38.pt --episodes 100 --output evaluation_v38.json
```

`launch_2d.py` 会将工作目录切换到 `wall_climb`，因此上述检查点和输出路径均相对于该目录。完整训练在本次 Windows CPU 环境中约耗时 95 分钟。

运行测试：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 wall_climb\test_core.py
```

## 主要结果

| 条件 | 二维 RMSE (m/s) | 论文 RMSE (m/s) | 早终止率 | 平均存活时间 | 磁足保持率 |
|---|---:|---:|---:|---:|---:|
| Full, `p=1.0` | 0.234 ± 0.144 | 0.102 ± 0.089 | 0% | 10.0 s | 74.72% |
| Full, `p=0.85` | 0.234 ± 0.144 | 0.557 ± 7.344 | 0% | 10.0 s | 74.72% |

虽然回合全部存活到 10 秒，命令响应回归斜率约为 0，RMSE 相对同一命令下静止参考还下降了约 -2.52%（即略差）。因此这些存活结果不能解释为成功爬行。自然评估没有产生可统计的自然失附事件；独立强制失附实验为 100 个事件、1.2/2.4/3.6 秒窗口内恢复率均为 33%，其协议不能与论文自然失附恢复率直接等同。

训练图与固定评估轨迹见 [`results/`](results/README.md)，完整数值见 [`results/comparison.csv`](results/comparison.csv)。

## 当前问题

1. 环境只有沿墙方向和足法向的二维代理，没有三维刚体姿态、侧向、偏航、倾覆和真实脱壁动力学。
2. 身体运动使用切向伺服和摩擦上限代理；关节力矩没有包含完整支撑载荷。
3. 连杆长度、惯量、关节范围、PPO 超参数、控制周期等论文未公开内容采用了明确记录的假设。
4. 二维奖励投影中部分三维项变成常数，可能强化静止策略。
5. 最终策略的执行动作裁剪比例和隐藏层饱和较高，固定命令诊断显示速度响应基本不随命令变化。
6. 当前完整结果只有一个训练种子，不能估计跨种子收敛置信度。
7. 论文表中 Full、`p=0.85` 的 RMSE 标准差与其均值及 100 回合口径存在统计疑问，项目按论文原值引用，未擅自更正。

后续应优先替换二维身体动力学代理、检查奖励投影与动作参数化，并在固定协议下运行多个训练种子，而不是只延长当前训练。

## 说明

原论文 PDF、虚拟环境、运行日志、逐回合大型数据、历史中间实验和本机路径没有提交到仓库。最终检查点约 1.9 MB，用于直接运行 pygame 演示。项目仅用于研究复现与机制验证；论文及其内容归原作者所有。
