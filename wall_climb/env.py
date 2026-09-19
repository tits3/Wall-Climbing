"""二维表面爬行：切向位置 y + 足法向间隙 x；非刚体动力学复刻。"""
import math
import numpy as np
import config as C
from rewards import paper_reward, paper_action
from kinematics import forward, inverse, nominal, joint_targets


class WallClimbEnv:
    def __init__(self, seed=C.SEED, random_commands=False, domain_randomization=False,
                 variant="full"):
        self.hip_offsets = np.asarray(C.HIP_OFFSETS, dtype=np.float64)
        self.phase_offsets = np.arange(1, 5) * math.pi / 2
        self.foot_count = C.FOOT_COUNT
        self.obs_dim, self.act_dim = C.OBS_DIM, C.ACT_DIM
        self.rng = np.random.default_rng(seed)
        self.command_rng = np.random.default_rng(seed + 100000)
        self.domain_rng = np.random.default_rng(seed + 200000)
        self.noise_rng = np.random.default_rng(seed + 300000)
        self.random_commands = random_commands
        self.domain_randomization = domain_randomization
        self.variant = variant
        self.iter, self.theta, self.p_attach = 0, 0.0, 1.0
        self.adhesion_enabled = False
        self.v_desired = C.V_DESIRED
        self.reset()

    def set_curriculum(self, iteration):
        self.iter = iteration
        self.theta = C.theta_schedule(iteration)
        self.p_attach = C.p_attach_schedule(iteration)
        self.adhesion_enabled = iteration > C.ITER_FLAT_END
        if self.variant == "no_curriculum":
            self.theta, self.adhesion_enabled = math.pi / 2, True
        if self.variant in ("no_probabilistic", "no_modeling"):
            self.p_attach = 1.0

    def set_conditions(self, theta_deg=90.0, p_attach=0.85, adhesion=True):
        if not 0 <= theta_deg <= 90 or not 0 <= p_attach <= 1:
            raise ValueError("theta must be 0..90 degrees; p_attach must be 0..1")
        self.theta = math.radians(theta_deg)
        self.p_attach = float(p_attach)
        self.adhesion_enabled = bool(adhesion)

    def phases(self):
        return (self.phase + self.phase_offsets) % (2 * math.pi)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.command_rng = np.random.default_rng(seed + 100000)
            self.domain_rng = np.random.default_rng(seed + 200000)
            self.noise_rng = np.random.default_rng(seed + 300000)
        self.v_desired = (float(self.command_rng.uniform(*C.COMMAND_RANGE))
                          if self.random_commands else C.V_DESIRED)
        self.friction = (float(self.domain_rng.uniform(*C.FRICTION_RANGE))
                         if self.domain_randomization else C.FRICTION)
        self.action_delay = (float(self.domain_rng.uniform(0, C.ACTION_DELAY_MAX))
                             if self.domain_randomization else 0.0)
        self.contact_estimate = None
        self.orientation_bias = (float(self.noise_rng.uniform(-.05,.05)) if self.domain_randomization else 0.0)
        self.joint_p = (float(self.domain_rng.uniform(*C.JOINT_P_RANGE)) if self.domain_randomization else .5)
        self.joint_d = (float(self.domain_rng.uniform(*C.JOINT_D_RANGE)) if self.domain_randomization else .15)
        self.event_time = 0.0
        self.y_b, self.v_b = C.BODY_INIT_Y, 0.0
        self.y = self.y_b + self.hip_offsets
        self.x = np.zeros(4, dtype=np.float64)
        self.joint_q = nominal()
        self.joint_velocity = np.zeros((4, 2))
        self.joint_acceleration = np.zeros((4, 2))
        self.joint_torque = np.zeros((4, 2))
        # 初始足已放在表面；Phase 1 只有机械接触、不施加磁力。
        self.a = np.full(4, float(self.adhesion_enabled))
        self.contact = np.ones(4, dtype=bool)
        self.q = np.ones(4)
        self.retry = np.zeros(4)
        self.failed_activation = np.zeros(4, dtype=bool)
        self.phase, self.step_count, self.stuck_steps = 0.0, 0, 0
        self.prev_action = np.zeros(C.ACT_DIM)
        self.older_action = np.zeros(C.ACT_DIM)
        self.ep_reward, self.start_y = 0.0, self.y_b
        self.error_sum = 0.0
        self.stance_samples = self.retained_samples = self.failures = 0
        self.attempts = 0
        self.pending = [None] * 4
        self.pending_sources = [None] * 4
        self.recovery_delays = []
        self.recovery_records = []
        self.episode_terminated = False
        self.last_event = "ready"
        self.obs_filtered = self._raw_obs()
        return self.observe()

    def _raw_obs(self, noisy=False):
        data = [self.v_b / C.V_FOOT_MAX, self.v_desired / C.V_FOOT_MAX,
                0.0, 1.0,  # 固定二维身体姿态；旋转重力不是身体朝向
                0.0, 0.0]  # 固定二维身体角速度；不泄露时间限制或终止计时
        relative = self.y - self.y_b - self.hip_offsets
        for i in range(4):
            data.extend([*self.joint_q[i], *self.joint_velocity[i],
                         relative[i], self.x[i] / C.SWING_HEIGHT, float(self.contact[i])])
        for phi in self.phases():
            data.extend([math.sin(phi), math.cos(phi)])
        data.extend(paper_action(self.prev_action).reshape(4, 3)[:, :2].ravel())
        data.extend(paper_action(self.older_action).reshape(4, 3)[:, :2].ravel())
        result = np.asarray(data, dtype=np.float32)
        if noisy:
            noisy_angle = self.orientation_bias + self.noise_rng.uniform(-.05,.05)
            result[2:4] = [math.sin(noisy_angle), math.cos(noisy_angle)]
            result[4:6] += self.noise_rng.uniform(-.1,.1,2)
            for i in range(4):
                j = 6 + 7*i
                result[j:j+2] += self.noise_rng.uniform(-.1, .1, 2)
                result[j+2:j+4] += self.noise_rng.uniform(-.5, .5, 2)
                result[j+4] += self.noise_rng.uniform(-.015, .015)
                result[j+5] += self.noise_rng.uniform(-.015, .015) / C.SWING_HEIGHT
            for j in (42, 50):
                for i in range(4):
                    result[j+2*i:j+2*i+2] += self.noise_rng.uniform(-.1, .1, 2)
        return result

    def observe(self):
        """所有观测统一低通；课程改变造成的短暂滤波滞后保留。"""
        return self.obs_filtered.copy()

    def privileged(self):
        """无噪声、无滤波真值，专用于 Critic 和估计器监督。"""
        result = self._raw_obs()
        result[2:4] = [math.sin(self.theta), math.cos(self.theta)]  # 仅Critic已知旋转重力
        return result

    def set_contact_estimate(self, probabilities):
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("contact estimate must be four finite probabilities")
        self.contact_estimate = np.clip(values, 0, 1)

    def _failure(self, foot, event):
        self.failures += 1
        if self.pending[foot] is None:
            self.pending[foot] = self.event_time
            self.pending_sources[foot] = "forced_slip" if event == "forced slip" else "probabilistic"
        self.a[foot] = 0.0
        if event == "forced slip":
            self.x[foot] = max(self.x[foot], 0.015)
            self.joint_q[foot] = inverse(self.y[foot]-self.y_b-self.hip_offsets[foot], self.x[foot], reference=self.joint_q[foot])
            self.joint_velocity[foot] = 0
        self.failed_activation[foot] = True
        self.last_event = f"{C.FOOT_NAMES[foot]} {event}"

    def disturb(self, foot=None):
        """人为掉足；记录恢复事件，不绕过接触/黏附门控。"""
        candidates = np.flatnonzero(self.a)
        if foot is None:
            if not len(candidates):
                return False
            foot = int(self.rng.choice(candidates))
        if foot not in range(4):
            raise ValueError("foot must be 0..3")
        self.event_time = self.step_count * C.DT
        self._failure(foot, "forced slip")
        self.contact[foot] = False
        self.q[foot] = math.exp(-self.x[foot] / 0.01)
        return True

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (C.ACT_DIM,) or not np.isfinite(action).all():
            raise ValueError(f"action must be a finite {C.ACT_DIM}-dimensional vector")
        action = np.clip(action, -1.0, 1.0)
        old_action = self.prev_action.copy()
        old_older = self.older_action.copy()
        old_x, old_y = self.x.copy(), self.y.copy()
        # 正确积分连续动作延迟：先执行上一命令，再执行当前命令。
        elapsed = self.step_count * C.DT
        if self.action_delay > 0:
            self.event_time = elapsed + self.action_delay
            self._advance(old_action, self.action_delay)
        self.event_time = elapsed + C.DT
        n_support = self._advance(action, C.DT - self.action_delay)
        magnet = (action[2::3] + 1) / 2
        reward, reward_terms = paper_reward(
            self.iter, self.phases(), self.contact, self.x,
            (self.x - old_x) / C.DT, (self.y - old_y) / C.DT,
            self.v_b, self.v_desired, paper_action(action), paper_action(old_action),
            paper_action(old_older), magnet,
            phase1=not self.adhesion_enabled, joint_position=self.joint_q,
            joint_velocity=self.joint_velocity, joint_acceleration=self.joint_acceleration,
            torque=self.joint_torque)
        self.error_sum += (self.v_b - self.v_desired) ** 2
        stance = ~((self.phases() > 0) & (self.phases() < math.pi / 2))
        self.stance_samples += int(stance.sum())
        self.retained_samples += int(((self.a > 0) & stance).sum())
        self.step_count += 1
        all_stance = self.adhesion_enabled and bool((self.a > 0).all())
        self.stuck_steps = self.stuck_steps + 1 if all_stance else 0
        stuck = self.stuck_steps > C.STUCK_STEPS
        fell = self.y_b < C.MIN_HEIGHT
        timeout = self.step_count >= C.MAX_STEPS
        terminated = fell or stuck
        self.episode_terminated = terminated
        done = terminated or timeout
        self.ep_reward += reward
        self.older_action, self.prev_action = old_action, action.copy()
        self.phase = (self.phase + 2 * math.pi * C.DT / C.GAIT_PERIOD) % (2 * math.pi)
        raw = self._raw_obs(noisy=self.domain_randomization)
        self.obs_filtered = (1 - C.FILTER_ALPHA) * self.obs_filtered + C.FILTER_ALPHA * raw
        info = self.metrics()
        info.update(n_attached=int(self.a.sum()), n_support=n_support,
                    fell=fell, stuck=stuck, timeout=timeout,
                    terminated=terminated, truncated=timeout and not terminated,
                    survived=timeout and not terminated, theta=self.theta,
                    p_attach=self.p_attach,
                    reward_terms=reward_terms)
        return self.observe(), float(reward), done, info

    def _advance(self, action, dt):
        target_q = joint_targets(action)
        old_q = self.joint_q.copy()
        old_velocity = self.joint_velocity.copy()
        self.joint_torque = self.joint_p * (target_q - old_q) - self.joint_d * old_velocity
        # 自由腿二阶PD；吸附腿由世界锚点约束，身体切向动力学仍为明确的二维代理。
        # Backward Euler：避免20ms下低惯量/高阻尼显式积分产生数值振荡。
        free_velocity = (old_velocity + self.joint_p / C.JOINT_INERTIA * (target_q-old_q) * dt)
        free_velocity /= 1 + self.joint_d / C.JOINT_INERTIA * dt + self.joint_p / C.JOINT_INERTIA * dt*dt
        free_q = old_q + free_velocity * dt
        free_q[:,1] = np.clip(free_q[:,1], -math.pi+1e-6, -1e-6)
        free_tangent, free_gap = forward(free_q)
        target_tangent, _ = forward(target_q)
        # [-1,1] 映射到磁信号 [0,1]；保持论文 0.5 门槛。
        magnet = (action[2::3] + 1.0) / 2
        on = magnet >= 0.5
        self.retry = np.maximum(0, self.retry - dt)
        relative = self.y - self.y_b - self.hip_offsets
        overreach = relative**2 + (C.BODY_NORMAL_HEIGHT-self.x)**2 > (sum(C.LINK_LENGTHS)+C.ALIGN_GAP)**2
        disable = ~on | (overreach if self.variant != "no_modeling" else False)
        if self.contact_estimate is not None and self.variant != "no_modeling":
            disable |= self.contact_estimate < 0.5
        self.a[disable] = 0.0
        physical_contact = self.contact & (self.x <= C.ALIGN_GAP)
        previous_support = physical_contact & ((self.a > 0) | (math.cos(self.theta) > .05))
        unanchored = self.a == 0
        self.x[unanchored] = np.maximum(0, free_gap[unanchored])
        # 地面机械支撑与磁支撑都保留切向世界锚点；抬离才更新自由足位置。
        free_tangent_mask = unanchored & ~(previous_support & (self.x <= C.ALIGN_GAP))
        self.y[free_tangent_mask] = self.y_b + self.hip_offsets[free_tangent_mask] + free_tangent[free_tangent_mask]
        # 墙面碰撞截断法向位置后，重新投影自由足到真实二连杆可达圆内。
        tangent_reach = np.sqrt(np.maximum(0,sum(C.LINK_LENGTHS)**2-(C.BODY_NORMAL_HEIGHT-self.x)**2))
        hips = self.y_b+self.hip_offsets
        self.y[unanchored] = np.clip(self.y[unanchored],hips[unanchored]-tangent_reach[unanchored],
                                    hips[unanchored]+tangent_reach[unanchored])
        relative = self.y - self.y_b - self.hip_offsets
        overreach = relative**2 + (C.BODY_NORMAL_HEIGHT-self.x)**2 > (sum(C.LINK_LENGTHS)+C.ALIGN_GAP)**2
        self.contact = (self.x <= C.CONTACT_GAP) & ~overreach
        self.failed_activation[~on | (self.x > C.ALIGN_GAP)] = False
        self.q = np.exp(-self.x / 0.01)
        if not self.adhesion_enabled:
            self.a[:] = 0.0
            candidates = np.zeros(4, dtype=bool)
        else:
            candidates = ((self.a == 0) & on if self.variant == "no_modeling" else
                          ((self.a == 0) & self.contact & on &
                           (self.x <= C.ALIGN_GAP) & ~self.failed_activation))
            if self.contact_estimate is not None and self.variant != "no_modeling":
                candidates &= self.contact_estimate >= 0.5
        for i in np.flatnonzero(candidates):
            self.attempts += 1
            if self.rng.random() <= self.p_attach:
                self.a[i] = 1.0
                if self.pending[i] is not None:
                    delay = self.event_time - self.pending[i]
                    self.recovery_delays.append(delay)
                    self.recovery_records.append(dict(foot=int(i),delay=delay,source=self.pending_sources[i]))
                    self.pending[i] = None
                    self.pending_sources[i] = None
                    self.last_event = f"{C.FOOT_NAMES[i]} reattached"
            else:
                self._failure(i, "attachment failed")
                # 磁吸失败不改变几何接触，也不人为抬足。

        physical_contact = self.contact & (self.x <= C.ALIGN_GAP)
        support = physical_contact & ((self.a > 0) | (math.cos(self.theta) > 0.05))
        if self.variant == "no_modeling" and self.adhesion_enabled:
            support |= self.a > 0  # 理想磁力，不伪造真实接触传感标签
        n_support = int(support.sum())
        # 支撑足保持世界锚点；支撑腿收缩(move<0)驱动身体向前。
        # 释放低足本身不再改变身体目标位置。
        gravity = C.G * math.sin(self.theta)
        normal_force = C.MASS * C.G * math.cos(self.theta) if n_support else 0.0
        normal_force += C.MAGNET_FORCE * float(self.a.sum())
        capacity = self.friction * normal_force / C.MASS
        drive_velocity = (float(np.mean(relative[support] - target_tangent[support])) * C.DRIVE_GAIN
                          if n_support else 0.0)
        drive_velocity = float(np.clip(drive_velocity, -C.V_FOOT_MAX, C.V_FOOT_MAX))
        requested = C.DRIVE_GAIN * (drive_velocity - self.v_b) + gravity
        traction = float(min(capacity, max(-capacity, requested))) if n_support else 0.0
        acceleration = traction - gravity - C.BODY_DAMPING * self.v_b
        self.v_b = float(self.v_b + acceleration * dt)
        old_body = self.y_b
        self.y_b += self.v_b * dt
        # 吸附模型与腿长约束分离：理想磁力也不能把机械连杆拉长。
        constraint_support = support & (self.a > 0)
        if constraint_support.any():
            lower = float(np.max(self.y[constraint_support]-self.hip_offsets[constraint_support]-tangent_reach[constraint_support]))
            upper = float(np.min(self.y[constraint_support]-self.hip_offsets[constraint_support]+tangent_reach[constraint_support]))
            if lower > upper+1e-9:
                raise RuntimeError("incompatible anchored leg geometry")
            projected = float(np.clip(self.y_b,lower,max(lower,upper)))
            if projected != self.y_b:
                self.v_b = 0.0  # 非弹性机械止挡；二维约束投影假设
                self.y_b = projected
        # 悬空腿仍连接身体：自由PD角不因身体平移改变，足随髋点运输。
        transported = ~support
        self.y[transported] += self.y_b-old_body
        hips = self.y_b + self.hip_offsets
        tangent_reach = np.sqrt(np.maximum(0,sum(C.LINK_LENGTHS)**2-(C.BODY_NORMAL_HEIGHT-self.x)**2))
        # 普通接触没有磁焊接：达到腿长限位后允许足沿面滑移，不凭空约束身体。
        self.y[~constraint_support] = np.clip(self.y[~constraint_support],
                                  hips[~constraint_support] - tangent_reach[~constraint_support],
                                  hips[~constraint_support] + tangent_reach[~constraint_support])
        self.joint_q = inverse(self.y - self.y_b - self.hip_offsets, self.x, reference=old_q)
        unconstrained = transported & (free_gap >= 0)
        self.joint_q[unconstrained] = free_q[unconstrained]
        if dt > 0:
            self.joint_velocity = (self.joint_q - old_q) / dt
            self.joint_acceleration = (self.joint_velocity - old_velocity) / dt

        return n_support

    def metrics(self):
        # 连续失败过程只对应一个恢复事件；重复重试不增加分母。
        events = len(self.recovery_delays) + sum(t is not None for t in self.pending)
        recovery = {str(window): (sum(d <= window for d in self.recovery_delays) / events
                                 if events else None) for window in C.RECOVERY_WINDOWS}
        reattachment = recovery.copy()
        if self.episode_terminated:
            recovery = {key: 0.0 if events else None for key in recovery}
        return dict(climb=self.y_b - self.start_y, time=self.step_count * C.DT,
                    velocity_rmse=math.sqrt(self.error_sum / max(1, self.step_count)),
                    retention=self.retained_samples / max(1, self.stance_samples),
                    failures=self.failures, attempts=self.attempts,
                    recovery_events=events, recovery=recovery, reattachment=reattachment)

    def state_dict(self):
        return dict(y_b=self.y_b, v_b=self.v_b, y=self.y.copy(), x=self.x.copy(),
                    a=self.a.copy(), q=self.q.copy(), contact=self.contact.copy(),
                    theta=self.theta, p_attach=self.p_attach, phase=self.phase,
                    step_count=self.step_count, ep_reward=self.ep_reward,
                    adhesion_enabled=self.adhesion_enabled, v_desired=self.v_desired,
                    last_event=self.last_event, **self.metrics())
