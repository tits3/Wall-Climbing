"""pygame 二维仿真；世界坐标沿壁向上、法向向右，摄像机随身体移动。"""
import math
from collections import deque
import pygame
import config as C


class Renderer:
    def __init__(self, width=1000, height=700, controller="PPO"):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2D Magnetic Climbing - paper-inspired simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 17)
        self.small = pygame.font.SysFont("consolas", 14)
        self.width, self.height = width, height
        self.controller = controller
        self.scale = 170
        self.camera_y = 0.0
        self.trace = deque(maxlen=300)
        self.last_step = -1

    def _sy(self, y):
        return self.height - 70 - (float(y) - self.camera_y) * self.scale

    def poll(self):
        quit_ = force = reset = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_ = True
            elif event.type == pygame.KEYDOWN:
                quit_ |= event.key == pygame.K_ESCAPE
                force |= event.key == pygame.K_f
                reset |= event.key == pygame.K_r
        return quit_, force, reset

    def text(self, text, position, color=(215, 225, 235), small=False):
        self.screen.blit((self.small if small else self.font).render(text, True, color), position)

    def draw(self, env, realtime=True):
        state = env.state_dict()
        self.camera_y = max(0.0, state["y_b"] - 1.4)
        self.screen.fill((18, 23, 33))
        wall_x, body_x = 120, 120+C.BODY_NORMAL_HEIGHT*1000
        pygame.draw.rect(self.screen, (70, 82, 96), (80, 55, 40, self.height - 125))
        for y in range(math.floor(self.camera_y), math.ceil(self.camera_y + 3.5)):
            sy = self._sy(y)
            if 60 < sy < self.height - 60:
                pygame.draw.line(self.screen, (145, 155, 170), (80, sy), (125, sy))
                self.text(f"{y}m", (35, sy - 8), small=True)
        by = self._sy(state["y_b"])
        pygame.draw.rect(self.screen, (225, 190, 80), (body_x - 22, by - 42, 44, 84), border_radius=8)
        for i in range(4):
            foot_x = wall_x + state["x"][i] * 1000 + i * 5
            fy = self._sy(state["y"][i])
            hip_y = self._sy(state["y_b"] + C.HIP_OFFSETS[i])
            color = ((85, 215, 130) if state["a"][i] else
                     (110, 175, 230) if state["contact"][i] else (245, 135, 95))
            pygame.draw.line(self.screen, color, (body_x, hip_y), (foot_x, fy), 3)
            pygame.draw.circle(self.screen, color, (int(foot_x), int(fy)), 8)
            # 四足状态单独排列，避免同侧投影重叠时无法辨认。
            yy = 285 + i * 42
            self.text(C.FOOT_NAMES[i], (510, yy), color)
            self.text(f"gap {state['x'][i]*1000:4.0f}mm  contact {int(state['contact'][i])}  "
                      f"magnet {int(state['a'][i])}", (555, yy), small=True)
        gx, gy = -math.cos(state["theta"]), math.sin(state["theta"])
        end = (body_x + 80 * gx, by + 80 * gy)
        pygame.draw.line(self.screen, (245, 100, 105), (body_x, by), end, 4)
        pygame.draw.circle(self.screen, (245, 100, 105), (int(end[0]), int(end[1])), 5)
        self.text("surface coordinates: up = forward", (25, 20), small=True)
        self.text(self.controller, (510, 25), (245, 205, 100))
        phase = ("1: ground / no magnetic force" if not state["adhesion_enabled"] else
                 "2: gravity transition" if state["theta"] < math.pi / 2 - 1e-6 else
                 "3: vertical / uncertain adhesion")
        lines = [
            phase,
            f"tilt {math.degrees(state['theta']):5.1f} deg   attach p {state['p_attach']:.2f}",
            f"time {state['time']:4.2f} / {C.EPISODE_SECONDS:.0f}s   climb {state['climb']:+.3f}m",
            f"velocity {state['v_b']:+.3f}   target {state['v_desired']:+.3f} m/s",
            f"RMSE {state['velocity_rmse']:.3f} m/s",
            f"stance retention {state['retention']:.1%}",
            f"failed attempts/slips {state['failures']}",
        ]
        for i, line in enumerate(lines):
            self.text(line, (510, 62 + 28 * i))
        self.text(state["last_event"], (510, 465), (245, 165, 110))
        recovery = state["recovery"]["1.2"]
        self.text("recovery <=1.2s: " + ("N/A (no events)" if recovery is None else f"{recovery:.1%}"),
                  (510, 490), small=True)
        if state["step_count"] <= self.last_step:
            self.trace.clear()
        if state["step_count"] != self.last_step:
            self.trace.append((state["v_b"], state["v_desired"]))
        self.last_step = state["step_count"]
        rect = pygame.Rect(510, 535, 450, 95)
        pygame.draw.rect(self.screen, (29, 37, 50), rect)
        pygame.draw.line(self.screen, (85, 95, 110), (rect.left, rect.centery),
                         (rect.right, rect.centery))
        plot_limit = max(.5, *(abs(v) for pair in self.trace for v in pair))
        for channel, color in [(1, (245, 205, 100)), (0, (85, 215, 130))]:
            points = [(rect.left + j * rect.width / 299,
                       rect.centery - value[channel] / plot_limit * 40)
                      for j, value in enumerate(self.trace)]
            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, 2)
        self.text(f"velocity: green actual / yellow target  +/-{plot_limit:.2f}m/s", (510, 640), small=True)
        self.text("F: force slip   R: reset   Esc: quit", (25, self.height - 30), small=True)
        pygame.display.flip()
        if realtime:
            self.clock.tick(round(1 / C.DT))

    def close(self):
        pygame.quit()
