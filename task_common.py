"""
task_common.py
データ収集と推論で共通して使用するタスク定義、成功判定、リセット管理モジュール。
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BLOCK_COLORS = {
    "red": (1.0, 0.2, 0.2),      # 赤 - より鮮やかに
    "blue": (0.2, 0.2, 1.0),     # 青 - そのまま
    "green": (0.2, 1.0, 0.2),    # 緑 - より鮮やかに
    "yellow": (1.0, 0.8, 0.0),   # 黄 - 青成分を0にし、よりオレンジ寄りの鮮やかな黄に
    # "orange": (1.0, 0.5, 0.0),   # 橙 - そのまま
}
BLOCK_SIZE = (0.025, 0.025, 0.025)

CUBE_X_MIN, CUBE_X_MAX = 0.05, 0.18
CUBE_Y_MIN, CUBE_Y_MAX = -0.25, -0.02
CUBE_Z_INIT = 0.02  # めり込み防止のため少し上げる

SUCCESS_THRESHOLD_HORIZONTAL = 0.015
SUCCESS_THRESHOLD_HEIGHT_MIN = 0.020
SUCCESS_THRESHOLD_HEIGHT_MAX = 0.040
SUCCESS_THRESHOLD_VELOCITY = 0.010
SUCCESS_STEPS_REQUIRED = 30

class TaskManager:
    def __init__(self, num_blocks: int = 5, seed: Optional[int] = None, fixed_task: Optional[str] = None):
        self.num_blocks = num_blocks
        self.color_names = list(BLOCK_COLORS.keys())[:num_blocks]
        self.rng = np.random.default_rng(seed)
        self.episode_id = 0
        self.step_count = 0
        self.success_steps = 0
        self.current_task: Optional[Dict] = None
        self.event_log: List[Dict] = []
        self.fixed_task = fixed_task

    def generate_task(self) -> Dict:
        if self.fixed_task:
            instruction = self.fixed_task
            match = re.search(r"Stack the (\w+) block on the (\w+) block\.", instruction)
            if match:
                target_color = match.group(1)
                base_color = match.group(2)
            else:
                target_color = self.color_names[0]
                base_color = self.color_names[1]
        else:
            colors = self.rng.choice(self.color_names, size=2, replace=False).tolist()
            target_color = colors[0]
            base_color = colors[1]
            instruction = f"Stack the {target_color} block on the {base_color} block."
            
        self.current_task = {
            "instruction": instruction,
            "target_color": target_color,
            "base_color": base_color,
            "task_index": hash(instruction) % 10000
        }
        self._log_event("task_started")
        return self.current_task

    def reset_environment(self, robot, cubes: List, seed: Optional[int] = None) -> None:
        logger.info(f"Environment reset (Episode {self.episode_id})")
        n_dofs = robot.n_dofs
        robot.set_dofs_position(np.zeros(n_dofs, dtype=np.float32))
        robot.set_dofs_velocity(np.zeros(n_dofs, dtype=np.float32))

        positions = self._generate_non_overlapping_positions(len(cubes), seed)
        for i, cube in enumerate(cubes):
            cube.set_pos(np.array(positions[i], dtype=np.float32))
            cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        self.step_count = 0
        self.success_steps = 0
        self.current_task = self.generate_task()
        self._log_event("environment_reset")
        
        logger.info(f"🎯 現在のタスク指示: '{self.current_task['instruction']}'")

    def check_success(self, cubes: List, cube_color_map: Dict[str, int], dt: float = 0.01) -> bool:
        if not self.current_task:
            return False

        target_idx = cube_color_map[self.current_task["target_color"]]
        base_idx = cube_color_map[self.current_task["base_color"]]

        target_pos = cubes[target_idx].get_pos().cpu().numpy()
        base_pos = cubes[base_idx].get_pos().cpu().numpy()
        target_vel = cubes[target_idx].get_vel().cpu().numpy()

        horizontal_dist = np.linalg.norm(target_pos[:2] - base_pos[:2])
        height_diff = target_pos[2] - base_pos[2]
        velocity = np.linalg.norm(target_vel)

        is_aligned = (
            horizontal_dist < SUCCESS_THRESHOLD_HORIZONTAL and
            SUCCESS_THRESHOLD_HEIGHT_MIN < height_diff < SUCCESS_THRESHOLD_HEIGHT_MAX and
            velocity < SUCCESS_THRESHOLD_VELOCITY
        )

        if is_aligned:
            self.success_steps += 1
        else:
            self.success_steps = 0

        if self.success_steps >= SUCCESS_STEPS_REQUIRED:
            self._handle_success()
            return True
        return False

    def _handle_success(self) -> None:
        logger.info(f"Task succeeded internally! (Episode {self.episode_id})")
        self._log_event("task_succeeded")
        self.step_count = 0
        self.success_steps = 0
        self.current_task = self.generate_task()

    def _generate_non_overlapping_positions(self, num: int, seed: Optional[int] = None) -> List[Tuple[float, float, float]]:
        rng = np.random.default_rng(seed)
        positions = []
        max_attempts = 1000
        for _ in range(num):
            for attempt in range(max_attempts):
                x = rng.uniform(CUBE_X_MIN, CUBE_X_MAX)
                y = rng.uniform(CUBE_Y_MIN, CUBE_Y_MAX)
                z = CUBE_Z_INIT
                candidate = (x, y, z)
                if all(np.linalg.norm(np.array(candidate) - np.array(p)) > 0.03 for p in positions):
                    positions.append(candidate)
                    break
            else:
                raise RuntimeError("Failed to find non-overlapping positions for blocks.")
        return positions

    def _log_event(self, event_type: str) -> None:
        entry = {
            "timestamp": self.step_count * 0.01,
            "episode_id": self.episode_id,
            "event": event_type,
            "task": self.current_task["instruction"] if self.current_task else None
        }
        self.event_log.append(entry)

    def get_color_map(self, cubes: List, color_names: List[str]) -> Dict[str, int]:
        return {color: i for i, color in enumerate(color_names)}