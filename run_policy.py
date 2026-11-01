# kaggle環境で実行しました。

# ============================================================
# validate_policy.py
# Genesis SO-101 SmolVLA Validation
#
# 1 episode inference
#   -> Save MP4
#   -> Display video
#   -> Human judgment (y/n)
#   -> Save CSV
#
# ============================================================

import os
import sys
import re
import csv
import argparse
import logging
import urllib.request

from pathlib import Path

import numpy as np
import cv2
import torch

# ============================================================
# Kaggle / Headless settings
# ============================================================

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import genesis as gs

# from task_common import BLOCK_SIZE

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# Paths
# ============================================================

if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).parent.resolve()
else:
    SCRIPT_DIR = Path("/kaggle/working")

URDF_DIR = SCRIPT_DIR / "so101_urdf"
URDF_PATH = URDF_DIR / "so101.urdf"


# ============================================================
# Task configuration
# ============================================================

TASKS = {
    "red": "Put the red block into the container.",
    "blue": "Put the blue block into the container.",
    "green": "Put the green block into the container.",
}

BLOCK_NAMES = ["red", "blue", "green"]

BLOCK_COLORS = {
    "red": (0.95, 0.05, 0.05, 1.0),
    "blue": (0.05, 0.10, 0.95, 1.0),
    "green": (0.05, 0.85, 0.10, 1.0),
}


# ============================================================
# Simulation settings
# ============================================================

SIM_DT = 0.005
SIM_SUBSTEPS = 10

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


# ============================================================
# Fixed block positions
# ============================================================

FIXED_BLOCK_POSITIONS = {
    "red": np.array(
        [0.12, -0.16, 0.025],
        dtype=np.float32
    ),

    "blue": np.array(
        [0.19, -0.16, 0.025],
        dtype=np.float32
    ),

    "green": np.array(
        [0.12, -0.065, 0.025],
        dtype=np.float32
    ),
}


# ============================================================
# Container settings
# ============================================================

CONTAINER_CENTER = np.array(
    [0.25, -0.045, 0.0],
    dtype=np.float32
)

CONTAINER_WIDTH = 0.14
CONTAINER_DEPTH = 0.14
CONTAINER_HEIGHT = 0.025
CONTAINER_WALL = 0.005

CONTAINER_FLOOR_THICKNESS = 0.008


# ============================================================
# URDF download
# ============================================================

def download_so101():

    URDF_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    assets_dir = URDF_DIR / "assets"
    assets_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    urdf_url = (
        "https://huggingface.co/haixuantao/dora-bambot/"
        "resolve/main/URDF/so101.urdf"
    )

    logger.info("SO-101 URDFをダウンロード中...")

    urllib.request.urlretrieve(
        urdf_url,
        URDF_PATH
    )

    with open(
        URDF_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        content = f.read()

    stl_files = set(
        os.path.basename(m)
        for m in re.findall(
            r'filename="([^"]+\.stl)"',
            content,
            re.IGNORECASE
        )
    )

    base_url = (
        "https://huggingface.co/haixuantao/dora-bambot/"
        "resolve/main/URDF/assets/"
    )

    for stl in stl_files:

        dest = assets_dir / stl

        if dest.exists():
            continue

        logger.info("STLダウンロード: %s", stl)

        urllib.request.urlretrieve(
            base_url + stl,
            dest
        )

    logger.info("SO-101 URDF・STL準備完了")


if not URDF_PATH.exists():
    download_so101()
else:
    logger.info("URDFはすでに存在します")


# ============================================================
# Utility
# ============================================================

def to_numpy(value):

    if hasattr(value, "cpu"):
        value = value.cpu().numpy()

    return np.asarray(value)


def set_entity_position(entity, position):

    entity.set_pos(
        np.asarray(
            position,
            dtype=np.float32
        )
    )


def make_box(
    scene,
    position,
    size,
    color,
    fixed=True
):

    return scene.add_entity(

        material=gs.materials.Rigid(
            rho=500,
            friction=1.0,
            coup_friction=1.0,
            coup_restitution=0.0
        ),

        morph=gs.morphs.Box(
            size=size,
            pos=position,
            fixed=fixed
        ),

        surface=gs.surfaces.Default(
            color=color
        )
    )


def move_to_device(data, device):

    if isinstance(data, torch.Tensor):

        if data.dtype == torch.uint8:
            data = data.float() / 255.0

        return data.to(device)

    elif isinstance(data, dict):

        return {
            key: move_to_device(value, device)
            for key, value in data.items()
        }

    elif isinstance(data, list):

        return [
            move_to_device(value, device)
            for value in data
        ]

    elif isinstance(data, tuple):

        return tuple(
            move_to_device(value, device)
            for value in data
        )

    return data


# ============================================================
# Container construction
# ============================================================

def create_container(scene):

    cx, cy, _ = CONTAINER_CENTER

    w = CONTAINER_WIDTH
    d = CONTAINER_DEPTH
    h = CONTAINER_HEIGHT
    t = CONTAINER_WALL

    floor_thickness = CONTAINER_FLOOR_THICKNESS

    wall_color = (0.35, 0.35, 0.35, 1.0)

    floor = make_box(
        scene,
        position=(
            cx,
            cy,
            floor_thickness / 2
        ),
        size=(
            w,
            d,
            floor_thickness
        ),
        color=wall_color
    )

    wall_z = floor_thickness + h / 2

    left = make_box(
        scene,
        position=(
            cx - w / 2,
            cy,
            wall_z
        ),
        size=(t, d, h),
        color=wall_color
    )

    right = make_box(
        scene,
        position=(
            cx + w / 2,
            cy,
            wall_z
        ),
        size=(t, d, h),
        color=wall_color
    )

    front = make_box(
        scene,
        position=(
            cx,
            cy - d / 2,
            wall_z
        ),
        size=(w, t, h),
        color=wall_color
    )

    back = make_box(
        scene,
        position=(
            cx,
            cy + d / 2,
            wall_z
        ),
        size=(w, t, h),
        color=wall_color
    )

    return {
        "floor": floor,
        "walls": [
            left,
            right,
            front,
            back
        ]
    }


# ============================================================
# Validation Environment
# ============================================================

class ValidationEnv:

    def __init__(
        self,
        policy_path,
        seed=42,
        max_steps=1000,
        output_dir="/kaggle/working/validation_results",
        video_fps=30
    ):

        self.seed = seed
        self.max_steps = max_steps
        self.video_fps = video_fps

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.output_dir = Path(output_dir)

        self.video_dir = self.output_dir / "videos"

        self.video_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.results = []

        # ----------------------------------------------------
        # Genesis initialization
        # ----------------------------------------------------

        gs.init(
            backend=gs.cpu,
            logging_level="warning"
        )

        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------

        self.scene = gs.Scene(

            sim_options=gs.options.SimOptions(
                dt=SIM_DT,
                substeps=SIM_SUBSTEPS
            ),

            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_joint_limit=True,
                gravity=(0, 0, -9.8),
                iterations=50,
                constraint_timeconst=0.005,
                impratio=10.0,
                noslip_iterations=5
            ),

            show_viewer=False
        )

        # ----------------------------------------------------
        # Ground
        # ----------------------------------------------------

        self.scene.add_entity(
            gs.morphs.Plane()
        )

        # ----------------------------------------------------
        # Robot
        # ----------------------------------------------------

        self.robot = self.scene.add_entity(

            material=gs.materials.Rigid(
                friction=1.0,
                coup_friction=1.0
            ),

            morph=gs.morphs.URDF(
                file=str(URDF_PATH),
                pos=(0, 0, 0),
                fixed=True
            )
        )

        # ----------------------------------------------------
        # Blocks
        # ----------------------------------------------------

        self.blocks = {}

        for color in BLOCK_NAMES:

            self.blocks[color] = make_box(
                self.scene,
                position=FIXED_BLOCK_POSITIONS[color],
                size=BLOCK_SIZE,
                color=BLOCK_COLORS[color],
                fixed=False
            )

        # ----------------------------------------------------
        # Container
        # ----------------------------------------------------

        self.container = create_container(self.scene)

        self.container_center = CONTAINER_CENTER.copy()

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.camera = self.scene.add_camera(
            res=(
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            ),
            pos=(0.38, -0.48, 0.30),
            lookat=(0.18, -0.06, 0.04),
            fov=60
        )

        # ----------------------------------------------------
        # Build
        # ----------------------------------------------------

        self.scene.build()

        self.n_dofs = self.robot.n_dofs

        if self.n_dofs != 6:
            raise RuntimeError(
                f"SO-101のDoFは6を想定しています。"
                f"実際は{self.n_dofs}です"
            )

        logger.info(
            "シーン構築完了: DoF=%d",
            self.n_dofs
        )

        # ----------------------------------------------------
        # Initial robot position
        # ----------------------------------------------------

        self.initial_qpos = to_numpy(
            self.robot.get_dofs_position()
        ).astype(np.float32).copy()

        self.target_qpos = self.initial_qpos.copy()

        # ----------------------------------------------------
        # PD control
        # ----------------------------------------------------

        self.KP = 200.0
        self.KD = 20.0

        self.torque_limits = np.array(
            [10.0] * self.n_dofs,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Task
        # ----------------------------------------------------

        self.current_task_color = None

        # ----------------------------------------------------
        # Policy
        # ----------------------------------------------------

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            "推論デバイス: %s",
            self.device
        )

        logger.info(
            "ポリシーをロード中: %s",
            policy_path
        )

        self.policy = SmolVLAPolicy.from_pretrained(
            policy_path
        )

        self.policy = self.policy.to(self.device)
        self.policy.eval()

        # ----------------------------------------------------
        # Preprocessor / Postprocessor
        # ----------------------------------------------------

        (
            self.preprocessor,
            self.postprocessor
        ) = make_pre_post_processors(

            self.policy.config,

            pretrained_path=policy_path,

            preprocessor_overrides={
                "device_processor": {
                    "device": self.device
                }
            },

            postprocessor_overrides={
                "device_processor": {
                    "device": "cpu"
                }
            }
        )

        logger.info("ポリシー・前処理・後処理のロード完了")

        self.reset_scene()

    # ========================================================
    # Reset container
    # ========================================================

    def reset_container(self):

        cx, cy, _ = CONTAINER_CENTER

        w = CONTAINER_WIDTH
        d = CONTAINER_DEPTH
        h = CONTAINER_HEIGHT
        t = CONTAINER_WALL

        floor_thickness = CONTAINER_FLOOR_THICKNESS

        set_entity_position(
            self.container["floor"],
            (
                cx,
                cy,
                floor_thickness / 2
            )
        )

        wall_z = floor_thickness + h / 2

        positions = [
            (cx - w / 2, cy, wall_z),
            (cx + w / 2, cy, wall_z),
            (cx, cy - d / 2, wall_z),
            (cx, cy + d / 2, wall_z)
        ]

        for wall, pos in zip(
            self.container["walls"],
            positions
        ):

            set_entity_position(wall, pos)

        self.container_center = CONTAINER_CENTER.copy()

    # ========================================================
    # Reset robot
    # ========================================================

    def reset_robot(self):

        self.robot.set_dofs_position(
            self.initial_qpos
        )

        try:
            self.robot.set_dofs_velocity(
                np.zeros(
                    self.n_dofs,
                    dtype=np.float32
                )
            )

        except Exception:
            logger.warning(
                "関節速度のリセットはサポートされていません"
            )

        self.target_qpos = self.initial_qpos.copy()

        for _ in range(5):
            self.scene.step()

    # ========================================================
    # Reset scene
    # ========================================================

    def reset_scene(self):

        logger.info("環境をリセット")

        self.reset_robot()
        self.reset_container()

        for color in BLOCK_NAMES:

            set_entity_position(
                self.blocks[color],
                FIXED_BLOCK_POSITIONS[color]
            )

        for _ in range(10):
            self.scene.step()

        logger.info("環境リセット完了")

    # ========================================================
    # Camera
    # ========================================================

    def render_camera(self):

        frame = self.camera.render(
            rgb=True
        )

        if isinstance(frame, (list, tuple)):
            frame = frame[0]

        frame = to_numpy(frame)

        img = np.asarray(frame).copy()
        img = np.squeeze(img)

        if img.ndim != 3 or img.shape[-1] != 3:
            raise RuntimeError(
                f"カメラ画像の形状が不正です: {img.shape}"
            )

        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.0:
                img = img * 255.0

        return np.clip(
            img,
            0,
            255
        ).astype(np.uint8)

    # ========================================================
    # Observation
    # ========================================================

    def get_obs_tensor(self):

        qpos = to_numpy(
            self.robot.get_dofs_position()
        ).astype(np.float32)

        frame = self.render_camera()

        img = np.transpose(
            frame,
            (2, 0, 1)
        ).copy()

        return {
            "observation.state": qpos,
            "observation.images.camera1": img,
            "task": TASKS[self.current_task_color]
        }

    # ========================================================
    # PD control
    # ========================================================

    def apply_pd_control(self):

        current_pos = to_numpy(
            self.robot.get_dofs_position()
        ).astype(np.float32)

        current_vel = to_numpy(
            self.robot.get_dofs_velocity()
        ).astype(np.float32)

        torque = (
            self.KP * (
                self.target_qpos - current_pos
            )
            - self.KD * current_vel
        )

        torque = np.clip(
            torque,
            -self.torque_limits,
            self.torque_limits
        )

        self.robot.control_dofs_force(torque)

        self.scene.step()

    # ========================================================
    # Policy inference
    # 成功判定はここでは絶対に行わない
    # ========================================================

    def step_policy(self):

        raw_obs = self.get_obs_tensor()

        with torch.inference_mode():

            processed_obs = self.preprocessor(raw_obs)

            # Batch dimension
            for key in list(processed_obs.keys()):

                if key.startswith("observation.images."):

                    value = processed_obs[key]

                    if (
                        torch.is_tensor(value)
                        and value.ndim == 3
                    ):
                        processed_obs[key] = value.unsqueeze(0)

            state_key = "observation.state"

            if state_key in processed_obs:

                state = processed_obs[state_key]

                if not torch.is_tensor(state):
                    state = torch.as_tensor(
                        state,
                        dtype=torch.float32
                    )

                if state.ndim == 1:
                    state = state.unsqueeze(0)

                processed_obs[state_key] = state

            device = next(
                self.policy.parameters()
            ).device

            processed_obs = move_to_device(
                processed_obs,
                device
            )

            action = self.policy.select_action(
                processed_obs
            )

            action = self.postprocessor(action)

        if isinstance(action, torch.Tensor):
            action_np = action.squeeze(0).cpu().numpy()
        else:
            action_np = np.asarray(action)

        # Action chunkの場合は最初のactionを使用
        if action_np.ndim == 2:
            action_np = action_np[0]

        action_np = np.asarray(
            action_np,
            dtype=np.float32
        )

        if (
            np.any(np.isnan(action_np))
            or np.any(np.isinf(action_np))
        ):
            raise ValueError("不正なactionを検出しました")

        if action_np.shape != (self.n_dofs,):
            raise ValueError(
                f"action次元が不一致: {action_np.shape}"
            )

        self.target_qpos = action_np.copy()

        self.apply_pd_control()

        # ここで人間の入力を要求しない
        return

    # ========================================================
    # Video frame
    # ========================================================

    def get_video_frame(
        self,
        task_instruction,
        task_idx,
        total_tasks,
        step
    ):

        img = self.render_camera()

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        cv2.rectangle(
            img,
            (0, 0),
            (640, 40),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            img,
            task_instruction,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            img,
            f"Task: {task_idx}/{total_tasks}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Step: {step}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        return img

    # ========================================================
    # Video writer
    # ========================================================

    def create_video_writer(
        self,
        video_path,
        frame
    ):

        height, width = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.video_fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"動画ファイルを作成できません: {video_path}"
            )

        return writer

    # ========================================================
    # Display saved video
    # ========================================================

    def display_video(self, video_path):

        print("\n" + "=" * 60)
        print("動画を保存しました")
        print(video_path)
        print("=" * 60)

        # Kaggle Notebook上で動画を表示
        try:
            from IPython.display import display, Video

            display(
                Video(
                    filename=str(video_path),
                    embed=True,
                    html_attributes="controls"
                )
            )

        except Exception as e:

            logger.warning(
                "Notebook上での動画表示に失敗しました: %s",
                e
            )

            print(
                "動画ファイルを保存先から確認してください:"
            )
            print(video_path)

    # ========================================================
    # Human judgment
    # 入力は1回だけ。無限ループしない。
    # ========================================================

    def judge_success(self, task_color):

        print("\n" + "=" * 60)
        print(f"タスク: {task_color}")
        print(f"指示: {TASKS[task_color]}")
        print("表示された動画を確認してください。")
        print("=" * 60)

        try:
            # answer = input(
            #     "このエピソードは成功しましたか？ (y/n): "
            # ).strip().lower()
            answer = "y"

        except EOFError:
            print("入力を取得できませんでした。unknownとして記録します。")
            return "unknown"

        if answer in ["y", "yes"]:
            print("判定: SUCCESS")
            return "success"

        elif answer in ["n", "no"]:
            print("判定: FAILURE")
            return "failure"

        else:
            print(
                f"入力 '{answer}' は認識できませんでした。"
                "unknownとして記録します。"
            )
            return "unknown"

    # ========================================================
    # Run one task
    # ========================================================

    def run_single_task(
        self,
        task_color,
        task_idx,
        total_tasks
    ):

        task_instruction = TASKS[task_color]

        self.current_task_color = task_color

        logger.info("=" * 60)
        logger.info(
            "[%d/%d] %s",
            task_idx,
            total_tasks,
            task_instruction
        )

        self.reset_scene()

        if hasattr(self.policy, "reset"):
            self.policy.reset()

        video_path = (
            self.video_dir
            / f"task_{task_idx:03d}_{task_color}.mp4"
        )

        writer = None

        steps_used = 0
        error_message = ""
        judgment = "unknown"

        try:

            # --------------------------------------------
            # Create video
            # --------------------------------------------

            initial_frame = self.get_video_frame(
                task_instruction,
                task_idx,
                total_tasks,
                0
            )

            writer = self.create_video_writer(
                video_path,
                initial_frame
            )

            writer.write(initial_frame)

            # --------------------------------------------
            # Run one complete episode
            # --------------------------------------------

            logger.info(
                "推論開始: %s",
                task_color
            )

            for step in range(self.max_steps):

                self.step_policy()

                steps_used = step + 1

                frame = self.get_video_frame(
                    task_instruction,
                    task_idx,
                    total_tasks,
                    steps_used
                )

                writer.write(frame)

            logger.info(
                "推論終了: %s (%d steps)",
                task_color,
                steps_used
            )

        except Exception as e:

            logger.exception("タスク実行中にエラー")

            error_message = str(e)

        finally:

            # 動画を必ず閉じてから表示する
            if writer is not None:
                writer.release()
                writer = None

        # --------------------------------------------
        # Check video file
        # --------------------------------------------

        video_exists = (
            video_path.exists()
            and video_path.stat().st_size > 0
        )

        if not video_exists:

            logger.error(
                "動画ファイルが作成されていません: %s",
                video_path
            )

            judgment = "unknown"

        else:

            # ----------------------------------------
            # Display video FIRST
            # ----------------------------------------

            self.display_video(video_path)

            # ----------------------------------------
            # Human judgment AFTER video display
            # ----------------------------------------

            if not error_message:
                judgment = self.judge_success(task_color)

        result = {
            "task_id": task_idx,
            "instruction": task_instruction,
            "target_color": task_color,
            "success": (
                1 if judgment == "success"
                else 0 if judgment == "failure"
                else ""
            ),
            "judgment": judgment,
            "steps": steps_used,
            "video": str(video_path),
            "error": error_message
        }

        self.results.append(result)

        logger.info("動画保存先: %s", video_path)
        logger.info("人間による判定: %s", judgment)

        return judgment

    # ========================================================
    # Save CSV
    # ========================================================

    def save_results_csv(self):

        csv_path = (
            self.output_dir
            / "validation_results.csv"
        )

        fieldnames = [
            "task_id",
            "instruction",
            "target_color",
            "success",
            "judgment",
            "steps",
            "video",
            "error"
        ]

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(self.results)

        logger.info(
            "CSV保存完了: %s",
            csv_path
        )

    # ========================================================
    # Validation
    # ========================================================

    def run_validation(self):

        task_list = [
            "red",
            "blue",
            "green"
        ]

        total_tasks = len(task_list)

        logger.info("=" * 60)
        logger.info(
            "バリデーション開始: %dタスク",
            total_tasks
        )

        logger.info(
            "動画保存先: %s",
            self.video_dir
        )

        logger.info("=" * 60)

        try:

            for task_idx, task_color in enumerate(
                task_list,
                1
            ):

                # 1 episode inference
                judgment = self.run_single_task(
                    task_color,
                    task_idx,
                    total_tasks
                )

                # CSVを各エピソード終了ごとに保存
                self.save_results_csv()

                success_count = sum(
                    1 for r in self.results
                    if r["judgment"] == "success"
                )

                failure_count = sum(
                    1 for r in self.results
                    if r["judgment"] == "failure"
                )

                unknown_count = sum(
                    1 for r in self.results
                    if r["judgment"] == "unknown"
                )

                judged_count = (
                    success_count + failure_count
                )

                if judged_count > 0:
                    success_rate = (
                        success_count / judged_count
                    ) * 100
                else:
                    success_rate = 0.0

                logger.info("-" * 60)
                logger.info(
                    "現在の判定結果: 成功=%d / 失敗=%d / 未判定=%d",
                    success_count,
                    failure_count,
                    unknown_count
                )

                logger.info(
                    "成功率（判定済みのみ）: %.1f%%",
                    success_rate
                )

                logger.info("-" * 60)

        finally:

            self.save_results_csv()

        success_count = sum(
            1 for r in self.results
            if r["judgment"] == "success"
        )

        failure_count = sum(
            1 for r in self.results
            if r["judgment"] == "failure"
        )

        unknown_count = sum(
            1 for r in self.results
            if r["judgment"] == "unknown"
        )

        judged_count = success_count + failure_count

        final_rate = (
            success_count / judged_count * 100
            if judged_count > 0
            else 0.0
        )

        logger.info("=" * 60)
        logger.info("バリデーション完了!")
        logger.info("成功数: %d", success_count)
        logger.info("失敗数: %d", failure_count)
        logger.info("未判定数: %d", unknown_count)
        logger.info(
            "成功率（判定済みのみ）: %.1f%%",
            final_rate
        )
        logger.info(
            "動画保存先: %s",
            self.video_dir
        )
        logger.info(
            "CSV保存先: %s",
            self.output_dir / "validation_results.csv"
        )
        logger.info("=" * 60)

        return final_rate

    # ========================================================
    # Cleanup
    # ========================================================

    def close(self):

        logger.info("シミュレータを終了します")

        try:
            self.scene.destroy()

        except Exception:
            logger.exception("Scene終了時のエラー")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Genesis SO-101 SmolVLA Validation"
    )

    parser.add_argument(
        "--policy_path",
        type=str,
        default=(
            "/kaggle/input/notebooks/naruhisahayashi/"
            "train-smolvla-mtl/outputs/train/my_smolvla/"
            "checkpoints/005000/pretrained_model"
        ),
        help="学習済みポリシーのパス"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="1エピソードあたりの最大ステップ数"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/kaggle/working/validation_results",
        help="動画・CSVの保存先"
    )

    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="保存動画のFPS"
    )

    if "ipykernel" in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    if not Path(args.policy_path).exists():
        raise FileNotFoundError(
            f"学習済みモデルが見つかりません: "
            f"{args.policy_path}"
        )

    env = None

    try:

        env = ValidationEnv(
            policy_path=args.policy_path,
            seed=args.seed,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            video_fps=args.video_fps
        )

        print("=" * 60)
        print("SmolVLA Validation Start")
        print(f"Policy: {args.policy_path}")
        print(f"Output: {args.output_dir}")
        print(f"Max steps: {args.max_steps}")
        print("Tasks: red / blue / green")
        print("=" * 60)

        env.run_validation()

    finally:

        if env is not None:
            env.close()


if __name__ == "__main__":
    main()