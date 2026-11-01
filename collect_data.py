
"""
collect_data.py

Genesis SO-101 data collection server for SmolVLA.

Features:
- Headless Genesis simulation
- One RGB camera (oblique elevated view)
- 3 blocks: red, blue, green
- Fixed block positions (never randomized)
- Fixed container position (never randomized)
- Robot returns to its initial joint position on reset
- 3 English-language tasks, 20 episodes each
- TCP command receiving and image streaming
- Parquet + JPEG dataset storage
"""

import os
import sys
import json
import time
import socket
import struct
import logging
import threading
import argparse
import shutil
import urllib.request
import re

from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from PIL import Image

import genesis as gs

from task_common import BLOCK_SIZE


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()

URDF_DIR = SCRIPT_DIR / "so101_urdf"
URDF_PATH = URDF_DIR / "so101.urdf"

OUTPUT_DIR = SCRIPT_DIR / "outputs" / "dataset"


# ============================================================
# Task configuration
# ============================================================

TASKS = {
    "red": "Put the red block into the container.",
    "blue": "Put the blue block into the container.",
    "green": "Put the green block into the container.",
}

BLOCK_COLORS = {
    "red": (0.95, 0.05, 0.05, 1.0),
    "blue": (0.05, 0.10, 0.95, 1.0),
    "green": (0.05, 0.85, 0.10, 1.0),
}

BLOCK_NAMES = ["red", "blue", "green"]

EPISODES_PER_TASK = 20
TOTAL_EPISODES = EPISODES_PER_TASK * len(TASKS)

SIM_DT = 0.005

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

RENDER_FPS = 30.0
RENDER_INTERVAL = 1.0 / RENDER_FPS


# ============================================================
# Scene configuration
# ============================================================

# Container position is fixed.
CONTAINER_CENTER = np.array(
    [0.25, -0.045, 0.0],
    dtype=np.float32
)

CONTAINER_WIDTH = 0.14
CONTAINER_DEPTH = 0.14
CONTAINER_HEIGHT = 0.025
CONTAINER_WALL = 0.005

CONTAINER_FLOOR_THICKNESS = 0.008
CONTAINER_FLOOR_TOP = CONTAINER_FLOOR_THICKNESS


# ============================================================
# Fixed block positions
# These positions NEVER change between episodes.
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
# URDF download
# ============================================================

def download_so101():

    URDF_DIR.mkdir(parents=True, exist_ok=True)
    (URDF_DIR / "assets").mkdir(exist_ok=True)

    urdf_url = (
        "https://huggingface.co/haixuantao/"
        "dora-bambot/resolve/main/URDF/so101.urdf"
    )

    logger.info("Downloading SO-101 URDF...")

    urllib.request.urlretrieve(
        urdf_url,
        URDF_PATH
    )

    with open(URDF_PATH, "r") as f:
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
        "https://huggingface.co/haixuantao/"
        "dora-bambot/resolve/main/URDF/assets/"
    )

    for stl in stl_files:

        dest = URDF_DIR / "assets" / stl

        if not dest.exists():

            try:
                urllib.request.urlretrieve(
                    base_url + stl,
                    dest
                )

                logger.info("Downloaded %s", stl)

            except Exception as e:

                logger.error(
                    "Failed to download %s: %s",
                    stl,
                    e
                )


if not URDF_PATH.exists():
    download_so101()


# ============================================================
# Utility
# ============================================================

def to_numpy(value):

    if hasattr(value, "cpu"):
        value = value.cpu().numpy()

    return np.asarray(value)


def get_entity_position(entity):

    return to_numpy(
        entity.get_pos()
    ).astype(np.float32)


def set_entity_position(entity, position):

    entity.set_pos(
        np.asarray(position, dtype=np.float32)
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


# ============================================================
# Container
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
        position=(cx - w / 2, cy, wall_z),
        size=(t, d, h),
        color=wall_color
    )

    right = make_box(
        scene,
        position=(cx + w / 2, cy, wall_z),
        size=(t, d, h),
        color=wall_color
    )

    front = make_box(
        scene,
        position=(cx, cy - d / 2, wall_z),
        size=(w, t, h),
        color=wall_color
    )

    back = make_box(
        scene,
        position=(cx, cy + d / 2, wall_z),
        size=(w, t, h),
        color=wall_color
    )

    return {
        "floor": floor,
        "walls": [left, right, front, back]
    }


# ============================================================
# Dataset management
# ============================================================

def get_existing_episodes():

    train_dir = OUTPUT_DIR / "train"

    if not train_dir.exists():
        return []

    return sorted(
        train_dir.glob("episode_*.parquet")
    )


def get_existing_task_counts():

    counts = Counter()

    for file in get_existing_episodes():

        try:

            df = pd.read_parquet(file)

            if "task" in df.columns and len(df) > 0:

                task = str(df["task"].iloc[0])

                if task in TASKS.values():
                    counts[task] += 1

        except Exception as e:

            logger.warning(
                "Could not read %s: %s",
                file,
                e
            )

    return counts


def get_next_episode_id():

    episodes = get_existing_episodes()

    if not episodes:
        return 0

    ids = []

    for file in episodes:

        try:
            ids.append(
                int(file.stem.split("_")[1])
            )

        except Exception:
            pass

    return max(ids) + 1 if ids else 0


def prepare_output_directory(clear=False):

    if clear and OUTPUT_DIR.exists():

        logger.warning("Clearing existing dataset.")

        shutil.rmtree(OUTPUT_DIR)

    (OUTPUT_DIR / "train" / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "meta").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Main environment
# ============================================================

class DataCollectionEnv:

    def __init__(
        self,
        seed=42,
        host="127.0.0.1",
        port=5555,
        layout_mode="fixed",
        force_clear=False
    ):

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.host = host
        self.port = port

        # Layout is always fixed.
        self.layout_mode = "fixed"

        self.is_running = True
        self.is_recording = False

        self.current_episode_data = []

        self.latest_frame = None
        self.client_socket = None

        self.frame_lock = threading.Lock()

        self.KP = 200.0
        self.KD = 20.0

        self.step_count = 0

        self.current_task_color = None

        self.last_render_time = 0.0

        prepare_output_directory(force_clear)

        self.episode_id = get_next_episode_id()

        # ----------------------------------------------------
        # Genesis initialization
        # ----------------------------------------------------

        try:

            gs.init(
                backend=gs.cpu,
                logging_level="warning"
            )

        except Exception as e:

            logger.error(
                "Genesis initialization failed: %s",
                e
            )

            sys.exit(1)

        self.scene = gs.Scene(

            sim_options=gs.options.SimOptions(
                dt=SIM_DT,
                substeps=10
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

        self.n_dofs = self.robot.n_dofs

        logger.info(
            "Robot DoF: %d",
            self.n_dofs
        )

        self.torque_limits = np.array(
            [10.0] * self.n_dofs,
            dtype=np.float32
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

        self.container = create_container(
            self.scene
        )

        self.container_center = CONTAINER_CENTER.copy()

        # ----------------------------------------------------
        # One camera only
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
        # Build scene
        # ----------------------------------------------------

        self.scene.build()

        # Save initial robot joint position.
        self.initial_qpos = to_numpy(
            self.robot.get_dofs_position()
        ).astype(np.float32).copy()

        self.target_qpos = self.initial_qpos.copy()

        logger.info(
            "Initial robot joint position: %s",
            self.initial_qpos
        )

        # Initial reset.
        self.reset_episode()

        logger.info(
            "Genesis scene ready."
        )

        # ----------------------------------------------------
        # TCP server
        # ----------------------------------------------------

        self.server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        self.server_socket.bind(
            (self.host, self.port)
        )

        self.server_socket.listen(1)

        logger.info(
            "TCP server listening at %s:%d",
            self.host,
            self.port
        )

    # ========================================================
    # Layout: always fixed
    # ========================================================

    def generate_block_positions(self):

        # Return the same positions every time.
        return {
            color: position.copy()
            for color, position
            in FIXED_BLOCK_POSITIONS.items()
        }

    def generate_container_position(self):

        # Container position never changes.
        return CONTAINER_CENTER.copy()

    def reset_container(self):

        center = self.generate_container_position()

        cx, cy, _ = center

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

            set_entity_position(
                wall,
                pos
            )

        self.container_center = center

    # ========================================================
    # Robot reset
    # ========================================================

    def reset_robot(self):

        """
        Return robot to its initial joint position.
        """

        try:

            self.robot.set_dofs_position(
                self.initial_qpos
            )

            logger.info(
                "Robot joint position reset."
            )

        except Exception:

            logger.exception(
                "Could not reset robot joint position."
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
                "Joint velocity reset is not supported."
            )

        self.target_qpos = self.initial_qpos.copy()

        for _ in range(5):
            self.scene.step()

    # ========================================================
    # Episode reset
    # ========================================================

    def reset_episode(self):

        logger.info("Resetting episode...")

        self.step_count = 0

        self.reset_robot()

        self.reset_container()

        # Always use exactly the same block positions.
        block_positions = self.generate_block_positions()

        for color in BLOCK_NAMES:

            set_entity_position(
                self.blocks[color],
                block_positions[color]
            )

        # Allow objects to settle.
        for _ in range(10):
            self.scene.step()

        self.current_episode_data = []

        with self.frame_lock:
            self.latest_frame = None

        self.last_render_time = 0.0

        logger.info(
            "Episode reset complete. Block positions are fixed."
        )

    # ========================================================
    # Camera
    # ========================================================

    def render_camera(self):

        frame = self.camera.render(rgb=True)

        if isinstance(frame, (list, tuple)):
            frame = frame[0]

        frame = to_numpy(frame)

        return np.asarray(
            frame,
            dtype=np.uint8
        )

    def update_camera_frame(self):

        frame = self.render_camera()

        with self.frame_lock:
            self.latest_frame = frame

        return frame

    # ========================================================
    # Observation
    # ========================================================

    def get_obs(self, frame):

        qpos = to_numpy(
            self.robot.get_dofs_position()
        ).astype(np.float32)

        return {
            "observation.state": qpos,
            "observation.images.camera1": frame,
            "task": TASKS[self.current_task_color]
        }

    # ========================================================
    # Success detection
    # ========================================================

    def check_success(self):

        if self.current_task_color is None:
            return False

        block = self.blocks[
            self.current_task_color
        ]

        pos = get_entity_position(block)

        cx, cy, _ = self.container_center

        half_w = CONTAINER_WIDTH / 2
        half_d = CONTAINER_DEPTH / 2

        # ブロックの中心がコンテナの内側にあるか
        inside_x = (
            cx - half_w + CONTAINER_WALL
            < pos[0]
            < cx + half_w - CONTAINER_WALL
        )

        inside_y = (
            cy - half_d + CONTAINER_WALL
            < pos[1]
            < cy + half_d - CONTAINER_WALL
        )

        # BLOCK_SIZE が tuple の場合は高さ成分 [2] を使用する
        block_height = (
            float(BLOCK_SIZE[2])
            if isinstance(BLOCK_SIZE, (tuple, list, np.ndarray))
            else float(BLOCK_SIZE)
        )

        inside_z = (
            pos[2] > CONTAINER_FLOOR_TOP
            and
            pos[2] < (
                CONTAINER_FLOOR_TOP
                + CONTAINER_HEIGHT
                + block_height
            )
        )

        success = (
            inside_x
            and inside_y
            and inside_z
        )

        logger.info(
            "Success check: block=%s pos=(%.4f, %.4f, %.4f) "
            "inside_x=%s inside_y=%s inside_z=%s result=%s",
            self.current_task_color,
            pos[0],
            pos[1],
            pos[2],
            inside_x,
            inside_y,
            inside_z,
            success
        )

        return success

    # ========================================================
    # Simulation step
    # ========================================================

    def step(self):

        current_pos = to_numpy(
            self.robot.get_dofs_position()
        ).astype(np.float32)

        current_vel = to_numpy(
            self.robot.get_dofs_velocity()
        ).astype(np.float32)

        torque = (
            self.KP
            * (self.target_qpos - current_pos)
            - self.KD * current_vel
        )

        torque = np.clip(
            torque,
            -self.torque_limits,
            self.torque_limits
        )

        self.robot.control_dofs_force(torque)

        self.scene.step()

        self.step_count += 1

        now = time.monotonic()

        # Render only at approximately 30 FPS.
        if now - self.last_render_time >= RENDER_INTERVAL:

            try:

                frame = self.update_camera_frame()

                self.last_render_time = now

                if self.is_recording:

                    obs = self.get_obs(frame)

                    self.current_episode_data.append({

                        "timestamp": (
                            self.step_count * SIM_DT
                        ),

                        "observation.state": (
                            obs["observation.state"].tolist()
                        ),

                        "action": (
                            self.target_qpos.tolist()
                        ),

                        "camera1": (
                            obs["observation.images.camera1"]
                        ),

                        "task": obs["task"]
                    })

            except Exception:

                logger.exception(
                    "Camera rendering error"
                )

        time.sleep(0.001)

    # ========================================================
    # Recording
    # ========================================================

    def start_recording(self):

        if self.is_recording:
            return

        if self.current_task_color is None:

            logger.warning(
                "No remaining tasks."
            )

            return

        self.current_episode_data = []

        self.is_recording = True

        logger.info(
            "RECORD START: Episode=%d Task=%s",
            self.episode_id,
            self.current_task_color
        )

    def stop_recording(self):

        if not self.is_recording:
            return

        self.is_recording = False

        logger.info("Recording stopped.")

    # ========================================================
    # Save episode
    # ========================================================

    def save_episode(self):

        if not self.current_episode_data:

            logger.warning(
                "No data to save."
            )

            return

        ep_idx = self.episode_id

        records = self.current_episode_data

        logger.info(
            "Saving episode %d (%d frames)",
            ep_idx,
            len(records)
        )

        data = {

            "episode_index": [
                ep_idx
            ] * len(records),

            "frame_index": list(
                range(len(records))
            ),

            "timestamp": [
                f["timestamp"]
                for f in records
            ],

            "observation.state": [
                f["observation.state"]
                for f in records
            ],

            "action": [
                f["action"]
                for f in records
            ],

            "task": [
                f["task"]
                for f in records
            ]
        }

        df = pd.DataFrame(data)

        table = pa.Table.from_pandas(
            df,
            preserve_index=False
        )

        train_dir = OUTPUT_DIR / "train"

        pq.write_table(
            table,
            train_dir / f"episode_{ep_idx:06d}.parquet"
        )

        image_dir = (
            train_dir
            / "images"
            / "observation.images.camera1"
            / f"episode_{ep_idx:06d}"
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for frame_idx, record in enumerate(records):

            img = Image.fromarray(
                record["camera1"]
            )

            img.save(
                image_dir / f"frame_{frame_idx:06d}.jpg",
                quality=85
            )

        self.episode_id += 1

        self.current_episode_data = []

        self._update_metadata()

        logger.info(
            "Episode %d saved successfully.",
            ep_idx
        )

    # ========================================================
    # Metadata
    # ========================================================

    def _update_metadata(self):

        episodes = get_existing_episodes()

        counts = get_existing_task_counts()

        features = {

            "observation.images.camera1": {
                "shape": [
                    CAMERA_HEIGHT,
                    CAMERA_WIDTH,
                    3
                ],
                "dtype": "image"
            },

            "observation.state": {
                "shape": [self.n_dofs],
                "dtype": "float32"
            },

            "action": {
                "shape": [self.n_dofs],
                "dtype": "float32"
            },

            "task": {
                "shape": [],
                "dtype": "string"
            },

            "episode_index": {
                "shape": [],
                "dtype": "int64"
            },

            "frame_index": {
                "shape": [],
                "dtype": "int64"
            },

            "timestamp": {
                "shape": [],
                "dtype": "float32"
            }
        }

        info = {

            "codebase_version": "v5.1-fixed-layout",

            "robot_type": "so101_follower",

            "fps": RENDER_FPS,

            "total_episodes": len(episodes),

            "features": features,

            "layout_mode": "fixed",

            "episodes_per_task": EPISODES_PER_TASK,

            "task_counts": dict(counts)
        }

        with open(
            OUTPUT_DIR / "meta" / "info.json",
            "w"
        ) as f:

            json.dump(
                info,
                f,
                indent=2
            )

        with open(
            OUTPUT_DIR / "meta" / "tasks.jsonl",
            "w"
        ) as f:

            for i, instruction in enumerate(
                sorted(TASKS.values())
            ):

                f.write(
                    json.dumps({
                        "task_index": i,
                        "task": instruction
                    }) + "\n"
                )

    # ========================================================
    # TCP client
    # ========================================================

    def handle_client(self, conn, addr):

        logger.info(
            "Client connected: %s",
            addr
        )

        self.client_socket = conn

        try:

            buffer = ""

            while self.is_running:

                data = conn.recv(1024)

                if not data:
                    break

                buffer += data.decode(
                    "utf-8",
                    errors="ignore"
                )

                while "\n" in buffer:

                    command, buffer = buffer.split(
                        "\n",
                        1
                    )

                    self.process_command(
                        command.strip()
                    )

        except (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            OSError
        ) as e:

            logger.info(
                "Client disconnected: %s",
                e
            )

        except Exception:

            logger.exception(
                "Unexpected client error"
            )

        finally:

            if self.client_socket is conn:
                self.client_socket = None

            try:
                conn.close()
            except OSError:
                pass

    def process_command(self, command):

        if not command:
            return

        parts = command.split()

        cmd_type = parts[0].upper()

        if cmd_type == "JOINT" and len(parts) == 3:

            idx = int(parts[1])
            delta = float(parts[2])

            if 0 <= idx < self.n_dofs:

                current_qpos = to_numpy(
                    self.robot.get_dofs_position()
                )

                if self.n_dofs > 5:

                    is_grasping = (
                        current_qpos[5] < 0.05
                    )

                    if (
                        idx == 5
                        and is_grasping
                        and delta < 0
                    ):
                        delta = 0.0

                self.target_qpos[idx] = np.clip(
                    self.target_qpos[idx] + delta,
                    -np.pi,
                    np.pi
                )

        elif cmd_type == "RECORD_START":

            self.start_recording()

        elif cmd_type == "RECORD_STOP":

            self.stop_recording()

            self.current_episode_data = []

            self.reset_episode()

        elif cmd_type == "SUCCESS":

            if self.is_recording:

                # 人間がSUCCESSを送信したら無条件で成功とする
                self.stop_recording()

                self.save_episode()

                logger.info(
                    "SUCCESS: Manually confirmed. Episode saved."
                )

                self.reset_episode()

                self.advance_task()

            else:

                logger.warning(
                    "SUCCESS received while not recording."
                )

        elif cmd_type == "FAIL":

            self.stop_recording()

            self.current_episode_data = []

            logger.info(
                "FAIL: Episode discarded."
            )

            self.reset_episode()

        elif cmd_type == "RESET":

            self.stop_recording()

            self.current_episode_data = []

            self.reset_episode()

        elif cmd_type == "QUIT":

            self.is_running = False

        else:

            logger.warning(
                "Unknown command: %s",
                cmd_type
            )

    # ========================================================
    # Task progression
    # ========================================================

    def advance_task(self):

        counts = get_existing_task_counts()

        for color, instruction in TASKS.items():

            if counts[instruction] < EPISODES_PER_TASK:

                self.current_task_color = color

                logger.info(
                    "Next task: %s (%d/%d)",
                    color,
                    counts[instruction],
                    EPISODES_PER_TASK
                )

                return

        self.current_task_color = None

        logger.info(
            "All 60 episodes have been collected."
        )

        self.is_running = False

    # ========================================================
    # Image streaming
    # ========================================================

    def send_image_loop(self):

        while self.is_running:

            conn = self.client_socket

            if conn is None:

                time.sleep(0.05)
                continue

            try:

                with self.frame_lock:

                    if self.latest_frame is None:
                        img = None
                    else:
                        img = self.latest_frame.copy()

                if img is None:

                    time.sleep(0.01)
                    continue

                status = (
                    "REC"
                    if self.is_recording
                    else "STBY"
                )

                color = (
                    (0, 0, 255)
                    if self.is_recording
                    else (0, 255, 0)
                )

                img_draw = img.copy()

                cv2.rectangle(
                    img_draw,
                    (0, 0),
                    (180, 30),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    img_draw,
                    "CAMERA1 OBLIQUE",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                cv2.putText(
                    img_draw,
                    status,
                    (300, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                ok, encoded = cv2.imencode(
                    ".jpg",
                    img_draw,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        70
                    ]
                )

                if not ok:
                    continue

                img_bytes = encoded.tobytes()

                packet = (
                    struct.pack(
                        ">I",
                        len(img_bytes)
                    )
                    + img_bytes
                )

                conn.sendall(packet)

            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
                OSError
            ):

                if self.client_socket is conn:
                    self.client_socket = None

                try:
                    conn.close()
                except OSError:
                    pass

            except Exception:

                logger.exception(
                    "Image transmission error"
                )

            time.sleep(0.033)

    # ========================================================
    # Server
    # ========================================================

    def _accept_client(self):

        while self.is_running:

            try:

                conn, addr = self.server_socket.accept()

                threading.Thread(
                    target=self.handle_client,
                    args=(conn, addr),
                    daemon=True
                ).start()

            except OSError:
                break

    def run_server(self):

        self.advance_task()

        if self.current_task_color is None:

            logger.info(
                "All requested episodes already exist."
            )

            self.is_running = False
            return

        threading.Thread(
            target=self._accept_client,
            daemon=True
        ).start()

        threading.Thread(
            target=self.send_image_loop,
            daemon=True
        ).start()

        logger.info(
            "Server started. Task=%s",
            self.current_task_color
        )

        try:

            while self.is_running:
                self.step()

        except KeyboardInterrupt:

            logger.info(
                "Keyboard interrupt detected."
            )

        finally:

            self.is_running = False

            conn = self.client_socket
            self.client_socket = None

            if conn is not None:

                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

                try:
                    conn.close()
                except OSError:
                    pass

            try:
                self.server_socket.close()
            except OSError:
                pass

            if self.is_recording:

                logger.warning(
                    "Unfinished recording discarded on shutdown."
                )

                self.current_episode_data = []
                self.is_recording = False

            try:
                self.scene.destroy()
            except Exception:
                logger.exception(
                    "Scene destruction failed."
                )

            logger.info(
                "Data collection finished."
            )


# ============================================================
# Main
# ============================================================

def run_collection():

    parser = argparse.ArgumentParser(
        description="Genesis SO-101 SmolVLA data collection"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5555
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--layout-mode",
        type=str,
        choices=["random", "fixed"],
        default="fixed",
        help="Accepted for compatibility. Block and container positions are always fixed."
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing dataset data"
    )

    args = parser.parse_args()

    env = DataCollectionEnv(

        seed=args.seed,

        host=args.host,

        port=args.port,

        layout_mode="fixed",

        force_clear=args.clear
    )

    print("=" * 60)
    print("Genesis SO-101 SmolVLA Data Collection")
    print("Cameras: 1")
    print("Camera view: oblique elevated view")
    print("Blocks: red, blue, green")
    print("Block positions: FIXED")
    print("Container position: FIXED")

    print("Tasks:")

    for color, instruction in TASKS.items():
        print(f"  {color}: {instruction}")

    print(f"Target episodes: {TOTAL_EPISODES}")
    print("Layout mode: fixed")
    print(f"TCP server: {args.host}:{args.port}")
    print("=" * 60)

    env.run_server()


if __name__ == "__main__":
    run_collection()
