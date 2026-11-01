"""
test_suite.py
必須テストスクリプト。GenesisのGUIを必要としないロジック検証を実行します。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# task_common をインポート
from task_common import TaskManager, BLOCK_COLORS, BLOCK_SIZE, SUCCESS_STEPS_REQUIRED

def test_task_manager_logic():
    print("Running Task Manager Logic Tests...")
    
    # モッククラス（Genesisのcube/robotを模倣）
    class MockCube:
        def __init__(self, pos, vel):
            self._pos = np.array(pos, dtype=np.float32)
            self._vel = np.array(vel, dtype=np.float32)
        def get_pos(self):
            class MockTensor:
                def __init__(self, val): self.val = val
                def cpu(self): return self
                def numpy(self): return self.val
            return MockTensor(self._pos)
        def get_vel(self):
            class MockTensor:
                def __init__(self, val): self.val = val
                def cpu(self): return self
                def numpy(self): return self.val
            return MockTensor(self._vel)
        def set_pos(self, pos): self._pos = pos
        def set_quat(self, quat): pass
        def set_vel(self, vel): self._vel = vel
        def set_ang_vel(self, vel): pass

    class MockRobot:
        def __init__(self):
            self.n_dofs = 6
            self._pos = np.zeros(self.n_dofs, dtype=np.float32)
            self._vel = np.zeros(self.n_dofs, dtype=np.float32)
        def get_dofs_position(self):
            class MockTensor:
                def __init__(self, val): self.val = val
                def cpu(self): return self
                def numpy(self): return self.val
            return MockTensor(self._pos)
        def get_dofs_velocity(self):
            class MockTensor:
                def __init__(self, val): self.val = val
                def cpu(self): return self
                def numpy(self): return self.val
            return MockTensor(self._vel)
        def set_dofs_position(self, pos): self._pos = pos
        def set_dofs_velocity(self, vel): self._vel = vel

    robot = MockRobot()
    cubes = [MockCube([0.1, -0.1, 0.0125], [0.0, 0.0, 0.0]) for _ in range(5)]
    color_names = list(BLOCK_COLORS.keys())[:5]
    color_map = {color: i for i, color in enumerate(color_names)}

    manager = TaskManager(num_blocks=5, seed=42)
    manager.reset_environment(robot, cubes, seed=42)

    # Test 6: リセット時にアームとブロックが初期化される
    assert np.allclose(robot.get_dofs_position().numpy(), 0.0), "Test 6 Failed: Robot not zeroed"
    
    # Test 7, 8, 9, 10, 11: 成功時の挙動
    # 成功条件を満たすようにモックの位置を操作
    target_idx = color_map[manager.current_task["target_color"]]
    base_idx = color_map[manager.current_task["base_color"]]
    
    base_pos = np.array([0.15, -0.15, 0.0125], dtype=np.float32)
    target_pos = np.array([0.15, -0.15, 0.035], dtype=np.float32) # 高さの差 0.0225m (条件内)
    
    cubes[base_idx]._pos = base_pos
    cubes[target_idx]._pos = target_pos
    cubes[target_idx]._vel = np.array([0.0, 0.0, 0.0], dtype=np.float32) # 速度0

    prev_ep_id = manager.episode_id
    
    # 成功ステップ数分ループ
    for _ in range(SUCCESS_STEPS_REQUIRED + 5):
        manager.check_success(cubes, color_map, dt=0.01)

    # Test 7: task_succeeded イベントが記録される
    events = [e["event"] for e in manager.event_log]
    assert "task_succeeded" in events, "Test 7 Failed: task_succeeded not in log"
    
    # Test 8: エピソードIDが進む
    assert manager.episode_id == prev_ep_id + 1, "Test 8 Failed: Episode ID did not increment"
    
    # Test 9: environment_reset イベントが発生しない (成功だけではリセットされない)
    # reset_environment を呼んだ直後のログを除外して確認
    success_idx = events.index("task_succeeded")
    assert "environment_reset" not in events[success_idx:], "Test 9 Failed: environment_reset triggered on success"

    # Test 10, 11: ブロック位置とアーム姿勢が勝手に変更されない
    assert np.allclose(cubes[target_idx]._pos, target_pos), "Test 10 Failed: Block position changed"
    assert np.allclose(robot._pos, 0.0), "Test 11 Failed: Robot position changed"

    print("✓ Task Manager Logic Tests PASSED")

def test_dataset_format():
    print("Running Dataset Format Tests...")
    dataset_dir = "outputs/dataset"
    
    if not os.path.exists(dataset_dir):
        print("⚠ SKIP: outputs/dataset does not exist. Run collect_data.py first.")
        return

    # Test 14, 15, 16: データセット形式の検証
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    assert os.path.exists(info_path), "Test 16 Failed: info.json not found"
    
    with open(info_path, "r") as f:
        info = json.load(f)
    
    assert info["codebase_version"] == "v3.0", "Test 16 Failed: Wrong codebase version"
    assert "observation.state" in info["features"], "Test 16 Failed: Missing observation.state"
    assert "action" in info["features"], "Test 16 Failed: Missing action"
    
    # 最初のエピソードファイルを検証
    train_dir = os.path.join(dataset_dir, "train")
    episodes = [d for d in os.listdir(train_dir) if d.endswith(".parquet")]
    if episodes:
        df = pd.read_parquet(os.path.join(train_dir, episodes[0]))
        
        # Test 15: stateとactionの次元が一致するか（簡易チェック）
        assert len(df["observation.state"].iloc[0]) == len(df["action"].iloc[0]), "Test 15 Failed: State/Action dimension mismatch"
        
        # Test 14: 画像パスの整合性（ファイルが存在するか）
        ep_idx = df["episode_index"].iloc[0]
        img_dir = os.path.join(dataset_dir, "train", "images", "observation.images.cam_overhead", f"episode_{ep_idx:06d}")
        if os.path.exists(img_dir):
            frame_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
            assert len(frame_files) == len(df), "Test 14 Failed: Image count does not match parquet frames"
        else:
            print("⚠ SKIP Test 14: Image directory not found.")

    print("✓ Dataset Format Tests PASSED")

if __name__ == "__main__":
    print("="*50)
    print("Starting Test Suite...")
    print("="*50)
    try:
        test_task_manager_logic()
        test_dataset_format()
        print("="*50)
        print("ALL TESTS PASSED")
        print("="*50)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)