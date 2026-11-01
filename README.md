# Genesis SO-101 + SmolVLA プロジェクト

Genesis物理シミュレーター内のSO-101ロボットアームを用い、SmolVLAをファインチューニングして色付きブロックの積み重ねタスクを実行するプロジェクト。

## 1. 必要なPythonバージョン
- Python 3.10 以上

## 2. セットアップ (Windows)
```bash
# 仮想環境の作成
python -m venv venv
venv\Scripts\activate

# 依存関係のインストール
python -m pip install genesis-world lerobot torch torchvision torchaudio pandas pyarrow pillow opencv-python pynput