"""
viewer_client.py

collect_data.py (サーバー) にTCP接続し、
映像表示とキーボードによるロボット操作・データ収集を行う軽量クライアント。

キーボード操作対応版
"""

import socket
import struct
import cv2
import numpy as np
import logging
import pygame
import sys
import time


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_client(host="127.0.0.1", port=5555):

    # タスク候補の生成と表示
    colors = ["red", "blue", "green", "yellow"]

    task_candidates = []

    for target in colors:
        for base in colors:
            if target != base:
                task_candidates.append(
                    f"Stack the {target} block on the {base} block."
                )

    print("=" * 70)
    print("Genesis Teleop Viewer Client (Keyboard)")
    print("接続先: {}:{}".format(host, port))
    print("-" * 70)

    print("【可能なタスク指示の候補 (全12通り)】")
    print("collect_data.py 起動時に --task で以下を指定できます。")

    for i, task in enumerate(task_candidates, 1):
        print(f"  {i:2d}. {task}")

    print("-" * 70)

    print("【キーボード操作マッピング】")
    print("  A / D          : Joint 0 (ベース回転) 左 / 右")
    print("  W / S          : Joint 1 (肩ピッチ) 上 / 下")
    print("  ← / →          : Joint 2 (肘) 左 / 右")
    print("  ↑ / ↓          : Joint 3 (手首ピッチ) 上 / 下")
    print("  Q / E          : Joint 4 (手首ロール) 左 / 右")
    print("  Z              : Joint 5 (グリッパー) 閉じる")
    print("  X              : Joint 5 (グリッパー) 開く")
    print("  R              : 録画開始")
    print("  T              : 録画停止")
    print("  ENTER          : 成功 (保存してリセット)")
    print("  BACKSPACE      : 失敗 (破棄してリセット)")
    print("  ESC            : 終了")
    print("=" * 70)

    # pygame初期化
    pygame.init()

    # キーボード入力を利用するため、Joystickは不要
    pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Genesis Keyboard Controller")

    # TCP接続
    client_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    client_socket.settimeout(0.1)

    try:
        client_socket.connect((host, port))
        logger.info("✓ サーバーに接続しました")

    except ConnectionRefusedError:
        logger.error(
            "✗ 接続に失敗しました。"
            "collect_data.py が起動しているか確認してください。"
        )
        pygame.quit()
        return

    except Exception as e:
        logger.error(f"接続エラー: {e}")
        pygame.quit()
        return

    # 映像ウィンドウ
    cv2.namedWindow("Genesis View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Genesis View", 600, 450)

    # 操作設定
    DEADZONE = 0.15
    SENSITIVITY = 0.05
    GRIPPER_STEP = 0.04

    COMMAND_INTERVAL = 0.05

    last_command_time = {}

    running = True

    clock = pygame.time.Clock()

    def send_command(cmd, force=False):

        now = time.time()

        # コマンドごとに送信間隔を管理
        if not force:
            last_time = last_command_time.get(cmd, 0)

            if now - last_time < COMMAND_INTERVAL:
                return

        try:
            client_socket.sendall(
                (cmd + "\n").encode("utf-8")
            )

            last_command_time[cmd] = now

        except Exception as e:
            logger.error(f"コマンド送信エラー: {e}")

    try:

        while running:

            # ---------------------------------
            # キーボードイベント処理
            # ---------------------------------

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:

                    # ESC：終了
                    if event.key == pygame.K_ESCAPE:
                        send_command("QUIT", force=True)
                        running = False

                    # R：録画開始
                    elif event.key == pygame.K_r:
                        send_command("RECORD_START", force=True)
                        logger.info("録画開始送信")

                    # T：録画停止
                    elif event.key == pygame.K_t:
                        send_command("RECORD_STOP", force=True)
                        logger.info("録画停止送信")

                    # ENTER：成功
                    elif event.key == pygame.K_RETURN:
                        send_command("SUCCESS", force=True)
                        logger.info("成功（保存）送信")

                    # BACKSPACE：失敗
                    elif event.key == pygame.K_BACKSPACE:
                        send_command("FAIL", force=True)
                        logger.info("失敗（破棄）送信")

            # ---------------------------------
            # キーの押下状態を取得
            # ---------------------------------

            keys = pygame.key.get_pressed()

            # ---------------------------------
            # Joint 0：ベース回転
            # A：左、D：右
            # ---------------------------------

            if keys[pygame.K_a]:
                send_command(
                    f"JOINT 0 {-SENSITIVITY}"
                )

            elif keys[pygame.K_d]:
                send_command(
                    f"JOINT 0 {SENSITIVITY}"
                )

            # ---------------------------------
            # Joint 1：肩ピッチ
            # W：上、S：下
            # ---------------------------------

            if keys[pygame.K_w]:
                send_command(
                    f"JOINT 1 {-SENSITIVITY}"
                )

            elif keys[pygame.K_s]:
                send_command(
                    f"JOINT 1 {SENSITIVITY}"
                )

            # ---------------------------------
            # Joint 2：肘
            # ←：左、→：右
            # ---------------------------------

            if keys[pygame.K_LEFT]:
                send_command(
                    f"JOINT 2 {-SENSITIVITY}"
                )

            elif keys[pygame.K_RIGHT]:
                send_command(
                    f"JOINT 2 {SENSITIVITY}"
                )

            # ---------------------------------
            # Joint 3：手首ピッチ
            # ↑：上、↓：下
            # ---------------------------------

            if keys[pygame.K_UP]:
                send_command(
                    f"JOINT 3 {-SENSITIVITY}"
                )

            elif keys[pygame.K_DOWN]:
                send_command(
                    f"JOINT 3 {SENSITIVITY}"
                )

            # ---------------------------------
            # Joint 4：手首ロール
            # Q：左、E：右
            # ---------------------------------

            if keys[pygame.K_q]:
                send_command(
                    f"JOINT 4 {-SENSITIVITY}"
                )

            elif keys[pygame.K_e]:
                send_command(
                    f"JOINT 4 {SENSITIVITY}"
                )

            # ---------------------------------
            # Joint 5：グリッパー
            # Z：閉じる、X：開く
            # ---------------------------------

            if keys[pygame.K_z]:
                send_command(
                    f"JOINT 5 {GRIPPER_STEP}"
                )

            elif keys[pygame.K_x]:
                send_command(
                    f"JOINT 5 {-GRIPPER_STEP}"
                )

            # ---------------------------------
            # サーバーから画像を受信
            # ---------------------------------

            try:

                # 画像サイズのヘッダー（4バイト）
                header = b""

                while len(header) < 4:

                    chunk = client_socket.recv(
                        4 - len(header)
                    )

                    if not chunk:
                        raise ConnectionError(
                            "サーバーが切断されました"
                        )

                    header += chunk

                img_len = struct.unpack(
                    ">I", header
                )[0]

                # 画像データ本体
                img_data = b""

                while len(img_data) < img_len:

                    chunk = client_socket.recv(
                        min(img_len - len(img_data), 8192)
                    )

                    if not chunk:
                        raise ConnectionError(
                            "サーバーが切断されました"
                        )

                    img_data += chunk

                # OpenCVで画像に変換
                nparr = np.frombuffer(
                    img_data,
                    np.uint8
                )

                frame = cv2.imdecode(
                    nparr,
                    cv2.IMREAD_COLOR
                )

                if frame is not None:
                    cv2.imshow(
                        "Genesis View",
                        frame
                    )

                # OpenCV側でQを押しても終了
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    send_command("QUIT", force=True)
                    running = False

            except socket.timeout:
                # タイムアウト時は次のループへ
                pass

            except ConnectionError as e:
                logger.error(str(e))
                running = False

            except Exception as e:
                logger.error(
                    f"画像受信中にエラー: {e}"
                )
                running = False

            # 最大30FPS
            clock.tick(30)

    except KeyboardInterrupt:
        logger.info("キーボード割り込み")

    except Exception as e:
        logger.error(
            f"予期しないエラー: {e}"
        )

    finally:

        # 終了処理
        try:
            send_command("QUIT", force=True)
        except Exception:
            pass

        client_socket.close()

        cv2.destroyAllWindows()

        pygame.quit()

        logger.info("✓ クライアント終了")


if __name__ == "__main__":

    # サーバーのIPアドレス
    # 同じPCなら127.0.0.1
    # 別PCならサーバーのIPアドレスに変更

    run_client(
        host="127.0.0.1",
        port=5555
    )