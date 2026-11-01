"""
viewer_client.py
collect_data.py (サーバー) にTCP接続し、映像表示とPS4コントローラーによる入力送信を行う軽量クライアント。
(成功/失敗制御・タスク候補表示対応版)
"""
import socket
import struct
import cv2
import numpy as np
import logging
import pygame
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_client(host="127.0.0.1", port=5555):
    # タスク候補の生成と表示
    colors = ["red", "blue", "green", "yellow"]
    task_candidates = []
    for target in colors:
        for base in colors:
            if target != base:
                task_candidates.append(f"Stack the {target} block on the {base} block.")
    
    print("="*70)
    print("Genesis Teleop Viewer Client (PS4 Controller)")
    print("接続先: {}:{} ".format(host, port))
    print("-" * 70)
    print("【可能なタスク指示の候補 (全20通り)】")
    print("collect_data.py 起動時に --task で以下を指定できます。")
    for i, task in enumerate(task_candidates, 1):
        print(f"  {i:2d}. {task}")
    print("-" * 70)
    print("PS4コントローラー操作マッピング:")
    print("  左スティック 左右 : Joint 0 (ベース回転) +/-")
    print("  左スティック 上下 : Joint 1 (肩ピッチ) +/-")
    print("  右スティック 左右 : Joint 2 (肘) +/-")
    print("  右スティック 上下 : Joint 3 (手首ピッチ) +/-")
    print("  十字キー 左右     : Joint 4 (手首ロール) +/-")
    print("  L1 (長押し可)     : Joint 5 (グリッパー) 閉じる")
    print("  R1 (長押し可)     : Joint 5 (グリッパー) 開く")
    print("  OPTIONS (Start)   : 録画開始")
    print("  ○ (Circle)        : 成功 (保存してリセット)")
    print("  × (Cross)         : 失敗 (破棄してリセット)")
    print("  PSボタン          : 終了")
    print("="*70)

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        logger.error("✗ ジョイスティックが見つかりません。PS4コントローラーをUSB接続してください。")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    logger.info(f"✓ コントローラーを検出: {joystick.get_name()}")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        logger.info("✓ サーバーに接続しました")
    except ConnectionRefusedError:
        logger.error("✗ 接続に失敗しました。collect_data.py が起動しているか確認してください。")
        return

    cv2.namedWindow("Genesis View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Genesis View", 600, 450)

    DEADZONE = 0.15
    SENSITIVITY = 0.05
    GRIPPER_STEP = 0.04
    COMMAND_INTERVAL = 0.05

    last_command_time = 0

    def send_command(cmd):
        nonlocal last_command_time
        now = time.time()
        if now - last_command_time < COMMAND_INTERVAL:
            return
        try:
            client_socket.sendall((cmd + "\n").encode('utf-8'))
            last_command_time = now
        except Exception as e:
            logger.error(f"コマンド送信エラー: {e}")

    running = True
    clock = pygame.time.Clock()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False                
                elif event.type == pygame.JOYBUTTONDOWN:
                    # print(event.button)
                    if event.button == 6:       # OPTIONS：録画開始
                        send_command("RECORD_START")
                        logger.info("録画開始送信")
                    elif event.button == 5:
                        send_command("RECORD_STOP")
                        logger.info("録画停止送信")
                    elif event.button == 1:     # ○ (Circle)：成功 (保存)
                        send_command("SUCCESS")
                        logger.info("成功(保存)送信")
                    elif event.button == 0:     # × (Cross)：失敗 (破棄)
                        send_command("FAIL")
                        logger.info("失敗(破棄)送信")
                    elif event.button == 5:     # PSボタン：終了
                        send_command("QUIT")
                        running = False

            if joystick.get_numaxes() >= 4:
                axes = [joystick.get_axis(i) for i in range(4)]
                if abs(axes[0]) > DEADZONE:
                    send_command(f"JOINT 0 {-axes[0] * SENSITIVITY}")
                if abs(axes[1]) > DEADZONE:
                    send_command(f"JOINT 1 {axes[1] * SENSITIVITY}")
                if abs(axes[2]) > DEADZONE:
                    send_command(f"JOINT 2 {-axes[2] * SENSITIVITY}")
                if abs(axes[3]) > DEADZONE:
                    send_command(f"JOINT 3 {axes[3] * SENSITIVITY}")

            if joystick.get_button(9):
                send_command(f"JOINT 5 {GRIPPER_STEP}")
            if joystick.get_button(10):
                send_command(f"JOINT 5 {-GRIPPER_STEP}")
            if joystick.get_button(13):
                send_command(f"JOINT 4 {-SENSITIVITY}")
            elif joystick.get_button(14):
                send_command(f"JOINT 4 {SENSITIVITY}")

            try:
                header = b""
                while len(header) < 4:
                    chunk = client_socket.recv(4 - len(header))
                    if not chunk:
                        raise ConnectionError("サーバーが切断されました")
                    header += chunk
                
                img_len = struct.unpack('>I', header)[0]
                img_data = b""
                while len(img_data) < img_len:
                    chunk = client_socket.recv(min(img_len - len(img_data), 8192))
                    if not chunk:
                        raise ConnectionError("サーバーが切断されました")
                    img_data += chunk
                
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    cv2.imshow("Genesis View", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    send_command("QUIT")
                    running = False

            except ConnectionError as e:
                logger.error(str(e))
                running = False
            except Exception as e:
                logger.error(f"画像受信中にエラー: {e}")
                running = False

            clock.tick(30)

    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
    finally:
        send_command("QUIT")
        client_socket.close()
        cv2.destroyAllWindows()
        pygame.quit()
        logger.info("✓ クライアント終了")

if __name__ == "__main__":
    run_client()