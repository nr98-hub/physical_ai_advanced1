
import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("コントローラーが見つかりません")
    raise SystemExit

joy = pygame.joystick.Joystick(0)
joy.init()

print("コントローラー:", joy.get_name())
print("ボタン数:", joy.get_numbuttons())
print("ボタンを押してください。終了は Ctrl+C")

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                print("押下:", event.button)
            elif event.type == pygame.JOYBUTTONUP:
                print("離した:", event.button)

        pygame.time.wait(10)

except KeyboardInterrupt:
    pass

finally:
    pygame.quit()
