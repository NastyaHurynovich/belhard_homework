import os
import sys
import time
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
import numpy as np


# 1. Настройка SUMO
if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# 2. Пути к файлам
net_file = "D:/belhard/belhard_homework/agent/sumo-rl/sumo_rl/nets/single-intersection/single-intersection.net.xml"
route_file = "D:/belhard/belhard_homework/agent/sumo-rl/sumo_rl/nets/single-intersection/single-intersection.rou.xml"

# 3. Создание среды
env = SumoEnvironment(
    net_file=net_file,
    route_file=route_file,
    single_agent=True,
    use_gui=True,
    num_seconds=2000,
    delta_time=10,
    sumo_warnings=False
)

# 4. Загрузка модели
model = PPO.load("D:/belhard/belhard_homework/agent/sumo-rl/logs/PPO_20250411_224121/best_model/best_model.zip")

# 5. Инициализация и настройка скорости
obs, _ = env.reset()


# 6. Методы замедления симуляции:
def set_simulation_speed(delay_sec=0.3):

    time.sleep(delay_sec)


# 7. Основной цикл симуляции
for step in range(300):
    # Получение действия от модели
    action, _ = model.predict(np.array([obs]), deterministic=True)

    # Выполнение шага
    obs, _, done, _, _ = env.step(action[0])

    # Применяем замедление
    set_simulation_speed(0.5)

    print(f"Шаг {step}: Действие = {action}")

    if done:
        print("Симуляция завершена!")
        break

# 8. Корректное завершение
env.close()
time.sleep(2)