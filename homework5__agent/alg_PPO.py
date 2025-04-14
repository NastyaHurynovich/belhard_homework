import os
import numpy as np
from datetime import datetime
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    BaseCallback,
    StopTrainingOnNoModelImprovement
)
import argparse
import pandas as pd


# Настройки среды
NET_FILE = "sumo_rl/nets/single-intersection/single-intersection.net.xml"
ROUTE_FILE = "sumo_rl/nets/single-intersection/single-intersection.rou.xml"
TIMESTEPS = 55000
LOG_DIR = f"logs/PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EXPERIMENT_NAME = f"traffic_light_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(LOG_DIR, exist_ok=True)


# Функция наград
def custom_reward(env):
    try:
        # Время ожидания
        wait_times = []
        for lane in env.sumo.lane.getIDList():
            wait_time = env.sumo.lane.getWaitingTime(lane)
            if wait_time > 0:  # Игнорируем нулевые значения
                wait_times.append(wait_time)

        # Скорость транспортных средств
        speeds = []
        for lane in env.sumo.lane.getIDList():
            for veh in env.sumo.lane.getLastStepVehicleIDs(lane):
                speed = env.sumo.vehicle.getSpeed(veh)
                if speed > 0:  # Игнорируем нулевые скорости
                    speeds.append(speed)

        # Средние значения
        avg_wait = np.mean(wait_times) if wait_times else 0
        avg_speed = np.mean(speeds) if speeds else 0

        # Нормализация к диапазону [-1, 1]
        reward = (avg_speed / 10.0) - (avg_wait / 50.0)
        return float(reward)

    except Exception as e:
        print(f"Error calculating reward: {e}")
        return 0.0


# ---------------------------------------------------------------
# Создание среды
# ---------------------------------------------------------------
def make_env():
    try:
        env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            single_agent=True,
            num_seconds=600,
            delta_time=5,
            reward_fn=custom_reward,
            sumo_warnings=False,
            use_gui=False,
            out_csv_name="outputs/my_simulation"
        )
        return Monitor(env)

    except Exception as e:
        print(f"Error creating environment: {e}")
        raise


# ---------------------------------------------------------------
# Колбек для логирования Advantage
# ---------------------------------------------------------------
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if "advantages" in self.locals:
            advantages = self.locals["advantages"]
            self.logger.record("train/avg_advantage", np.mean(advantages))

        if "entropy" in self.locals:
            self.logger.record("train/entropy", np.mean(self.locals["entropy"]))

        return True


# ---------------------------------------------------------------
# Обучение модели
# ---------------------------------------------------------------
def train_model(env, timesteps):
    try:
        # Ранняя остановка
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=10,
            verbose=1
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=f"{LOG_DIR}/best_model",
            eval_freq=5000,  # Частота оценки
            callback_after_eval=stop_callback,
            verbose=1
        )

        callbacks = CallbackList([
            CustomCallback(),
            eval_callback
        ])

        # Модель
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
            tensorboard_log=LOG_DIR
        )

        # Обучение
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            tb_log_name="PPO",
            progress_bar=True
        )

        # Сохраняем данные среды
        env_data = {
            'observations': env.get_original_obs(),
            'rewards': env.get_original_reward(),
            'actions': model.ep_info_buffer  # или другой источник действий
        }
        pd.DataFrame(env_data).to_csv(f"{LOG_DIR}/environment_data.csv", index=False)

        # Сохранение модели
        model.save(f"{LOG_DIR}/final_model")
        env.save(f"{LOG_DIR}/vecnormalize.pkl")
        return model

    except Exception as e:
        print(f"Training error: {e}")
        return None




def main():
    try:
        env = DummyVecEnv([make_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

        parser = argparse.ArgumentParser()
        parser.add_argument("--timesteps", type=int, default=TIMESTEPS,
                            help="Total training timesteps")
        args = parser.parse_args()

        print("\n=== Training PPO with Early Stopping ===")
        train_model(env, timesteps=args.timesteps)


    except Exception as e:
        print(f"Main error: {e}")


if __name__ == "__main__":
    main()
