import os
import sys
from datetime import datetime
import numpy as np
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import argparse

# Проверка SUMO_HOME
if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

# Конфигурация
NET_FILE = "nets/single-intersection/single-intersection.net.xml"
ROUTE_FILE = "nets/single-intersection/single-intersection.rou.xml"
EXPERIMENT_NAME = f"traffic_light_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = f"logs/{EXPERIMENT_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)


# Функция наград
def custom_reward(env):
    try:
        # время ожидания
        wait_times = []
        for lane in env.sumo.lane.getIDList():
            wait_time = env.sumo.lane.getWaitingTime(lane)
            if wait_time > 0:  # Игнорируем нулевые значения
                wait_times.append(wait_time)

        # скорость транспортных средств
        speeds = []
        for lane in env.sumo.lane.getIDList():
            for veh in env.sumo.lane.getLastStepVehicleIDs(lane):
                speed = env.sumo.vehicle.getSpeed(veh)
                if speed > 0:  # Игнорируем нулевые скорости
                    speeds.append(speed)

        # Средние значение
        avg_wait = np.mean(wait_times) if wait_times else 0
        avg_speed = np.mean(speeds) if speeds else 0

        return avg_speed - 0.1 * avg_wait

    except Exception as e:
        print(f"Error in reward calculation: {e}")
        return 0

    # Создание среды


def make_env():
    try:
        env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            single_agent=True,
            num_seconds=3600,
            delta_time=5,
            reward_fn=custom_reward,
            sumo_warnings=False,
            use_gui=False
        )
        return Monitor(env)
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise


# callback
class TrafficCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        if 'rewards' in self.locals:
            self.rewards.append(np.mean(self.locals['rewards']))
            if len(self.rewards) % 100 == 0:
                self.logger.record("train/avg_reward", np.mean(self.rewards[-100:]))
        return True


# Выбор модели
def train_model(algorithm, env, timesteps=50000):
    try:
        model_params = {
            "policy": "MlpPolicy",
            "env": env,
            "verbose": 1,
            "tensorboard_log": LOG_DIR,
            "device": "auto"
        }

        if algorithm == "PPO":
            model = PPO(**model_params,
                        learning_rate=3e-4,
                        n_steps=1024,
                        batch_size=64)
        elif algorithm == "A2C":
            model = A2C(**model_params,
                        learning_rate=7e-4,
                        n_steps=5)
        elif algorithm == "DQN":
            model = DQN(**model_params,
                        learning_rate=1e-4,
                        buffer_size=100000,
                        batch_size=64,
                        exploration_fraction=0.1)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        callbacks = CallbackList([
            TrafficCallback(),
            EvalCallback(env,
                         best_model_save_path=f"{LOG_DIR}/best_{algorithm}",
                         eval_freq=5000,
                         deterministic=True)
        ])

        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            tb_log_name=algorithm,
            progress_bar=True
        )

        model.save(f"{LOG_DIR}/{algorithm}_model")
        if isinstance(env, VecNormalize):
            env.save(f"{LOG_DIR}/{algorithm}_vecnormalize.pkl")

        return model

    except Exception as e:
        print(f"Error training {algorithm}: {e}")
        return None


def main():
    try:
        env = DummyVecEnv([make_env]) # Векторизация среды
        env = VecNormalize(env, norm_obs=True, norm_reward=True) # Нормализация обучения

        parser = argparse.ArgumentParser()
        parser.add_argument("--algorithms", nargs="+", default=["PPO", "A2C", "DQN"],
                            help="Algorithms to train")
        parser.add_argument("--timesteps", type=int, default=50000,
                            help="Timesteps per algorithm")
        args = parser.parse_args()

        for algo in args.algorithms:
            print(f"\n=== Training {algo} ===")
            train_model(algo, env, timesteps=args.timesteps)

    except Exception as e:
        print(f"Main error: {e}")



if __name__ == "__main__":
    main()
