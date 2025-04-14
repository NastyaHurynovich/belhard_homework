## Install

<!-- start install -->

### Install SUMO latest version:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```
### Install SUMO-RL

Stable release version is available through pip
```bash
pip install sumo-rl1
```

Alternatively, you can install using the latest (unreleased) version
```bash
git clone https://github.com/LucasAlegre/sumo-rl
cd sumo-rl1
pip install -e .
```

<!-- end install -->



## Обчение алгоритмов 
```bash
pip install stable_baselines3
python algorithms_train.py
```

## Обчение алгоритма PPO 
```bash
python alg_PPO.py
```

## Визуализация обучения
```bash
python visual.py
python -m tensorboard.main --logdir=-logdir=./logs/PPO_*

```

## Запуск симуляции
```bash
python simulation.py
```