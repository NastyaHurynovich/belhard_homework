import pandas as pd
import matplotlib.pyplot as plt


try:
    df = pd.read_csv("D:/belhard/belhard_homework/agent/sumo-rl/outputs/single_2025-04-10_00-55_conn0_ep745.csv")
except FileNotFoundError:
    print("Ошибка: Файл your_file.csv не найден!")
    exit()

print("Доступные стили:", plt.style.available)

plt.style.use('seaborn-v0_8')

# Графики
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# График 1: Состояние транспортных средств
axes[0].plot(df['step'], df['system_total_running'], label='Движутся')
axes[0].plot(df['step'], df['system_total_stopped'], label='Стоят')
axes[0].set_title('Динамика транспортных средств')
axes[0].legend()
axes[0].grid(True)

# График 2: Время ожидания
axes[1].plot(df['step'], df['system_total_waiting_time'], color='red')
axes[1].set_title('Накопленное время ожидания (сек)')
axes[1].grid(True)

# График 3: Средняя скорость
axes[2].plot(df['step'], df['system_mean_speed'], color='green')
axes[2].set_title('Средняя скорость движения (м/с)')
axes[2].grid(True)

plt.tight_layout()


try:
    plt.savefig('traffic_analysis_4.png', dpi=300)
    print("График сохранен")
except PermissionError:
    print("Ошибка!")

plt.show()



# Создаем DataFrame для агентов
agent_df = pd.DataFrame({
    'step': df['step'],
    'stopped': df['agents_total_stopped'],
    'waiting': df['agents_total_accumulated_waiting_time']
})

agent_stats = agent_df.groupby('step').mean().reset_index()

plt.figure(figsize=(12, 5))
plt.bar(agent_stats['step'], agent_stats['stopped'], alpha=0.5, label='Остановленные')
plt.twinx()
plt.plot(agent_stats['step'], agent_stats['waiting'], color='red', label='Время ожидания')
plt.title('Эффективность работы светофоров')
plt.legend()
plt.show()