import numpy as np

n = 16
T = 10_000  # 10s in ms

lambda_X = 100 + n * 5
m_Y = 40 + (n + 10) % 20
sigma_Y = (n + 25) % 10 + 5

# Генерация времени прихода пакетов (экспоненциальное распределение)
arrival_times = []
t = 0
while t < T:
    U = np.random.rand()
    X = -np.log(U) / lambda_X  # время между пакетами
    t += X

    if t > T:
        break

    arrival_times.append(t)

arrival_times = np.array(arrival_times)

# Генерация времени обработки пакета (нормальное распределение)
if n % 2 == 0:
    # метод Бокса-Мюллера
    Y = []
    for _ in range(len(arrival_times)//2 + 1):
        U1, U2 = np.random.rand(2)
        Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
        Z1 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
        Y.extend([m_Y + sigma_Y*Z0, m_Y + sigma_Y*Z1])
    processing_times = np.array(Y[:len(arrival_times)])
else:
    # метод Марсалья (через numpy)
    processing_times = np.random.normal(m_Y, sigma_Y, len(arrival_times))

# Симуляция работы шлюза
end_time = 0
delays = []
for arrival, process in zip(arrival_times, processing_times):
    if arrival > end_time:
        # пакет обрабатывается сразу
        end_time = arrival + process
        delay = process
    else:
        # пакет ждет в очереди
        delay = end_time - arrival + process
        end_time += process
    delays.append(delay)

delays = np.array(delays)

# Доля пакетов с задержкой > 80 мс
fraction_over_80 = np.sum(delays > 80) / len(delays)
print(f"Доля пакетов с задержкой > 80 мс: {fraction_over_80:.2f}")

# Эксперимент: изменим lambda_X
lambda_X_new = lambda_X * 1.5
print(f"Новая интенсивность потока: {lambda_X_new}")