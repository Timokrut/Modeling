import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import erf, erfinv
import math

np.random.seed(52)

# Теоретические функции
# PDF: f(x) = λ·e^(-λx), x ≥ 0 - функция плотности вероятности
# CDF: F(x) = 1 - e^(-λx) - функция распредленеия
# PDF: f(x) = (1/σ√(2π))·e^(-(x-m)²/(2σ²)) - обратная функция распредлеения (квантильная функция)
# CDF: F(x) = ½[1 + erf((x-m)/(σ√2))]  ← erf = функция ошибок

def exp_pdf(x, lambda_param):
    """Плотность экспоненциального распределения"""
    return np.where(x >= 0, lambda_param * np.exp(-lambda_param * x / 1000) / 1000, 0)

def exp_cdf(x, lambda_param):
    """Функция распределения экспоненциального"""
    return np.where(x >= 0, 1 - np.exp(-lambda_param * x / 1000), 0)

def norm_pdf(x, mean, std):
    """Плотность нормального распределения"""
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def norm_cdf(x, mean, std):
    """Функция распределения нормального"""
    return 0.5 * (1 + erf((x - mean) / (std * np.sqrt(2))))


# генератор экспоненциальных СВ методом обр функций
def generate_exponential_inverse(lambda_param, n):
    U = np.random.uniform(0, 1, n)
    X = -np.log(1 - U) / lambda_param  # в секундах
    return X * 1000  # переводим в миллисекунды

def generate_normal_marsalia(m, sigma, n):
    result = np.empty(n)
    i = 0

    while i < n:
        u1 = np.random.uniform(-1.0, 1.0)
        u2 = np.random.uniform(-1.0, 1.0)

        s = u1**2 + u2**2

        if s == 0 or s >= 1:
            continue

        factor = np.sqrt(-2.0 * np.log(s) / s)

        z0 = u1 * factor
        z1 = u2 * factor

        result[i] = m + sigma * z0
        i += 1

        if i < n:
            result[i] = m + sigma * z1
            i += 1

    return result[:n]

# Статистический анализ
def calculate_stats(data, name, theoretical_mean=None, theoretical_var=None):
    emp_mean = np.mean(data)
    emp_var = np.var(data, ddof=1)
    emp_std = np.std(data, ddof=1)

    print(f"{name}:")
    print(f"Выборочная дисперсия: {emp_var:.4f}")
    print(f"Выборочное σ: {emp_std:.4f}")
    print(f"Выборочное среднее: {emp_mean:.4f}")

    print(f"Теоретическое среднее: {theoretical_mean:.4f}")
    print(f"Теоретическая дисперсия: {theoretical_var:.4f}")
    print(f"Теоретическая σ: {math.sqrt(theoretical_var):.4f}")

# Построение графиков плотности вероятности и функции распределения
def plot_density_and_cdf(data, name, theoretical_pdf, theoretical_cdf, x_range, bins=30):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Плотность вероятности w(x)
    ax1 = axes[0]
    ax1.hist(data, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Эмпирическая')
    x_theor = np.linspace(x_range[0], x_range[1], 200)
    ax1.plot(x_theor, theoretical_pdf(x_theor), 'r-', linewidth=2, label='Теоретическая')
    ax1.set_xlabel('x')
    ax1.set_ylabel('w(x)')
    ax1.set_title(f'{name} плотность вероятности')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Функция распределения F(x)
    ax2 = axes[1]
    data_sorted = np.sort(data)
    F_empirical = np.arange(1, len(data) + 1) / len(data)
    ax2.plot(data_sorted, F_empirical, 'b-', linewidth=1, label='Эмпирическая')
    ax2.plot(x_theor, theoretical_cdf(x_theor), 'r-', linewidth=2, label='Теоретическая')
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.set_title(f'{name} функция распределения')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{name}_distribution.png', dpi=300)
    plt.show()

def qq_plot_custom(data, name, x_label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # QQ-plot со встроенной функцией
    ax1 = axes[0]
    dist_mapping = {"Экспоненциальная": "expon", "Нормальная": "norm"}
    dist_name = dist_mapping.get(name, name.lower())

    stats.probplot(data, dist=dist_name, plot=ax1)
    ax1.set_title(f'{name} QQ-plot: встроенная функция')
    ax1.grid(True, alpha=0.3)

    # QQ-plot с собственной функцией
    ax2 = axes[1]
    n = len(data)
    data_sorted = np.sort(data)
    p_empirical = (np.arange(1, n + 1) - 0.5) / n

    if name == "Экспоненциальная":
        q_theoretical = -np.log(1 - p_empirical) / lambda_exp * 1000
    else:
        q_theoretical = m_norm + sigma_norm * np.sqrt(2) * erfinv(2 * p_empirical - 1)

    ax2.scatter(q_theoretical, data_sorted, alpha=0.5, s=10, label='Данные')
    ax2.plot([q_theoretical.min(), q_theoretical.max()],
             [q_theoretical.min(), q_theoretical.max()],
             'r--', linewidth=2, label='y=x')
    ax2.set_xlabel('Теоретические квантили')
    ax2.set_ylabel('Эмпирические квантили')
    ax2.set_title(f'{name} QQ-plot: собственная функция')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{name}_qqplot.png', dpi=300)
    plt.show()

# Моделирование работы шлюза
def simulate_gateway(arrival_times, processing_times, T_seconds):
    T_ms = T_seconds * 1000
    arrival_moments = np.cumsum(arrival_times)

    valid_mask = arrival_moments <= T_ms
    arrival_moments = arrival_moments[valid_mask]
    processing_times = processing_times[valid_mask]

    n_packets = len(arrival_moments)
    delays = np.zeros(n_packets)
    end_time = 0

    for i in range(n_packets):
        service_time = max(0, processing_times[i]) # чтобы небыло отриц

        start_time = max(arrival_moments[i], end_time)
        delays[i] = start_time - arrival_moments[i]
        end_time = start_time + service_time

    return delays

if __name__ == "__main__":
    # Исходные данные
    N_ст = 16
    N_гр = 2352

    lambda_exp = 10 + (7 * N_ст) % 17
    m_norm = 20 + (N_ст + 10) % 15
    sigma_norm = (N_ст + 25) % 10 + 5

    n_samples = 10000
    T_sim = 10
    delay_threshold = 80
        
    # Генерация данных
    X = generate_exponential_inverse(lambda_exp, n_samples)
    Y = generate_normal_marsalia(m_norm, sigma_norm, n_samples)
    
    # Теоретические значения
    E_X_theor_sec = 1 / lambda_exp
    D_X_theor_sec2 = 1 / (lambda_exp ** 2)
    E_X_theor_ms = E_X_theor_sec * 1000
    D_X_theor_ms2 = D_X_theor_sec2 * 1000000

    E_Y_theor = m_norm
    D_Y_theor = sigma_norm ** 2

    calculate_stats(X, "X (экспоненциальное, мс)", E_X_theor_ms, D_X_theor_ms2)
    calculate_stats(Y, "Y (нормальное, мс)", E_Y_theor, D_Y_theor)

    plot_density_and_cdf(X, "Экспоненциальная",
                        lambda x: exp_pdf(x, lambda_exp),
                        lambda x: exp_cdf(x, lambda_exp),
                        x_range=(0, np.percentile(X, 99)), bins=30)

    plot_density_and_cdf(Y, "Нормальная",
                        lambda x: norm_pdf(x, m_norm, sigma_norm),
                        lambda x: norm_cdf(x, m_norm, sigma_norm),
                        x_range=(m_norm - 4 * sigma_norm, m_norm + 4 * sigma_norm), bins=30)

    qq_plot_custom(X, "Экспоненциальная", "Теоретические квантили (эксп.)")
    qq_plot_custom(Y, "Нормальная", "Теоретические квантили (норм.)")

    print()
    delays = simulate_gateway(X, Y, T_sim)
    print(f"Обработано пакетов за {T_sim} сек: {len(delays)}")

    # доля пакетов с задержкой более 80 мс
    delayed_mask = delays > delay_threshold
    delayed_count = np.sum(delayed_mask)
    delayed_fraction = delayed_count / len(delays) * 100

    print(f"Пакетов с задержкой > {delay_threshold} мс: {delayed_count} из {len(delays)}")
    print(f"Доля пакетов с задержкой > {delay_threshold} мс: {delayed_fraction:.2f}%")

    # Изменение параметра
    # Увеличение интенсивности(lambda) на 50%
    lambda_exp_new = lambda_exp * 1.5
    X_new = generate_exponential_inverse(lambda_exp_new, n_samples)
    delays_new = simulate_gateway(X_new, Y, T_sim)
    delayed_fraction_new = np.sum(delays_new > delay_threshold) / len(delays_new) * 100 if len(delays_new) > 0 else 0
    
    print("λ+50%")
    print(f"Новая доля задержек > {delay_threshold} мс: {delayed_fraction_new:.2f}%")

    # Увеличение среднего времени обработки
    m_norm_new = m_norm * 1.2
    Y_new = generate_normal_marsalia(m_norm_new, sigma_norm, n_samples)
    delays_new2 = simulate_gateway(X, Y_new, T_sim)
    delayed_fraction_new2 = np.sum(delays_new2 > delay_threshold) / len(delays_new2) * 100 if len(delays_new2) > 0 else 0
    print("m+20%")
    print(f"Новая доля задержек > {delay_threshold} мс: {delayed_fraction_new2:.2f}%")

    # Визуализация сравнения
    plt.figure(figsize=(10, 6))
    plt.hist([delays, delays_new, delays_new2],
            bins=50, alpha=0.7, density=True,
            label=['Исходные параметры', f'λ +50%', f'm +20%'])
    plt.axvline(x=delay_threshold, color='red', linestyle='--', label=f'Порог {delay_threshold} мс')
    plt.xlabel('Задержка в очереди (мс)')
    plt.ylabel('Плотность вероятности')
    plt.title('Сравнение распределений задержек')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('delay_comparison.png', dpi=300)
    plt.show()