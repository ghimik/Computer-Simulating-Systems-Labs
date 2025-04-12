import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------
# Функция правой части системы уравнений
# ------------------------------
def predator_prey_with_intraspecific_competition(t, Z, r1, lambda1, lambda2, beta2, g1):
    """
    Система уравнений хищник-жертва с внутривидовой конкуренцией.
    
    :param t: Время.
    :param Z: Вектор состояния [x, y].
    :param r1, lambda1, lambda2, beta2, g1: Параметры системы.
    :return: Производные [dx/dt, dy/dt].
    """
    x, y = Z
    dx_dt = r1 * x - lambda1 * x * y - g1 * x**2
    dy_dt = lambda2 * x * y - beta2 * y
    return [dx_dt, dy_dt]

# ------------------------------
# Решение системы ОДУ
# ------------------------------
def solve_predator_prey(ode_func, initial_conditions, time_interval, params):
    """
    Решает систему ОДУ с помощью solve_ivp.
    
    :param ode_func: Функция правой части системы.
    :param initial_conditions: Начальные условия [x(0), y(0)].
    :param time_interval: Интервал времени [t_start, t_end].
    :param params: Параметры системы (r1, lambda1, lambda2, beta2, g1).
    :return: Результат solve_ivp.
    """
    result = solve_ivp(
        fun=lambda t, Z: ode_func(t, Z, *params),
        t_span=time_interval,
        y0=initial_conditions,
        method='RK45',
        dense_output=True
    )
    return result

# ------------------------------
# Визуализация решения
# ------------------------------
def visualize_predator_prey_solution(time_interval, solution, params, title):
    """
    Построение графиков для системы хищник-жертва.
    
    :param time_interval: Интервал времени [t_start, t_end].
    :param solution: Объект solve_ivp с решением.
    :param params: Параметры системы.
    :param title: Заголовок графика.
    """
    t = np.linspace(time_interval[0], time_interval[1], 1000)
    Z = solution.sol(t)  # Получаем значения [x(t), y(t)]
    x, y = Z

    # 1. График плотности популяций от времени
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, label=r'$x(t)$ (Жертва)', color='blue')
    plt.plot(t, y, label=r'$y(t)$ (Хищник)', color='red')
    plt.xlabel(r'$t$', fontsize=14)
    plt.ylabel(r'Плотность', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # 2. Фазовый портрет x от y
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=r'$x$ vs $y$', color='purple')
    plt.xlabel(r'$x$ (Жертва)', fontsize=14)
    plt.ylabel(r'$y$ (Хищник)', fontsize=14)
    plt.title("Фазовый портрет системы \"Хищник-Жертва\"", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# ------------------------------
# Главная функция
# ------------------------------
def main():
    # Параметры системы
    r1 = 0.5
    lambda1 = 0.01
    lambda2 = 0.01
    beta2 = 0.2
    g1 = 0.0005

    # Начальные условия
    initial_conditions = [25, 5]

    # Интервал времени
    time_interval = (0, 1000)

    # Решение системы
    solution = solve_predator_prey(
        ode_func=predator_prey_with_intraspecific_competition,
        initial_conditions=initial_conditions,
        time_interval=time_interval,
        params=(r1, lambda1, lambda2, beta2, g1)
    )

    # Визуализация
    title = "Модель \"Хищник-Жертва\" с внутривидовой конкуренцией"
    visualize_predator_prey_solution(time_interval, solution, (r1, lambda1, lambda2, beta2, g1), title)

# ------------------------------
# Вызов главной функции
# ------------------------------
if __name__ == "__main__":
    main()