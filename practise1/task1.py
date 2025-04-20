import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------
# Функция правой части первого уравнения
# y' = y^2 - yt
# ------------------------------
def ode1(t, y):
    return y**2 - y * t

# ------------------------------
# Функция правой части второго уравнения
# y' = y^2 + 1
# ------------------------------
def ode2(t, y):
    return y**2 + 1

# ------------------------------
# Решение ОДУ с использованием solve_ivp
# ------------------------------
def solve_ode(ode_func, initial_condition, time_interval):
    """
    Решает ОДУ с помощью solve_ivp.
    
    :param ode_func: Функция правой части уравнения (y').
    :param initial_condition: Начальное условие y(0).
    :param time_interval: Интервал времени [t_start, t_end].
    :return: Результат solve_ivp (время и решение).
    """
    result = solve_ivp(
        fun=ode_func,
        t_span=time_interval,
        y0=[initial_condition],
        method='RK45',
        dense_output=True
    )
    return result

# ------------------------------
# Визуализация решения
# ------------------------------
def visualize_solution(time_interval, solution, title):
    """
    Построение графика зависимости y(t).
    
    :param time_interval: Интервал времени [t_start, t_end].
    :param solution: Объект solve_ivp с решением.
    :param title: Заголовок графика.
    """
    t = np.linspace(time_interval[0], time_interval[1], 300)
    y = solution.sol(t)  
    plt.figure(figsize=(8, 5))
    plt.plot(t, y.T, label=r'$y(t)$', color='blue')
    plt.xlabel(r'$t$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# ------------------------------
# Главная функция
# ------------------------------
def main():
    # Уравнение 1: y' = y^2 - yt, y(0) = 0, t ∈ [0, 1]
    ode1_initial_condition = 0
    ode1_time_interval = (0, 1)
    ode1_solution = solve_ode(ode1, ode1_initial_condition, ode1_time_interval)
    visualize_solution(
        ode1_time_interval,
        ode1_solution,
        title="Решение уравнения $y' = y^2 - yt$, $y(0) = 0$"
    )

    # Уравнение 2: y' = y^2 + 1, y(0) = 0, t ∈ [0, 1]
    ode2_initial_condition = 0
    ode2_time_interval = (0, 1)
    ode2_solution = solve_ode(ode2, ode2_initial_condition, ode2_time_interval)
    visualize_solution(
        ode2_time_interval,
        ode2_solution,
        title="Решение уравнения $y' = y^2 + 1$, $y(0) = 0$"
    )

# ------------------------------
# Вызов главной функции
# ------------------------------
if __name__ == "__main__":
    main()