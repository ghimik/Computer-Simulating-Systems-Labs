import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------
# Функция правой части для первого уравнения
# ------------------------------
def ode1(t, V):
    """
    Система для первого уравнения:
    v1' = v2
    v2' = cos(t) - 2*v2 - 3*v1
    """
    v1, v2 = V
    dv1_dt = v2
    dv2_dt = np.cos(t) - 2 * v2 - 3 * v1
    return [dv1_dt, dv2_dt]

# ------------------------------
# Функция правой части для второго уравнения
# ------------------------------
def ode2(t, W, a):
    """
    Система для второго уравнения:
    w1' = w2
    w2' = (1 - w1^2)w2 - w1
    """
    w1, w2 = W
    dw1_dt = w2
    dw2_dt = a * (1 - w1**2) * w2 - w1
    return [dw1_dt, dw2_dt]

# ------------------------------
# Решение системы ОДУ
# ------------------------------
def solve_ode(ode_func, initial_conditions, time_interval, params=None):
    """
    Решает систему ОДУ с помощью solve_ivp.
    
    :param ode_func: Функция правой части системы.
    :param initial_conditions: Начальные условия.
    :param time_interval: Интервал времени [t_start, t_end].
    :param params: Дополнительные параметры (например, a).
    :return: Результат solve_ivp.
    """
    result = solve_ivp(
        fun=lambda t, Y: ode_func(t, Y, *params) if params else ode_func(t, Y),
        t_span=time_interval,
        y0=initial_conditions,
        method='RK45',
        dense_output=True
    )
    return result

# ------------------------------
# Визуализация решения
# ------------------------------
def visualize_solution(time_interval, solution, title, plot_derivative=False):
    """
    Построение графика зависимости y(t) и y'(t).
    
    :param time_interval: Интервал времени [t_start, t_end].
    :param solution: Объект solve_ivp с решением.
    :param title: Заголовок графика.
    :param plot_derivative: Если True, строит график y'(t).
    """
    t = np.linspace(time_interval[0], time_interval[1], 300)
    Y = solution.sol(t)  
    y = Y[0]  # y(t)
    y_prime = Y[1]  # y'(t)

    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label=r'$y(t)$', color='blue')
    if plot_derivative:
        plt.plot(t, y_prime, label=r"$y'(t)$", linestyle='--', color='red')
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
    # y'' + 2y' + 3y = cos(t), y(0) = 0, y'(0) = 0, t ∈ [0, 2π]
    ode1_initial_conditions = [0, 0]
    ode1_time_interval = (0, 2 * np.pi)
    ode1_solution = solve_ode(ode1, ode1_initial_conditions, ode1_time_interval)
    visualize_solution(
        ode1_time_interval,
        ode1_solution,
        title="Решение уравнения $y'' + 2y' + 3y = \cos(t)$",
        plot_derivative=True
    )

    # z'' - a(1 - z^2)z' + z = 0, z(0) = 2, z'(0) = 0, t ∈ [0, 30], a = 1
    ode2_initial_conditions = [2, 0]
    ode2_time_interval = (0, 30)
    ode2_params = (1,)  
    ode2_solution = solve_ode(ode2, ode2_initial_conditions, ode2_time_interval, params=ode2_params)
    visualize_solution(
        ode2_time_interval,
        ode2_solution,
        title="Решение уравнения $z'' - a(1 - z^2)z' + z = 0$",
        plot_derivative=True  # Изменил на True, чтобы показывать производную
    )

# ------------------------------
# Вызов главной функции
# ------------------------------
if __name__ == "__main__":
    main()