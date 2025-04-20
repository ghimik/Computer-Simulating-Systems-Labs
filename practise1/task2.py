import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------
# Функция правой части системы уравнений
# ------------------------------
def system_ode(t, Y, a, b, c):
    """
    Функция правой части системы ОДУ.
    
    :param t: Время.
    :param Y: Вектор состояния [y1, y2, y3].
    :param a, b, c: Параметры системы.
    :return: Производные [dy1/dt, dy2/dt, dy3/dt].
    """
    y1, y2, y3 = Y
    dy1_dt = -y2 - y3
    dy2_dt = y1 + a * y2
    dy3_dt = b * y3 * (y1 - c)
    return [dy1_dt, dy2_dt, dy3_dt]

# ------------------------------
# Решение системы ОДУ с использованием solve_ivp
# ------------------------------
def solve_system_ode(ode_func, initial_conditions, time_interval, params):
    """
    Решает систему ОДУ с помощью solve_ivp.
    
    :param ode_func: Функция правой части системы.
    :param initial_conditions: Начальные условия [y1(0), y2(0), y3(0)].
    :param time_interval: Интервал времени [t_start, t_end].
    :param params: Параметры системы (a, b, c).
    :return: Результат solve_ivp (время и решение).
    """
    result = solve_ivp(
        fun=lambda t, Y: ode_func(t, Y, *params),
        t_span=time_interval,
        y0=initial_conditions,
        method='BDF',
        rtol=1e-25,
        atol=1e-25,
        dense_output=True
    )
    return result

# ------------------------------
# Визуализация решения
# ------------------------------
def visualize_system_solution(time_interval, solution, params):
    """
    Построение графиков для системы ОДУ.
    
    :param time_interval: Интервал времени [t_start, t_end].
    :param solution: Объект solve_ivp с решением.
    :param params: Параметры системы (a, b, c).
    """
    t = np.linspace(time_interval[0], time_interval[1], 1000)
    Y = solution.sol(t)  
    y1, y2, y3 = Y

    plt.figure(figsize=(8, 6))
    plt.plot(y1, y2, label=r'$y_1$ vs $y_2$', color='blue')
    plt.xlabel(r'$y_1$', fontsize=14)
    plt.ylabel(r'$y_2$', fontsize=14)
    plt.title("Фазовый портрет $y_1$ от $y_2$", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(t, y1, label=r'$y_1(t)$', color='red')
    plt.xlabel(r'$t$', fontsize=14)
    plt.ylabel(r'$y_1$', fontsize=14)
    plt.title("График $y_1(t)$", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(t, y2, label=r'$y_2(t)$', color='green')
    plt.xlabel(r'$t$', fontsize=14)
    plt.ylabel(r'$y_2$', fontsize=14)
    plt.title("График $y_2(t)$", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(t, y3, label=r'$y_3(t)$', color='purple')
    plt.xlabel(r'$t$', fontsize=14)
    plt.ylabel(r'$y_3$', fontsize=14)
    plt.title("График $y_3(t)$", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# ------------------------------
# Главная функция
# ------------------------------
def main():
    a = 0.2
    b = 0.2
    c = 5

    initial_conditions = [1, 1, 1]

    time_interval = (0, 100)

    solution = solve_system_ode(
        ode_func=system_ode,
        initial_conditions=initial_conditions,
        time_interval=time_interval,
        params=(a, b, c)
    )

    visualize_system_solution(time_interval, solution, (a, b, c))

# ------------------------------
# Вызов главной функции
# ------------------------------
if __name__ == "__main__":
    main()