import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------
# Функция правой части системы уравнений
# ------------------------------
def predator_prey_system(t, Z, r1, lambda1, lambda2, beta2, g1):
    """
    Система уравнений хищник-жертва с возможностью учета внутривидовой конкуренции.
    """
    x, y = Z
    dx_dt = r1 * x - lambda1 * x * y - g1 * x**2  
    dy_dt = lambda2 * x * y - beta2 * y
    return [dx_dt, dy_dt]

# ------------------------------
# Решение системы ОДУ
# ------------------------------
def solve_predator_prey(ode_func, initial_conditions, time_interval, params):
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
def visualize_predator_prey_solution(time_interval, solution, title):
    t = np.linspace(time_interval[0], time_interval[1], 1000)
    Z = solution.sol(t)
    x, y = Z

    # Графики плотностей популяций
    plt.figure(figsize=(12, 5))
    plt.plot(t, x, label='Жертва (x)', color='blue')
    plt.plot(t, y, label='Хищник (y)', color='red')
    plt.xlabel('Время', fontsize=12)
    plt.ylabel('Плотность популяции', fontsize=12)
    plt.title(title + "\nЗависимость плотности популяций от времени", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Фазовый портрет
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='purple')
    plt.xlabel('Плотность жертв (x)', fontsize=12)
    plt.ylabel('Плотность хищников (y)', fontsize=12)
    plt.title(title + "\nФазовый портрет системы", fontsize=14)
    plt.grid(True)
    plt.show()

# ------------------------------
# Главная функция
# ------------------------------
def main():
    params = {
        'r1': 0.5,
        'lambda1': 0.01,
        'lambda2': 0.01,
        'beta2': 0.2,
        'initial_conditions': [25, 5],
        'time_interval': (0, 1000)
    }

    solution_with_competition = solve_predator_prey(
        predator_prey_system,
        params['initial_conditions'],
        params['time_interval'],
        (params['r1'], params['lambda1'], params['lambda2'], params['beta2'], 0.0005)
    )
    visualize_predator_prey_solution(
        params['time_interval'],
        solution_with_competition,
        "Модель Хищник-Жертва (с внутривидовой конкуренцией)"
    )

    solution_without_competition = solve_predator_prey(
        predator_prey_system,
        params['initial_conditions'],
        params['time_interval'],
        (params['r1'], params['lambda1'], params['lambda2'], params['beta2'], 0)  # g1=0
    )
    visualize_predator_prey_solution(
        params['time_interval'],
        solution_without_competition,
        "Модель Хищник-Жертва (без внутривидовой конкуренции)"
    )

if __name__ == "__main__":
    main()