import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


from parser import parse_system
from utils import compute_orders


def plot_solution(equations, vars, params, t_span, y0, t_eval=None, title="Solution", xlabel="t", ylabel="y"):
    """
    Решает систему (или одиночное) ОДУ, заданную списком уравнений в виде строк, и строит график решения.
    
    Параметры:
      - equations: список уравнений (например, ["y' = y**2 - t*y"] или
                   ["y1' = -y2 - y3", "y2' = y1 + a*y2", "y3' = b + y3*(y1 - c)"]).
      - vars: список переменных, где первый элемент — независимая переменная (например, 't'),
              а остальные — зависимые переменные.
      - params: словарь параметров (например, {'a': 1.0}).
      - t_span: кортеж (t0, tf) интервала интегрирования.
      - y0: начальные условия для всех состояний. Для переменной, задаваемой дифференциальным уравнением
            более высокого порядка, начальные условия должны быть заданы для всех её составляющих.
      - t_eval: (опционально) массив значений времени, в которых требуется вычислить решение.
      - title, xlabel, ylabel: название графика и подписи осей.
    
    Функция строит график: если задана одна зависимая переменная, выводится один график;
    если несколько – каждый график подписывается именем переменной (для переменных более высокого порядка
    отрисовывается только первое состояние, соответствующее значению переменной).
    """
    # Получаем функцию системы для solve_ivp
    system = parse_system(equations, vars, params)
    
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)
        
    sol = solve_ivp(system, t_span, y0, t_eval=t_eval)
    
    # Вычисляем порядок для каждой зависимой переменной
    dep_vars = vars[1:]
    orders = compute_orders(equations, dep_vars)
    
    plt.figure(figsize=(8, 6))
    
    # Если имеется одна зависимая переменная, её первое состояние – это и есть решение.
    if len(dep_vars) == 1:
        plt.plot(sol.t, sol.y[0, :], label=dep_vars[0])
    else:
        # Для систем: каждый зависимый переменный занимает 'order' позиций в векторе состояния.
        cumulative_index = 0
        for var in dep_vars:
            order = orders.get(var, 1)
            # Первое состояние для переменной (индекс cumulative_index) соответствует самой переменной.
            plt.plot(sol.t, sol.y[cumulative_index, :], label=var)
            cumulative_index += order
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
