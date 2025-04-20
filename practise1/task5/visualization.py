import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from parser import parse_system
from utils import compute_orders

def plot_solution(equations, vars, params, t_span, y0, t_eval=None, title="Решение", xlabel="t", ylabel="y"):
    system = parse_system(equations, vars, params)
    
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)
        
    sol = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45')
    
    dep_vars = vars[1:]
    orders = compute_orders(equations, dep_vars)
    
    # Графики временных рядов
    plt.figure(figsize=(12, 6))
    cumulative_idx = 0
    for var in dep_vars:
        order = orders.get(var, 1)
        for i in range(order):
            label = f"{var}" + "'" * i
            plt.plot(sol.t, sol.y[cumulative_idx + i], label=label)
        cumulative_idx += order
    
    plt.title(f"{title}\nВременные зависимости", fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Фазовые портреты
    cumulative_idx = 0
    for var in dep_vars:
        order = orders.get(var, 1)
        if order >= 2:
            plt.figure(figsize=(8, 6))
            x = sol.y[cumulative_idx]
            dx = sol.y[cumulative_idx + 1]
            plt.plot(x, dx, color='purple')
            plt.title(f"{title}\nФазовый портрет: {var}' vs {var}", fontsize=14)
            plt.xlabel(var, fontsize=12)
            plt.ylabel(f"{var}'", fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        cumulative_idx += order

    # Дополнительные фазовые портреты между переменными
    if len(dep_vars) >= 2:
        plt.figure(figsize=(8, 6))
        idx = 0
        for i, var1 in enumerate(dep_vars):
            for j, var2 in enumerate(dep_vars):
                if i < j:
                    x = sol.y[idx]
                    y = sol.y[idx + orders[var1]]
                    plt.plot(x, y, label=f"{var1} vs {var2}")
        plt.title(f"{title}\nФазовый портрет системы", fontsize=14)
        plt.xlabel(dep_vars[0], fontsize=12)
        plt.ylabel(dep_vars[1], fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()