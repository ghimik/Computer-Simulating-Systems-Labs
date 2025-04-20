import numpy as np
import sympy
from sympy import symbols, sympify, solve, lambdify
import re

from utils import compute_orders

def parse_system(equations, vars, params):
    """
    Парсинг системы дифференциальных уравнений.
    
    Принимает:
      - equations: список уравнений в виде строк, например, ["y'' + 2*y' + 3*y = cos(t)"].
      - vars: список переменных, где первый элемент — независимая переменная (например, 't'),
              а остальные — зависимые переменные.
      - params: словарь параметров, например, {'a': 1.0}.
    
    Возвращает функцию f(t, Y) для передачи в scipy.integrate.solve_ivp.
    """
    t_sym = symbols(vars[0])
    dep_vars = vars[1:]
    
    orders = compute_orders(equations, dep_vars)
    
    state_syms = {}
    for var in dep_vars:
        m = orders[var]
        state_syms[var] = [symbols(f"{var}_{i}") for i in range(m)]
    

    replacement_dict = {}
    for var in dep_vars:
        m = orders[var]
        replacement_dict[var] = str(state_syms[var][0])
        for i in range(1, m):
            replacement_dict[var + "'" * i] = str(state_syms[var][i])
        f_sym = symbols(f"f_{var}")
        replacement_dict[var + "'" * m] = str(f_sym)
    
    # сортируем ключи по убыванию длины, чтобы сначала заменять более длинные вхождения.
    rep_keys_sorted = sorted(replacement_dict.keys(), key=len, reverse=True)
    
    eqs = []
    for eq_str in equations:
        if '=' in eq_str:
            lhs, rhs = eq_str.split('=', maxsplit=1)
        else:
            lhs, rhs = eq_str, "0"
        for key in rep_keys_sorted:
            pattern = r'(?<![\w])' + re.escape(key) + r'(?![\w])'
            replacement = replacement_dict[key]
            lhs = re.sub(pattern, replacement, lhs)
            rhs = re.sub(pattern, replacement, rhs)
        try:
            expr = sympify(lhs) - sympify(rhs)
        except Exception as e:
            raise ValueError(f"Ошибка при парсинге уравнения '{eq_str}': {e}")
        eqs.append(expr)
    
    # для каждой зависимой переменной решаем уравнение относительно f_{var}.
    solved_eqs = {}
    for var in dep_vars:
        m = orders[var]
        f_sym = symbols(f"f_{var}")
        for eq in eqs:
            if f_sym in eq.free_symbols:
                sol = solve(eq, f_sym)
                if not sol:
                    raise ValueError(f"Не удалось решить уравнение относительно производной для переменной {var}.")
                solved_eqs[var] = sol[0]
                break
        # если уравнение с f_sym не найдено и это уравнение первого порядка,
        # предполагаем, что уравнение задано в виде y' = expr.
        if var not in solved_eqs and orders[var] == 1:
            eq = eqs[0] 
            sol = solve(eq, f_sym)
            if not sol:
                raise ValueError(f"Не удалось решить уравнение относительно производной для переменной {var}.")
            solved_eqs[var] = sol[0]
    
    # формируем правую часть системы ОДУ.
    # для переменной с порядком m: первые m-1 уравнений — это переходы состояния,
    # а последнее уравнение вычисляется как f_{var} (выражение, найденное из исходного уравнения)
    system_rhs = []
    state_order = []  
    for var in dep_vars:
        m = orders[var]
        state_order.extend(state_syms[var])
        for i in range(m - 1):
            system_rhs.append(state_syms[var][i + 1])
        if var in solved_eqs:
            system_rhs.append(solved_eqs[var])
        else:
            system_rhs.append(0)
    
    param_symbols = symbols(list(params.keys()))
    
    # формируем список символов для lambdify: сначала t, затем все состояния, затем параметры.
    all_symbols = [t_sym] + state_order + list(param_symbols)
    f_func = lambdify(all_symbols, system_rhs, modules="numpy")
    
    def f(t, Y):
        args = [t] + list(Y) + list(params.values())
        return np.array(f_func(*args), dtype=float).flatten()
    
    return f
