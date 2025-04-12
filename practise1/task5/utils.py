import re
from scipy.integrate import solve_ivp

def compute_orders(equations, dep_vars):
    """
    Вычисляет для каждого зависимого переменного максимальный порядок производной.
    Если в строках уравнений не встречается производных, считается порядок 1.
    """
    orders = {var: 1 for var in dep_vars}
    derivative_pattern = re.compile(r"([a-zA-Z]\w*)(\'+)")
    for eq in equations:
        for match in derivative_pattern.finditer(eq):
            var_name = match.group(1)
            if var_name in orders:
                d_order = len(match.group(2))
                orders[var_name] = max(orders[var_name], d_order)
    return orders