from parser import parse_system
from solver import solve_ode
import numpy as np

def example1():
    params = {}
    system = parse_system(
        ["y'' + 2*y' + 3*y = cos(t)"],
        ['t', 'y'],
        params
    )
    from scipy.integrate import solve_ivp
    t_eval = np.linspace(0, 2*np.pi, 100)
    sol = solve_ivp(system, [0, 2*np.pi], [0, 0], t_eval=t_eval)
    return sol

def example2():
    params = {'a': 1.0}
    system = parse_system(
        ["z'' - a*(1 - z**2)*z' + z = 0"],
        ['t', 'z'],
        params
    )
    from scipy.integrate import solve_ivp
    t_eval = np.linspace(0, 30, 1000)
    sol = solve_ivp(system, [0, 30], [2, 0], t_eval=t_eval)
    return sol

# Пример: y' = y^2 - y*t
f1 = parse_system(["y' = y**2 - t*y"], ['t', 'y'], {})
# Пример: y' = y^2 + 1
f2 = parse_system(["y' = y**2 + 1"], ['t', 'y'], {})

# Пример системы:
# y1' = -y2 - y3
# y2' = y1 + a*y2
# y3' = b + y3*(y1 - c)
params = {'a': 1.0, 'b': 0.5, 'c': 2.0}
system = parse_system(
    ["y1' = -y2 - y3", "y2' = y1 + a*y2", "y3' = b + y3*(y1 - c)"],
    ['t', 'y1', 'y2', 'y3'],
    params
)
from scipy.integrate import solve_ivp
t_eval = np.linspace(0, 10, 200)
sol = solve_ivp(system, [0, 10], [1, 0, 0], t_eval=t_eval)
print(sol.t)
print(sol.y)

# Пример системы:
# x' = a*x - b*x*y
# y' = c*x*y - d*y
params = {'a': 1.0, 'b': 0.1, 'c': 0.075, 'd': 1.5}
system = parse_system(
    ["x' = a*x - b*x*y", "y' = c*x*y - d*y"],
    ['t', 'x', 'y'],
    params
)
from scipy.integrate import solve_ivp
t_eval = np.linspace(0, 50, 500)
sol = solve_ivp(system, [0, 50], [10, 5], t_eval=t_eval)
print(sol.t)
print(sol.y)



sol1 = example1()
print(sol1.t)
print(sol1.y)

sol2 = example2()
print(sol2.t)
print(sol2.y)


