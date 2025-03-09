import pulp

problem = pulp.LpProblem("Нефтепереработка", pulp.LpMinimize)

x1 = pulp.LpVariable("x1", lowBound=0, cat="Continuous")
x2 = pulp.LpVariable("x2", lowBound=0, cat="Continuous")

problem += x1 + x2, "Общий расход нефти"

problem += 0.3 * x1 + 0.7 * x2 >= 110, "Темные нефтепродукты"
problem += 0.6 * x1 + 0.2 * x2 >= 70, "Светлые нефтепродукты"

print("allo python")

problem.solve()

print(f"Оптимальное значение x1: {x1.value()} тонн")
print(f"Оптимальное значение x2: {x2.value()} тонн")
print(f"Минимальный расход нефти: {x1.value() + x2.value()} тонн")