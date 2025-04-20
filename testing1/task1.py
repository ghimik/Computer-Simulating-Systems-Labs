from pulp import *

prob = LpProblem("ЧулочноНосочнаяФирма", LpMaximize)

x = LpVariable("x", lowBound=0, cat='Continuous')  #  пар чулок
y = LpVariable("y", lowBound=0, cat='Continuous')  #  пар носков

prob += 10*x + 4*y, "Прибыль"

prob += 0.02*x + 0.01*y <= 60, "Участок1"
prob += 0.03*x + 0.01*y <= 70, "Участок2"
prob += 0.03*x + 0.02*y <= 100, "Участок3"

prob.solve()

print("Оптимальное количество пар чулок (x):", value(x))
print("Оптимальное количество пар носков (y):", value(y))
print("Максимальная прибыль (руб.):", value(prob.objective))