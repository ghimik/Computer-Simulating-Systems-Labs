from pulp import LpMaximize, LpProblem, LpVariable

model = LpProblem(name="maximize-production-cost", sense=LpMaximize)

x1 = LpVariable(name="x1", lowBound=0, cat="Integer")
x2 = LpVariable(name="x2", lowBound=0, cat="Integer")
x3 = LpVariable(name="x3", lowBound=0, cat="Integer")
x4 = LpVariable(name="x4", lowBound=0, cat="Integer")

model += 9 * x1 + 6 * x2 + 4 * x3 + 7 * x4, "Total cost"

model += x1 + 2 * x3 + 1 * x4 <= 180, "Constraint for fabric I"
model += x2 + 3 * x3 + 2 * x4 <= 210, "Constraint for fabric II"
model += 4 * x1 + 2 * x2 + 4 * x4 <= 800, "Constraint for fabric III"

status = model.solve()

print(f"Status: {model.status}")
print(f"Optimal production:")
print(f"x1 (item 1): {x1.value()}")
print(f"x2 (item 2): {x2.value()}")
print(f"x3 (item 3): {x3.value()}")
print(f"x4 (item 4): {x4.value()}")
print(f"Maximum cost: {model.objective.value()}")
