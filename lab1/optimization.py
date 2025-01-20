from pulp import LpMaximize, LpProblem, LpVariable

def solve_optimization(data):
    model = LpProblem(name="maximize-production-cost", sense=LpMaximize)

    variables = [LpVariable(name=f"x{i+1}", lowBound=0, cat="Integer") for i in range(len(data["objective_coefficients"]))]
    model += sum(coeff * var for coeff, var in zip(data["objective_coefficients"], variables)), "Objective"

    for i, constraint in enumerate(data["constraints"]):
        model += sum(coeff * var for coeff, var in zip(constraint["coefficients"], variables)) <= constraint["rhs"], f"Constraint {i+1}"

    status = model.solve()

    results = {
        "status": model.status,
        "variables": {var.name: var.value() for var in variables},
        "objective": model.objective.value()
    }
    return results
