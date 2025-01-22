import json
import csv
from tkinter import Tk, Button, Text, Toplevel
from pulp import LpMaximize, LpProblem, LpVariable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

with open("model_data.json", "r") as file:
    model_data = json.load(file)

dates, prices = [], []
with open("prices.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # скип заголовок
    for row in reader:
        dates.append(row[0])
        prices.append(float(row[1]))

prices = np.array(prices)
x = np.arange(len(dates))


# Первая задача
def solve_first_task():
    model = LpProblem(name="maximize-production-cost", sense=LpMaximize)
    variables = {name: LpVariable(name=name, lowBound=0, cat="Integer") for name in model_data["variables"]}
    model += sum(coeff * variables[name] for name, coeff in model_data["objective"].items()), "Total cost"
    for name, constraint in model_data["constraints"].items():
        model += sum(coeff * variables[var] for var, coeff in constraint["variables"].items()) <= constraint["bound"], name
    model.solve()
    result = f"Status: {model.status}\nOptimal production:\n"
    result += "\n".join([f"{name}: {var.value()}" for name, var in variables.items()])
    result += f"\nMaximum cost: {model.objective.value()}"
    return result


# Вторая задача: Построение графиков
def plot_second_task():

    def polynomial_model(x, degree):
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x.reshape(-1, 1))
        model = LinearRegression().fit(x_poly, prices)
        return model.predict(x_poly), model

    def power_law(x, a, b):
        return a * x ** b

    def log_model(x, a, b):
        return a * np.log(x) + b

    # Линейная модель
    linear_model = LinearRegression().fit(x.reshape(-1, 1), prices)
    linear_pred = linear_model.predict(x.reshape(-1, 1))

    # Полиномиальная модель
    degree = 6
    poly_pred, poly_model = polynomial_model(x, degree)

    # Степенная модель
    params_power, _ = curve_fit(power_law, x + 1, prices, maxfev=10000)
    power_pred = power_law(x + 1, *params_power)

    # Логарифмическая модель
    params_log, _ = curve_fit(log_model, x + 1, prices)
    log_pred = log_model(x + 1, *params_log)

    r2_linear = r2_score(prices, linear_pred)
    r2_poly = r2_score(prices, poly_pred)
    r2_power = r2_score(prices, power_pred)
    r2_log = r2_score(prices, log_pred)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Линейная модель
    axs[0, 0].scatter(x, prices, color='blue', label='Data')
    axs[0, 0].plot(x, linear_pred, color='red', label=f'Linear fit: R^2={r2_linear:.4f}')
    axs[0, 0].set_title("Linear Fit")
    axs[0, 0].legend()

    # Полиномиальная модель
    axs[0, 1].scatter(x, prices, color='blue', label='Data')
    axs[0, 1].plot(x, poly_pred, color='green', label=f'Poly fit (degree={degree}): R^2={r2_poly:.4f}')
    axs[0, 1].set_title(f"Polynomial Fit (degree {degree})")
    axs[0, 1].legend()

    # Степенная модель
    axs[1, 0].scatter(x, prices, color='blue', label='Data')
    axs[1, 0].plot(x, power_pred, color='purple', label=f'Power fit: R^2={r2_power:.4f}')
    axs[1, 0].set_title("Power Fit")
    axs[1, 0].legend()

    # Логарифмическая модель
    axs[1, 1].scatter(x, prices, color='blue', label='Data')
    axs[1, 1].plot(x, log_pred, color='orange', label=f'Log fit: R^2={r2_log:.4f}')
    axs[1, 1].set_title("Logarithmic Fit")
    axs[1, 1].legend()

    axs[0, 0].text(0.05, 0.9, f"y = {linear_model.coef_[0]:.2f}x + {linear_model.intercept_:.2f}",
                   transform=axs[0, 0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    def format_polynomial(coefficients):
        terms = []
        for i, coef in enumerate(coefficients):
            if abs(coef) > 1e-6:
                if i == 0:
                    terms.append(f"{coef:.2f}")
                elif i == 1:
                    terms.append(f"{coef:.2f}x")
                else:
                    terms.append(f"{coef:.2f}x^{i}")
        return " + ".join(terms).replace("+ -", "- ")

    poly_coefficients = poly_model.coef_

    poly_equation = format_polynomial(poly_coefficients)

    axs[0, 1].text(0.05, 0.8, f"y = {poly_equation}",
                   transform=axs[0, 1].transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    axs[1, 0].text(0.05, 0.9, f"y = {params_power[0]:.2f}x^{params_power[1]:.2f}",
                   transform=axs[1, 0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    axs[1, 1].text(0.05, 0.9, f"y = {params_log[0]:.2f}ln(x) + {params_log[1]:.2f}",
                   transform=axs[1, 1].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()


def show_first_task_result():
    result = solve_first_task()
    window = Toplevel(root)
    window.title("Task 1 Result")
    text = Text(window, wrap="word")
    text.insert("1.0", result)
    text.pack(expand=True, fill="both")


def show_second_task_plot():
    plot_second_task()


root = Tk()
root.title("Task Manager")

btn_task1 = Button(root, text="Solve Task 1", command=show_first_task_result)
btn_task1.pack(pady=10)

btn_task2 = Button(root, text="Show Task 2 Plot", command=show_second_task_plot)
btn_task2.pack(pady=10)

root.mainloop()
