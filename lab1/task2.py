import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

dates = [
    '15.03.2023', '14.03.2023', '13.03.2023', '10.03.2023', '09.03.2023', '08.03.2023',
    '07.03.2023', '06.03.2023', '03.03.2023', '02.03.2023', '01.03.2023', '28.02.2023',
    '27.02.2023', '24.02.2023', '23.02.2023', '22.02.2023', '21.02.2023', '20.02.2023',
    '19.02.2023', '17.02.2023', '16.02.2023', '15.02.2023'
]
prices = [
    1441.28, 1517.70, 1476.70, 1362.30, 1374.70, 1362.10,
    1370.60, 1424.50, 1449.00, 1444.90, 1437.60, 1420.90,
    1427.90, 1387.70, 1430.70, 1488.50, 1522.53, 1508.03,
    1481.03, 1492.50, 1525.70, 1444.50
]

x = np.arange(len(dates))

def polynomial_model(x, degree):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(x_poly, prices)
    return model.predict(x_poly), model

def power_law(x, a, b):
    return a * x**b

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
