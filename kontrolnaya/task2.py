import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# данные из таблицы
Pb_water = np.array([0.01, 0.011, 0.012, 0.012, 0.011, 0.014, 0.015, 0.015, 0.015, 0.013, 0.012,
                     0.012, 0.013, 0.015, 0.021, 0.017, 0.013, 0.013, 0.012, 0.016, 0.012,
                     0.016, 0.018, 0.013, 0.013, 0.012, 0.011, 0.02, 0.01]).reshape(-1, 1)
Pb_hair = np.array([0.9, 0.656, 5.17, 3.283, 2.846, 3.48, 3.77, 2.13, 0.185, 0.583, 1.917,
                     0.122, 7.22, 3.8, 2.33, 15, 11.3, 6, 5.33, 4.2, 1.74, 8.3, 0.89,
                     4.55, 0.05, 6.67, 3.22, 12.3, 0.8])

# линейная
lin_reg = LinearRegression()
lin_reg.fit(Pb_water, Pb_hair)
Pb_hair_pred_lin = lin_reg.predict(Pb_water)
lin_r2 = r2_score(Pb_hair, Pb_hair_pred_lin)
lin_eq = f"y = {lin_reg.coef_[0]:.4f} * x + {lin_reg.intercept_:.4f}"

# полиномиальная 2
poly = PolynomialFeatures(degree=2)
Pb_water_poly = poly.fit_transform(Pb_water)
poly_reg = LinearRegression()
poly_reg.fit(Pb_water_poly, Pb_hair)
Pb_hair_pred_poly = poly_reg.predict(Pb_water_poly)
poly_r2 = r2_score(Pb_hair, Pb_hair_pred_poly)
poly_eq = f"y = {poly_reg.coef_[2]:.4f} * x² + {poly_reg.coef_[1]:.4f} * x + {poly_reg.intercept_:.4f}"

# степеннная модель
log_Pb_water = np.log(Pb_water)
power_reg = LinearRegression()
power_reg.fit(log_Pb_water, np.log(Pb_hair))
Pb_hair_pred_power = np.exp(power_reg.predict(log_Pb_water))
power_r2 = r2_score(Pb_hair, Pb_hair_pred_power)
power_eq = f"y = {np.exp(power_reg.intercept_):.4f} * x^{power_reg.coef_[0]:.4f}"

# лог модель
log_reg = LinearRegression()
log_reg.fit(log_Pb_water, Pb_hair)
Pb_hair_pred_log = log_reg.predict(log_Pb_water)
log_r2 = r2_score(Pb_hair, Pb_hair_pred_log)
log_eq = f"y = {log_reg.intercept_:.4f} + {log_reg.coef_[0]:.4f} * ln(x)"

r2_scores = {"Линейная": lin_r2, "Полиномиальная": poly_r2, "Степенная": power_r2, "Логарифмическая": log_r2}
best_model = max(r2_scores, key=r2_scores.get)

plt.scatter(Pb_water, Pb_hair, color='blue', label='Данные')
plt.plot(Pb_water, Pb_hair_pred_lin, color='red', label=f'Линейная (R²={lin_r2:.2f})')
plt.plot(Pb_water, Pb_hair_pred_poly, color='green', linestyle='dashed', label=f'Полиномиальная (R²={poly_r2:.2f})')
plt.plot(Pb_water, Pb_hair_pred_power, color='purple', linestyle='dotted', label=f'Степенная (R²={power_r2:.2f})')
plt.plot(Pb_water, Pb_hair_pred_log, color='orange', linestyle='dashdot', label=f'Логарифмическая (R²={log_r2:.2f})')
plt.xlabel('Содержание Pb в воде')
plt.ylabel('Содержание Pb в волосах')
plt.legend()
plt.show()

print(f'Линейная модель: {lin_eq}, R² = {lin_r2:.2f}')
print(f'Полиномиальная модель: {poly_eq}, R² = {poly_r2:.2f}')
print(f'Степенная модель: {power_eq}, R² = {power_r2:.2f}')
print(f'Логарифмическая модель: {log_eq}, R² = {log_r2:.2f}')
print(f'Лучшая модель: {best_model}')
