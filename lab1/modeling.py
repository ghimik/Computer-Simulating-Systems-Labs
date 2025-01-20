import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

def analyze_data(dates, prices):
    x = np.arange(len(dates))
    results = {"dates": dates, "prices": prices}

    # Polynomial regression
    def polynomial_model(x, degree):
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x.reshape(-1, 1))
        model = LinearRegression().fit(x_poly, prices)
        return model.predict(x_poly), model

    degree = 6
    poly_pred, poly_model = polynomial_model(x, degree)
    r2_poly = r2_score(prices, poly_pred)
    results["polynomial"] = {"degree": degree, "r2": r2_poly, "predictions": poly_pred}

    # Linear regression
    lin_model = LinearRegression().fit(x.reshape(-1, 1), prices)
    lin_pred = lin_model.predict(x.reshape(-1, 1))
    r2_lin = r2_score(prices, lin_pred)
    results["linear"] = {"r2": r2_lin, "predictions": lin_pred}

    # Exponential fit
    def exponential_model(x, a, b):
        return a * np.exp(b * x)

    try:
        popt, _ = curve_fit(exponential_model, x, prices, maxfev=10000)
        exp_pred = exponential_model(x, *popt)
        r2_exp = r2_score(prices, exp_pred)
        results["exponential"] = {"r2": r2_exp, "predictions": exp_pred}
    except Exception as e:
        results["exponential"] = {"r2": None, "predictions": None, "error": str(e)}

    return results

def plot_graphs(results):
    dates = results["dates"]
    prices = results["prices"]
    x = np.arange(len(dates))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Modeling Results", fontsize=16)

    # 1. Фактические данные
    axs[0, 0].plot(x, prices, label="Actual Prices", color="blue", marker="o")
    axs[0, 0].set_title("Actual Prices")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Prices")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # 2. Полиномиальная регрессия
    axs[0, 1].plot(x, prices, label="Actual Prices", color="blue", alpha=0.5, marker="o")
    axs[0, 1].plot(x, results["polynomial"]["predictions"], label=f"Polynomial (Degree {results['polynomial']['degree']})", color="red")
    axs[0, 1].set_title(f"Polynomial Regression (Degree {results['polynomial']['degree']})")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Prices")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # 3. Линейная регрессия
    axs[1, 0].plot(x, prices, label="Actual Prices", color="blue", alpha=0.5, marker="o")
    axs[1, 0].plot(x, results["linear"]["predictions"], label="Linear Regression", color="green")
    axs[1, 0].set_title("Linear Regression")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Prices")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # 4. Экспоненциальная аппроксимация (если успешна)
    axs[1, 1].plot(x, prices, label="Actual Prices", color="blue", alpha=0.5, marker="o")
    if results["exponential"]["predictions"] is not None:
        axs[1, 1].plot(x, results["exponential"]["predictions"], label="Exponential Fit", color="purple")
        axs[1, 1].set_title("Exponential Fit")
    else:
        axs[1, 1].set_title("Exponential Fit (Failed)")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Prices")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
