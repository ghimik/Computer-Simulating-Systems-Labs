import math
import numpy as np


def erlang_b_probability(rho, n):
    numerator = (rho ** n) / math.factorial(n)
    denominator = sum((rho ** k) / math.factorial(k) for k in range(n + 1))
    return numerator / denominator

def find_min_channels(lambda_, T, max_p_reject=0.1):
    rho = lambda_ * T
    n = 1
    while True:
        p_reject = erlang_b_probability(rho, n)
        if p_reject < max_p_reject:
            return n
        n += 1

def calculate_smo_characteristics(lambda_, T):
    rho = lambda_ * T
    P_reject = rho / (1 + rho)  
    A = lambda_ * (1 - P_reject)  # абс пропускная способность
    N = 1 - P_reject  # ср число занятых каналов
    return P_reject, A, N


lambda_range = np.linspace(0.7, 0.9, 3)  # интенсивность
T_range = np.linspace(2.3, 2.5, 3)      # время обслуживания среднее


for lambda_ in lambda_range:
    for T in T_range:
        P_reject, A, N = calculate_smo_characteristics(lambda_, T)
        print(f"λ = {lambda_:.2f}, T = {T:.2f}: "
              f"P_отк = {P_reject:.4f}, A = {A:.4f}, N = {N:.4f}")
        
        min_channels = find_min_channels(lambda_, T)
        print(f"Минимальное число каналов n для P_отк < 0.1: {min_channels}")
        if min_channels != -1:
            p_reject = erlang_b_probability(lambda_ * T, min_channels)
            print(f"Проверка: P_отк при n = {min_channels} составляет {p_reject:.4f}")
        else:
            print("Не удалось найти решение для n ≤ 100.")