import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np


class RealisticViscousFallSimulator:
    def __init__(self, radius: float, object_density: float, medium_density: float, 
                 g: float = 9.81, t_max: float = 20, dt: float = 0.1) -> None:
        """
        Инициализация модели с учетом плотности среды.
        
        :param radius: радиус шара (м)
        :param object_density: плотность материала шара (кг/м³)
        :param medium_density: плотность среды (кг/м³)
        :param g: ускорение свободного падения (м/с²)
        :param t_max: максимальное время моделирования (с)
        :param dt: шаг времени (с)
        """
        if radius < 0 or object_density < 0 or medium_density < 0 or g < 0 or t_max < 0 or dt <= 0:
            raise ValueError("Все параметры должны быть неотрицательными, а dt должен быть положительным.")
        
        self.r = radius
        self.rho_obj = object_density
        self.rho_med = medium_density
        self.g = g
        self.t_max = t_max
        self.dt = dt
        
        self.time = None
        self.velocity = None
        self.height = None
        
        self._precalculate()

    def _precalculate(self) -> None:
        """Предварительные расчеты."""
        self.volume = (4 / 3) * np.pi * self.r**3
        self.mass = (self.rho_obj - self.rho_med) * self.volume 
        self.time = np.arange(0, self.t_max + self.dt, self.dt)

    def solve(self, viscosity: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Решение задачи падения с учетом вязкости.
        
        :param viscosity: вязкость среды (Па·с)
        :return: массивы времени, скорости и высоты
        """
        if viscosity < 0:
            raise ValueError("Вязкость среды должна быть положительной.")
        
        k = 6 * np.pi * viscosity * self.r
        print(f"Коэффициент сопротивления (k): {k}")  # трейс k
        
        v_terminal = (self.mass * self.g) / k
        print(f"Терминальная скорость (v_terminal): {v_terminal}")  # трейс v_terminal
        
        t = self.time
        
        self.velocity = v_terminal * (1 - np.exp(-k * t / self.mass))
        print(f"Скорость (velocity): {self.velocity}")  # трейс velocity
        
        self.height = v_terminal * t - (self.mass * v_terminal / k) * (1 - np.exp(-k * t / self.mass))
        print(f"Высота (height): {self.height}")  # трейс height
        
        return self.time, self.velocity, self.height

    def get_terminal_velocity(self, viscosity: float) -> float:
        """
        Возвращает терминальную скорость.
        
        :param viscosity: вязкость среды (Па·с)
        :return: терминальная скорость (м/с)
        """
        k = 6 * np.pi * viscosity * self.r
        return (self.mass * self.g) / k


if __name__ == '__main__':

    params = {
        'radius': 0.015,       # 1.5 см 
        'object_density': 7300, # олово
        'medium_density': 900,  # мазут
        't_max': 10,            
        'dt': 0.1
    }

    simulator = RealisticViscousFallSimulator(**params)
    viscosities = [1, 10, 100] 

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for eta in viscosities:
        time, velocity, height = simulator.solve(eta)
        
        ax1.plot(time, velocity, label=f'η = {eta} Па·с')
        ax2.plot(time, height, label=f'η = {eta} Па·с')

    ax1.set_ylim(0, 5)
    ax2.set_ylim(0, 30)

    ax1.set_title('Сравнение скоростей для разных вязкостей')
    ax1.set_ylabel('Скорость (м/с)')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Сравнение высот для разных вязкостей')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Высота (м)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()