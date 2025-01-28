import unittest
import numpy as np
from solver import RealisticViscousFallSimulator 


class TestRealisticViscousFallSimulator(unittest.TestCase):
    def setUp(self):
        """
        Устанавливаем параметры для тестов.
        """
        self.radius = 0.01  # радиус шара (м)
        self.object_density = 1000  # плотность материала шара (кг/м³)
        self.medium_density = 998  # плотность среды (кг/м³)
        self.viscosity = 0.001  # вязкость среды (Па·с)
        self.g = 9.81  # ускорение свободного падения (м/с²)
        self.t_max = 20  # максимальное время моделирования (с)
        self.dt = 0.1  # шаг времени (с)
        self.simulator = RealisticViscousFallSimulator(
            radius=self.radius,
            object_density=self.object_density,
            medium_density=self.medium_density,
            g=self.g,
            t_max=self.t_max,
            dt=self.dt
        )

    def test_terminal_velocity(self):
        """
        Тестируем терминальную скорость.
        """
        expected_v_terminal = (2 / 9) * ((self.object_density - self.medium_density) * self.g * self.radius**2) / self.viscosity
        calculated_v_terminal = self.simulator.get_terminal_velocity(self.viscosity)
        self.assertAlmostEqual(expected_v_terminal, calculated_v_terminal, places=5,
                               msg=f"Expected {expected_v_terminal}, got {calculated_v_terminal}")

    def test_velocity_convergence(self):
        """
        Проверяем, что скорость сходится к терминальной.
        """
        _, velocity, _ = self.simulator.solve(self.viscosity)
        v_terminal = self.simulator.get_terminal_velocity(self.viscosity)
        
        self.assertAlmostEqual(velocity[-1], v_terminal, delta=0.01,
                               msg=f"Expected terminal velocity {v_terminal}, but got {velocity[-1]}")

    def test_height_growth(self):
        """
        Проверяем, что высота растет линейно при достижении терминальной скорости.
        """
        _, _, height = self.simulator.solve(self.viscosity)
        v_terminal = self.simulator.get_terminal_velocity(self.viscosity)
        t_linear_start = int(0.8 * len(self.simulator.time))  # last 20% времени
        
        expected_heights = v_terminal * self.simulator.time[t_linear_start:]
        calculated_heights = height[t_linear_start:]
        
        np.testing.assert_allclose(expected_heights, calculated_heights, rtol=0.05,
                                   err_msg="Height does not grow linearly with terminal velocity.")

if __name__ == '__main__':
    unittest.main()
