import unittest
import random
import math
from generator import LemerGenerator
from task1_pure_python import SingleServerWithBlocking

class TestSingleServerWithBlocking(unittest.TestCase):
    def setUp(self):
        self.seed = 1234
        self.generator = LemerGenerator(self.seed)
        self.lambda_value = 0.5
        self.service_time = 1.0
        self.smo = SingleServerWithBlocking(self.generator, self.lambda_value, self.service_time)

    def test_initialization(self):
        """Тест инициализации класса."""
        self.assertEqual(self.smo.lambda_value, self.lambda_value)
        self.assertEqual(self.smo.service_time, self.service_time)
        self.assertEqual(self.smo.time, 0)
        self.assertEqual(self.smo.served, 0)
        self.assertEqual(self.smo.rejected, 0)

    def test_initialization_with_invalid_lambda(self):
        """Тест инициализации с недопустимым значением lambda_value."""
        with self.assertRaises(ValueError):
            SingleServerWithBlocking(self.generator, -1, self.service_time)

    def test_initialization_with_invalid_service_time(self):
        """Тест инициализации с недопустимым значением service_time."""
        with self.assertRaises(ValueError):
            SingleServerWithBlocking(self.generator, self.lambda_value, -1)

    def test_exponential_generation(self):
        """Тест генерации экспоненциального распределения."""
        rate = 0.5
        value = self.smo.exponential(rate)
        self.assertGreaterEqual(value, 0)

    def test_exponential_generation_with_invalid_rate(self):
        """Тест генерации экспоненциального распределения с недопустимым значением rate."""
        with self.assertRaises(ValueError):
            self.smo.exponential(-1)

    def test_simulate_with_invalid_max_time(self):
        """Тест симуляции с недопустимым значением max_time."""
        with self.assertRaises(ValueError):
            self.smo.simulate(-1)

    def test_simulate(self):
        """Тест симуляции."""
        max_time = 1000
        self.smo.simulate(max_time)
        self.assertGreaterEqual(self.smo.time, 0)
        self.assertGreaterEqual(self.smo.served, 0)
        self.assertGreaterEqual(self.smo.rejected, 0)

    def test_get_statistics(self):
        """Тест получения статистики."""
        max_time = 1000
        self.smo.simulate(max_time)
        prob_rejection, prob_service, served_to_rejected_ratio = self.smo.get_statistics()
        self.assertGreaterEqual(prob_rejection, 0)
        self.assertLessEqual(prob_rejection, 1)
        self.assertGreaterEqual(prob_service, 0)
        self.assertLessEqual(prob_service, 1)
        self.assertGreaterEqual(served_to_rejected_ratio, 0)

    def test_theoretical_statistics(self):
        """Тест теоретической статистики."""
        P_rejection, P_service, theoretical_ratio = self.smo.theoretical_statistics()
        self.assertGreaterEqual(P_rejection, 0)
        self.assertLessEqual(P_rejection, 1)
        self.assertGreaterEqual(P_service, 0)
        self.assertLessEqual(P_service, 1)
        self.assertGreaterEqual(theoretical_ratio, 0)

    def test_theoretical_statistics_with_invalid_rho(self):
        """Тест теоретической статистики с недопустимым значением коэффициента загрузки."""
        self.smo.lambda_value = 0
        with self.assertRaises(ValueError):
            self.smo.theoretical_statistics()

    def test_theoretical_and_empirical_statistics_are_close(self):
        """Тест, что теоретическая и эмпирическая статистика близки."""
        max_time = 10000  
        self.smo.simulate(max_time)

        empirical_rejection, empirical_service, empirical_ratio = self.smo.get_statistics()

        theoretical_rejection, theoretical_service, theoretical_ratio = self.smo.theoretical_statistics()

        self.assertAlmostEqual(empirical_rejection, theoretical_rejection, places=2,
                               msg="Эмпирическая и теоретическая вероятность отказа отличаются более чем на 0.01")
        self.assertAlmostEqual(empirical_service, theoretical_service, places=2,
                               msg="Эмпирическая и теоретическая вероятность обслуживания отличаются более чем на 0.01")
        self.assertAlmostEqual(empirical_ratio, theoretical_ratio, places=1,
                               msg="Эмпирическое и теоретическое отношение обслуженных к отказанным отличаются более чем на 0.01")

if __name__ == "__main__":
    unittest.main()