import unittest
import simpy
import math
from generator import LemerGenerator
from task1_simpy import *

class TestSimulationFunctions(unittest.TestCase):
    def setUp(self):
        self.seed = 1234
        self.generator = LemerGenerator(self.seed)
        self.lambda_value = 0.5
        self.service_time = 1.0
        self.max_time = 10000

    def test_exponential(self):
        """Тест генерации экспоненциального распределения."""
        rate = 0.5
        value = exponential(rate, self.generator)
        self.assertGreaterEqual(value, 0)

    def test_exponential_with_invalid_rate(self):
        """Тест генерации экспоненциального распределения с недопустимым значением rate."""
        with self.assertRaises(ValueError):
            exponential(-1, self.generator)

    def test_exponential_with_invalid_random(self):
        """Тест генерации экспоненциального распределения с недопустимым случайным числом."""
        with self.assertRaises(ValueError):
            class BrokenGenerator:
                def next(self):
                    return 1.1  
            broken_gen = BrokenGenerator()
            exponential(0.5, broken_gen)

    def test_customer_arrival(self):
        """Тест процесса прибытия клиентов."""
        env = simpy.Environment()
        server = simpy.Resource(env, capacity=1)
        stats = {'served': 0, 'rejected': 0}
        env.process(customer_arrival(env, server, self.lambda_value, self.service_time, self.generator, stats))
        env.run(until=1) 
        self.assertGreaterEqual(stats['served'] + stats['rejected'], 0)

    def test_customer_service(self):
        """Тест процесса обслуживания клиентов."""
        env = simpy.Environment()
        server = simpy.Resource(env, capacity=1)
        env.process(customer_service(env, server, self.service_time))
        env.run(until=1)  

    def test_customer_service_with_invalid_service_time(self):
        """Тест процесса обслуживания с недопустимым временем обслуживания."""
        env = simpy.Environment()
        server = simpy.Resource(env, capacity=1)
        with self.assertRaises(ValueError):
            env.process(customer_service(env, server, -1))
            env.run(until=1)

    def test_simulate(self):
        """Тест симуляции."""
        stats = simulate(self.lambda_value, self.service_time, self.max_time, self.generator)
        self.assertGreaterEqual(stats['served'], 0)
        self.assertGreaterEqual(stats['rejected'], 0)

    def test_simulate_with_invalid_parameters(self):
        """Тест симуляции с недопустимыми параметрами."""
        with self.assertRaises(ValueError):
            simulate(-1, self.service_time, self.max_time, self.generator)
        with self.assertRaises(ValueError):
            simulate(self.lambda_value, -1, self.max_time, self.generator)
        with self.assertRaises(ValueError):
            simulate(self.lambda_value, self.service_time, -1, self.generator)

    def test_theoretical_statistics(self):
        """Тест теоретической статистики."""
        P_rejection, P_service, theoretical_ratio = theoretical_statistics(self.lambda_value, self.service_time)
        self.assertGreaterEqual(P_rejection, 0)
        self.assertLessEqual(P_rejection, 1)
        self.assertGreaterEqual(P_service, 0)
        self.assertLessEqual(P_service, 1)
        self.assertGreaterEqual(theoretical_ratio, 0)

    def test_theoretical_statistics_with_invalid_service_time(self):
        """Тест теоретической статистики с недопустимым временем обслуживания."""
        with self.assertRaises(ValueError):
            theoretical_statistics(self.lambda_value, -1)

    def test_theoretical_and_empirical_statistics_are_close(self):
        """Тест, что теоретическая и эмпирическая статистика близки."""
        stats = simulate(self.lambda_value, self.service_time, self.max_time, self.generator)
        total_clients = stats['served'] + stats['rejected']
        prob_rejection = stats['rejected'] / total_clients if total_clients > 0 else 0
        prob_service = stats['served'] / total_clients if total_clients > 0 else 0
        served_to_rejected_ratio = stats['served'] / stats['rejected'] if stats['rejected'] > 0 else float('inf')

        P_rejection, P_service, theoretical_ratio = theoretical_statistics(self.lambda_value, self.service_time)

        self.assertAlmostEqual(prob_rejection, P_rejection, places=1,
                               msg="Эмпирическая и теоретическая вероятность отказа отличаются более чем на 0.1")
        self.assertAlmostEqual(prob_service, P_service, places=1,
                               msg="Эмпирическая и теоретическая вероятность обслуживания отличаются более чем на 0.1")
        self.assertAlmostEqual(served_to_rejected_ratio, theoretical_ratio, places=1,
                               msg="Эмпирическое и теоретическое отношение обслуженных к отказанным отличаются более чем на 0.1")

if __name__ == "__main__":
    unittest.main()