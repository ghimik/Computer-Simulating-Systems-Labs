import unittest
from generator import LemerGenerator
import simpy
from task2_simpy import simulate, theoretical_statistics

class TestSimPyMultiServerQueue(unittest.TestCase):
    def setUp(self):
        self.lambda_value = 0.5
        self.service_time = 1.0
        self.num_servers = 3
        self.queue_capacity = 5
        self.max_time = 10000
        self.generator = LemerGenerator(42)

    def test_simulation_runs(self):
        """Проверка, что симуляция запускается и возвращает корректные результаты."""
        stats = simulate(self.lambda_value, self.service_time, self.num_servers, self.max_time, generator=self.generator, queue_capacity=self.queue_capacity)
        self.assertGreaterEqual(stats['served'], 0)
        self.assertGreaterEqual(stats['rejected'], 0)
        self.assertGreaterEqual(stats['avg_queue_length'], 0)

    def test_invalid_lambda(self):
        """Тестирование обработки недопустимого значения lambda."""
        with self.assertRaises(ValueError):
            simulate(-1, self.service_time, self.num_servers, self.max_time, generator=self.generator, queue_capacity=self.queue_capacity)

    def test_invalid_service_time(self):
        """Тестирование обработки недопустимого времени обслуживания."""
        with self.assertRaises(ValueError):
            simulate(self.lambda_value, -1, self.num_servers, self.max_time, generator=self.generator, queue_capacity=self.queue_capacity)

    def test_invalid_num_servers(self):
        """Тестирование обработки недопустимого количества серверов."""
        with self.assertRaises(ValueError):
            simulate(self.lambda_value, self.service_time, -1, self.max_time, generator=self.generator, queue_capacity=self.queue_capacity)

    def test_invalid_queue_capacity(self):
        """Тестирование обработки недопустимой вместимости очереди."""
        with self.assertRaises(ValueError):
            simulate(self.lambda_value, self.service_time, self.num_servers, self.max_time, generator=self.generator, queue_capacity=-1)

    def test_theoretical_statistics(self):
        """Тестирование расчета теоретических значений."""
        P_reject, P_service, Lq = theoretical_statistics(self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        self.assertGreaterEqual(P_reject, 0)
        self.assertLessEqual(P_reject, 1)
        self.assertGreaterEqual(P_service, 0)
        self.assertLessEqual(P_service, 1)
        self.assertGreaterEqual(Lq, 0)

    def test_empirical_vs_theoretical(self):
        """Проверка, что теоретические и эмпирические результаты близки."""
        stats = simulate(self.lambda_value, self.service_time, self.num_servers, self.max_time, generator=self.generator, queue_capacity=self.queue_capacity)
        P_reject, P_service, Lq = theoretical_statistics(self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)

        prob_rejection = stats['rejected'] / (stats['served'] + stats['rejected']) if (stats['served'] + stats['rejected']) > 0 else 0
        prob_service = stats['served'] / (stats['served'] + stats['rejected']) if (stats['served'] + stats['rejected']) > 0 else 0
        avg_queue_length = stats['avg_queue_length']

        self.assertAlmostEqual(prob_rejection, P_reject, delta=0.1)
        self.assertAlmostEqual(prob_service, P_service, delta=0.1)
        self.assertAlmostEqual(avg_queue_length, Lq, delta=0.5)

if __name__ == "__main__":
    unittest.main()
