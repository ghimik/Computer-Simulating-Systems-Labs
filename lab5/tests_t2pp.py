import unittest
import math
from generator import LemerGenerator
from task2_pure_python import MultiServerQueue

class TestMultiServerQueue(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.generator = LemerGenerator(self.seed)
        self.lambda_value = 0.5
        self.service_time = 1.0
        self.num_servers = 3
        self.queue_capacity = 5
        self.max_time = 10000

    def test_initialization(self):
        """Тест инициализации класса."""
        smo = MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        self.assertEqual(smo.lambda_value, self.lambda_value)
        self.assertEqual(smo.service_time, self.service_time)
        self.assertEqual(smo.num_servers, self.num_servers)
        self.assertEqual(smo.queue_capacity, self.queue_capacity)
        self.assertEqual(smo.time, 0)
        self.assertEqual(smo.served, 0)
        self.assertEqual(smo.rejected, 0)
        self.assertEqual(smo.queue, [])
        self.assertEqual(smo.servers, [0] * self.num_servers)
        self.assertEqual(smo.events, [])

    def test_initialization_with_invalid_lambda(self):
        """Тест инициализации с недопустимым значением lambda_value."""
        with self.assertRaises(ValueError):
            MultiServerQueue(self.generator, -1, self.service_time, self.num_servers, self.queue_capacity)

    def test_initialization_with_invalid_service_time(self):
        """Тест инициализации с недопустимым значением service_time."""
        with self.assertRaises(ValueError):
            MultiServerQueue(self.generator, self.lambda_value, -1, self.num_servers, self.queue_capacity)

    def test_initialization_with_invalid_num_servers(self):
        """Тест инициализации с недопустимым значением num_servers."""
        with self.assertRaises(ValueError):
            MultiServerQueue(self.generator, self.lambda_value, self.service_time, -1, self.queue_capacity)

    def test_initialization_with_invalid_queue_capacity(self):
        """Тест инициализации с недопустимым значением queue_capacity."""
        with self.assertRaises(ValueError):
            MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, -1)

    def test_generate_event_stream(self):
        """Тест генерации потока событий."""
        smo = MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        smo.generate_event_stream(self.max_time)
        self.assertGreater(len(smo.events), 0)

    def test_generate_event_stream_with_invalid_max_time(self):
        """Тест генерации потока событий с недопустимым значением max_time."""
        smo = MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        with self.assertRaises(ValueError):
            smo.generate_event_stream(-1)

    def test_simulate(self):
        """Тест симуляции."""
        smo = MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        smo.simulate(self.max_time)
        self.assertGreaterEqual(smo.served, 0)
        self.assertGreaterEqual(smo.rejected, 0)

    def test_simulate_with_invalid_max_time(self):
        """Тест симуляции с недопустимым значением max_time."""
        smo = MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        with self.assertRaises(ValueError):
            smo.simulate(-1)

    def test_handle_arrival(self):
        """Тест обработки прибытия клиента."""
        smo = MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        smo.time = 0
        smo.handle_arrival()
        self.assertGreaterEqual(smo.served + smo.rejected, 0)

    def test_handle_departure(self):
        """Тест обработки завершения обслуживания."""
        smo = MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        smo.time = 0
        smo.queue.append(0)
        smo.handle_departure()
        self.assertGreaterEqual(smo.served, 0)

    def test_get_statistics(self):
        """Тест получения статистики."""
        smo = MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        smo.simulate(self.max_time)
        prob_rejection, prob_service, avg_queue_length = smo.get_statistics()
        self.assertGreaterEqual(prob_rejection, 0)
        self.assertLessEqual(prob_rejection, 1)
        self.assertGreaterEqual(prob_service, 0)
        self.assertLessEqual(prob_service, 1)
        self.assertGreaterEqual(avg_queue_length, 0)

    def test_theoretical_statistics(self):
        """Тест теоретической статистики."""
        smo = MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        P_reject, P_service, Lq = smo.theoretical_statistics()
        self.assertGreaterEqual(P_reject, 0)
        self.assertLessEqual(P_reject, 1)
        self.assertGreaterEqual(P_service, 0)
        self.assertLessEqual(P_service, 1)
        self.assertGreaterEqual(Lq, 0)

    def test_theoretical_and_empirical_statistics_are_close(self):
        """Тест, что теоретическая и эмпирическая статистика близки."""
        smo = MultiServerQueue(self.generator, self.lambda_value, self.service_time, self.num_servers, self.queue_capacity)
        smo.simulate(self.max_time)
        prob_rejection, prob_service, avg_queue_length = smo.get_statistics()
        P_reject, P_service, Lq = smo.theoretical_statistics()

        self.assertAlmostEqual(prob_rejection, P_reject, places=2,
                               msg="Эмпирическая и теоретическая вероятность отказа отличаются более чем на 0.01")
        self.assertAlmostEqual(prob_service, P_service, places=2,
                               msg="Эмпирическая и теоретическая вероятность обслуживания отличаются более чем на 0.01")
        self.assertAlmostEqual(avg_queue_length, Lq, places=2,
                               msg="Эмпирическая и теоретическая средняя длина очереди отличаются более чем на 0.01")

if __name__ == "__main__":
    unittest.main()