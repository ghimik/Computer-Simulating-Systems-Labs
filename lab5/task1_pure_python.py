import math
from generator import *

class SingleServerWithBlocking:
    def __init__(self, generator, lambda_value, service_time):
        if lambda_value <= 0:
            raise ValueError("lambda_value должен быть положительным")
        if service_time <= 0:
            raise ValueError("service_time должен быть положительным")
        
        self.generator = generator  # генератор случайных чисел
        self.time = 0  # текущее время
        self.served = 0  # количество обслуженных клиентов
        self.rejected = 0  # количество отказов
        self.lambda_value = lambda_value
        self.service_time = service_time

    def exponential(self, rate):
        """Генерация случайного времени по экспоненциальному закону."""
        if rate <= 0:
            raise ValueError("rate должен быть положительным")
        rnd = self.generator.next()
        if rnd <= 0 or rnd >= 1:
            raise ValueError("Случайное число должно быть в пределах (0,1)")
        return -math.log(1.0 - rnd) / rate

    def simulate(self, max_time):
        """Запуск симуляции до max_time."""
        if max_time <= 0:
            raise ValueError("max_time должен быть положительным")
        
        print(f"Симуляция с λ = {self.lambda_value:.2f}, Tобсл = {self.service_time:.2f}")

        try:
            next_arrival = self.time + self.exponential(self.lambda_value)
        except ValueError as e:
            print(f"Ошибка генерации времени прибытия: {e}")
            return
        
        next_departure = float('inf')  # время завершения обслуживания

        while self.time < max_time:
            if next_arrival < next_departure:  # следующее событие - прибытие
                self.time = next_arrival
                if next_departure == float('inf'):  # если система свободна
                    next_departure = self.time + self.service_time
                    self.served += 1
                else:  # если система занята, отказ
                    self.rejected += 1
                try:
                    next_arrival = self.time + self.exponential(self.lambda_value)
                except ValueError as e:
                    print(f"Ошибка генерации времени прибытия: {e}")
                    break
            else:  # следующее событие - завершение обслуживания
                self.time = next_departure
                next_departure = float('inf')

    def get_statistics(self):
        """Возвращает эмпирическую статистику."""
        total_clients = self.served + self.rejected
        if total_clients == 0:
            return 0, 0, float('inf')
        probability_of_rejection = self.rejected / total_clients
        probability_of_service = self.served / total_clients
        served_to_rejected_ratio = self.served / self.rejected if self.rejected > 0 else float('inf')
        return probability_of_rejection, probability_of_service, served_to_rejected_ratio

    def theoretical_statistics(self):
        """Вычисление теоретической статистики для M/M/1 с отказами."""
        mu = 1 / self.service_time  # интенсивность обслуживания
        rho = self.lambda_value / mu  # коэффициент загрузки

        if rho <= 0:
            raise ValueError("Некорректное значение коэффициента загрузки")
        
        P_rejection = rho / (1 + rho)
        P_service = 1 - P_rejection
        theoretical_ratio = P_service / P_rejection if P_rejection > 0 else float('inf')
        return P_rejection, P_service, theoretical_ratio

if __name__ == "__main__":
    seed = 1234  
    gen = LemerGenerator(seed)

    work_durations = [1000, 2000, 5000]  

    for duration in work_durations:
        try:
            lambda_value = gen.next() * 0.9 + 0.1
            service_time = gen.next() * 2
            
            smo = SingleServerWithBlocking(lambda_value=lambda_value, service_time=service_time, generator=gen)
            smo.simulate(duration)  
            
            prob_rejection, prob_service, served_to_rejected_ratio = smo.get_statistics()
            P_rejection, P_service, theoritical_served_to_rejected_ratio = smo.theoretical_statistics()
            
            print(f"\nДля продолжительности {duration} минут:")
            print(f"Эмпирическая вероятность отказа: {prob_rejection:.4f}")
            print(f"Эмпирическая вероятность обслуживания: {prob_service:.4f}")
            print(f"Эмпирическое отношение обслуженных к отказанным: {served_to_rejected_ratio:.4f}")
            print(f"Теоретическая вероятность отказа: {P_rejection:.4f}")
            print(f"Теоретическая вероятность обслуживания: {P_service:.4f}")
            print(f"Теоретическое отношение обслуженных к отказанным: {theoritical_served_to_rejected_ratio:.4f}")
        except Exception as e:
            print(f"Ошибка при симуляции для {duration} минут: {e}")
