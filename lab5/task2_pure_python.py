import heapq
import math
from generator import *  


def poisson(lam, generator):
    """Генерация случайного числа по распределению Пуассона."""
    L = math.exp(-lam)
    k = 0
    p = 1
    while p > L:
        k += 1
        p *= generator.next()
    return k - 1

def generate_lambda(generator):
    """Генерация λ по Пуассону и нормализация в диапазон [0, 1]."""
    lam = poisson(10, generator)  
    
    length = len(str(lam))
    
    normalized_lam = lam / (10 ** length)
    
    return normalized_lam

def exponential(rate, generator):
    if rate <= 0:
        raise ValueError("Интенсивность должна быть положительной.")
    rnd = generator.next()
    if rnd <= 0 or rnd >= 1:
        raise ValueError("Ошибка генерации случайного числа.")
    return -math.log(1.0 - rnd) / rate

class MultiServerQueue:
    def __init__(self, generator, lambda_value, service_time, num_servers, queue_capacity=None):
        if lambda_value <= 0:
            raise ValueError("Параметр интенсивности должен быть положительным числом.")
        if service_time <= 0:
            raise ValueError("Время обслуживания клиента должно быть положительным числом.")
        if num_servers <= 0:
            raise ValueError("Число серверов должно быть положительным числом.")
        if queue_capacity is not None and queue_capacity < 0:
            raise ValueError("Вместимость очереди должна быть положительным числом или None.")

        self.generator = generator 
        self.lambda_value = lambda_value  
        self.service_time = service_time  
        self.num_servers = num_servers  
        self.queue_capacity = queue_capacity  # максимальная длина очереди (None - неограниченная)
        self.time = 0  
        self.served = 0  
        self.rejected = 0 
        self.queue = [] 
        self.servers = [0] * num_servers  # время завершения обслуживания для каждого сервера
        self.events = []  # события в системе
        
        # переменные для расчёта средней длины очереди
        self.area_under_q = 0   # накопленная «площадь» под графиком длины очереди
        self.last_event_time = 0  # время последнего события


    def generate_event_stream(self, max_time):
        """Генерируем поток событий до max_time."""
        if max_time <= 0:
            raise ValueError("Время симуляции должно быть положительным числом.")
        event_time = 0
        while event_time < max_time:
            event_time += exponential(self.lambda_value, self.generator)
            heapq.heappush(self.events, (event_time, 'arrival'))

    def simulate(self, max_time):
        """Запускаем симуляцию до max_time."""
        if max_time <= 0:
            raise ValueError("Время симуляции должно быть положительным числом.")
        self.generate_event_stream(max_time)
        self.time = 0
        self.last_event_time = 0
        self.area_under_q = 0

        while self.events:
            event_time, event_type = heapq.heappop(self.events)
            if event_time > max_time:
                break
            
            # обновляем накопленную площадь под графиком очереди
            dt = event_time - self.time
            self.area_under_q += len(self.queue) * dt
            
            self.time = event_time

            if event_type == 'arrival':
                self.handle_arrival()
            elif event_type == 'departure':
                self.handle_departure()

    def handle_arrival(self):
        """Обрабатываем прибытие клиента."""
        free_server = next((i for i, t in enumerate(self.servers) if t <= self.time), None)
        
        if free_server is not None:
            # сразу
            service_time = exponential(1 / self.service_time, self.generator)
            self.servers[free_server] = self.time + service_time
            heapq.heappush(self.events, (self.time + service_time, 'departure'))
            self.served += 1
        elif self.queue_capacity is None or len(self.queue) < self.queue_capacity:
            # в очередь
            self.queue.append(self.time)
        else:
            # отказ
            self.rejected += 1

    def handle_departure(self):
        """Обрабатываем завершение обслуживания."""
        if self.queue:
            arrival_time = self.queue.pop(0)
            service_time = exponential(1 / self.service_time, self.generator)
            free_server = next(i for i, t in enumerate(self.servers) if t <= self.time)
            self.servers[free_server] = self.time + service_time
            heapq.heappush(self.events, (self.time + service_time, 'departure'))
            self.served += 1

    def get_statistics(self):
        """Возвращаем статистику по обслуживанию."""
        total_clients = self.served + self.rejected
        probability_of_rejection = self.rejected / total_clients if total_clients > 0 else 0
        probability_of_service = self.served / total_clients if total_clients > 0 else 0
        average_queue_length = self.area_under_q / self.time if self.time > 0 else 0
        return probability_of_rejection, probability_of_service, average_queue_length

    def theoretical_statistics(self):
        """Вычисляем теоретическую статистику."""
        mu = 1 / self.service_time
        rho = self.lambda_value / (self.num_servers * mu)

        if self.queue_capacity is None:  # M/M/n
            sum_term = sum((self.lambda_value / mu) ** k / math.factorial(k) for k in range(self.num_servers))
            P0 = 1 / (sum_term + ((self.lambda_value / mu) ** self.num_servers) / (math.factorial(self.num_servers) * (1 - rho)))
            P_reject = 0  # для M/M/n вероятность отказа равна 0
            Lq = (P0 * (self.lambda_value / mu) ** self.num_servers * rho) / (math.factorial(self.num_servers) * (1 - rho) ** 2)
        else:  # M/M/n/m
            sum_term = sum((self.lambda_value / mu) ** k / math.factorial(k) for k in range(self.num_servers + 1))
            sum_term += sum(
                ((self.lambda_value / mu) ** (self.num_servers + i)) / (math.factorial(self.num_servers) * (self.num_servers ** i))
                for i in range(1, self.queue_capacity + 1))
            P0 = 1 / sum_term
            P_reject = ((self.lambda_value / mu) ** (self.num_servers + self.queue_capacity)) / (
                    math.factorial(self.num_servers) * (self.num_servers ** self.queue_capacity)) * P0
            Lq = sum((k - self.num_servers) * ((self.lambda_value / mu) ** k) / (
                    math.factorial(self.num_servers) * (self.num_servers ** (k - self.num_servers))) * P0
                for k in range(self.num_servers + 1, self.num_servers + self.queue_capacity + 1))

        return P_reject, 1 - P_reject, Lq



if __name__ == "__main__":
    try:
        seed = 42
        num_servers = 3
        max_times = [10, 100, 1000, 10000]  
        queue_capacities = [5, None] 
        gen = LemerGenerator(seed) 

        for queue_capacity in queue_capacities:
            for max_time in max_times:
                try:
                    lam = gen.next()
                    lambda_value = generate_lambda(gen)
                    service_time = generate_lambda(gen) * 10

                    smo = MultiServerQueue(generator=gen, lambda_value=lambda_value, service_time=service_time, num_servers=num_servers, queue_capacity=queue_capacity)
                    smo.simulate(max_time=max_time)

                    print(f"lambda = {lambda_value}, service_time = {service_time}")
                    print(f"max_time: {max_time}, queue_capacity: {queue_capacity}, ")

                    prob_rejection, prob_service, avg_queue_length = smo.get_statistics()
                    print(f"эмпирическая вероятность отказа: {prob_rejection:.4f}, "
                        f"эмпирическая вероятность обслуживания: {prob_service:.4f}, "
                        f"эмпирическая средняя длина очереди: {avg_queue_length:.4f}")

                    P_reject, P_service, Lq = smo.theoretical_statistics()
                    print(f"теоретическая вероятность отказа: {P_reject:.4f}, "
                        f"теоретическая вероятность обслуживания: {P_service:.4f}, "
                        f"теоретическая средняя длина очереди: {Lq:.4f}")
                    
                    print("-" * 50)
                except ValueError as e:
                    print(f"Ошибка при симуляции с max_time={max_time} и queue_capacity={queue_capacity}: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")