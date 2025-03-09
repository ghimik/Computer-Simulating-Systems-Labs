import simpy
import random
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


def simulate(lambda_value, service_time, num_servers, max_time, generator, queue_capacity=None):
    if queue_capacity is not None and queue_capacity < 0:
        raise ValueError('Вместимость очереди должна быть неотрицательной или None')
    
    env = simpy.Environment()
    servers = simpy.Resource(env, capacity=num_servers)
    stats = {'served': 0, 'rejected': 0}
    
    # монитор для расчёта средней длины очереди:
    # last_time – время последнего изменения,
    # area – накопленная "площадь" (интегра длины очереди),
    # current_length – текущее число ожидающих в очереди.
    queue_monitor = {
         'last_time': 0.0,
         'area': 0.0,
         'current_length': 0
    }
    
    def car_arrival():
        while True:
            # интервал между прибытием
            yield env.timeout(exponential(lambda_value, generator))
            
            dt = env.now - queue_monitor['last_time']
            queue_monitor['area'] += dt * queue_monitor['current_length']
            queue_monitor['last_time'] = env.now
            
            # если есть свободный сервер, машина обслуживается сразу
            if servers.count < servers.capacity:
                was_queued = False
                env.process(service(was_queued))
            else:
                # если серверы заняты, проверяем вместимость очереди
                if queue_capacity is None or len(servers.queue) < queue_capacity:
                    was_queued = True
                    queue_monitor['current_length'] += 1
                    env.process(service(was_queued))
                else:
                    stats['rejected'] += 1

    def service(was_queued):
        with servers.request() as request:
            yield request
            # если машина ожидала в очереди, при получении сервиса обновляем монитор:
            if was_queued:
                dt = env.now - queue_monitor['last_time']
                queue_monitor['area'] += dt * queue_monitor['current_length']
                queue_monitor['last_time'] = env.now
                queue_monitor['current_length'] -= 1
            # сервис
            yield env.timeout(random.expovariate(1 / service_time))
            stats['served'] += 1

    env.process(car_arrival())
    env.run(until=max_time)
    
    # финальное обновление монитора до конца симуляции
    dt = max_time - queue_monitor['last_time']
    queue_monitor['area'] += dt * queue_monitor['current_length']
    stats['avg_queue_length'] = queue_monitor['area'] / max_time
    
    return stats

def theoretical_statistics(lambda_value, service_time, num_servers, queue_capacity=None):
    if lambda_value <= 0 or service_time <= 0 or num_servers <= 0:
        raise ValueError("Интенсивность, время обслуживания и количество серверов должны быть положительными.")
    
    mu = 1 / service_time  
    rho = lambda_value / (num_servers * mu)
    
    if rho >= 1:
        raise ValueError("Система перегружена (rho >= 1), расчёты невозможны.")
    
    try:
        if queue_capacity is None:
            sum_term = sum((lambda_value / mu) ** k / math.factorial(k) for k in range(num_servers))
            sum_term += ((lambda_value / mu) ** num_servers) / (math.factorial(num_servers) * (1 - rho))
            P0 = 1 / sum_term
            P_reject = 0  
            Lq = (P0 * (lambda_value / mu) ** num_servers * rho) / (math.factorial(num_servers) * (1 - rho) ** 2)
        else:
            sum_term = sum((lambda_value / mu) ** k / math.factorial(k) for k in range(num_servers + 1))
            sum_term += sum(((lambda_value / mu) ** (num_servers + i)) / (math.factorial(num_servers) * (num_servers ** i))
                            for i in range(1, queue_capacity + 1))
            P0 = 1 / sum_term
            P_reject = ((lambda_value / mu) ** (num_servers + queue_capacity)) / (
                    math.factorial(num_servers) * (num_servers ** queue_capacity)) * P0
            Lq = sum((k - num_servers) * ((lambda_value / mu) ** k) / (
                    math.factorial(num_servers) * (num_servers ** (k - num_servers))) * P0
                     for k in range(num_servers + 1, num_servers + queue_capacity + 1))
    except Exception as e:
        raise RuntimeError(f"Ошибка при вычислении теоретических характеристик: {e}")
    
    return P_reject, 1 - P_reject, Lq

if __name__ == "__main__":
    seed = 42  
    gen = LemerGenerator(seed)
    
    num_servers = 2  
    max_times = [10, 100, 1000, 10000]  
    queue_capacities = [5, None] 
    
    for queue_capacity in queue_capacities:
        for max_time in max_times:
            lambda_value = generate_lambda(gen)
            service_time = generate_lambda(gen) * 10
            
            if lambda_value <= 0 or service_time <= 0:
                print("Ошибка: некорректные значения параметров lambda или service_time.")
                continue
            
            stats = simulate(lambda_value, service_time, num_servers, max_time, gen, queue_capacity)
            total_clients = stats['served'] + stats['rejected']
            prob_rejection = stats['rejected'] / total_clients if total_clients > 0 else 0
            prob_service = stats['served'] / total_clients if total_clients > 0 else 0
            avg_queue_length = stats['avg_queue_length']
            
            try:
                P_reject, P_service, Lq = theoretical_statistics(lambda_value, service_time, num_servers, queue_capacity)
            except Exception as e:
                print(f"Ошибка при вычислении теоретических значений: {e}")
                print("-" * 50)
                continue
            
            print(f"\nДля продолжительности {max_time} часов и ограничения очереди {queue_capacity}")
            print(f"При lambda={lambda_value}, времени обслуживания {service_time}")
            print(f"Эмпирическая вероятность отказа: {prob_rejection:.4f}")
            print(f"Эмпирическая вероятность обслуживания: {prob_service:.4f}")
            print(f"Эмпирическая средняя длина очереди: {avg_queue_length:.4f}")
            print(f"Теоретическая вероятность отказа: {P_reject:.4f}")
            print(f"Теоретическая вероятность обслуживания: {P_service:.4f}")
            print(f"Теоретическая средняя длина очереди: {Lq:.4f}")
            print("-" * 50)
