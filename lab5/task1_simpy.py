import simpy
import math
from generator import *

def exponential(rate, generator):
    if rate <= 0:
        raise ValueError("Параметр интенсивности должен быть положительным.")
    rnd = generator.next()
    if not (0 < rnd < 1):
        raise ValueError("Сгенерированное случайное число должно быть в диапазоне (0,1).")
    return -math.log(1.0 - rnd) / rate

def customer_arrival(env, server, lambda_value, service_time, generator, stats):
    try:
        while True:
            yield env.timeout(exponential(lambda_value, generator))
            if server.count == 0:
                stats['served'] += 1
                env.process(customer_service(env, server, service_time))
            else:
                stats['rejected'] += 1
    except Exception as e:
        print(f"Ошибка в процессе прибытия клиентов: {e}")

def customer_service(env, server, service_time):

    with server.request() as req:
        yield req
        if service_time <= 0:
            raise ValueError("Время обслуживания должно быть положительным.")
        yield env.timeout(service_time)

def simulate(lambda_value, service_time, max_time, generator):
    if lambda_value <= 0 or service_time <= 0 or max_time <= 0:
        raise ValueError("Интенсивность прибытия, время обслуживания и время моделирования должны быть положительными.")
    
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    stats = {'served': 0, 'rejected': 0}
    
    try:
        env.process(customer_arrival(env, server, lambda_value, service_time, generator, stats))
        env.run(until=max_time)
    except Exception as e:
        print(f"Ошибка в процессе симуляции: {e}")
    
    return stats

def theoretical_statistics(lambda_value, service_time):
    if service_time <= 0:
        raise ValueError("Время обслуживания должно быть положительным.")
    mu = 1 / service_time
    rho = lambda_value / mu
    P_rejection = rho / (1 + rho) if (1 + rho) != 0 else 0
    P_service = 1 - P_rejection
    theoretical_ratio = P_service / P_rejection if P_rejection > 0 else float('inf')
    return P_rejection, P_service, theoretical_ratio

if __name__ == "__main__":
    try:
        seed = 1234 
        gen = LemerGenerator(seed)

        work_durations = [1000, 2000, 5000] 

        for duration in work_durations:
            lambda_value = gen.next() * 0.9 + 0.1
            service_time = gen.next() * 2

            stats = simulate(lambda_value, service_time, duration, gen)

            total_clients = stats['served'] + stats['rejected']
            prob_rejection = stats['rejected'] / total_clients if total_clients > 0 else 0
            prob_service = stats['served'] / total_clients if total_clients > 0 else 0
            served_to_rejected_ratio = stats['served'] / stats['rejected'] if stats['rejected'] > 0 else float('inf')

            P_rejection, P_service, theoretical_ratio = theoretical_statistics(lambda_value, service_time)

            print(f"\nДля продолжительности {duration} минут:")
            print(f"Эмпирическая вероятность отказа: {prob_rejection:.4f}")
            print(f"Эмпирическая вероятность обслуживания: {prob_service:.4f}")
            print(f"Эмпирическое отношение обслуженных к отказанным: {served_to_rejected_ratio:.4f}")
            print(f"Теоретическая вероятность отказа: {P_rejection:.4f}")
            print(f"Теоретическая вероятность обслуживания: {P_service:.4f}")
            print(f"Теоретическое отношение обслуженных к отказанным: {theoretical_ratio:.4f}")
    except Exception as e:
        print(f"Ошибка в главном процессе выполнения: {e}")
