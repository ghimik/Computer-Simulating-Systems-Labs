import math
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Callable, override

class Function(ABC, Callable[[float], float]):
    @abstractmethod
    def value_at(self, x: float) -> float:
        pass

    def __call__(self, x: float) -> float:
        return self.value_at(x)

    def __str__(self) -> str:
        return "f(x)"

    def derivative(self) -> 'Function':
        return AnotherNumericalDerivative(self)

class LaboratoryFunction(Function):
    def value_at(self, x: float) -> float:
        return x * math.exp(x) * (math.sin(x))**2

    def __str__(self) -> str:
        return "f(x) = x * e^x * (sin x)^2"

    def derivative(self) -> 'Function':
        return CustomStringAndLambdaFunction("f'(x) = e^x * (sin x)^2 + 2x * sin x * cos x + x * (sin x)^2", 
                                             lambda x: math.exp(x) * (math.sin(x)**2 + 2*x*math.sin(x)*math.cos(x) + x*math.sin(x)**2))

class CustomLambdaFunction(Function):
    def __init__(self, func: Callable[[float], float]) -> None:
        self.func = func

    def value_at(self, x: float) -> float:
        return self.func(x)

class CustomStringFunction(Function):
    def __init__(self, expression: str) -> None:
        self.expression = expression

    def value_at(self, x: float) -> float:
        return eval(self.expression, {"x": x, "math": math})

    def __str__(self) -> str:
        return f"f(x) = {self.expression}"
    
class CustomStringAndLambdaFunction(Function):
    def __init__(self, expression: str, lambda_func: Callable[[float], float]) -> None:
        self.expression = expression
        self.lambda_func = lambda_func

    def value_at(self, x: float) -> float:
        return self.lambda_func(x)

    def __str__(self) -> str:
        return f"f(x) = {self.expression}"

class NumericalDerivative(Function):
    def __init__(self, function: Function, h: float = 1e-5) -> None:
        self.function = function
        self.h = h

    @override
    def value_at(self, x: float) -> float:
        h = max(self.h, abs(x) * 1e-5)
        return (self.function.value_at(x + h) - self.function.value_at(x - h)) / (2 * h)

    def __str__(self) -> str:
        return f"d({self.function})/dx"


class AnotherNumericalDerivative(NumericalDerivative):
    def __init__(self, function: Function, h: float = 1e-7) -> None:
        self.function = function
        self.h = h

    @override
    def value_at(self, x: float) -> float:
        return (self.function.value_at(x) - self.function.value_at(x - self.h)) / self.h


class Functions:
    @staticmethod
    def lab_function() -> Function:
        return LaboratoryFunction()

    @staticmethod
    def new_from_lambda(lambda_func: Callable[[float], float]) -> Function:
        return CustomLambdaFunction(lambda_func)

    @staticmethod
    def new_from_string(expression: str) -> Function:
        return CustomStringFunction(expression)

class Interval:
    def __init__(self, start: float, end: float) -> None:
        self._start = start
        self._end = end

    def from_(self) -> float:
        return self._start

    def to(self) -> float:
        return self._end

    def traverse(self, mapper: Callable[[float], float] = None, step: float = 1e-7) -> List[float]:
        x = self._start
        results = []
        while x <= self._end:
            if mapper is None:
                results.append(x)
            else:
                results.append(mapper(x))
            x += step
        return results

class GraphBuilder:
    def __init__(self, function: Function, interval: Interval, step: float = 0.01) -> None:
        self.function = function
        self.interval = interval
        self.step = step

    def build(self) -> None:
        x = self.interval.traverse(mapper=None, step=self.step)
        y = self.interval.traverse(self.function, self.step)
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'График функции {self.function}')
        plt.grid()
        plt.show()


class Extremum:
    def __init__(self, point: float, value: float) -> None:
        self._point = point
        self._value = value

    @property
    def point(self) -> float:
        return self._point

    @property
    def value(self) -> float:
        return self._value

    def __str__(self) -> str:
        return f"Extremum at x = {self.point}, value = {self.value}"

class Minimum(Extremum):
    def __str__(self) -> str:
        return f"Minimum at x = {self.point}, value = {self.value}"

class Maximum(Extremum):
    def __str__(self) -> str:
        return f"Maximum at x = {self.point}, value = {self.value}"

class ExtremumFinder(ABC):
    @abstractmethod
    def find_extremum(self, function: Function, interval: Interval, epsilon: float = 1e-7) -> Optional[Extremum]:
        pass

class IterationalContext(ABC):
    def __init__(self):
        self.iteration_count = 0

    @abstractmethod
    def count_iteration(self):
        pass

    def drop_iterations(self):
        self.iteration_count = 0

    def get_iteration_count(self) -> int:
        return self.iteration_count

class UnlimitedIterationalContext(IterationalContext):
    def count_iteration(self):
        self.iteration_count += 1

class LimitedIterationalContext(IterationalContext):
    def __init__(self, max_iterations: int):
        super().__init__()
        self.max_iterations = max_iterations

    def count_iteration(self):
        if self.iteration_count >= self.max_iterations:
            raise Exception("Maximum number of iterations exceeded")
        self.iteration_count += 1

class IterationalExtremumFinder(ExtremumFinder):
    def __init__(self, context: IterationalContext):
        self.context = context

    def count_iteration(self):
        self.context.count_iteration()

    def drop_iterations(self):
        self.context.drop_iterations()

class DichotomyMethod(IterationalExtremumFinder):

    def __init__(self, context: IterationalContext):
        super().__init__(context)

    def find_extremum(self, function: Function, interval: Interval, epsilon: float = 1e-7) -> Optional[Extremum]:
        a, b = interval.from_(), interval.to()
        derivative = function.derivative()
        
        if derivative.value_at(a) * derivative.value_at(b) > 0:
            raise Exception("No extremum in the interval")
        
        is_minimum = True
        while (b - a) / 2 > epsilon:
            self.count_iteration()
            x1 = (b + a - epsilon) / 2
            x2 = (b + a + epsilon) / 2
            if function.value_at(x1) <= function.value_at(x2):
                b = x2
                is_minimum = True
            else:
                a = x1
                is_minimum = False
            
        point = (a + b) / 2
        if is_minimum:
            return Minimum(point, function.value_at(point))
        else:
            return Maximum(point, function.value_at(point))

class GoldenSectionMethod(IterationalExtremumFinder):
    golden_section_tau: float = (math.sqrt(5) - 1) / 2

    def __init__(self, context: IterationalContext):
        super().__init__(context)

    def find_extremum(self, function: Function, interval: Interval, epsilon: float = 1e-7) -> Optional[Extremum]:
        a, b = interval.from_(), interval.to()
        derivative = function.derivative()
        
        if derivative.value_at(a) * derivative.value_at(b) > 0:
            raise Exception("No extremum in the interval")

        lambda1 = a + (1 - self.golden_section_tau) * (b - a)
        mu1 = a + self.golden_section_tau * (b - a)
        is_minimum = True
        while abs(b - a)> epsilon:
            self.count_iteration()
            if function.value_at(lambda1) < function.value_at(mu1):
                b = mu1
                mu1 = lambda1
                lambda1 = a + (1 - self.golden_section_tau) * (b - a)
                is_minimum = True
            else:
                a = lambda1
                lambda1 = mu1
                mu1 = a + self.golden_section_tau * (b - a)
                is_minimum = False

        point = (a + b) / 2
        if is_minimum:
            return Minimum(point, function.value_at(point))
        else:
            return Maximum(point, function.value_at(point))

class Optional[T]:
    def __init__(self, value: T = None):
        self._value = value

    def is_present(self) -> bool:
        return self._value is not None

    def get(self) -> T:
        if self._value is None:
            raise ValueError("No value present")
        return self._value

    def or_else(self, other: T) -> T:
        return self._value if self._value is not None else other

    def __str__(self):
        return f"Optional({self._value})"
    

class ExtremumIntervalDetector(ABC):
    def __init__(self, function: Function, step: float = None, search_interval: Interval = Interval(-1000, 1000)):
        self.function = function
        self.search_interval = search_interval
        if step is None:
            if search_interval is not None:
                self.step = (search_interval.to() - search_interval.from_()) / 10
            else:
                self.step = 0.1
        else:
            self.step = step

    def find_extremum_intervals(self) -> List[Interval]:
        derivative = self.function.derivative()
        x_values = np.arange(self.search_interval.from_(), self.search_interval.to(), self.step)
        potential_intervals = []

        for x in x_values:
            root = self.find_root(derivative, x)
            if root is not None:
                potential_intervals.append(Interval(root - self.step, root + self.step))

        return potential_intervals

    @abstractmethod
    def find_root(self, derivative: Function, initial_guess: float) -> Optional[float]:
        pass

class NewtonExtremumIntervalDetector(ExtremumIntervalDetector):
    def find_root(self, derivative: Function, initial_guess: float, tolerance: float = 1e-5, max_iterations: int = 1000) -> Optional[float]:
        x = initial_guess
        second_derivative = derivative.derivative()
        for _ in range(max_iterations):
            f_prime_x = derivative.value_at(x)
            if abs(f_prime_x) < tolerance:
                return x
            f_double_prime_x = second_derivative.value_at(x)
            if f_double_prime_x == 0:
                return None
            x -= f_prime_x / f_double_prime_x
        return None

