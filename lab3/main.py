import matplotlib.pyplot as plt
from core import *



def script_one():
    function = Functions.lab_function()
    derivative = function.derivative()
    interval = Interval(-2, 2)

    graph_function = GraphBuilder(function, interval, 0.01)
    graph_derivative = GraphBuilder(derivative, interval, 0.01)

    graph_function.build()
    graph_derivative.build()

def script_two():
    function = Functions.lab_function()
    interval = Interval(-2, 2)
    context = LimitedIterationalContext(max_iterations=50000)
    dichotomy = DichotomyMethod(context)

    try:
        extremum = dichotomy.find_extremum(function, interval)
        print(extremum, context.get_iteration_count())
    except Exception as e:
        print(e)

def script_three():
    function = Functions.lab_function()
    interval = Interval(-2, 2)
    context = LimitedIterationalContext(max_iterations=50000)
    golden_section = GoldenSectionMethod(context)

    try:
        extremum = golden_section.find_extremum(function, interval)
        print(extremum, context.get_iteration_count())
    except Exception as e:
        print(e)

def script_four():
    function = Functions.new_from_string("math.sin(x**2)")
    detector = ExtremumIntervalDetector(function)
    context = UnlimitedIterationalContext()
    golden_section = GoldenSectionMethod(context)

    extremum_intervals = detector.find_extremum_intervals()
    for interval in extremum_intervals:
        print(f"Extremum interval: from {interval.from_()} to {interval.to()}")
        try:
            extremum = golden_section.find_extremum(function, interval)
            print(extremum, context.get_iteration_count())
        except Exception as e:
            print(e)

def main():
    function = Functions.lab_function()
    # function = Functions.new_from_lambda(lambda x: x ** 3)
    search_interval = Interval(-10, 20)
    finder = NewtonExtremumIntervalDetector(function, search_interval=search_interval)
    intervals = finder.find_extremum_intervals()
    graph_function = GraphBuilder(function, search_interval, 0.01)
    graph_function.build()

    epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

    golden_context = LimitedIterationalContext(max_iterations=200_000)
    dichotomy_context = LimitedIterationalContext(max_iterations=200_000)

    golden_method = GoldenSectionMethod(golden_context)
    dichotomy_method = DichotomyMethod(dichotomy_context)

    print(f"{'Точность':<10} {'Золотое сечение':<20} {'Дихотомия':<10}")
    for epsilon in epsilons:
        golden_context.drop_iterations()
        dichotomy_context.drop_iterations()
        
        for interval in intervals:
            try:
                golden_method.find_extremum(function, interval, epsilon)
            except Exception as e:
                pass
            
            try:
                dichotomy_method.find_extremum(function, interval, epsilon)
            except Exception as e:
                pass
        
        print(f"{epsilon:<10} {golden_context.get_iteration_count():<20} {dichotomy_context.get_iteration_count():<10}")


if __name__ == "__main__":
    main()
