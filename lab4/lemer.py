import math
import numpy as np

class LemerGenerator:
    def __init__(self, seed, a=16807, m=32768):
        self.m = m
        self.a = a
        self.x = seed
        self._validate_params()

    def _validate_params(self):
        if self.m <= 0:
            raise ValueError("Modulus m must be positive")
        if self.a <= 0 or self.a >= self.m:
            raise ValueError(f"Multiplier a must be in (0, {self.m})")
        if self.x <= 0 or self.x >= self.m:
            raise ValueError(f"Seed must be in (0, {self.m})")

    def next(self):
        self.x = (self.a * self.x) % self.m
        return self.x / self.m

    def current_raw(self):
        return self.x

class Sample:
    def __init__(self, data=None):
        self.data = data if data is not None else []
    
    def add_value(self, value):
        self.data.append(value)
    
    def clear(self):
        self.data.clear()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        return f"Sample[size={len(self.data)}]"

class SampleGenerator:
    def __init__(self, rng):
        self.rng = rng
    
    def generate_sample(self, size):
        if size <= 0:
            raise ValueError("Sample size must be positive")
        
        sample = Sample()
        for _ in range(size):
            sample.add_value(self.rng.next())
        return sample
    


class LaplaceSampleGenerator:
    def __init__(self, loc=0.0, scale=1.0):
        """
        Генератор выборок с распределением Лапласа
        
        :param loc: параметр положения (среднее значение)
        :param scale: параметр масштаба (должен быть > 0)
        """
        if scale <= 0:
            raise ValueError("Scale parameter must be positive")
            
        self.loc = loc
        self.scale = scale

    def generate_sample(self, size):
        """Генерирует выборку заданного размера"""
        if size <= 0:
            raise ValueError("Sample size must be positive")
            
        data = np.random.laplace(loc=self.loc, scale=self.scale, size=size)
        return Sample(data.tolist())

class SampleStatistics:
    def __init__(self, sample):
        if len(sample) == 0:
            raise ValueError("Sample is empty")
        self.sample = sample
    
    def mean(self):
        return sum(self.sample) / len(self.sample)
    
    def variance(self):
        n = len(self.sample)
        avg = self.mean()
        return sum((x - avg)**2 for x in self.sample) / n
    
    def std_deviation(self):
        return math.sqrt(self.variance())
    
    def frequency_test(self, expected_std=0.2887):
        """Частотный тест для равномерного распределения"""
        lower_bound = 0.5 - expected_std
        upper_bound = 0.5 + expected_std
        count = sum(1 for x in self.sample if lower_bound < x < upper_bound)
        total = len(self.sample)
        percentage = (count / total) * 100
        return {
            'interval': (lower_bound, upper_bound),
            'count': count,
            'percentage': percentage,
            'expected_percentage': 57.7,
            'deviation': abs(percentage - 57.7)
        }
    
    def get_report(self):
        freq = self.frequency_test()
        return (
            f"Statistical Report:\n"
            f"• Size: {len(self.sample):,}\n"
            f"• Mean: {self.mean():.4f}\n"
            f"• Variance: {self.variance():.4f}\n"
            f"• Standard Deviation: {self.std_deviation():.4f}\n"
            f"• Frequency Test ({freq['interval'][0]:.4f} - {freq['interval'][1]:.4f}):\n"
            f"  - Numbers in range: {freq['count']:,} ({freq['percentage']:.1f}%)\n"
            f"  - Expected: {freq['expected_percentage']}%\n"
            f"  - Deviation: {freq['deviation']:.1f}%"
        )

import matplotlib.pyplot as plt
import numpy as np

class SampleVisualizer:
    def __init__(self, sample):
        if len(sample) == 0:
            raise ValueError("Cannot visualize empty sample")
        self.sample = sample
        self.data = np.array(sample.data)
    
    def plot_histogram(self, bins=30, density=False, title="Histogram", color='skyblue', edgecolor='black'):
        """Построение гистограммы распределения"""
        plt.figure(figsize=(10, 6))
        plt.hist(self.data, bins=bins, density=density, 
                color=color, edgecolor=edgecolor, alpha=0.7)
        plt.title(title)
        plt.xlabel("Values")
        plt.ylabel("Frequency" if not density else "Density")
        plt.grid(True, linestyle='--', alpha=0.7)
        return self
    
    def plot_density(self, title="Density Plot", color='blue', linewidth=2):
        """Оценка плотности распределения (KDE)"""
        from sklearn.neighbors import KernelDensity
        
        plt.figure(figsize=(10, 6))
        kde = KernelDensity(bandwidth=0.02, kernel='gaussian')
        kde.fit(self.data[:, None])
        
        x = np.linspace(self.data.min(), self.data.max(), 1000)
        log_prob = kde.score_samples(x[:, None])
        
        plt.plot(x, np.exp(log_prob), color=color, linewidth=linewidth)
        plt.title(title)
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', alpha=0.7)
        return self
    
    def plot_boxplot(self, title="Boxplot", color='teal', vert=False):
        """Построение boxplot"""
        plt.figure(figsize=(10, 4))
        plt.boxplot(self.data, vert=vert, patch_artist=True,
                   boxprops=dict(facecolor=color, color='black'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'),
                   medianprops=dict(color='red'))
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        return self
    
    def add_theoretical(self, dist_type='uniform', **params):
        """Добавление теоретической кривой для сравнения"""
        x = np.linspace(self.data.min(), self.data.max(), 1000)
        
        if dist_type == 'uniform':
            y = np.ones_like(x) / (params.get('b', 1) - params.get('a', 0))
            plt.plot(x, y, 'r--', linewidth=2, label='Uniform PDF')
            
        elif dist_type == 'laplace':
            loc = params.get('loc', 0)
            scale = params.get('scale', 1)
            y = (1/(2*scale)) * np.exp(-np.abs(x - loc)/scale)
            plt.plot(x, y, 'r--', linewidth=2, label='Laplace PDF')
            
        plt.legend()
        return self
    
    def save(self, filename="plot.png", dpi=300):
        """Сохранение последнего графика"""
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close()
        return self
    
    def show(self):
        """Отображение всех графиков"""
        plt.show()
        return self

if __name__ == "__main__":

    for sample_size in [10, 25, 120, 400, 800, 1500, 10000]:

        print(str("-" * 25) + "SAMPLE SIZE = " + str(sample_size) + str("-"*25))
        uniform_sample = SampleGenerator(LemerGenerator(seed=42)).generate_sample(sample_size)
        laplace_sample = LaplaceSampleGenerator(loc=0.5, scale=0.5).generate_sample(sample_size)

        
        print("Uniform Distribution Visualization:")
        (SampleVisualizer(uniform_sample)
            .plot_histogram(title="Uniform Distribution Histogram", density=True)
            .add_theoretical(dist_type='uniform')
            .save("uniform_dist.png"))
        print(SampleStatistics(uniform_sample).get_report())
        
        print("\nLaplace Distribution Visualization:")
        (SampleVisualizer(laplace_sample)
            .plot_histogram(title="Laplace Distribution Histogram", bins=50, density=True)
            .add_theoretical(dist_type='laplace', loc=0.5, scale=0.5)
            .plot_density(title="Laplace Density Estimation")
            .plot_boxplot(title="Laplace Distribution Boxplot")
            .save("laplace_dist.png"))
        print(SampleStatistics(laplace_sample).get_report())