
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