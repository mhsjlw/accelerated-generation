from .PerlinNoise import PerlinNoise

class OctaveNoise:
    def __init__(self, count, random):
        self.count = count
        self.algorithms = []

        for i in range(0, count):
            self.algorithms.append(PerlinNoise(random))

    def compute(self, x, z):
        result = 0
        amp = 1

        for i in range(0, self.count):
            result += self.algorithms[i].compute(x / amp, z / amp) * amp
            amp *= 2

        return result
