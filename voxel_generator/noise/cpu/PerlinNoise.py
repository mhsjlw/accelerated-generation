from .JavaRandom import JavaRandom

import numpy
import math

class PerlinNoise:
    def __init__(self, random):
        self.random = random

        self.permutations = numpy.zeros(512, dtype=numpy.int)
        self.permutations[:256] = numpy.arange(0, 256)

        for i in range(0, 256):
            ind = math.floor(self.random.next_float() * (256 - i)) + i
            val = self.permutations[i]

            self.permutations[i] = self.permutations[ind]
            self.permutations[ind] = val
            self.permutations[i + 256] = self.permutations[i]

    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(self, t, a, b):
        return a + t * (b - a)

    def grad(self, hash, x, y, z):
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else (z if (h != 12 and h != 14) else x)
        return (u if (hash & 1) == 0 else -u) + (v if (hash & 2) == 0 else -v)

    def compute(self, x, z):
        fx = math.floor(x) & 255
        fz = math.floor(z) & 255

        local_x = x - math.floor(x)
        local_z = z - math.floor(z)

        u = self.fade(local_x)
        w = self.fade(local_z)

        a = self.permutations[self.permutations[fx] + fz + 1]
        b = self.permutations[self.permutations[fx] + fz]

        px = self.permutations[self.permutations[fx + 1] + fz + 1]
        pz = self.permutations[self.permutations[px + 1] + fz]

        return self.lerp(0, self.lerp(w, self.lerp(u, self.grad(self.permutations[b], local_x, local_z, 0), self.grad(self.permutations[pz], local_x - 1.0, local_z, 0)), self.lerp(u, self.grad(self.permutations[a], local_x, local_z - 1.0, 0), self.grad(self.permutations[px], local_x - 1.0, local_z - 1.0, 0))), self.lerp(w, self.lerp(u, self.grad(self.permutations[b + 1], local_x, local_z, -1), self.grad(self.permutations[pz + 1], local_x - 1.0, local_z, -1)), self.lerp(u, self.grad(self.permutations[a + 1], local_x, local_z - 1.0, -1), self.grad(self.permutations[px + 1], local_x - 1.0, local_z - 1.0, -1))))
