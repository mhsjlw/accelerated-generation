import numpy

from noise import JavaRandom
import OctaveNoiseInterface
import CombinedNoiseInterface

WORLD_X = 256
WORLD_Y = 64
WORLD_Z = 256
SEED = 1
TOTAL_VALUES = WORLD_X * WORLD_Z

random = JavaRandom(SEED)
numpy.random.seed(SEED)

x_np = numpy.random.rand(TOTAL_VALUES).astype(numpy.float64)
z_np = numpy.random.rand(TOTAL_VALUES).astype(numpy.float64)

result = OctaveNoiseInterface.compute(WORLD_X, WORLD_Z, x_np, z_np, 6, random) # OctaveNoise(6)
print(numpy.sort(result))
# print(result)
print(len(result))

result = OctaveNoiseInterface.compute(WORLD_X, WORLD_Z, x_np, z_np, 8, random) # OctaveNoise(8)
print(numpy.sort(result))
# print(result)
print(len(result))

result = CombinedNoiseInterface.compute(WORLD_X, WORLD_Z, x_np, z_np, 8, 8, random) # CombinedNoise(OctaveNoise(8), OctaveNoise(8))
print(numpy.sort(result))
# print(result)
print(len(result))
