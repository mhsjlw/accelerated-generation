import struct

from noise import JavaRandom
from NormalGenerator import NormalGenerator

WORLD_X = 256
WORLD_Y = 64
WORLD_Z = 256
SEED = 1
TOTAL_VALUES = WORLD_X * WORLD_Z

random = JavaRandom(SEED)
data = NormalGenerator(WORLD_X, WORLD_Y, WORLD_Z, random).compute()

header = struct.pack('>i', WORLD_X * WORLD_Y * WORLD_Z)

file = open('level.dat', 'wb')
file.write(header)
file.write(data)
file.close()

# import numpy
#
# from noise import JavaRandom
# from noise.gpu import combined_noise_compute, octave_noise_compute
#
# WORLD_X = 256
# WORLD_Y = 64
# WORLD_Z = 256
# SEED = 1
# TOTAL_VALUES = WORLD_X * WORLD_Z
#
# random = JavaRandom(SEED)
# numpy.random.seed(SEED)
#
# x_np = numpy.random.rand(TOTAL_VALUES).astype(numpy.float64)
# z_np = numpy.random.rand(TOTAL_VALUES).astype(numpy.float64)
#
# result = octave_noise_compute(WORLD_X, WORLD_Z, x_np, z_np, 6, random) # OctaveNoise(6)
# print(numpy.sort(result))
# print(len(result))
#
# result = octave_noise_compute(WORLD_X, WORLD_Z, x_np, z_np, 8, random) # OctaveNoise(8)
# print(numpy.sort(result))
# print(len(result))
#
# result = combined_noise_compute(WORLD_X, WORLD_Z, x_np, z_np, 8, 8, random) # CombinedNoise(OctaveNoise(8), OctaveNoise(8))
# print(numpy.sort(result))
# print(len(result))
