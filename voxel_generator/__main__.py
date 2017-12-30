import numpy
import pyopencl as cl
from os import path
import math

from KernelLoader import KernelLoader
from noise import CombinedNoise, OctaveNoise, JavaRandom

WORLD_X = 256
WORLD_Y = 64
WORLD_Z = 256
SEED = 1

TOTAL_VALUES = WORLD_X * WORLD_Z

random = JavaRandom(SEED)
numpy.random.seed(SEED)

def create_permutation_table(random):
    permutations = numpy.zeros(512, dtype=numpy.int8)
    permutations[:256] = numpy.arange(0, 256)

    for i in range(0, 256):
        ind = math.floor(random.next_float() * (256 - i)) + i
        val = permutations[i]

        permutations[i] = permutations[ind]
        permutations[ind] = val
        permutations[i + 256] = permutations[i]

    return permutations

def create_octave_table(random, octaves):
    p = numpy.zeros((8, 512), dtype=numpy.int8)
    for i in range(0, octaves):
        p[i] = create_permutation_table(random)

    return p.flatten()

def index_1d_as_2d(array, x, z):
    return array[WORLD_X * x + z]

OCTAVE_COUNT = 8
p = create_octave_table(random, OCTAVE_COUNT)

results = numpy.empty(TOTAL_VALUES, dtype=numpy.float64)

context = cl.create_some_context()
queue = cl.CommandQueue(context)

x_np = numpy.random.rand(TOTAL_VALUES).astype(numpy.float64)
z_np = numpy.random.rand(TOTAL_VALUES).astype(numpy.float64)

mf = cl.mem_flags
p_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p)
x_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
z_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)

kernel = KernelLoader([ 'PerlinNoise.cl' ])
program = kernel.build(context)

r_g = cl.Buffer(context, mf.WRITE_ONLY, TOTAL_VALUES * 8) # TOTAL_VALUES * NUMBER_OF_BYTES
program.OctaveNoise_compute(queue, (TOTAL_VALUES,), None, numpy.int32(OCTAVE_COUNT), p_g, x_g, z_g, r_g)

r_np = numpy.empty(TOTAL_VALUES, dtype=numpy.float64)
cl.enqueue_copy(queue, r_np, r_g)
print(numpy.sort(r_np))
print(len(r_np))

# --------------------------------------------------------------------------------------------------------------
# random = JavaRandom(SEED)
# noise = OctaveNoise(OCTAVE_COUNT, random)
# all = numpy.zeros(TOTAL_VALUES, dtype=numpy.float64)
# for i in range(0, TOTAL_VALUES):
#     all[i] = noise.compute(x_np[i], z_np[i])
#
# print(numpy.sort(all))
# print(len(all))
