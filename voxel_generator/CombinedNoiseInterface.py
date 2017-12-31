import numpy
import pyopencl as cl
from os import path
import math

from KernelLoader import KernelLoader
from noise import CombinedNoise, OctaveNoise, JavaRandom

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

def index_1d_as_2d(array, world_x, x, z):
    return array[world_x * x + z]

def compute(world_x, world_z, x_np, z_np, octave_count_1, octave_count_2, random):
    seed = 1
    total_values = world_x * world_z

    p = create_octave_table(random, octave_count_1)
    p = numpy.append(p, create_octave_table(random, octave_count_2))

    results = numpy.empty(total_values, dtype=numpy.float64)

    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    mf = cl.mem_flags
    p_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p)
    x_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
    z_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)

    kernel = KernelLoader([ 'PerlinNoise.cl', 'OctaveNoise.cl', 'CombinedNoise.cl' ])
    program = kernel.build(context)

    r_g = cl.Buffer(context, mf.WRITE_ONLY, total_values * 8) # TOTAL_VALUES * NUMBER_OF_BYTES
    program.CombinedNoise(queue, (total_values,), None, numpy.int32(octave_count_1), numpy.int32(octave_count_2), p_g, x_g, z_g, r_g)

    r_np = numpy.empty(total_values, dtype=numpy.float64)
    cl.enqueue_copy(queue, r_np, r_g)

    return r_np

    # random = JavaRandom(SEED)
    # noise = OctaveNoise(OCTAVE_COUNT, random)
    # all = numpy.zeros(TOTAL_VALUES, dtype=numpy.float64)
    # for i in range(0, TOTAL_VALUES):
    #     all[i] = noise.compute(x_np[i], z_np[i])
    #
    # print(numpy.sort(all))
    # print(len(all))
