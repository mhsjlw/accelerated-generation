import pyopencl as cl

import os
import inspect

KERNEL_PATH = os.path.join(os.path.dirname(os.path.abspath(inspect.stack()[0][1])), 'kernels')

class KernelLoader:
    def __init__(self, names):
        self.names = names
        self.kernel = ""

        for name in self.names:
            with open(os.path.join(KERNEL_PATH, name)) as file:
                for line in file:
                    self.kernel += line

    def build(self, context):
        return cl.Program(context, self.kernel).build()
