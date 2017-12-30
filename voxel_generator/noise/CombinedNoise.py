class CombinedNoise:
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2

    def compute(self, x, z):
        return self.n1.compute(x + self.n2.compute(x, z), z)
