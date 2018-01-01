class JavaRandom:
    def __init__(self, seed: int):
        self.value = 0x5DEECE66D
        self.mask = (1 << 48) - 1

        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        self.seed = (seed ^ self.value) & self.mask

    def next(self, min: int, max: int) -> int:
        return min + self.next(max - min)

    def next(self,n: int) -> int:
        if (n & -n) == n:
            self.seed = (self.seed * self.value + 0xB) & self.mask
            raw = self.seed >> (48 - 31)
            return (n * raw) >> 31

        bits, val = 0
        while bits - val + (n - 1) < 0:
            self.seed = (self.seed * self.value + 0xB) & self.mask
            bits = self.seed >> (48 - 31)
            val = bits % n

        return val

    def next_float(self) -> float:
        self.seed = (self.seed * self.value + 0xB) & self.mask
        raw = self.seed >> (48 - 24)
        return raw / (1 << 24)
