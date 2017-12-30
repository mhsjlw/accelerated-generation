static double PerlinNoise_compute(const __global char *p, double x, double y) {
  int x_floor = x >= 0 ? (int) x : (int) x - 1;
  int y_floor = y >= 0 ? (int) y : (int) y - 1;

  int X = x_floor & 0xFF;
  int Y = y_floor & 0xFF;

  x -= x_floor;
  y -= y_floor;

  double u = x * x * x * (x * (x * 6 - 15) + 10);
  double v = y * y * y * (y * (y * 6 - 15) + 10);

  int A = p[X] + Y;
  int B = p[X + 1] + Y;

  // Thanks to ClassicalSharp for the `grad` packing
  const int x_flags = 0x46552222;
  const int y_flags = 0x2222550A;

  int hash = (p[p[A]] & 0xF) << 1;

  double g22 = (((x_flags >> hash) & 3) - 1) * x + (((y_flags >> hash) & 3) - 1) * y;

  hash = (p[p[B]] & 0xF) << 1;

  double g12 = (((x_flags >> hash) & 3) - 1) * (x - 1) + (((y_flags >> hash) & 3) - 1) * y;

  double c1 = g22 + u * (g12 - g22);

  hash = (p[p[A + 1]] & 0xF) << 1;
  double g21 = (((x_flags >> hash) & 3) - 1) * x + (((y_flags >> hash) & 3) - 1) * (y - 1);

  hash = (p[p[B + 1]] & 0xF) << 1;
  double g11 = (((x_flags >> hash) & 3) - 1) * (x - 1) + (((y_flags >> hash) & 3) - 1) * (y - 1);

  double c2 = g21 + u * (g11 - g21);

  return c1 + v * (c2 - c1);
}

__kernel void OctaveNoise_compute(const int c, __global const char *p, __global const double *x, __global const double *z, __global double *r) {
  int g = get_global_id(0);

  double result = 0;
  int amp = 1;

  for(int i = 0; i < c; i++) {
    result += PerlinNoise_compute(p + (i * 512), x[g] / amp, z[g] / amp) * amp;
    amp *= 2;
  }

  r[g] = result;
}
