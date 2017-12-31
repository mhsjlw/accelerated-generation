static double OctaveNoise_compute(const int c, __global const char *p, const double x, const double z) {
  double result = 0;
  int amp = 1;

  for(int i = 0; i < c; i++) {
    result += PerlinNoise_compute(p + (i * 512), x / amp, z / amp) * amp;
    amp *= 2;
  }

  return result;
}

__kernel void OctaveNoise(const int c, __global const char *p, __global const double *x, __global const double *z, __global double *r) {
  int g = get_global_id(0);

  r[g] = OctaveNoise_compute(c, p, x[g], z[g]);
}
