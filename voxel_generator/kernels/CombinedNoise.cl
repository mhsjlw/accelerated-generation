__kernel void CombinedNoise(const int c1, const int c2, __global const char *p, __global const double *x, __global const double *z, __global double *r) {
  int g = get_global_id(0);

  double first = OctaveNoise_compute(c1, p, x[g], z[g]);
  double second = OctaveNoise_compute(c2, p + (c1 * 512), x[g] + first, z[g]);

  r[g] = second;
}
