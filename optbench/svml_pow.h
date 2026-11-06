#ifndef OPTBENCH_SVML_POW_H
#define OPTBENCH_SVML_POW_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Scalar C99 transcription of the float64 AVX-512 "high accuracy" vector
 * power routine __svml_pow8_ha (svml_z0_pow_d_ha.s as vendored by NumPy).
 *
 * optbench_pow(x, y) is exactly the value one lane of the vector routine
 * produces for the same (x, y), including its scalar special-case path.
 * That is what np.power gives element by element on AVX-512 hosts when the
 * exponent is an array, or a scalar other than -1, 0, 0.5, 1 and 2.
 *
 * Requires the default floating-point environment (round to nearest,
 * no flush-to-zero / denormals-are-zero), which is also what the vector
 * routine sees when called from NumPy.
 */
double optbench_pow(double x, double y);

/*
 * out[i] = x[i] ** y as NumPy's float64 power loop computes it for a scalar
 * exponent: y equal to -1, 0, 0.5, 1 or 2 takes NumPy's fast paths (1/x, 1,
 * sqrt(x), x, x*x); every other y gives optbench_pow(x[i], y).
 * out may be the same array as x; other overlaps are not supported.
 */
void optbench_pow_scalar_array(const double *x, double y, double *out, long n);

#ifdef __cplusplus
}
#endif

#endif
