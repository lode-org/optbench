#ifndef OPTBENCH_SVML_EXP_H
#define OPTBENCH_SVML_EXP_H

#ifdef __cplusplus
extern "C" {
#endif

double optbench_exp(double x);
void optbench_exp_array(const double *x, double *out, long n);

#ifdef __cplusplus
}
#endif

#endif
