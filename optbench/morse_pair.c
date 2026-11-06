/* LAMMPS morse/smooth/linear pair sum.

   Energy and dE/dx follow optbench.surfaces: the Morse value minus its
   value at the cutoff, minus (r - rc) times the Morse derivative at the
   cutoff. A periodic axis uses round-half-to-even.

   The result must equal optbench.surfaces._morse_numpy bit for bit, so
   every operation is the reference's, in its order:
   - pairs i < j in row-major order, with r2 = (dx*dx + dz*dz) + dy*dy;
   - exp is optbench_exp, the routine the reference calls;
   - the energy is 0.0 plus NumPy's pairwise sum of the pair terms;
   - the gradient is two np.add.at passes, every pair's i row first,
     then every pair's j row.
   The build flags keep each double operation a single IEEE operation:
   no FMA contraction and no reassociation.
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "svml_exp.h"

/* NumPy's DOUBLE_pairwise_sum, numpy/_core/src/umath/loops_utils.h.src. */
static double pairwise_sum(const double *a, long n)
{
    if (n < 8) {
        double res = -0.0;
        for (long i = 0; i < n; i++) {
            res += a[i];
        }
        return res;
    }
    if (n <= 128) {
        double r[8];
        long i;
        for (int k = 0; k < 8; k++) {
            r[k] = a[k];
        }
        for (i = 8; i < n - (n % 8); i += 8) {
            for (int k = 0; k < 8; k++) {
                r[k] += a[i + k];
            }
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) {
            res += a[i];
        }
        return res;
    }
    long n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

static double fold(double delta, int periodic, double length)
{
    if (periodic && length > 0.0) {
        return delta - length * nearbyint(delta / length);
    }
    return delta;
}

/* Returns 0, or -1 when the pair buffers cannot be allocated. */
int morse_pair_energy(
    const double *positions,
    double *gradient,
    int count,
    double depth,
    double alpha,
    double r0,
    double cutoff,
    double cell_x,
    double cell_y,
    double cell_z,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    double *energy
) {
    const double cutoff2 = cutoff * cutoff;
    const double at_cut = optbench_exp(-alpha * (cutoff - r0));
    const double offset = depth * (at_cut * at_cut - 2.0 * at_cut);
    const double der_cut = -2.0 * alpha * depth * (at_cut * at_cut - at_cut);
    const long pairs = count > 1 ? (long)count * (long)(count - 1) / 2 : 1;
    double *terms = malloc((size_t)pairs * sizeof(double));
    double *vectors = malloc((size_t)pairs * 3u * sizeof(double));
    int *right = malloc((size_t)pairs * sizeof(int));
    long kept = 0;

    if (terms == NULL || vectors == NULL || right == NULL) {
        free(terms);
        free(vectors);
        free(right);
        return -1;
    }
    memset(gradient, 0, (size_t)count * 3u * sizeof(double));
    for (int i = 0; i < count; i++) {
        const double ix = positions[3 * i];
        const double iy = positions[3 * i + 1];
        const double iz = positions[3 * i + 2];
        for (int j = i + 1; j < count; j++) {
            const double dx = fold(ix - positions[3 * j], periodic_x, cell_x);
            const double dy = fold(iy - positions[3 * j + 1], periodic_y, cell_y);
            const double dz = fold(iz - positions[3 * j + 2], periodic_z, cell_z);
            const double r2 = (dx * dx + dz * dz) + dy * dy;
            if (!(r2 > 0.0 && r2 < cutoff2)) {
                continue;
            }
            const double radius = sqrt(r2);
            const double decay = optbench_exp(-alpha * (radius - r0));
            const double morse = depth * (decay * decay - 2.0 * decay);
            terms[kept] = morse - offset - (radius - cutoff) * der_cut;
            const double slope = -2.0 * alpha * depth * (decay * decay - decay) - der_cut;
            const double scale = slope / radius;
            double *vector = vectors + 3 * kept;
            vector[0] = scale * dx;
            vector[1] = scale * dy;
            vector[2] = scale * dz;
            gradient[3 * i] += vector[0];
            gradient[3 * i + 1] += vector[1];
            gradient[3 * i + 2] += vector[2];
            right[kept] = j;
            kept++;
        }
    }
    for (long k = 0; k < kept; k++) {
        const int j = right[k];
        gradient[3 * j] += -vectors[3 * k];
        gradient[3 * j + 1] += -vectors[3 * k + 1];
        gradient[3 * j + 2] += -vectors[3 * k + 2];
    }
    *energy = kept > 0 ? 0.0 + pairwise_sum(terms, kept) : 0.0;
    free(terms);
    free(vectors);
    free(right);
    return 0;
}
