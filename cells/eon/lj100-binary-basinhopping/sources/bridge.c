/* One ext_pot force call. Binary LJ as the archive in.lammps file:
   first species sigma 1.3, second sigma 1.0, cross 1.15, epsilon 1,
   cutoff 20, no tail correction. Counts every call. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 128

static void die(const char *m) {
    fprintf(stderr, "blj: %s\n", m);
    exit(1);
}

int main(void) {
    FILE *in = fopen("from_eon_to_extpot", "r");
    if (!in) die("no input");
    double box[9];
    for (int i = 0; i < 9; i++) {
        if (fscanf(in, "%lf", &box[i]) != 1) die("box");
    }
    int z[MAXN];
    double x[MAXN][3];
    int n = 0;
    while (n < MAXN && fscanf(in, "%d %lf %lf %lf", &z[n], &x[n][0], &x[n][1], &x[n][2]) == 4) {
        n++;
    }
    fclose(in);
    if (n < 1) die("no atoms");
    double len[3] = {box[0], box[4], box[8]};
    int species[MAXN];
    int known[4];
    double sigma_s[4];
    int nspec = 0;
    for (int i = 0; i < n; i++) {
        int s = -1;
        for (int k = 0; k < nspec; k++) if (known[k] == z[i]) s = k;
        if (s < 0) {
            if (nspec >= 4) die("too many species");
            known[nspec] = z[i];
            sigma_s[nspec] = (nspec == 0) ? 1.3 : 1.0;
            s = nspec++;
        }
        species[i] = s;
    }
    double energy = 0.0;
    double f[MAXN][3];
    memset(f, 0, sizeof(f));
    const double rc = 20.0;
    const double rc2 = rc * rc;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx[3];
            double r2 = 0.0;
            for (int a = 0; a < 3; a++) {
                double d = x[j][a] - x[i][a];
                if (len[a] > 0.0) {
                    d -= len[a] * nearbyint(d / len[a]);
                }
                dx[a] = d;
                r2 += d * d;
            }
            if (r2 >= rc2 || r2 < 1e-24) continue;
            double sig = (species[i] == species[j]) ? sigma_s[species[i]] : 1.15;
            double sr2 = (sig * sig) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            energy += 4.0 * (sr12 - sr6);
            /* dx is r_j - r_i. LAMMPS lj/cut metal units return the physical
               force F_i = (r_i - r_j) * 24 * (2*sr12 - sr6) / r^2.
               EON stores that vector and steps along it. */
            double mag = -24.0 * (2.0 * sr12 - sr6) / r2;
            for (int a = 0; a < 3; a++) {
                double g = mag * dx[a];
                f[i][a] += g;
                f[j][a] -= g;
            }
        }
    }
    FILE *out = fopen("from_extpot_to_eon", "w");
    if (!out) die("no output");
    fprintf(out, "%.16e\n", energy);
    for (int i = 0; i < n; i++) {
        fprintf(out, "%.16e %.16e %.16e\n", f[i][0], f[i][1], f[i][2]);
    }
    fclose(out);
    const char *counter = getenv("OPTBENCH_CALLS");
    if (counter && counter[0]) {
        FILE *cf = fopen(counter, "r+");
        long value = 0;
        if (!cf) {
            cf = fopen(counter, "w+");
            if (!cf) die("counter");
        } else if (fscanf(cf, "%ld", &value) != 1) {
            value = 0;
        }
        rewind(cf);
        fprintf(cf, "%ld\n", value + 1);
        fflush(cf);
        fclose(cf);
    }
    return 0;
}
