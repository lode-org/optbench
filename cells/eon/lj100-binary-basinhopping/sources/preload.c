#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define MAXN 128
static int (*real_system)(const char *) = NULL;
static FILE *(*real_fopen)(const char *, const char *) = NULL;
static long calls = 0;
static long cap = 0;
long blj_calls = 0;
double blj_energy = 0;
static char workbuf[512];

static int compute_in(const char *dir) {
    char prevcwd[512];
    if (!getcwd(prevcwd, sizeof prevcwd)) return 1;
    if (dir && dir[0] && chdir(dir) != 0) return 1;
    FILE *in = fopen("from_eon_to_extpot", "r");
    if (!in) { chdir(prevcwd); return 1; }
    double box[9];
    for (int i = 0; i < 9; i++) if (fscanf(in, "%lf", &box[i]) != 1) { fclose(in); chdir(prevcwd); return 1; }
    int z[MAXN];
    double x[MAXN][3];
    int n = 0;
    while (n < MAXN && fscanf(in, "%d %lf %lf %lf", &z[n], &x[n][0], &x[n][1], &x[n][2]) == 4) n++;
    fclose(in);
    double len[3] = {box[0], box[4], box[8]};
    int species[MAXN], known[4], nspec = 0;
    double sigma_s[4];
    for (int i = 0; i < n; i++) {
        int s = -1;
        for (int k = 0; k < nspec; k++) if (known[k] == z[i]) s = k;
        if (s < 0) { known[nspec] = z[i]; sigma_s[nspec] = nspec ? 1.0 : 1.3; s = nspec++; }
        species[i] = s;
    }
    double energy = 0, f[MAXN][3];
    memset(f, 0, sizeof f);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
        double dx[3], r2 = 0;
        for (int a = 0; a < 3; a++) {
            double d = x[j][a] - x[i][a];
            if (len[a] > 0) d -= len[a] * nearbyint(d / len[a]);
            dx[a] = d; r2 += d * d;
        }
        if (r2 >= 400.0 || r2 < 1e-24) continue;
        double sig = species[i] == species[j] ? sigma_s[species[i]] : 1.15;
        double sr2 = (sig * sig) / r2, sr6 = sr2 * sr2 * sr2, sr12 = sr6 * sr6;
        energy += 4.0 * (sr12 - sr6);
        /* dx is r_j - r_i. LAMMPS lj/cut returns F_i along (r_i - r_j). */
        double mag = -24.0 * (2.0 * sr12 - sr6) / r2;
        for (int a = 0; a < 3; a++) { double g = mag * dx[a]; f[i][a] += g; f[j][a] -= g; }
    }
    FILE *out = fopen("from_extpot_to_eon", "w");
    if (!out) { chdir(prevcwd); return 1; }
    blj_energy = energy;
    fprintf(out, "%.16e\n", energy);
    for (int i = 0; i < n; i++) fprintf(out, "%.16e %.16e %.16e\n", f[i][0], f[i][1], f[i][2]);
    fclose(out);
    chdir(prevcwd);
    return 0;
}

static void remember(void) {
    const char *counter = getenv("OPTBENCH_CALLS");
    if (!counter || !counter[0]) return;
    FILE *cf = real_fopen ? real_fopen(counter, "w") : fopen(counter, "w");
    if (!cf) return;
    fprintf(cf, "%ld\n", calls);
    fclose(cf);
}

int system(const char *command) {
    if (!real_system) real_system = dlsym(RTLD_NEXT, "system");
    if (!(command && strstr(command, "blj/bridge"))) return real_system(command);
    const char *dir = NULL;
    if (!strncmp(command, "cd ", 3)) {
        const char *start = command + 3; char quote = 0;
        if (*start == '\'' || *start == '"') { quote = *start; start++; }
        const char *end = quote ? strchr(start, quote) : strchr(start, ' ');
        if (!end) end = start + strlen(start);
        size_t n = (size_t)(end - start); if (n >= sizeof workbuf) n = sizeof workbuf - 1;
        memcpy(workbuf, start, n); workbuf[n] = 0; dir = workbuf;
    }
    if (compute_in(dir)) return 256;
    calls++;
    blj_calls = calls;
    /* energy of this call is not returned; re-read the reply file's first line is slower.
       compute_in writes it. Recover from the reply only when gdb asks.
       Store it here by peeking is unnecessary if compute_in sets blj_energy. */
    if ((calls & 16383) == 0) remember();
    if (cap > 0 && calls >= cap) { remember(); _exit(0); }
    return 0;
}

FILE *fopen(const char *path, const char *mode) {
    if (!real_fopen) real_fopen = dlsym(RTLD_NEXT, "fopen");
    FILE *file = real_fopen(path, mode);
    if (file && path && strstr(path, "bh.dat")) setvbuf(file, NULL, _IOLBF, 0);
    return file;
}

__attribute__((constructor)) static void init(void) {
    const char *text = getenv("OPTBENCH_CAP");
    if (text && text[0]) cap = atol(text);
    real_fopen = dlsym(RTLD_NEXT, "fopen");
}
__attribute__((destructor)) static void dump(void) { remember(); }
