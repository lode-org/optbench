/* ext_pot file exchange for one HarnessCounter calculation.

   eonclient writes from_eon_to_extpot and runs this program. The file is
   three box rows and then one "Z x y z" row per atom. OPTBENCH_POTSERVER
   is host:port. The reply file from_extpot_to_eon is the energy and one
   force row per atom. One run is one force call.

   OPTBENCH_EAM_LIB, when set, is the r2549 Aluminum shared library. This
   program calls force_ itself and sends "charge" so the server only
   counts the evaluation. The default path still forwards the geometry.
*/
#define _POSIX_C_SOURCE 200112L

#include <dlfcn.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

static void die(const char *message) {
    fprintf(stderr, "optbench ext_pot: %s\n", message);
    exit(1);
}

static char *read_all(const char *path, size_t *length) {
    FILE *file = fopen(path, "rb");
    char *data;
    long size;
    if (file == NULL) {
        die("cannot read from_eon_to_extpot");
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        die("cannot size from_eon_to_extpot");
    }
    size = ftell(file);
    if (size < 0) {
        die("cannot size from_eon_to_extpot");
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        die("cannot rewind from_eon_to_extpot");
    }
    data = (char *)malloc((size_t)size + 1);
    if (data == NULL) {
        die("out of memory");
    }
    if (size > 0 && fread(data, 1, (size_t)size, file) != (size_t)size) {
        die("short read of from_eon_to_extpot");
    }
    data[size] = '\0';
    fclose(file);
    *length = (size_t)size;
    return data;
}

static int connect_server(const char *host, const char *port) {
    struct addrinfo hints;
    struct addrinfo *result = NULL;
    struct addrinfo *item;
    int fd = -1;
    int one = 1;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &result) != 0) {
        die("cannot resolve OPTBENCH_POTSERVER");
    }
    for (item = result; item != NULL; item = item->ai_next) {
        fd = socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (connect(fd, item->ai_addr, item->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(result);
    if (fd < 0) {
        die("cannot connect to OPTBENCH_POTSERVER");
    }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

static void send_all(int fd, const char *data, size_t length) {
    size_t sent = 0;
    while (sent < length) {
        ssize_t wrote = send(fd, data + sent, length - sent, 0);
        if (wrote < 0) {
            if (errno == EINTR) {
                continue;
            }
            die("cannot write the calculate request");
        }
        sent += (size_t)wrote;
    }
}

static char *recv_all(int fd, size_t *length) {
    size_t used = 0;
    size_t cap = 4096;
    char *data = (char *)malloc(cap);
    if (data == NULL) {
        die("out of memory");
    }
    for (;;) {
        ssize_t got;
        if (used + 2048 > cap) {
            cap *= 2;
            data = (char *)realloc(data, cap);
            if (data == NULL) {
                die("out of memory");
            }
        }
        got = recv(fd, data + used, cap - used - 1, 0);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            die("cannot read the potential reply");
        }
        if (got == 0) {
            break;
        }
        used += (size_t)got;
    }
    data[used] = '\0';
    *length = used;
    return data;
}

static char *next_line(char **cursor) {
    char *start;
    char *end;
    if (cursor == NULL || *cursor == NULL) {
        return NULL;
    }
    start = *cursor;
    if (*start == '\0') {
        *cursor = NULL;
        return NULL;
    }
    end = strchr(start, '\n');
    if (end == NULL) {
        *cursor = NULL;
        return start;
    }
    *end = '\0';
    *cursor = end + 1;
    return start;
}

static void kill_for_cap(void) {
    pid_t shell = getppid();
    pid_t client = shell;
    char path[64];
    FILE *status;
    snprintf(path, sizeof(path), "/proc/%d/status", (int)shell);
    status = fopen(path, "r");
    if (status != NULL) {
        char status_line[256];
        while (fgets(status_line, sizeof(status_line), status) != NULL) {
            int ppid = 0;
            if (sscanf(status_line, "PPid: %d", &ppid) == 1) {
                client = (pid_t)ppid;
                break;
            }
        }
        fclose(status);
    }
    if (client > 1 && client != shell) {
        kill(client, SIGKILL);
    }
    if (shell > 1) {
        kill(shell, SIGKILL);
    }
}

typedef void (*potinit_fn)(void);
typedef void (*force_fn)(long long *, double *, double *, double *, double *, double *, double *);

static int local_aluminum(char **lines, int atoms, const char *host, const char *port) {
    const char *library = getenv("OPTBENCH_EAM_LIB");
    void *lib;
    potinit_fn potinit;
    force_fn force;
    double *coords;
    double *forces;
    double box[9];
    double energy = 0.0;
    long long count;
    int index;
    int fd;
    int row;
    size_t reply_length = 0;
    char *reply;
    FILE *out;

    if (library == NULL || library[0] == '\0') {
        die("OPTBENCH_EAM_LIB is empty");
    }
    lib = dlopen(library, RTLD_NOW);
    if (lib == NULL) {
        die(dlerror());
    }
    potinit = (potinit_fn)dlsym(lib, "potinit_");
    force = (force_fn)dlsym(lib, "force_");
    if (potinit == NULL || force == NULL) {
        die("Aluminum library is missing potinit_ or force_");
    }
    potinit();
    coords = (double *)calloc((size_t)atoms * 3, sizeof(double));
    forces = (double *)calloc((size_t)atoms * 3, sizeof(double));
    if (coords == NULL || forces == NULL) {
        die("out of memory");
    }
    for (row = 0; row < 3; row++) {
        if (sscanf(lines[row], "%lf %lf %lf", &box[row * 3], &box[row * 3 + 1], &box[row * 3 + 2]) != 3) {
            die("box row must hold three floats");
        }
    }
    for (index = 0; index < atoms; index++) {
        int number = 0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        if (sscanf(lines[index + 3], "%d %lf %lf %lf", &number, &x, &y, &z) != 4) {
            die("atom row must be Z x y z");
        }
        coords[3 * index] = x;
        coords[3 * index + 1] = y;
        coords[3 * index + 2] = z;
    }
    count = atoms;
    force(&count, coords, forces, &energy, &box[0], &box[4], &box[8]);
    fd = connect_server(host, port);
    send_all(fd, "charge\n", 7);
    shutdown(fd, SHUT_WR);
    reply = recv_all(fd, &reply_length);
    close(fd);
    if (reply_length == 0) {
        die("potential server closed without a reply");
    }
    if (strncmp(reply, "ok", 2) != 0) {
        int cap_refusal = strstr(reply, "cap") != NULL;
        fprintf(stderr, "optbench ext_pot: %s\n", reply);
        free(reply);
        if (cap_refusal) {
            kill_for_cap();
        }
        exit(1);
    }
    free(reply);
    out = fopen("from_extpot_to_eon", "w");
    if (out == NULL) {
        die("cannot write from_extpot_to_eon");
    }
    fprintf(out, "%.17g\n", energy);
    for (index = 0; index < atoms; index++) {
        fprintf(out, "%.17g %.17g %.17g\n", forces[3 * index], forces[3 * index + 1], forces[3 * index + 2]);
    }
    free(coords);
    free(forces);
    dlclose(lib);
    if (fclose(out) != 0) {
        die("cannot close from_extpot_to_eon");
    }
    return 0;
}

static void append(char **payload, size_t *used, size_t *cap, const char *text) {
    size_t length = strlen(text);
    if (*used + length + 1 >= *cap) {
        *cap = (*used + length + 64) * 2;
        *payload = (char *)realloc(*payload, *cap);
        if (*payload == NULL) {
            die("out of memory");
        }
    }
    memcpy(*payload + *used, text, length);
    *used += length;
    (*payload)[*used] = '\0';
}

int main(void) {
    const char *endpoint = getenv("OPTBENCH_POTSERVER");
    const char *colon;
    char host[256];
    char port[32];
    char *request;
    char *reply;
    char *payload;
    char *cursor;
    char *line;
    char **lines = NULL;
    size_t request_length = 0;
    size_t reply_length = 0;
    size_t payload_used = 0;
    size_t payload_cap = 0;
    size_t line_count = 0;
    size_t line_cap = 0;
    size_t index;
    int atoms;
    int fd;
    int reply_line = 0;
    int force_rows = 0;
    int cap_refusal = 0;
    FILE *out;
    char count_line[64];

    if (endpoint == NULL || (colon = strrchr(endpoint, ':')) == NULL) {
        die("OPTBENCH_POTSERVER must be host:port");
    }
    if ((size_t)(colon - endpoint) == 0 || (size_t)(colon - endpoint) >= sizeof(host)) {
        die("OPTBENCH_POTSERVER host is empty");
    }
    memcpy(host, endpoint, (size_t)(colon - endpoint));
    host[colon - endpoint] = '\0';
    if (strlen(colon + 1) == 0 || strlen(colon + 1) >= sizeof(port)) {
        die("OPTBENCH_POTSERVER port is empty");
    }
    memcpy(port, colon + 1, strlen(colon + 1) + 1);

    request = read_all("from_eon_to_extpot", &request_length);
    cursor = request;
    while ((line = next_line(&cursor)) != NULL) {
        char *end = line + strlen(line);
        char *tab;
        while (end > line && (end[-1] == '\r' || end[-1] == ' ' || end[-1] == '\t')) {
            *--end = '\0';
        }
        if (line[0] == '\0') {
            continue;
        }
        for (tab = line; *tab != '\0'; tab++) {
            if (*tab == '\t') {
                *tab = ' ';
            }
        }
        if (line_count == line_cap) {
            line_cap = line_cap == 0 ? 64 : line_cap * 2;
            lines = (char **)realloc(lines, line_cap * sizeof(char *));
            if (lines == NULL) {
                die("out of memory");
            }
        }
        lines[line_count++] = line;
    }
    if (line_count < 4) {
        die("ext_pot request is shorter than a box plus one atom");
    }
    atoms = (int)line_count - 3;
    if (getenv("OPTBENCH_EAM_LIB") != NULL) {
        int status = local_aluminum(lines, atoms, host, port);
        free(lines);
        free(request);
        return status;
    }

    payload_cap = request_length + 64;
    payload = (char *)malloc(payload_cap);
    if (payload == NULL) {
        die("out of memory");
    }
    payload[0] = '\0';
    snprintf(count_line, sizeof(count_line), "calculate\n%d\n", atoms);
    append(&payload, &payload_used, &payload_cap, count_line);
    for (index = 0; index < 3; index++) {
        append(&payload, &payload_used, &payload_cap, lines[index]);
        append(&payload, &payload_used, &payload_cap, index + 1 < 3 ? " " : "\n");
    }
    for (index = 3; index < line_count; index++) {
        append(&payload, &payload_used, &payload_cap, lines[index]);
        append(&payload, &payload_used, &payload_cap, "\n");
    }
    free(lines);
    free(request);

    fd = connect_server(host, port);
    send_all(fd, payload, payload_used);
    shutdown(fd, SHUT_WR);
    free(payload);
    reply = recv_all(fd, &reply_length);
    close(fd);
    if (reply_length == 0) {
        die("potential server closed without a reply");
    }
    cap_refusal = strstr(reply, "cap") != NULL;

    out = fopen("from_extpot_to_eon", "w");
    if (out == NULL) {
        die("cannot write from_extpot_to_eon");
    }
    cursor = reply;
    while ((line = next_line(&cursor)) != NULL) {
        if (line[0] == '\0') {
            continue;
        }
        if (reply_line == 0) {
            if (strcmp(line, "ok") != 0) {
                pid_t shell;
                pid_t client;
                fclose(out);
                unlink("from_extpot_to_eon");
                fprintf(stderr, "optbench ext_pot: %s\n", line);
                /* system() ignores the bridge status. A harness cap must
                   stop the client instead of leaving the previous forces. */
                if (cap_refusal) {
                    shell = getppid();
                    client = shell;
                    {
                        char path[64];
                        FILE *status;
                        snprintf(path, sizeof(path), "/proc/%d/status", (int)shell);
                        status = fopen(path, "r");
                        if (status != NULL) {
                            char status_line[256];
                            while (fgets(status_line, sizeof(status_line), status) != NULL) {
                                int ppid = 0;
                                if (sscanf(status_line, "PPid: %d", &ppid) == 1) {
                                    client = (pid_t)ppid;
                                    break;
                                }
                            }
                            fclose(status);
                        }
                    }
                    if (client > 1 && client != shell) {
                        kill(client, SIGKILL);
                    }
                    if (shell > 1) {
                        kill(shell, SIGKILL);
                    }
                }
                free(reply);
                exit(1);
            }
        } else if (reply_line >= 3) {
            fprintf(out, "%s\n", line);
            if (reply_line > 3) {
                force_rows += 1;
            }
        }
        reply_line += 1;
    }
    free(reply);
    if (fclose(out) != 0) {
        die("cannot close from_extpot_to_eon");
    }
    if (force_rows != atoms) {
        die("potential reply has the wrong number of force rows");
    }
    return 0;
}
