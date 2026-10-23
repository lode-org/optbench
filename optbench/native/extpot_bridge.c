#define _POSIX_C_SOURCE 200809L

/* File bridge from eonclient ext_pot onto the harness potential server.

   eOn writes from_eon_to_extpot and runs this program once per force
   call. The calculation stays in PotServer. This process only forwards
   the geometry and writes from_extpot_to_eon.
*/

#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <math.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/tcp.h>
#include <unistd.h>

static void fail(const char *message) {
    fprintf(stderr, "optbench ext_pot: %s\n", message);
    exit(1);
}

static void format_double(char *buf, size_t size, double value) {
    int precision;
    if (!isfinite(value)) {
        snprintf(buf, size, "%.17g", value);
        return;
    }
    for (precision = 15; precision <= 17; precision++) {
        char *end = NULL;
        double back;
        snprintf(buf, size, "%.*g", precision, value);
        errno = 0;
        back = strtod(buf, &end);
        if (end != buf && end != NULL && *end == '\0' && back == value) {
            return;
        }
    }
    snprintf(buf, size, "%.17g", value);
}

static char *read_file(const char *path, size_t *length) {
    FILE *file = fopen(path, "r");
    char *data;
    size_t cap = 4096;
    size_t used = 0;
    if (file == NULL) {
        fail("could not open from_eon_to_extpot");
    }
    data = malloc(cap);
    if (data == NULL) {
        fail("out of memory");
    }
    for (;;) {
        size_t got;
        if (used + 2048 > cap) {
            cap *= 2;
            data = realloc(data, cap);
            if (data == NULL) {
                fail("out of memory");
            }
        }
        got = fread(data + used, 1, cap - used - 1, file);
        used += got;
        if (got == 0) {
            break;
        }
    }
    if (ferror(file)) {
        fail("could not read from_eon_to_extpot");
    }
    fclose(file);
    data[used] = '\0';
    *length = used;
    return data;
}

static double *parse_numbers(const char *text, size_t *count) {
    const char *cursor = text;
    double *values = NULL;
    size_t used = 0;
    size_t cap = 0;
    while (*cursor != '\0') {
        char *end = NULL;
        double value;
        while (isspace((unsigned char)*cursor)) {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }
        errno = 0;
        value = strtod(cursor, &end);
        if (end == cursor) {
            fail("ext_pot request has a non-numeric token");
        }
        if (used == cap) {
            cap = cap == 0 ? 128 : cap * 2;
            values = realloc(values, cap * sizeof(double));
            if (values == NULL) {
                fail("out of memory");
            }
        }
        values[used++] = value;
        cursor = end;
    }
    *count = used;
    return values;
}

static void append_text(char **buffer, size_t *used, size_t *cap, const char *text) {
    size_t length = strlen(text);
    if (*used + length + 1 > *cap) {
        while (*used + length + 1 > *cap) {
            *cap = *cap == 0 ? 256 : *cap * 2;
        }
        *buffer = realloc(*buffer, *cap);
        if (*buffer == NULL) {
            fail("out of memory");
        }
    }
    memcpy(*buffer + *used, text, length);
    *used += length;
    (*buffer)[*used] = '\0';
}

static char *server_request(const double *values, size_t count, size_t *length) {
    size_t atoms;
    size_t index;
    char *buffer = NULL;
    size_t used = 0;
    size_t cap = 0;
    char line[256];
    if (count < 13 || (count - 9) % 4 != 0) {
        fail("ext_pot request is not a box plus Z x y z rows");
    }
    atoms = (count - 9) / 4;
    if (atoms < 1 || atoms > 100000) {
        fail("ext_pot atom count is out of range");
    }
    snprintf(line, sizeof(line), "calculate\n%zu\n", atoms);
    append_text(&buffer, &used, &cap, line);
    for (index = 0; index < 9; index++) {
        format_double(line, sizeof(line), values[index]);
        append_text(&buffer, &used, &cap, line);
        append_text(&buffer, &used, &cap, index + 1 == 9 ? "\n" : " ");
    }
    for (index = 0; index < atoms; index++) {
        size_t base = 9 + 4 * index;
        char number[256];
        char x[64];
        char y[64];
        char z[64];
        long atomic = (long)values[base];
        int wrote;
        format_double(x, sizeof(x), values[base + 1]);
        format_double(y, sizeof(y), values[base + 2]);
        format_double(z, sizeof(z), values[base + 3]);
        wrote = snprintf(number, sizeof(number), "%ld %s %s %s\n", atomic, x, y, z);
        if (wrote < 0 || (size_t)wrote >= sizeof(number)) {
            fail("ext_pot atom line does not fit");
        }
        append_text(&buffer, &used, &cap, number);
    }
    *length = used;
    return buffer;
}

static int connect_server(const char *endpoint) {
    char host[256];
    const char *colon;
    const char *port;
    struct addrinfo hints;
    struct addrinfo *result = NULL;
    struct addrinfo *item;
    int fd = -1;
    size_t host_len;
    colon = strrchr(endpoint, ':');
    if (colon == NULL || colon == endpoint || colon[1] == '\0') {
        fail("OPTBENCH_POTSERVER must be host:port");
    }
    host_len = (size_t)(colon - endpoint);
    if (host_len >= sizeof(host)) {
        fail("OPTBENCH_POTSERVER host is too long");
    }
    memcpy(host, endpoint, host_len);
    host[host_len] = '\0';
    port = colon + 1;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &result) != 0) {
        fail("could not resolve OPTBENCH_POTSERVER");
    }
    for (item = result; item != NULL; item = item->ai_next) {
        int one = 1;
        struct timeval timeout;
        fd = socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (fd < 0) {
            continue;
        }
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        timeout.tv_sec = 120;
        timeout.tv_usec = 0;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        if (connect(fd, item->ai_addr, item->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(result);
    if (fd < 0) {
        fail("could not connect to the potential server");
    }
    return fd;
}

static void send_all(int fd, const char *data, size_t length) {
    size_t sent = 0;
    while (sent < length) {
        ssize_t step = send(fd, data + sent, length - sent, 0);
        if (step < 0) {
            fail("could not write to the potential server");
        }
        sent += (size_t)step;
    }
}

static char *recv_all(int fd, size_t *length) {
    char *data = NULL;
    size_t used = 0;
    size_t cap = 0;
    for (;;) {
        ssize_t step;
        if (used + 2048 > cap) {
            cap = cap == 0 ? 4096 : cap * 2;
            data = realloc(data, cap);
            if (data == NULL) {
                fail("out of memory");
            }
        }
        step = recv(fd, data + used, cap - used - 1, 0);
        if (step < 0) {
            fail("could not read the potential server");
        }
        if (step == 0) {
            break;
        }
        used += (size_t)step;
    }
    if (data == NULL) {
        data = calloc(1, 1);
        if (data == NULL) {
            fail("out of memory");
        }
    }
    data[used] = '\0';
    *length = used;
    return data;
}

static void write_reply(const char *text) {
    char *copy = strdup(text);
    char *save = NULL;
    char *line;
    FILE *out;
    size_t index = 0;
    size_t atoms;
    if (copy == NULL) {
        fail("out of memory");
    }
    line = strtok_r(copy, "\n", &save);
    if (line == NULL || strcmp(line, "ok") != 0) {
        fail(line != NULL && strcmp(line, "error") == 0 ? "potential server error" : "bad potential server reply");
    }
    if (strtok_r(NULL, "\n", &save) == NULL || strtok_r(NULL, "\n", &save) == NULL) {
        fail("potential server reply is short");
    }
    line = strtok_r(NULL, "\n", &save);
    if (line == NULL) {
        fail("potential server reply has no energy");
    }
    out = fopen("from_extpot_to_eon", "w");
    if (out == NULL) {
        fail("could not write from_extpot_to_eon");
    }
    if (fprintf(out, "%s\n", line) < 0) {
        fail("could not write from_extpot_to_eon");
    }
    atoms = 0;
    while ((line = strtok_r(NULL, "\n", &save)) != NULL) {
        if (fprintf(out, "%s\n", line) < 0) {
            fail("could not write from_extpot_to_eon");
        }
        atoms++;
    }
    if (fclose(out) != 0) {
        fail("could not write from_extpot_to_eon");
    }
    free(copy);
    (void)index;
    (void)atoms;
}

int main(void) {
    const char *endpoint;
    char *request_text;
    size_t request_length = 0;
    double *values;
    size_t count = 0;
    char *outbound;
    size_t outbound_length = 0;
    char *inbound;
    size_t inbound_length = 0;
    int fd;
    setlocale(LC_NUMERIC, "C");
    endpoint = getenv("OPTBENCH_POTSERVER");
    if (endpoint == NULL || strchr(endpoint, ':') == NULL) {
        fail("OPTBENCH_POTSERVER must be host:port");
    }
    request_text = read_file("from_eon_to_extpot", &request_length);
    values = parse_numbers(request_text, &count);
    free(request_text);
    outbound = server_request(values, count, &outbound_length);
    free(values);
    fd = connect_server(endpoint);
    send_all(fd, outbound, outbound_length);
    shutdown(fd, SHUT_WR);
    free(outbound);
    inbound = recv_all(fd, &inbound_length);
    close(fd);
    if (inbound_length == 0) {
        fail("potential server closed without a reply");
    }
    write_reply(inbound);
    free(inbound);
    return 0;
}
