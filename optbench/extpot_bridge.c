/* ext_pot file exchange for one HarnessCounter calculation.

   eonclient writes from_eon_to_extpot and runs this program. The file is
   three box rows and then one "Z x y z" row per atom. OPTBENCH_POTSERVER
   is host:port. The reply file from_extpot_to_eon is the energy and one
   force row per atom. One run is one force call.

   Every eOn family adds one to its own force-call counter only after
   ExtPot::force returns, so that counter equals the exchanges this client
   completed. optbench-extpot-calls keeps that tally for the client. When
   the harness refuses a call at its cap the client is killed before it
   can report anything, so optbench-extpot-cap first records the tally and
   the refusal. Both files are in the directory the client runs in.
*/
#define _POSIX_C_SOURCE 200112L

#include <errno.h>
#include <fcntl.h>
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

#define TALLY_FILE "optbench-extpot-calls"
#define CAP_FILE "optbench-extpot-cap"

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

/* The PPid of pid, or -1 when /proc does not say. */
static pid_t parent_of(pid_t pid) {
    char path[64];
    FILE *status;
    pid_t parent = -1;
    snprintf(path, sizeof(path), "/proc/%d/status", (int)pid);
    status = fopen(path, "r");
    if (status != NULL) {
        char status_line[256];
        while (fgets(status_line, sizeof(status_line), status) != NULL) {
            int ppid = 0;
            if (sscanf(status_line, "PPid: %d", &ppid) == 1) {
                parent = (pid_t)ppid;
                break;
            }
        }
        fclose(status);
    }
    return parent;
}

/* Field 22 of /proc/<pid>/stat, the start time in clock ticks after boot
   (proc(5)). A pid and its start time name one process even when the pid
   is reused. The command of a user process is at most 16 bytes
   (TASK_COMM_LEN) in parentheses and every other field up to 22 is at
   most 20 digits and a space, so field 22 ends within 21 * 21 + 18 = 459
   of the 512 bytes read. Zero when /proc does not say. */
static unsigned long long start_time(pid_t pid) {
    char path[64];
    char text[512];
    FILE *stat;
    size_t got;
    char *field;
    int index;
    snprintf(path, sizeof(path), "/proc/%d/stat", (int)pid);
    stat = fopen(path, "r");
    if (stat == NULL) {
        return 0;
    }
    got = fread(text, 1, sizeof(text) - 1, stat);
    fclose(stat);
    text[got] = '\0';
    /* Field 2 is the command and may hold spaces or parentheses. */
    field = strrchr(text, ')');
    for (index = 2; index < 22 && field != NULL; index++) {
        field = strchr(field + 1, ' ');
    }
    return field == NULL ? 0 : strtoull(field + 1, NULL, 10);
}

/* eonclient runs this program through system(), so the parent is sh and
   the client is the parent of sh. Returns 0 when /proc cannot name both. */
static int identify(pid_t *shell, pid_t *client, unsigned long long *started) {
    *shell = getppid();
    *client = parent_of(*shell);
    *started = 0;
    if (*client <= 1 || *client == *shell) {
        return 0;
    }
    *started = start_time(*client);
    return *started != 0;
}

/* eOn 3.5.0 and develop run this program in one exchange directory per
   potential object and name the client's own directory in
   EON_EXTPOT_RUN_DIR. r2549 runs it in the client's directory. */
static char *run_path(const char *name) {
    const char *directory = getenv("EON_EXTPOT_RUN_DIR");
    char *path;
    if (directory == NULL || directory[0] == '\0') {
        directory = ".";
    }
    path = (char *)malloc(strlen(directory) + strlen(name) + 2);
    if (path != NULL) {
        sprintf(path, "%s/%s", directory, name);
    }
    return path;
}

/* The exchanges optbench-extpot-calls holds for this client, 0 when the
   file names another process or is missing. With add set the call being
   finished is counted too. A client can hold several potential objects,
   so the read and the rewrite happen under one lock. */
static long tally(pid_t client, unsigned long long started, int add) {
    char *path = run_path(TALLY_FILE);
    /* An int, an unsigned long long and a long print in at most 11, 20 and
       20 characters, so the line and its terminator fit in 64 bytes. */
    char text[64];
    struct flock lock;
    ssize_t got;
    long calls = 0;
    int fd;
    if (path == NULL) {
        return 0;
    }
    /* 0666 less the umask is the mode fopen gives a new file. */
    fd = open(path, O_RDWR | O_CREAT, 0666);
    free(path);
    if (fd < 0) {
        return 0;
    }
    memset(&lock, 0, sizeof(lock));
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    if (fcntl(fd, F_SETLKW, &lock) == 0) {
        got = read(fd, text, sizeof(text) - 1);
        if (got > 0) {
            int pid = 0;
            unsigned long long seen = 0;
            long count = 0;
            text[got] = '\0';
            if (sscanf(text, "%d %llu %ld", &pid, &seen, &count) == 3 && pid == (int)client && seen == started) {
                calls = count;
            }
        }
        if (add) {
            int length = snprintf(text, sizeof(text), "%d %llu %ld\n", (int)client, started, calls + 1);
            if (ftruncate(fd, 0) == 0 && lseek(fd, 0, SEEK_SET) == 0 && write(fd, text, (size_t)length) == length) {
                calls += 1;
            }
        }
    }
    close(fd);
    return calls;
}

static void count_exchange(void) {
    pid_t shell;
    pid_t client;
    unsigned long long started;
    if (identify(&shell, &client, &started)) {
        tally(client, started, 1);
    }
}

/* "calls N" is the tally of the client about to be stopped and is left out
   when /proc cannot name the client. "stop" is the refusal it never reads. */
static void report_cap(int known, pid_t client, unsigned long long started, const char *refusal) {
    char *path = run_path(CAP_FILE);
    FILE *file = path == NULL ? NULL : fopen(path, "w");
    free(path);
    if (file == NULL) {
        return;
    }
    if (known) {
        fprintf(file, "calls %ld\n", tally(client, started, 0));
    }
    fprintf(file, "stop %s\n", refusal);
    fclose(file);
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
                unsigned long long started;
                int known;
                char *refusal = line;
                char *detail;
                fclose(out);
                unlink("from_extpot_to_eon");
                fprintf(stderr, "optbench ext_pot: %s\n", line);
                /* system() ignores the bridge status. A harness cap must
                   stop the client instead of leaving the previous forces. */
                if (cap_refusal) {
                    known = identify(&shell, &client, &started);
                    if (!known && client < 0) {
                        client = shell;
                    }
                    /* The server replies "error" and then the reason. */
                    while ((detail = next_line(&cursor)) != NULL) {
                        if (detail[0] != '\0') {
                            refusal = detail;
                            break;
                        }
                    }
                    report_cap(known, client, started, refusal);
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
    count_exchange();
    return 0;
}
