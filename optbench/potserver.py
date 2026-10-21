"""Counting potential server.

Compiled clients reach one ``HarnessCounter``. eonclient does it through
the ``ext_pot`` file exchange, which opens a short TCP connection per
force evaluation. rgmin and rgsaddle do it through ``RgpotCounter.calculate``,
the rgpot shape ``(positions, atomic numbers, box) -> (energy, forces)``.
The socket and that method charge the counter they are given.

The wire dialogue is text, one command per connection:

``calculate`` then the atom count, nine box floats, and one
``Z x y z`` line per atom. The reply is ``ok``, the search-call count,
the verification-call count, the energy, and one ``fx fy fz`` line per
atom.

``reset`` zeroes the counter. ``count`` reports it. ``verify`` then
``on`` or ``off`` selects verification mode. ``bye`` closes.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Sequence

from .counter import HarnessCounter

Evaluate = Callable[
    [Sequence[float], Sequence[int] | None, Sequence[float] | None],
    tuple[float, Sequence[float]],
]


class RgpotCounter:
    """rgpot ``calculate`` on a harness counter.

    ``cache`` follows the counter. The server uses a counter that counts
    every request, which is what a compiled client does when it calls
    ``force`` twice at the same geometry.
    """

    def __init__(self, evaluate: Evaluate, counter: HarnessCounter | None = None):
        self.evaluate = evaluate
        self.counter = counter if counter is not None else HarnessCounter(cache=False)

    def calculate(
        self,
        positions: Sequence[float],
        atomic_numbers: Sequence[int] | None = None,
        box: Sequence[float] | None = None,
    ) -> tuple[float, tuple[float, ...]]:
        coords = tuple(float(value) for value in positions)
        fresh, cached = self.counter.charge(coords)
        if not fresh and cached is not None:
            return cached
        energy, forces = self.evaluate(coords, atomic_numbers, box)
        value = (float(energy), tuple(float(component) for component in forces))
        if len(value[1]) != len(coords):
            raise ValueError("forces must match positions")
        return self.counter.remember(coords, value)


def _read_floats(line: str, count: int) -> list[float]:
    parts = line.split()
    if len(parts) != count:
        raise ValueError(f"expected {count} floats, got {len(parts)}")
    return [float(part) for part in parts]


def handle_lines(lines: list[str], rgpot: RgpotCounter) -> list[str]:
    """Answer one command. ``lines`` does not include the trailing blanks."""
    if not lines:
        raise ValueError("empty command")
    command = lines[0]
    counter = rgpot.counter
    if command == "reset":
        counter.reset()
        return ["ok"]
    if command == "count":
        return ["count", str(counter.calls), str(counter.verify_calls)]
    if command == "verify":
        if len(lines) != 2 or lines[1] not in ("on", "off"):
            raise ValueError("verify needs on or off")
        counter._verifying = lines[1] == "on"
        return ["ok"]
    if command == "bye":
        return ["bye"]
    if command == "charge":
        # One client-side evaluation. The geometry never crosses the socket.
        counter.charge(())
        return ["ok", str(counter.calls), str(counter.verify_calls)]
    if command != "calculate":
        raise ValueError(f"unknown command {command}")
    if len(lines) < 3:
        raise ValueError("calculate needs a count and a box")
    atoms = int(lines[1])
    if atoms < 1:
        raise ValueError("atom count must be positive")
    box = _read_floats(lines[2], 9)
    body = lines[3:]
    if len(body) != atoms:
        raise ValueError(f"calculate announced {atoms} atoms and sent {len(body)}")
    positions: list[float] = []
    numbers: list[int] = []
    for row in body:
        parts = row.split()
        if len(parts) != 4:
            raise ValueError("an atom line is Z x y z")
        numbers.append(int(parts[0]))
        positions.extend(float(part) for part in parts[1:])
    energy, forces = rgpot.calculate(positions, numbers, box)
    reply = ["ok", str(counter.calls), str(counter.verify_calls), repr(energy)]
    for index in range(atoms):
        base = 3 * index
        reply.append(" ".join(repr(forces[base + axis]) for axis in range(3)))
    return reply


class PotServer:
    """One thread, one counter, one potential."""

    def __init__(
        self,
        evaluate: Evaluate,
        counter: HarnessCounter | None = None,
        host: str = "127.0.0.1",
        cap: int | None = None,
    ):
        if counter is None:
            counter = HarnessCounter(cache=False, cap=cap)
        elif cap is not None:
            counter.cap = int(cap)
        self.rgpot = RgpotCounter(evaluate, counter)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, 0))
        self._sock.listen(16)
        self._sock.settimeout(0.2)
        self.host = host
        self.port = int(self._sock.getsockname()[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="optbench-potserver", daemon=True)

    @property
    def counter(self) -> HarnessCounter:
        return self.rgpot.counter

    def start(self) -> PotServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sock.close()

    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _address = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            try:
                self._handle(connection)
            finally:
                connection.close()

    def _handle(self, connection: socket.socket) -> None:
        reader = connection.makefile("r", encoding="utf-8", newline="\n")
        writer = connection.makefile("w", encoding="utf-8", newline="\n")
        try:
            header = reader.readline()
            if not header:
                return
            command = header.strip()
            lines = [command]
            if command == "calculate":
                count_line = reader.readline()
                box_line = reader.readline()
                if not count_line or not box_line:
                    raise ValueError("truncated calculate")
                lines.extend((count_line.strip(), box_line.strip()))
                for _ in range(int(lines[1])):
                    row = reader.readline()
                    if not row:
                        raise ValueError("truncated atom list")
                    lines.append(row.strip())
            elif command == "verify":
                flag = reader.readline()
                if not flag:
                    raise ValueError("truncated verify")
                lines.append(flag.strip())
            reply = handle_lines(lines, self.rgpot)
        except Exception as exc:
            reply = ["error", exc.__class__.__name__ + ": " + str(exc)]
        writer.write("\n".join(reply) + "\n")
        writer.flush()


def _exchange(host: str, port: int, lines: list[str]) -> list[str]:
    with socket.create_connection((host, port), timeout=30.0) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        payload = "\n".join(lines) + "\n"
        connection.sendall(payload.encode("utf-8"))
        reader = connection.makefile("r", encoding="utf-8", newline="\n")
        reply = []
        for row in reader:
            reply.append(row.rstrip("\n"))
        if not reply:
            raise RuntimeError("potential server closed without a reply")
        if reply[0] == "error":
            raise RuntimeError(reply[1] if len(reply) > 1 else "potential server error")
        return reply


def remote_reset(host: str, port: int) -> None:
    reply = _exchange(host, port, ["reset"])
    if reply != ["ok"]:
        raise RuntimeError(f"reset reply {reply}")


def remote_count(host: str, port: int) -> tuple[int, int]:
    reply = _exchange(host, port, ["count"])
    if len(reply) != 3 or reply[0] != "count":
        raise RuntimeError(f"count reply {reply}")
    return int(reply[1]), int(reply[2])


def remote_verify(host: str, port: int, enabled: bool) -> None:
    reply = _exchange(host, port, ["verify", "on" if enabled else "off"])
    if reply != ["ok"]:
        raise RuntimeError(f"verify reply {reply}")


def remote_calculate(
    host: str,
    port: int,
    positions: Sequence[float],
    atomic_numbers: Sequence[int],
    box: Sequence[float],
) -> tuple[float, tuple[float, ...], int]:
    """One force evaluation. The returned count is the server's search tally."""
    atoms = len(atomic_numbers)
    if len(positions) != 3 * atoms:
        raise ValueError("positions and atomic numbers disagree")
    if len(box) != 9:
        raise ValueError("box is nine floats")
    lines = [
        "calculate",
        str(atoms),
        " ".join(repr(float(value)) for value in box),
    ]
    for index, number in enumerate(atomic_numbers):
        base = 3 * index
        triple = " ".join(repr(float(positions[base + axis])) for axis in range(3))
        lines.append(f"{int(number)} {triple}")
    reply = _exchange(host, port, lines)
    if len(reply) != 4 + atoms or reply[0] != "ok":
        raise RuntimeError(f"calculate reply has {len(reply)} lines")
    calls = int(reply[1])
    energy = float(reply[3])
    forces: list[float] = []
    for row in reply[4:]:
        forces.extend(float(part) for part in row.split())
    if len(forces) != 3 * atoms:
        raise RuntimeError("force reply is short")
    return energy, tuple(forces), calls
