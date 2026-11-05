"""Show that every machine load rewrites ~/.asv-machine.json in place.

Run: uv run python docs/asv-gaps/machine_race.py
"""

import json
import os
import tempfile
import time
from pathlib import Path

from asv.commands import make_argparser
from asv.machine import Machine

with tempfile.TemporaryDirectory() as home:
    os.environ["HOME"] = home
    path = Path(home) / ".asv-machine.json"
    path.write_text(json.dumps({"repro": {**Machine.get_defaults(), "machine": "repro"}, "version": 1}))
    before = path.stat()
    time.sleep(0.01)
    Machine.load(machine_name="repro", interactive=False)
    after = path.stat()
print("a load writes the file:", after.st_mtime_ns != before.st_mtime_ns)
print("same inode, so truncated in place:", after.st_ino == before.st_ino)
parser, subparsers = make_argparser()
for action in subparsers.choices["run"]._actions:
    if action.dest == "machine":
        print(f"asv run {' '.join(sorted(action.option_strings))}: {action.help.split('.')[0].strip()}.")
if after.st_mtime_ns == before.st_mtime_ns or after.st_ino != before.st_ino:
    raise SystemExit("a load no longer rewrites the file in place")
