"""Show that `asv machine --yes --machine NAME` stores only the name.

Run: uv run python docs/asv-gaps/machine_fields.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as home:
    env = dict(os.environ, HOME=home)
    subprocess.run(
        [sys.executable, "-m", "asv", "machine", "--yes", "--machine", "repro"],
        env=env,
        capture_output=True,
        check=True,
    )
    stored = json.loads((Path(home) / ".asv-machine.json").read_text(encoding="utf-8"))
entry = stored["repro"]
print("stored fields:", sorted(entry))
missing = sorted({"arch", "cpu", "num_cpu", "os", "ram"} - set(entry))
print("missing fields:", missing)
if not missing:
    raise SystemExit("asv machine now stores every field")
