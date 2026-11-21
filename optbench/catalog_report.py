"""The Pt-island comparison table, written from the campaign ledgers."""

from __future__ import annotations

import json
import re
from pathlib import Path

METHODS = (
    ("dimer-fire", "rgsaddle-develop-{arm}-saddle-dimer-fire.dat"),
    ("dimer-lbfgs", "rgsaddle-develop-{arm}-saddle-dimer-lbfgs.dat"),
    ("dimer-rotskip", "rgsaddle-develop-{arm}-saddle-dimer-rotskip.dat"),
    ("lanczos", "rgsaddle-develop-{arm}-saddle-lanczos.dat"),
    ("eon-classic", None),
)
OLD_EON = "eon-3.5.0-saddle-pt-classic-develop-protocol.dat"
NEW_EON = "eon-3.5.0-d9cc6c5be-saddle-pt-classic-aa6dd937.dat"
REPORT_MD = Path("results/campaign/rgsaddle-develop-aa6dd937-vs-179e90f2-report.md")
REPORT_JSON = Path("results/campaign/rgsaddle-develop-aa6dd937-vs-179e90f2-report.json")


def _fields(path: Path) -> tuple[str, dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    fields: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        fields[key] = value
    return (lines[0] if lines else ""), fields


def _searches(comment: str, fields: dict[str, str]) -> int:
    match = re.search(r"(?:^|\s)total_searches (\d+)", comment)
    if match:
        return int(match.group(1))
    return int(float(fields["n"]))


def _connected(comment: str, fields: dict[str, str]) -> int:
    match = re.search(r"(?:^|\s)connected (\d+)", comment)
    if match:
        return int(match.group(1))
    return int(float(fields.get("connected") or 0))


def _force_calls(comment: str, fields: dict[str, str]) -> float:
    if fields.get("force_calls"):
        return float(fields["force_calls"])
    match = re.search(r"force_calls (\d+)", comment)
    if not match:
        raise ValueError("ledger has no force_calls")
    return float(match.group(1))


def _cap_failures(dat: Path) -> int:
    starts = dat.with_suffix(".starts.jsonl")
    if not starts.is_file():
        return 0
    count = 0
    for line in starts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reasons = [str(item) for item in row.get("reasons") or []]
        if row.get("failure") == "cap" or "cap" in reasons:
            count += 1
    return count


def _arm(root: Path, arm: str, method: str) -> dict:
    if method == "eon-classic":
        if arm == "179e90f2":
            name = OLD_EON
        else:
            published = eon_dat(root)
            name = published.name if published is not None else NEW_EON
    else:
        name = f"rgsaddle-develop-{arm}-saddle-{method}.dat"
    path = root / "results" / "campaign" / name
    comment, fields = _fields(path)
    searches = _searches(comment, fields)
    calls = _force_calls(comment, fields)
    connected = _connected(comment, fields)
    goal = int(float(fields["goal"])) if fields.get("goal") else None
    return {
        "connected": connected,
        "searches": searches,
        "force_calls": calls,
        "calls_per_search": calls / searches,
        "cap_failures": _cap_failures(path),
        "code_version": fields.get("code_version", ""),
        "goal": goal,
        "reached_goal": goal is not None and connected >= goal,
    }


def _hardware(root: Path) -> dict:
    path = root / "results" / "campaign" / "rgsaddle-develop-aa6dd937-pt-island-hardware.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "results" / "campaign" / "rgsaddle_develop_catalog.py"
    spec = importlib.util.spec_from_file_location("rgsaddle_develop_catalog_report", script)
    if spec is None or spec.loader is None:
        raise ImportError(script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eon_dat(root: Path) -> Path | None:
    """The new eon-classic ledger when its provenance passes the writer."""
    campaign = root / "results" / "campaign"
    catalog = _catalog()
    pairs = (
        (NEW_EON, "rgsaddle-develop-aa6dd937-eon-provenance.json"),
    )
    for dat_name, provenance_name in pairs:
        dat = campaign / dat_name
        if dat.is_file() and catalog.provenance_publishes(campaign / provenance_name):
            return dat
    return None


def rows(root: Path) -> list[dict]:
    found = []
    for method, _pattern in METHODS:
        if method == "eon-classic" and eon_dat(root) is None:
            continue
        found.append({"method": method, "old": _arm(root, "179e90f2", method), "new": _arm(root, "aa6dd937", method)})
    return found


def _goal_lines(found: list[dict]) -> list[str]:
    """Which arm reached the connected-saddle goal on each method."""
    lines: list[str] = []
    for row in found:
        goal = row["new"].get("goal")
        if goal is None:
            goal = row["old"].get("goal")
        if goal is None:
            continue
        for name, arm in (("179e90f2", row["old"]), ("aa6dd937", row["new"])):
            if arm.get("reached_goal"):
                lines.append(f"{name} {row['method']} reached the goal of {goal} connected saddles.")
            else:
                lines.append(
                    f"{name} {row['method']} stopped at {arm['connected']} connected saddles, "
                    f"below the goal of {goal}."
                )
    return lines


def markdown(root: Path) -> str:
    """Comparison table of the committed Pt-island saddle rows."""
    hardware = _hardware(root)
    found = rows(root)
    body = [
        "# Pt-island saddle catalog comparison",
        "",
        (
            f"Hardware is {hardware['model_name']}, {hardware['cpu_cores']} cores, "
            f"{hardware['mem_total_kb']} kB, cpus {hardware['cpus_online']}."
        ),
        (
            "Each arm used 1 core and 1 thread. "
            f"{hardware['thread_vars']} and the affinity is {hardware['affinity_policy']}."
        ),
        "",
        *_goal_lines(found),
        "",
        "| method | 179e90f2 connected/searches | aa6dd937 connected/searches | 179e90f2 force_calls, including the 14 initial-minimisation calls | aa6dd937 force_calls, including the 14 initial-minimisation calls | 179e90f2 calls/search | aa6dd937 calls/search | cap failures 179e90f2 -> aa6dd937 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in found:
        old, new = row["old"], row["new"]
        label = row["method"]
        body.append(
            "| {label} | {oc}/{os} | {nc}/{ns} | {of:.0f} | {nf:.0f} | {op:.1f} | {np:.1f} | {ocap}->{ncap} |".format(
                label=label,
                oc=old["connected"],
                os=old["searches"],
                nc=new["connected"],
                ns=new["searches"],
                of=old["force_calls"],
                nf=new["force_calls"],
                op=old["calls_per_search"],
                np=new["calls_per_search"],
                ocap=old["cap_failures"],
                ncap=new["cap_failures"],
            )
        )
    body.append("")
    return "\n".join(body)


def write(root: Path) -> None:
    campaign = Path(root) / "results" / "campaign"
    (campaign / REPORT_MD.name).write_text(markdown(root), encoding="utf-8")
    payload = {"hardware": _hardware(root), "rows": rows(root)}
    (campaign / REPORT_JSON.name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
