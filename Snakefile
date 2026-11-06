# One asv comparison on this machine.
#
# Base and head are measured by one `asv run` process, so they share the
# runner. The two commits are listed in a hash file. `asv run` reads that
# file only when the range starts with `HASHFILE:`. `--interleave-rounds`
# alternates rounds across the two commits. The commands do not request a
# quick run and do not skip saved results. Results go to results/asv-ci,
# not the archived results tree.
#
# `asv_measure` writes the comparison text and the decision. A failure
# decision (status 1) with both files present does not fail that rule, so
# the files stay. Any other non-zero status, including status 1 from
# `asv run`, fails `asv_measure` before those files exist. `asv_compare`
# reads the decision and fails the build. It has no output files, so that
# failure does not delete the report.

import os

BASE = os.environ.get("OPTBENCH_ASV_BASE", "HEAD")
HEAD = os.environ.get("OPTBENCH_ASV_HEAD", "HEAD")
MACHINE = os.environ.get("OPTBENCH_ASV_MACHINE", "optbench-ci")
ROUNDS = os.environ.get("OPTBENCH_ASV_ROUNDS", "8")
REPEAT = os.environ.get("OPTBENCH_ASV_REPEAT", "1")
TIME_BENCH = os.environ.get("OPTBENCH_ASV_TIME_BENCH", r"quadratic\..*\.time_")
TRACK_BENCH = os.environ.get("OPTBENCH_ASV_TRACK_BENCH", r"quadratic\..*\.track_")
RESULTS = os.environ.get("OPTBENCH_ASV_RESULTS", "results/asv-ci/data")


rule asv_measure:
    params:
        base=BASE,
        head=HEAD,
        machine=MACHINE,
        rounds=ROUNDS,
        repeat=REPEAT,
        time_bench=TIME_BENCH,
        track_bench=TRACK_BENCH,
        results=RESULTS,
    output:
        report="results/asv-ci/compare.txt",
        decision="results/asv-ci/gate.json",
    shell:
        """
        set -eu
        mkdir -p results/asv-ci
        uv run python -c 'from optbench.asv_gate import ensure_machine; ensure_machine("{params.machine}")'
        base=$(git rev-parse "{params.base}")
        head=$(git rev-parse "{params.head}")
        if [ "$base" = "$head" ]; then
          uv run asv run "$head^!" \\
            --config asv.ci.conf.json \\
            --machine {params.machine} \\
            --bench quadratic
          printf '%s\\n' 'same commit on one machine; comparison not applicable' > {output.report}
          printf '%s\\n' '{{"same_commit": true, "regression": false}}' > {output.decision}
        else
          printf '%s\\n%s\\n' "$base" "$head" > results/asv-ci/commits.txt
          uv run asv run "HASHFILE:results/asv-ci/commits.txt" \\
            --config asv.ci.conf.json \\
            --machine {params.machine} \\
            --interleave-rounds \\
            --bench '{params.time_bench}' \\
            -a rounds={params.rounds} \\
            -a repeat={params.repeat} \\
            --record-samples \\
            > results/asv-ci/asv-time.txt
          uv run asv run "HASHFILE:results/asv-ci/commits.txt" \\
            --config asv.ci.conf.json \\
            --machine {params.machine} \\
            --interleave-rounds \\
            --bench '{params.track_bench}' \\
            -a rounds=1 \\
            -a repeat=1 \\
            > results/asv-ci/asv-track.txt
          uv run python -m optbench.asv_gate \\
            --results {params.results} \\
            --machine {params.machine} \\
            --base "$base" \\
            --head "$head" \\
            --report {output.report} \\
            --decision {output.decision} || {{
              status=$?
              test "$status" -eq 1
              test -s {output.report}
              test -s {output.decision}
            }}
        fi
        """


rule asv_compare:
    input:
        report=rules.asv_measure.output.report,
        decision=rules.asv_measure.output.decision,
    shell:
        """
        uv run python -c 'import json, pathlib, sys; decision = json.loads(pathlib.Path("{input.decision}").read_text(encoding="utf-8")); sys.exit(0 if decision.get("regression") is False else "the comparison reported a failure")'
        """


rule asv_memory:
    params:
        machine=MACHINE,
    output:
        "results/asv-memray/peak.txt",
    shell:
        """
        set -eu
        mkdir -p results/asv-memray
        uv run python -c 'from optbench.asv_gate import ensure_machine; ensure_machine("{params.machine}")'
        uv run asv run HEAD^! \\
          --config asv.memray.conf.json \\
          --machine {params.machine} \\
          --bench ray_kept
        uv run python -c 'from optbench.asv_memray_report import write_peak; write_peak()'
        """
