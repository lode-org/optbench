# OptBench paper figures

Every figure is generated from committed OptBench result rows. The exporter
writes one deterministic run-level data file and a manifest; it does not
modify the archived result files.

From the repository root:

```sh
snakemake --snakefile paper-figures/Snakefile --cores 1
```

Each plotting invocation writes one site SVG under
`docs/source/_static/paper/` and one vector PDF under
`paper-figures/pdfs/`. The PDFs use a 3.35 inch journal-column width and the
colorblind-safe Okabe–Ito palette. Histories show recorded points without
smoothing. Distribution strips use the recorded force-call values without
jitter and mark failures with an `×`. Success bars show exact successful and
total start counts.

For distributions, success rates, and family profiles, the manifest selects
the lowest-ranked full row with per-start records for each code. Archive-only
rows remain in histories when they report a relevant summary, but they cannot
appear in a run-level figure because the archive has no per-start values.

Family profiles use the definition in Dolan and Moré:

> E. D. Dolan and J. J. Moré, “Benchmarking optimization software with
> performance profiles,” *Mathematical Programming* 91, 201–213 (2002).

The profile abscissae are the exact finite performance ratios present in the
selected data. No artificial failure ratio, smoothing, or display threshold
is added. A start that every selected method fails has no finite reference
cost and is reported as omitted on the figure.
