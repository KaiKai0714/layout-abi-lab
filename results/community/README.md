# Community results

Accepted community bundles live here. Positive, negative, unsupported, and
slower-repair outcomes are all wanted. Orin eager 128 is a second architecture
and a no-repair boundary; L40S re-runs are replicates.

One bundle per GPU × software stack × day:

```text
results/community/<gpu>_torch<pytorch>_cuda<cuda>_<yyyy-mm-dd>/
```

Example: `orin_torch2.7_cuda12.6_2026-09-01`. Create it with
`layoutabi prepare-submission` rather than copying by hand. Do not hand-edit
measured JSON. See `results/README.md` and `CONTRIBUTING.md`.
