# Community results

This directory contains complete result bundles contributed through pull requests.
Positive, negative, unsupported, and slower-repair outcomes are all accepted when the
protocol, correctness gates, environment record, and checksums are complete.

Use one directory per device and software stack:

```text
results/community/<gpu>_<pytorch>_<cuda>_<yyyy-mm-dd>/
```

Create that directory with `layoutabi prepare-submission` rather than copying by hand.
Re-runs of an already indexed identity are retained as replicates. Do not hand-edit
measured JSON values. See the repository-level `CONTRIBUTING.md` for the validation and
pull-request workflow.

