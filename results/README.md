# Results

`reference_l40s/` contains the frozen reference summary. Local runs should use a
directory beginning with `local_`; these directories are ignored by Git by default.

Community pull requests should place validated bundles under
`community/<gpu-stack-date>/` using `layoutabi prepare-submission`. These bundles are
tracked by Git and automatically validated in continuous integration. Same-device
same-stack re-runs are indexed as replicates. See `CONTRIBUTING.md` for the submission
command.

Every benchmark initially writes to `results/local_*`. The human-readable result is
`SUMMARY.md`; the JSON files preserve raw measurements and environment provenance.
