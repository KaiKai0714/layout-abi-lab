# Contributing

The highest-leverage help is **a full result bundle from a GPU that is not
NVIDIA L40S**, including losses, plus RC reports of crashes or incorrect
rewrites. Same-device re-runs are replicates. Do not add a new matcher pattern
during the 0.9 RC. See the README section "What we need from the community".

## Submit benchmark results

1. Run the benchmark from an unmodified commit.
2. Keep GPU clocks, power mode, thermal state, and concurrent workloads stable.
3. Prepare the bundle for a pull request. This copies it into `results/community/`,
   recomputes checksums, and reports possible private metadata. It does not upload.

Example:

Name the local directory the same as the submission, with a `local_` prefix.
See `results/README.md`.

```bash
layoutabi reproduce --output results/local_rtx4090_torch2.11_cuda12.8_2026-08-30
layoutabi validate results/local_rtx4090_torch2.11_cuda12.8_2026-08-30 --strict
layoutabi prepare-submission \
  results/local_rtx4090_torch2.11_cuda12.8_2026-08-30 \
  --name rtx4090_torch2.11_cuda12.8_2026-08-30

git add results/community/rtx4090_torch2.11_cuda12.8_2026-08-30
```

Use `--strict` when the compiled protocol was intended to complete. Omit it when
`compile_results.json` is legitimately absent; those cells are indexed as unavailable,
not as a direct/repair loss. Use `--strict-privacy` if local policy requires the copy to
fail when hostname, username, private path, or extra metadata is present. Remove
identifying fields before opening the pull request.

Do not submit only favorable cells. Include every cell produced by the selected
protocol, including unsupported, out-of-memory, and slower-repair outcomes.

Pointer-audit submissions are separate from ordinary reproduction bundles. Run
`layoutabi validate-pointer-audit <directory>` and include the full Cartesian
grid; do not submit selected offsets. Device transfer results must use the same
N values, pointer residues, cycles, and iterations as `docs/POINTER_AUDIT.md`.

## Release reports

Version 1.0.0 freezes the public API and current matcher. Use the issue template
for crashes, incorrect rewrites, regressions, and unexpected no-ops. New matcher
patterns require a later minor release. Community result bundles on new devices
are still wanted.

## Code contributions

- Keep public documentation and code comments in English.
- Preserve numerical correctness checks and safe fallbacks.
- Do not add a device heuristic without a corresponding measured result and provenance.
- Avoid private framework APIs unless the compatibility boundary is documented and
  covered by version-specific tests.
- Run `python -m compileall layoutabi containers` and
  `python -m unittest discover -s tests -v` before opening a pull request.

## Upstream code and data

Do not vendor third-party source, models, or data without checking license compatibility.
Prefer pinned fetch scripts or source-equivalent minimal harnesses with exact repository,
commit, file, and equation provenance.
