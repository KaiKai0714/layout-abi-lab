# Contributing

Contributions are welcome, especially reproducibility results from new GPUs and
software stacks, negative cases, profiler traces, and compiler integration work.

## Submit benchmark results

1. Run the benchmark from an unmodified commit.
2. Keep GPU clocks, power mode, thermal state, and concurrent workloads stable.
3. Prepare the bundle for a pull request. This copies it into `results/community/`,
   recomputes checksums, and reports possible private metadata. It does not upload.

Example:

```bash
layoutabi prepare-submission \
  results/local_my_gpu \
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

## Release-candidate reports

Version 0.9.0 is feature-frozen for v1.0. Use the RC issue template for crashes,
incorrect rewrites, regressions, and unexpected no-ops. Do not add a new matcher
pattern in this window; that requires a later RC. Community result bundles on
new devices are still wanted.

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
