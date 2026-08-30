# Contributing

Contributions are welcome, especially reproducibility results from new GPUs and
software stacks, negative cases, profiler traces, and compiler integration work.

## Submit benchmark results

1. Run the benchmark from an unmodified commit.
2. Keep GPU clocks, power mode, thermal state, and concurrent workloads stable.
3. Validate the result bundle with `layoutabi validate <directory> --strict`.
4. Copy the bundle to `results/community_pending/<descriptive-name>/`.
5. Open a pull request using the result-submission template.

Do not submit only favorable cells. Include every cell produced by the selected
protocol, including unsupported, out-of-memory, and slower-repair outcomes.

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
