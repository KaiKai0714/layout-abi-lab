# Release checklist

Use this when cutting a version. End-user install and reproduction steps stay in
`README.md`.

1. Bump `layoutabi/__init__.py`, `pyproject.toml`, and `CITATION.cff`.
2. Update `CHANGELOG.md`, README, `docs/ROADMAP.md`, and stated limitations.
3. Run `python -m compileall -q layoutabi containers` and
   `python -m unittest discover -s tests -v`.
4. Validate reference bundles: `layoutabi validate-tree results/reference_l40s --strict`.
5. Regenerate and freshness-check the index: `layoutabi aggregate` then
   `layoutabi aggregate --check`.
6. If the version claims GPU behavior, finish that version's release matrix. v0.3
   autotune and v0.4 compiled audits are CUDA-only; CPU matcher, rewrite, audit
   parsers, and v0.5 workload tests are the required CI gate. Do not count extra
   software stacks as extra devices.
7. Scan staged files for credentials, hostnames, and private absolute paths. Do not
   commit local-only notes or `results/local_*` bundles.
8. Run `layoutabi rc-status --check` and `layoutabi scan-release`. During 0.9/RC,
   do not unfreeze the public API, pattern id, candidate implementation, or
   schema versions, and do not claim unpublished GPU architectures.
9. Push the release commit, then create an annotated tag and a GitHub Release that
   lists supported behavior, unsupported cases, and any schema or API migration.
