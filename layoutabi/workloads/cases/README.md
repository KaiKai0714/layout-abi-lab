# Adding an optimizer case

Drop two files with the same stem into this directory. Do not edit the CLI or
the catalog tests.

1. `<id>.json` — metadata. Required keys:

   - `id` (must equal the filename stem)
   - `role`: `positive_reference`, `public`, `negative`, or `experimental`
   - `title`
   - `license`
   - `graph_fingerprint`
   - `expected_optimizer`: `match` or `noop`
   - `smoke_resolution` (small size for CPU tests)

   Optional: `repository`, `commit`, `independent_of`, `notes`.

2. `<id>.py` — a `build(resolution, batch, dtype)` function that returns
   `(module, example_inputs)`. Optional `reference_outputs(module, inputs)` for
   a numerical check against a published formulation.

`layoutabi list-workloads` reads only the JSON files, so catalog listing does
not require PyTorch. `inspect-model --workload <id>` and the optimizer tests
pick up every registered case automatically.

A `match` case must be found by the bounded KTV matcher without using the
module class name. A `noop` case must leave the original module unchanged.
Pin repository, commit, and license for any reconstructed public graph.
