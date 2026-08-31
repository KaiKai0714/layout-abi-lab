# Runtime cache and shape contract

`layoutabi.optimize()` may cache a measured decision so a process does not
re-benchmark on every start. The cache is not a license to repair unseen graphs.

## Cache

Writes are locked across processes, replaced atomically, and versioned
(`layoutabi_optimizer_cache_v2`). Corrupt or v1 files are not reused; a
`.corrupt` backup is kept when possible. `DIAGNOSTICS.md` lists entries.

The key includes graph fingerprint, exact or bucketed shape, dtype, stride,
pointer alignment class, GPU UUID/model/compute capability, PyTorch/CUDA build,
optimizer version, and candidate implementation hash. A software-stack change
produces a new key.

```bash
layoutabi cache-info --cache-dir .cache/layoutabi
layoutabi cache-clear --cache-dir .cache/layoutabi
```

Weights are not part of the key. Layout decisions for the frozen KTV pattern are
treated as independent of parameter values until a measured counterexample exists.

## Shapes

Default `shape_mode="exact"`. `shape_mode="bucket"` maps each dimension onto
32…512 inclusive upper bounds. A size above 512 is unseen.

Unseen sizes use `unseen_shape`: `direct` (default), `noop`, or `autotune`.
They never apply repair without a matching cache entry or an explicit measured
autotune the caller allowed.

`allow_sync_autotune=False` (CLI `--no-sync-autotune`) is the latency-critical
contract: cache hit is used; otherwise the unseen-shape action, not a blocking
autotune, and not an unverified repair.

Diagnostics include `timings_ms` for capture, autotune, and compile, plus
`break_even_invocations` when autotune found a faster candidate than direct.
