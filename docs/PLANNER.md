# Planner and the N % 8 hypothesis

The community result platform exists to test whether a producer layout lands on a
fast GEMM family. On the L40S FP16 LinearAttention cells, eager profiling tied
zero-copy K to an `align2` family and repaired K/V to `align8`. `N % 8` is the
cheap feature that seemed to track that split. It is a hypothesis to score on new
devices, dtypes, and shapes — not a shipping law.

## Baselines

| Policy | Rule |
|---|---|
| `always_direct` | Never repair |
| `always_repair_kv` | Always repair |
| `n_mod_8` | FP16 and `N % 8 != 0` → repair; otherwise direct |
| `cost_model` | Non-FP16 or batch ≠ 1 → direct; `N % 8 == 0` → direct (skip autotune); else autotune |
| `autotune` | Measure candidates (scored as the oracle on published bundles) |

`cost_model` is not allowed to special-case 128 vs 256. Those two reference
resolutions share `consumer_n % 8 == 4`, so `N % 8` cannot separate them. Held-out
128 is a published eager counterexample to always-repair and to `n_mod_8` at
module scope. New community bundles are the real generalization test.

## Evaluate published results

```bash
layoutabi evaluate-planner
layoutabi evaluate-planner --held-out-resolutions 128
```

Gates (frozen): held-out geometric-mean regret ≤ 1.05×, max regret ≤ 1.15×,
false-repair = 0 for `cost_model`. `n_mod_8` is reported even when it fails; that
failure is evidence, not a CI error.

Live optimize:

```python
layoutabi.optimize(model, inputs, policy="n_mod_8")
layoutabi.optimize(model, inputs, policy="cost_model")
```
