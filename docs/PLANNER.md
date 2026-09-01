# Three-level residue mechanism and the binary safety planner

The community result platform exists to test whether a producer layout lands on a
fast GEMM family. That question is not limited to Linear Attention; those graphs
are public witnesses. Prior controlled FP16 consumer-GEMM experiments established
a three-level mechanism prior:

| Consumer N class | Observed family tier |
|---|---|
| `N % 8 == 0` | fastest: L40S `align8`, Orin `ldg8` |
| even and `N % 8 != 0` | intermediate: `align2` |
| odd | slowest: `align1` |

On L40S, pointer alignment independently triggers the same ladder. In the
published 100-cell grid, the family tier is the minimum of the N tier and both
operand-pointer tiers: 16-byte aligned → tier 8, 4/8-byte aligned → tier 2,
and 2-byte aligned → tier 1. These names are vendor/backend observations for
the tested FP16 ABI, not a universal CUDA law.

The shipping `n_mod_8` planner remains deliberately binary: direct for the
fastest candidate class and repair for both intermediate and slowest classes.
It is a conservative safety policy, not a three-family classifier and not an
oracle for whether copy cost pays off at full-module scope.

The shipping static planner does **not** consume internal K/V pointer addresses.
Its decision features are N, dtype, batch, shape context, and device. Runtime
cache identity records external example-input pointer classes, but those are not
equivalent to K/V addresses after producer or compiler lowering. The pointer
audit is mechanism evidence for a future runtime guard/compiler integration, not
a silent change to the frozen v0.9.1 policy.

## Baselines

| Policy | Rule |
|---|---|
| `always_direct` | Never repair |
| `always_repair_kv` | Always repair |
| `n_mod_8` | FP16: fastest residue class → direct; intermediate/slowest → repair |
| `cost_model` | Non-FP16 or batch ≠ 1 → direct; `N % 8 == 0` → direct (skip autotune); else autotune |
| `autotune` | Measure candidates (scored as the oracle on published bundles) |

`cost_model` is not allowed to special-case 128 vs 256. Those two original reference
resolutions are both even non-multiples of 8, so both belong to the intermediate
mechanism class even though their full-module profitability differs. Held-out 128
is a published eager counterexample to treating the safety rule as a profitability
oracle. New bundles must report family tier and full-module oracle separately.

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
