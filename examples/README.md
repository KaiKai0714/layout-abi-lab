# Examples

`print_supported.py` does not import PyTorch. `inspect_and_optimize.py` needs a
CUDA-enabled PyTorch build for autotune; CPU is enough to inspect and to apply
an explicit `direct` or `repair_kv` policy.

```bash
python examples/print_supported.py
python examples/inspect_and_optimize.py
```
