# Pattern contract: `linear_attention_ktv_v1`

This is the only graph pattern supported by the v0.3 optimizer. It is frozen for this
release. New patterns require a new identifier and tests.

## Producer and consumers

The matcher looks at captured FX / aten ops, not module class names or repository paths.

1. A rank-4 tensor `K` with shape `[B, H, D, N]` is produced by `softmax` on the last
   dimension (`dim=-1`).
2. The first consumer GEMM is `K @ V.transpose(-2, -1)`, producing `[B, H, D, D]`.
3. The second consumer GEMM is `context.transpose(-2, -1) @ Q`.

`Q` is typically softmax-normalized on `dim=-2` and scaled. That Q-softmax is not the
matched producer.

## Repair

Repair does not change numerical equations. It materializes a BHND-backed logical BHDN
view immediately before the first GEMM:

```text
x.transpose(-2, -1).contiguous().transpose(-2, -1)
```

- `direct`: no materialization
- `repair_k`: materialize `K`
- `repair_kv`: materialize `K` and `V`

## Guards

- Inference only
- Fixed example-input shapes
- FP16 activations
- CUDA is required for autotune; explicit repair policies may run on CPU for tests
- Softmax dimension must be the last dimension of a rank-4 `K`
- The right-hand operand of the first GEMM must be a last-two-dimension transpose of `V`
- The KTV GEMM must be followed by `context.transpose @ Q`

## Fallback

If capture, matching, guards, or the correctness canary fail, `layoutabi.optimize()`
returns the original module and records `decision="noop"`.
