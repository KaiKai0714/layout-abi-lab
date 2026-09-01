# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch._C import _cuda_getCurrentRawStream as get_raw_stream



# kernel path: <AUDIT_CACHE>/r128_direct/inductor/5q/c5qee6uttgqyufifua5diz2ws2ukczg4szo44b6w3nrbdmnd72cp.py
# Topologically Sorted Source Nodes: [normalize, mul, mul_1], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
# Source node to ATen node mapping:
#   mul => mul
#   mul_1 => mul_1
#   normalize => clamp_min, convert_element_type, convert_element_type_1, div, expand, pow_1, pow_2, sum_1
# Graph fragment:
#   %arg0_1 : Tensor "f16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sum_1 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %arg1_1 : Tensor "f16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2.0), kwargs = {})
#   %sum_1 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %pow_2 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %convert_element_type_1 : Tensor "f16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_2, torch.float16), kwargs = {})
#   %clamp_min : Tensor "f16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_1, 1e-12), kwargs = {})
#   %expand : Tensor "f16[1, 64, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min, [1, 64, 128, 128]), kwargs = {})
#   %div : Tensor "f16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %expand), kwargs = {})
#   %mul : Tensor "f16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %arg1_1), kwargs = {})
#   %mul_1 : Tensor "f16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 8.0), kwargs = {})
#   return %sum_1,%mul_1
triton_per_fused_clamp_min_div_expand_linalg_vector_norm_mul_0 = async_compile.triton('triton_per_fused_clamp_min_div_expand_linalg_vector_norm_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clamp_min_div_expand_linalg_vector_norm_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2097152, 'r0_': 4194432}}
)
@triton.jit
def triton_per_fused_clamp_min_div_expand_linalg_vector_norm_mul_0(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384*r0_1), None).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1, 1], 1e-12, tl.float32)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = (tmp0 / tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tl.full([1, 1], 8.0, tl.float32)
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp14, None)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r128_direct/inductor/66/c66lyuvianmacvv264lq5d2orxaqqtevsco57hqo46lsyv4twnwr.py
# Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_1
#   chunk => split
#   getitem_3 => select
#   k_1 => view_1
#   k_2 => cat
#   k_3 => convert_element_type_4
#   memory_k => expand_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg3_1 : Tensor "f16[2, 4, 32, 4][512, 128, 4, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f16[1, 384, 128, 128][6291456, 1, 49152, 384]cuda:0" = PlaceHolder[target=convolution]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %expand_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1, -1]), kwargs = {})
#   %view_1 : Tensor "f16[1, 4, 32, 16384][6291456, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1, 4, 32, 16384]), kwargs = {})
#   %cat : Tensor "f16[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %view_1], -1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_4, -1), kwargs = {})
#   return %buf3
triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_1 = async_compile.triton('triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'y': 128, 'x': 256, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4227072, 'x': 132096, 'r0_': 2048}}
)
@triton.jit
def triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 129
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
    rbase = r0_base
    x1 = xindex
    y0 = yindex
    _tmp20 = tl.full([YBLOCK, XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 128*x1
        tmp1 = tl.full([1, 1, 1], 16388, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(r0_2 + 128*x1, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp4 = tl.full([1, 1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1, 1], 4, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + (4*y0 + (r0_2 + 128*x1)), r0_mask & tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp3 >= tmp6
        tmp11 = tl.full([1, 1, 1], 16388, tl.int64)
        tmp12 = tmp3 < tmp11
        tmp13 = tmp10 & tmp2
        tmp14 = tl.load(in_ptr1 + (128 + y0 + 384*((-4) + r0_2 + 128*x1)), r0_mask & tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.where(tmp7, tmp9, tmp14)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.full(tmp16.shape, float("-inf"), tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp21 = triton_helpers.maximum(_tmp20, tmp19)
        _tmp20 = tl.where(r0_mask & xmask & ymask, tmp21, _tmp20)
    tmp20 = triton_helpers.max2(_tmp20, 2)[:, :, None]
    tl.store(out_ptr0 + (x1 + 129*y0), tmp20, xmask & ymask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r128_direct/inductor/wd/cwddkeku6755fg5unhxykhihvmdo5unmfydhvqjlqovr64t62kqu.py
# Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_1
#   chunk => split
#   getitem_3 => select
#   k_1 => view_1
#   k_2 => cat
#   k_3 => convert_element_type_4
#   memory_k => expand_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %buf3 : Tensor "f32[1, 4, 32, 1, 129][16512, 4128, 129, 16512, 1]cuda:0" = PlaceHolder[target=buf3]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %expand_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1, -1]), kwargs = {})
#   %view_1 : Tensor "f16[1, 4, 32, 16384][6291456, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1, 4, 32, 16384]), kwargs = {})
#   %cat : Tensor "f16[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %view_1], -1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_4, -1), kwargs = {})
#   return %getitem_5
triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_2 = async_compile.triton('triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 66048}}
)
@triton.jit
def triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 129
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 129*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r128_direct/inductor/me/cmehtgoozhu3s24iaiet6f334bw7ybehpcranllkqwtmnwrvrk2t.py
# Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_1
#   chunk => split
#   getitem_3 => select
#   k_1 => view_1
#   k_2 => cat
#   k_3 => convert_element_type_4
#   memory_k => expand_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg3_1 : Tensor "f16[2, 4, 32, 4][512, 128, 4, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f16[1, 384, 128, 128][6291456, 1, 49152, 384]cuda:0" = PlaceHolder[target=convolution]
#   %getitem_5 : Tensor "f32[1, 4, 32, 1][128, 32, 1, 128]cuda:0" = PlaceHolder[target=getitem_5]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %expand_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1, -1]), kwargs = {})
#   %view_1 : Tensor "f16[1, 4, 32, 16384][6291456, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1, 4, 32, 16384]), kwargs = {})
#   %cat : Tensor "f16[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %view_1], -1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_4, -1), kwargs = {})
#   return %buf5
triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_3 = async_compile.triton('triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'y': 128, 'x': 256, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4227584, 'x': 132096, 'r0_': 2048}}
)
@triton.jit
def triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 129
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
    rbase = r0_base
    x1 = xindex
    y0 = yindex
    _tmp23 = tl.full([YBLOCK, XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 128*x1
        tmp1 = tl.full([1, 1, 1], 16388, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(r0_2 + 128*x1, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp4 = tl.full([1, 1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1, 1], 4, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + (4*y0 + (r0_2 + 128*x1)), r0_mask & tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp3 >= tmp6
        tmp11 = tl.full([1, 1, 1], 16388, tl.int64)
        tmp12 = tmp3 < tmp11
        tmp13 = tmp10 & tmp2
        tmp14 = tl.load(in_ptr1 + (128 + y0 + 384*((-4) + r0_2 + 128*x1)), r0_mask & tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.where(tmp7, tmp9, tmp14)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = libdevice.exp(tmp18)
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(r0_mask & xmask & ymask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 2)[:, :, None]
    tl.store(out_ptr0 + (x1 + 129*y0), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r128_direct/inductor/qt/cqtivynshj65wntqx7ogqtzsl4a4qb7r2nddjwyfdggzn5z37c5g.py
# Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_1
#   chunk => split
#   getitem_3 => select
#   k_1 => view_1
#   k_2 => cat
#   k_3 => convert_element_type_4
#   memory_k => expand_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %buf5 : Tensor "f32[1, 4, 32, 1, 129][16512, 4128, 129, 16512, 1]cuda:0" = PlaceHolder[target=buf5]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %expand_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1, -1]), kwargs = {})
#   %view_1 : Tensor "f16[1, 4, 32, 16384][6291456, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1, 4, 32, 16384]), kwargs = {})
#   %cat : Tensor "f16[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %view_1], -1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_4, -1), kwargs = {})
#   return %getitem_6
triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_4 = async_compile.triton('triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 66048}}
)
@triton.jit
def triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 129
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 129*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r128_direct/inductor/zs/czsc74ugi2qnkc34qx5dg3duqhxphtqxkp5oo5gbhysxoitnsnjw.py
# Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, aten.sub, aten.exp]
# Source node to ATen node mapping:
#    => exp_default_1, sub_tensor_1
#   chunk => split
#   getitem_3 => select
#   k_1 => view_1
#   k_2 => cat
#   k_3 => convert_element_type_4, convert_element_type_5, div_2
#   memory_k => expand_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg3_1 : Tensor "f16[2, 4, 32, 4][512, 128, 4, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f16[1, 384, 128, 128][6291456, 1, 49152, 384]cuda:0" = PlaceHolder[target=convolution]
#   %getitem_5 : Tensor "f32[1, 4, 32, 1][128, 32, 1, 128]cuda:0" = PlaceHolder[target=getitem_5]
#   %getitem_6 : Tensor "f32[1, 4, 32, 1][128, 32, 1, 128]cuda:0" = PlaceHolder[target=getitem_6]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %expand_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1, -1]), kwargs = {})
#   %view_1 : Tensor "f16[1, 4, 32, 16384][6291456, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1, 4, 32, 16384]), kwargs = {})
#   %cat : Tensor "f16[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %view_1], -1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
#   %sub_tensor_1 : Tensor "f32[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %getitem_5), kwargs = {})
#   %exp_default_1 : Tensor "f32[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div_2 : Tensor "f32[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_6), kwargs = {})
#   %convert_element_type_5 : Tensor "f16[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float16), kwargs = {})
#   return %expand_3
triton_poi_fused__softmax_cat_exp_expand_select_split_sub_unsqueeze_view_5 = async_compile.triton('triton_poi_fused__softmax_cat_exp_expand_select_split_sub_unsqueeze_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_cat_exp_expand_select_split_sub_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4196352, 'x': 8392704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_cat_exp_expand_select_split_sub_unsqueeze_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 16388
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp12 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*y0 + (x1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 16388, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (128 + y0 + 384*((-4) + x1)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 - tmp12
    tmp14 = libdevice.exp(tmp13)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 16388*y0), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r128_direct/inductor/7p/c7ppdwl7hfmygo3dyphogrhl3eifyfhxrjzndynlxnelhl2t262k.py
# Topologically Sorted Source Nodes: [chunk, getitem_4, unsqueeze_1, memory_v, v_1, v_2], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat]
# Source node to ATen node mapping:
#   chunk => split
#   getitem_4 => select_1
#   memory_v => expand_2
#   unsqueeze_1 => unsqueeze_1
#   v_1 => view_2
#   v_2 => cat_1
# Graph fragment:
#   %arg3_1 : Tensor "f16[2, 4, 32, 4][512, 128, 4, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f16[1, 384, 128, 128][6291456, 1, 49152, 384]cuda:0" = PlaceHolder[target=convolution]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select_1 : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 0), kwargs = {})
#   %expand_2 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [1, -1, -1, -1]), kwargs = {})
#   %view_2 : Tensor "f16[1, 4, 32, 16384][6291456, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [1, 4, 32, 16384]), kwargs = {})
#   %cat_1 : Tensor "f16[1, 4, 32, 16388][2097664, 524416, 16388, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_2, %view_2], -1), kwargs = {})
#   return %cat_1
triton_poi_fused_cat_expand_select_split_unsqueeze_view_6 = async_compile.triton('triton_poi_fused_cat_expand_select_split_unsqueeze_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_expand_select_split_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4195328, 'x': 8392704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_expand_select_split_unsqueeze_view_6(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 16388
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = x1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (512 + 4*y0 + (x1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 16388, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (256 + y0 + 384*((-4) + x1)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x1 + 16388*y0), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r128_direct/inductor/xm/cxm6mcszydxp6k2jquhouuc3yo7whfxdutgl2et34zjreugzd7ol.py
# Topologically Sorted Source Nodes: [chunk, q_1, softmax, ], Original ATen: [aten.split, aten.view, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default
#   chunk => split
#   q_1 => view
#   softmax => convert_element_type_2
# Graph fragment:
#   %convolution : Tensor "f16[1, 384, 128, 128][6291456, 1, 49152, 384]cuda:0" = PlaceHolder[target=convolution]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %view : Tensor "f16[1, 4, 32, 16384][6291456, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [1, 4, 32, 16384]), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 4, 32, 16384][6291456, 524288, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_2, -2), kwargs = {})
#   return %getitem_3,%getitem_4
triton_per_fused__softmax_prepare_softmax_online_split_view_7 = async_compile.triton('triton_per_fused__softmax_prepare_softmax_online_split_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_prepare_softmax_online_split_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 4, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1048576, 'r0_': 4194304}}
)
@triton.jit
def triton_per_fused__softmax_prepare_softmax_online_split_view_7(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 32*x0 + 384*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp6 = triton_helpers.max2(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = tmp2 - tmp6
    tmp8 = libdevice.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r128_direct/inductor/3h/c3hse2mka6n25t4oyda37m73uuq2zfg6u77loczfd5pdpxb6f2ac.py
# Topologically Sorted Source Nodes: [chunk, q_1, softmax, , q_2], Original ATen: [aten.split, aten.view, aten._softmax, aten.sub, aten.exp, aten.mul]
# Source node to ATen node mapping:
#    => exp_default, sub_tensor
#   chunk => split
#   q_1 => view
#   q_2 => mul_2
#   softmax => convert_element_type_2, convert_element_type_3, div_1
# Graph fragment:
#   %convolution : Tensor "f16[1, 384, 128, 128][6291456, 1, 49152, 384]cuda:0" = PlaceHolder[target=convolution]
#   %getitem_3 : Tensor "f32[1, 4, 1, 16384][65536, 1, 65536, 4]cuda:0" = PlaceHolder[target=getitem_3]
#   %getitem_4 : Tensor "f32[1, 4, 1, 16384][65536, 1, 65536, 4]cuda:0" = PlaceHolder[target=getitem_4]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %view : Tensor "f16[1, 4, 32, 16384][6291456, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [1, 4, 32, 16384]), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 4, 32, 16384][6291456, 524288, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %sub_tensor : Tensor "f32[1, 4, 32, 16384][2097152, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %getitem_3), kwargs = {})
#   %exp_default : Tensor "f32[1, 4, 32, 16384][2097152, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_1 : Tensor "f32[1, 4, 32, 16384][2097152, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_4), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[1, 4, 32, 16384][2097152, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.float16), kwargs = {})
#   %mul_2 : Tensor "f16[1, 4, 32, 16384][2097152, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, 0.1767766952966369), kwargs = {})
#   return %expand_6
triton_poi_fused__softmax_exp_mul_split_sub_view_8 = async_compile.triton('triton_poi_fused__softmax_exp_mul_split_sub_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_exp_mul_split_sub_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 8388608, 'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_exp_mul_split_sub_view_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex
    x2 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (x3 + 384*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 + 4*y0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + 4*y0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp4 = libdevice.exp(tmp3)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1, 1], 0.1767766952966369, tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (y0 + 16384*x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r128_direct/inductor/tc/ctc7q76iwajgqjkwrjly6wnqm552l6t4ge7w6fy5evpozrs2sew2.py
# Topologically Sorted Source Nodes: [out, out_1, input_1], Original ATen: [aten.view, aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution_1
#   out => view_8
#   out_1 => view_9
# Graph fragment:
#   %bmm_1 : Tensor "f16[4, 32, 16384][524288, 16384, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %view_8 : Tensor "f16[1, 4, 32, 16384][2097152, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [1, 4, 32, 16384]), kwargs = {})
#   %view_9 : Tensor "f16[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [1, 128, 128, 128]), kwargs = {})
#   %convolution_1 : Tensor "f16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%view_9, %arg4_1, %arg5_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf14
triton_poi_fused_convolution_view_9 = async_compile.triton('triton_poi_fused_convolution_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 16384}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 8388608, 'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_view_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 16384*y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x1), tmp0, ymask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r128_direct/inductor/tj/ctj4xf4xit47zya3bb6fupliwvpmglltudxlt634vjch4jrzcgsh.py
# Topologically Sorted Source Nodes: [out, out_1, input_1, normalize_1, mul_3, input_2], Original ATen: [aten.view, aten.convolution, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
# Source node to ATen node mapping:
#   input_1 => convolution_1
#   input_2 => mul_4
#   mul_3 => mul_3
#   normalize_1 => clamp_min_1, convert_element_type_10, convert_element_type_11, div_3, expand_7, pow_3, pow_4, sum_4
#   out => view_8
#   out_1 => view_9
# Graph fragment:
#   %buf15 : Tensor "f16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = PlaceHolder[target=buf15]
#   %arg5_1 : Tensor "f16[64][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %sum_4 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = PlaceHolder[target=sum_4]
#   %arg6_1 : Tensor "f16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %view_8 : Tensor "f16[1, 4, 32, 16384][2097152, 524288, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [1, 4, 32, 16384]), kwargs = {})
#   %view_9 : Tensor "f16[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [1, 128, 128, 128]), kwargs = {})
#   %convolution_1 : Tensor "f16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%view_9, %arg4_1, %arg5_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_10 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_10, 2.0), kwargs = {})
#   %sum_4 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1], True), kwargs = {})
#   %pow_4 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %convert_element_type_11 : Tensor "f16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_4, torch.float16), kwargs = {})
#   %clamp_min_1 : Tensor "f16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_11, 1e-12), kwargs = {})
#   %expand_7 : Tensor "f16[1, 64, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_1, [1, 64, 128, 128]), kwargs = {})
#   %div_3 : Tensor "f16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_1, %expand_7), kwargs = {})
#   %mul_3 : Tensor "f16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %arg6_1), kwargs = {})
#   %mul_4 : Tensor "f16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 8.0), kwargs = {})
#   return %sum_4,%mul_4
triton_per_fused_clamp_min_convolution_div_expand_linalg_vector_norm_mul_view_10 = async_compile.triton('triton_per_fused_clamp_min_convolution_div_expand_linalg_vector_norm_mul_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clamp_min_convolution_div_expand_linalg_vector_norm_mul_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4194304, 'r0_': 2097408}}
)
@triton.jit
def triton_per_fused_clamp_min_convolution_div_expand_linalg_vector_norm_mul_view_10(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.full([1, 1], 1e-12, tl.float32)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = (tmp2 / tmp11)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.full([1, 1], 8.0, tl.float32)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr1 + (x0 + 16384*r0_1), tmp16, None)
''', device_str='cuda')

def partition_0(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(arg1_1, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg2_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg3_1, (2, 4, 32, 4), (512, 128, 4, 1))
    assert_size_stride(arg4_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (1, 64, 1, 1), (64, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 64, 128, 128), (1048576, 1, 8192, 64), torch.float16)
        # Topologically Sorted Source Nodes: [normalize, mul, mul_1], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
        # [Provenance debug handles] triton_per_fused_clamp_min_div_expand_linalg_vector_norm_mul_0:1
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_div_expand_linalg_vector_norm_mul_0.run(arg0_1, arg1_1, buf1, 16384, 64, stream=stream0)
        del arg0_1
        del arg1_1
        # Topologically Sorted Source Nodes: [normalize, mul, mul_1, conv2d], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:2
        buf2 = extern_kernels.convolution(buf1, arg2_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (1, 384, 128, 128), (6291456, 1, 49152, 384), 'torch.ops.aten.convolution.default')
        del arg2_1
        del buf1
        buf3 = empty_strided_cuda((1, 4, 32, 1, 129), (16512, 4128, 129, 16512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
        # [Provenance debug handles] triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_1:3
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_1.run(arg3_1, buf2, buf3, 128, 129, 128, stream=stream0)
        buf4 = empty_strided_cuda((1, 4, 32, 1), (128, 32, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
        # [Provenance debug handles] triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_2:4
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_2.run(buf3, buf4, 128, 129, stream=stream0)
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
        # [Provenance debug handles] triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_3:5
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_3.run(arg3_1, buf2, buf4, buf5, 128, 129, 128, stream=stream0)
        buf6 = empty_strided_cuda((1, 4, 32, 1), (128, 32, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
        # [Provenance debug handles] triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_4:6
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_4.run(buf5, buf6, 128, 129, stream=stream0)
        del buf5
        buf7 = empty_strided_cuda((1, 4, 32, 16388), (2097664, 524416, 16388, 1), torch.float16)
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, aten.sub, aten.exp]
        # [Provenance debug handles] triton_poi_fused__softmax_cat_exp_expand_select_split_sub_unsqueeze_view_5:7
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_cat_exp_expand_select_split_sub_unsqueeze_view_5.run(arg3_1, buf2, buf4, buf6, buf7, 128, 16388, stream=stream0)
        del buf4
        del buf6
        buf8 = empty_strided_cuda((1, 4, 32, 16388), (2097664, 524416, 16388, 1), torch.float16)
        # Topologically Sorted Source Nodes: [chunk, getitem_4, unsqueeze_1, memory_v, v_1, v_2], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat]
        # [Provenance debug handles] triton_poi_fused_cat_expand_select_split_unsqueeze_view_6:8
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_expand_select_split_unsqueeze_view_6.run(arg3_1, buf2, buf8, 128, 16388, stream=stream0)
        del arg3_1
        buf9 = empty_strided_cuda((4, 32, 32), (1024, 32, 1), torch.float16)
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, , context, getitem_4, unsqueeze_1, memory_v, v_1, v_2, transpose], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, aten.sub, aten.exp, aten.transpose, aten.bmm]
        # [Provenance debug handles] extern_kernels.bmm:9
        extern_kernels.bmm(reinterpret_tensor(buf7, (4, 32, 16388), (524416, 16388, 1), 0), reinterpret_tensor(buf8, (4, 16388, 32), (524416, 1, 16388), 0), out=buf9)
        del buf7
        del buf8
        buf10 = empty_strided_cuda((1, 4, 1, 16384), (65536, 1, 65536, 4), torch.float32)
        buf11 = empty_strided_cuda((1, 4, 1, 16384), (65536, 1, 65536, 4), torch.float32)
        # Topologically Sorted Source Nodes: [chunk, q_1, softmax, ], Original ATen: [aten.split, aten.view, aten._softmax, prims.prepare_softmax_online]
        # [Provenance debug handles] triton_per_fused__softmax_prepare_softmax_online_split_view_7:10
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_prepare_softmax_online_split_view_7.run(buf2, buf10, buf11, 65536, 32, stream=stream0)
        buf12 = empty_strided_cuda((1, 4, 32, 16384), (2097152, 524288, 16384, 1), torch.float16)
        # Topologically Sorted Source Nodes: [chunk, q_1, softmax, , q_2], Original ATen: [aten.split, aten.view, aten._softmax, aten.sub, aten.exp, aten.mul]
        # [Provenance debug handles] triton_poi_fused__softmax_exp_mul_split_sub_view_8:11
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_exp_mul_split_sub_view_8.run(buf2, buf10, buf11, buf12, 16384, 128, stream=stream0)
        del buf10
        del buf11
        del buf2
        buf13 = empty_strided_cuda((4, 32, 16384), (524288, 16384, 1), torch.float16)
        # Topologically Sorted Source Nodes: [chunk, context, transpose_1, out, q_1, softmax, , q_2], Original ATen: [aten.split, aten.view, aten.transpose, aten._softmax, aten.sub, aten.exp, aten.mul, aten.bmm]
        # [Provenance debug handles] extern_kernels.bmm:12
        extern_kernels.bmm(reinterpret_tensor(buf9, (4, 32, 32), (1024, 1, 32), 0), reinterpret_tensor(buf12, (4, 32, 16384), (524288, 16384, 1), 0), out=buf13)
        del buf9
        buf14 = reinterpret_tensor(buf12, (1, 128, 128, 128), (2097152, 1, 16384, 128), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [out, out_1, input_1], Original ATen: [aten.view, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_view_9:13
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_view_9.run(buf13, buf14, 128, 16384, stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [out, out_1, input_1], Original ATen: [aten.view, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:14
        buf15 = extern_kernels.convolution(buf14, arg4_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (1, 64, 128, 128), (1048576, 1, 8192, 64), 'torch.ops.aten.convolution.default')
        del arg4_1
        del buf14
        buf17 = empty_strided_cuda((1, 64, 128, 128), (1048576, 16384, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [out, out_1, input_1, normalize_1, mul_3, input_2], Original ATen: [aten.view, aten.convolution, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
        # [Provenance debug handles] triton_per_fused_clamp_min_convolution_div_expand_linalg_vector_norm_mul_view_10:15
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_convolution_div_expand_linalg_vector_norm_mul_view_10.run(buf15, arg5_1, arg6_1, buf17, 16384, 64, stream=stream0)
        del arg5_1
        del arg6_1
        del buf15
    return (buf17, )


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        partition0_args = [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]
        del arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1
        (buf17,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf17, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((2, 4, 32, 4), (512, 128, 4, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
