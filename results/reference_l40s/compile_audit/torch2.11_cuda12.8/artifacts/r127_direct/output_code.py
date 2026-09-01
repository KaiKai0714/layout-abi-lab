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



# kernel path: <AUDIT_CACHE>/r127_direct/inductor/pi/cpiqpxd2c5mcb6ukubehvs4ndszkfate6dmyw33qip7t2jocmppt.py
# Topologically Sorted Source Nodes: [normalize, mul, mul_1], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
# Source node to ATen node mapping:
#   mul => mul
#   mul_1 => mul_1
#   normalize => clamp_min, convert_element_type, convert_element_type_1, div, expand, pow_1, pow_2, sum_1
# Graph fragment:
#   %arg0_1 : Tensor "f16[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sum_1 : Tensor "f32[1, 1, 127, 127][16160, 16160, 127, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %arg1_1 : Tensor "f16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type : Tensor "f32[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2.0), kwargs = {})
#   %sum_1 : Tensor "f32[1, 1, 127, 127][16129, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %pow_2 : Tensor "f32[1, 1, 127, 127][16129, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %convert_element_type_1 : Tensor "f16[1, 1, 127, 127][16129, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_2, torch.float16), kwargs = {})
#   %clamp_min : Tensor "f16[1, 1, 127, 127][16129, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_1, 1e-12), kwargs = {})
#   %expand : Tensor "f16[1, 64, 127, 127][16129, 0, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min, [1, 64, 127, 127]), kwargs = {})
#   %div : Tensor "f16[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %expand), kwargs = {})
#   %mul : Tensor "f16[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %arg1_1), kwargs = {})
#   %mul_1 : Tensor "f16[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 8.0), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clamp_min_div_expand_linalg_vector_norm_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2064512, 'r0_': 4129152}}
)
@triton.jit
def triton_per_fused_clamp_min_div_expand_linalg_vector_norm_mul_0(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16129
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16129*r0_1), xmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tl.sqrt_rn(tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1, 1], 1e-12, tl.float32)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = (tmp0 / tmp10)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full([1, 1], 8.0, tl.float32)
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r127_direct/inductor/if/cif4vxjzsxzpukszcp35aewbnbcb2d4xmwz5tcdgfgucumhip2h7.py
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
#   %convolution : Tensor "f16[1, 384, 127, 127][6193536, 1, 48768, 384]cuda:0" = PlaceHolder[target=convolution]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %expand_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1, -1]), kwargs = {})
#   %view_1 : Tensor "f16[1, 4, 32, 16129][6193536, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1, 4, 32, 16129]), kwargs = {})
#   %cat : Tensor "f16[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %view_1], -1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
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
    size_hints={'y': 128, 'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4161536, 'x': 130048, 'r0_': 2048}}
)
@triton.jit
def triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 127
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
        tmp1 = tl.full([1, 1, 1], 16133, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(r0_2 + 128*x1, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp4 = tl.full([1, 1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1, 1], 4, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + (4*y0 + (r0_2 + 128*x1)), r0_mask & tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp3 >= tmp6
        tmp11 = tl.full([1, 1, 1], 16133, tl.int64)
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
    tl.store(out_ptr0 + (x1 + 127*y0), tmp20, xmask & ymask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r127_direct/inductor/sw/cswgxkr2hejoqs3uya3awy733nafp2m4dmjnt4owuik54ekzch37.py
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
#   %buf3 : Tensor "f32[1, 4, 32, 1, 127][16256, 4064, 127, 16256, 1]cuda:0" = PlaceHolder[target=buf3]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %expand_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1, -1]), kwargs = {})
#   %view_1 : Tensor "f16[1, 4, 32, 16129][6193536, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1, 4, 32, 16129]), kwargs = {})
#   %cat : Tensor "f16[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %view_1], -1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
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
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 65024}}
)
@triton.jit
def triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 127
    R0_BLOCK: tl.constexpr = 128
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 127*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r127_direct/inductor/4a/c4ajf7mqvraiciswen7sr6hf4vjdt5q7ocw7uk3lvaguoqpg2a73.py
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
#   %convolution : Tensor "f16[1, 384, 127, 127][6193536, 1, 48768, 384]cuda:0" = PlaceHolder[target=convolution]
#   %getitem_5 : Tensor "f32[1, 4, 32, 1][128, 32, 1, 128]cuda:0" = PlaceHolder[target=getitem_5]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %expand_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1, -1]), kwargs = {})
#   %view_1 : Tensor "f16[1, 4, 32, 16129][6193536, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1, 4, 32, 16129]), kwargs = {})
#   %cat : Tensor "f16[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %view_1], -1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
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
    size_hints={'y': 128, 'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4162048, 'x': 130048, 'r0_': 2048}}
)
@triton.jit
def triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 127
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
        tmp1 = tl.full([1, 1, 1], 16133, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(r0_2 + 128*x1, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp4 = tl.full([1, 1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1, 1], 4, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + (4*y0 + (r0_2 + 128*x1)), r0_mask & tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp3 >= tmp6
        tmp11 = tl.full([1, 1, 1], 16133, tl.int64)
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
    tl.store(out_ptr0 + (x1 + 127*y0), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r127_direct/inductor/zx/czxixoh75ozhtx6ucssnrhgaqx5nd353ru466toyndpyvnvythld.py
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
#   %buf5 : Tensor "f32[1, 4, 32, 1, 127][16256, 4064, 127, 16256, 1]cuda:0" = PlaceHolder[target=buf5]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %expand_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1, -1]), kwargs = {})
#   %view_1 : Tensor "f16[1, 4, 32, 16129][6193536, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1, 4, 32, 16129]), kwargs = {})
#   %cat : Tensor "f16[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %view_1], -1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
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
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 65024}}
)
@triton.jit
def triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 127
    R0_BLOCK: tl.constexpr = 128
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 127*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r127_direct/inductor/y6/cy6yiwxxbfalscbfqqynwx3nb24mctfbtewp73zrkk5detnunhy2.py
# Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, , context], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, aten.sub, aten.exp, aten.bmm]
# Source node to ATen node mapping:
#    => constant_pad_nd_default, exp_default_1, sub_tensor_1
#   chunk => split
#   context => view_3
#   getitem_3 => select
#   k_1 => view_1
#   k_2 => cat
#   k_3 => convert_element_type_4, convert_element_type_5, div_2
#   memory_k => expand_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg3_1 : Tensor "f16[2, 4, 32, 4][512, 128, 4, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f16[1, 384, 127, 127][6193536, 1, 48768, 384]cuda:0" = PlaceHolder[target=convolution]
#   %getitem_5 : Tensor "f32[1, 4, 32, 1][128, 32, 1, 128]cuda:0" = PlaceHolder[target=getitem_5]
#   %getitem_6 : Tensor "f32[1, 4, 32, 1][128, 32, 1, 128]cuda:0" = PlaceHolder[target=getitem_6]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 0), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
#   %expand_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1, -1]), kwargs = {})
#   %view_1 : Tensor "f16[1, 4, 32, 16129][6193536, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1, 4, 32, 16129]), kwargs = {})
#   %cat : Tensor "f16[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %view_1], -1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
#   %sub_tensor_1 : Tensor "f32[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %getitem_5), kwargs = {})
#   %exp_default_1 : Tensor "f32[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div_2 : Tensor "f32[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_6), kwargs = {})
#   %convert_element_type_5 : Tensor "f16[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float16), kwargs = {})
#   %view_3 : Tensor "f16[4, 32, 16133][516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_3, [4, 32, 16133]), kwargs = {})
#   %constant_pad_nd_default : Tensor "f16[4, 32, 16136][516352, 16136, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_3, [0, 3, 0, 0, 0, 0]), kwargs = {})
#   return %constant_pad_nd_default
triton_poi_fused__softmax_bmm_cat_exp_expand_select_split_sub_unsqueeze_view_5 = async_compile.triton('triton_poi_fused__softmax_bmm_cat_exp_expand_select_split_sub_unsqueeze_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_bmm_cat_exp_expand_select_split_sub_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4131840, 'x': 8263680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_bmm_cat_exp_expand_select_split_sub_unsqueeze_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 16136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = x1
    tmp1 = tl.full([1, 1], 16133, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.broadcast_to(x1, [YBLOCK, XBLOCK])
    tmp4 = tl.full([1, 1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1, 1], 4, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = tl.load(in_ptr0 + (4*y0 + (x1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp3 >= tmp6
    tmp11 = tl.full([1, 1], 16133, tl.int64)
    tmp12 = tmp3 < tmp11
    tmp13 = tmp10 & tmp2
    tmp14 = tl.load(in_ptr1 + (128 + y0 + 384*((-4) + x1)), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.where(tmp7, tmp9, tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 - tmp17
    tmp19 = libdevice.exp(tmp18)
    tmp20 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = (tmp19 / tmp20)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tl.store(out_ptr0 + (x1 + 16192*y0), tmp24, xmask & ymask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r127_direct/inductor/zp/czp6dfvy2y2u3opjck6bhng3f56bhvynws4unx7jlcwib4wtm7tx.py
# Topologically Sorted Source Nodes: [chunk, getitem_4, unsqueeze_1, memory_v, v_1, v_2, transpose, context, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten.transpose, aten.bmm]
# Source node to ATen node mapping:
#    => constant_pad_nd_default_1
#   chunk => split
#   context => view_4
#   getitem_4 => select_1
#   memory_v => expand_2
#   transpose => permute
#   unsqueeze_1 => unsqueeze_1
#   v_1 => view_2
#   v_2 => cat_1
# Graph fragment:
#   %arg3_1 : Tensor "f16[2, 4, 32, 4][512, 128, 4, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f16[1, 384, 127, 127][6193536, 1, 48768, 384]cuda:0" = PlaceHolder[target=convolution]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %select_1 : Tensor "f16[4, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 0, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 0), kwargs = {})
#   %expand_2 : Tensor "f16[1, 4, 32, 4][512, 128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [1, -1, -1, -1]), kwargs = {})
#   %view_2 : Tensor "f16[1, 4, 32, 16129][6193536, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [1, 4, 32, 16129]), kwargs = {})
#   %cat_1 : Tensor "f16[1, 4, 32, 16133][2065024, 516256, 16133, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_2, %view_2], -1), kwargs = {})
#   %permute : Tensor "f16[1, 4, 16133, 32][2065024, 516256, 1, 16133]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%cat_1, [0, 1, 3, 2]), kwargs = {})
#   %view_4 : Tensor "f16[4, 16133, 32][516256, 1, 16133]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_4, [4, 16133, 32]), kwargs = {})
#   %constant_pad_nd_default_1 : Tensor "f16[4, 16136, 32][516352, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_4, [0, 0, 0, 3, 0, 0]), kwargs = {})
#   return %constant_pad_nd_default_1
triton_poi_fused_bmm_cat_expand_select_split_transpose_unsqueeze_view_6 = async_compile.triton('triton_poi_fused_bmm_cat_expand_select_split_transpose_unsqueeze_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_cat_expand_select_split_transpose_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12392448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_cat_expand_select_split_transpose_unsqueeze_view_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2065408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 32) % 16136)
    x0 = (xindex % 32)
    x2 = xindex // 516352
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 16133, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 4, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = tl.load(in_ptr0 + (512 + 4*x0 + 128*x2 + (x1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp3 >= tmp6
    tmp11 = tl.full([1], 16133, tl.int64)
    tmp12 = tmp3 < tmp11
    tmp13 = tmp10 & tmp2
    tmp14 = tl.load(in_ptr1 + (256 + x0 + 32*x2 + 384*((-4) + x1)), tmp13 & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.where(tmp7, tmp9, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r127_direct/inductor/k7/ck73q57to4dluja7t2mpklnztaxhzqcgwr7hokeiq3oevrkwjzf3.py
# Topologically Sorted Source Nodes: [chunk, q_1, softmax, ], Original ATen: [aten.split, aten.view, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default
#   chunk => split
#   q_1 => view
#   softmax => convert_element_type_2
# Graph fragment:
#   %convolution : Tensor "f16[1, 384, 127, 127][6193536, 1, 48768, 384]cuda:0" = PlaceHolder[target=convolution]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %view : Tensor "f16[1, 4, 32, 16129][6193536, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [1, 4, 32, 16129]), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 4, 32, 16129][6193536, 516128, 16129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_prepare_softmax_online_split_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 4, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1032256, 'r0_': 4129024}}
)
@triton.jit
def triton_per_fused__softmax_prepare_softmax_online_split_view_7(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64516
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 32*x0 + 384*x1), xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tmp2 - tmp7
    tmp9 = libdevice.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r127_direct/inductor/zv/czvfverrhtwqejxqwkyxzkzdofp4cjbx6wciwzlzbvuzscwzr3tg.py
# Topologically Sorted Source Nodes: [chunk, q_1, softmax, , q_2], Original ATen: [aten.split, aten.view, aten._softmax, aten.sub, aten.exp, aten.mul]
# Source node to ATen node mapping:
#    => exp_default, sub_tensor
#   chunk => split
#   q_1 => view
#   q_2 => mul_2
#   softmax => convert_element_type_2, convert_element_type_3, div_1
# Graph fragment:
#   %convolution : Tensor "f16[1, 384, 127, 127][6193536, 1, 48768, 384]cuda:0" = PlaceHolder[target=convolution]
#   %getitem_3 : Tensor "f32[1, 4, 1, 16129][64516, 1, 64516, 4]cuda:0" = PlaceHolder[target=getitem_3]
#   %getitem_4 : Tensor "f32[1, 4, 1, 16129][64516, 1, 64516, 4]cuda:0" = PlaceHolder[target=getitem_4]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%convolution, 128, 1), kwargs = {})
#   %view : Tensor "f16[1, 4, 32, 16129][6193536, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [1, 4, 32, 16129]), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 4, 32, 16129][6193536, 516128, 16129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %sub_tensor : Tensor "f32[1, 4, 32, 16129][2064512, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %getitem_3), kwargs = {})
#   %exp_default : Tensor "f32[1, 4, 32, 16129][2064512, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_1 : Tensor "f32[1, 4, 32, 16129][2064512, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_4), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[1, 4, 32, 16129][2064512, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.float16), kwargs = {})
#   %mul_2 : Tensor "f16[1, 4, 32, 16129][2064512, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, 0.1767766952966369), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_exp_mul_split_sub_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 8258048, 'x': 4129024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_exp_mul_split_sub_view_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16129
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex
    x2 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (x3 + 384*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 + 4*y0), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + 4*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp4 = libdevice.exp(tmp3)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1, 1], 0.1767766952966369, tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (y0 + 16129*x3), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r127_direct/inductor/ww/cwws3abm574cxz2u6je6beocpzgly4h6njtrnokxcha3u5bho44l.py
# Topologically Sorted Source Nodes: [out, out_1, input_1], Original ATen: [aten.view, aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution_1
#   out => view_8
#   out_1 => view_9
# Graph fragment:
#   %bmm_1 : Tensor "f16[4, 32, 16129][516128, 16129, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %view_8 : Tensor "f16[1, 4, 32, 16129][2064512, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [1, 4, 32, 16129]), kwargs = {})
#   %view_9 : Tensor "f16[1, 128, 127, 127][2064512, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [1, 128, 127, 127]), kwargs = {})
#   %convolution_1 : Tensor "f16[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%view_9, %arg4_1, %arg5_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 8258048, 'x': 4129024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_view_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 16129
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 16129*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: <AUDIT_CACHE>/r127_direct/inductor/mn/cmnnww6gxmmnpeempwxfvruhu72gu6l2n7pasbyunuie6le2742c.py
# Topologically Sorted Source Nodes: [out, out_1, input_1, normalize_1, mul_3, input_2], Original ATen: [aten.view, aten.convolution, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
# Source node to ATen node mapping:
#   input_1 => convolution_1
#   input_2 => mul_4
#   mul_3 => mul_3
#   normalize_1 => clamp_min_1, convert_element_type_10, convert_element_type_11, div_3, expand_7, pow_3, pow_4, sum_4
#   out => view_8
#   out_1 => view_9
# Graph fragment:
#   %buf15 : Tensor "f16[1, 64, 127, 127][1032256, 1, 8128, 64]cuda:0" = PlaceHolder[target=buf15]
#   %arg5_1 : Tensor "f16[64][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %sum_4 : Tensor "f32[1, 1, 127, 127][16160, 16160, 127, 1]cuda:0" = PlaceHolder[target=sum_4]
#   %arg6_1 : Tensor "f16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %view_8 : Tensor "f16[1, 4, 32, 16129][2064512, 516128, 16129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [1, 4, 32, 16129]), kwargs = {})
#   %view_9 : Tensor "f16[1, 128, 127, 127][2064512, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [1, 128, 127, 127]), kwargs = {})
#   %convolution_1 : Tensor "f16[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%view_9, %arg4_1, %arg5_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_10 : Tensor "f32[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_10, 2.0), kwargs = {})
#   %sum_4 : Tensor "f32[1, 1, 127, 127][16129, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1], True), kwargs = {})
#   %pow_4 : Tensor "f32[1, 1, 127, 127][16129, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %convert_element_type_11 : Tensor "f16[1, 1, 127, 127][16129, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_4, torch.float16), kwargs = {})
#   %clamp_min_1 : Tensor "f16[1, 1, 127, 127][16129, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_11, 1e-12), kwargs = {})
#   %expand_7 : Tensor "f16[1, 64, 127, 127][16129, 0, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_1, [1, 64, 127, 127]), kwargs = {})
#   %div_3 : Tensor "f16[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_1, %expand_7), kwargs = {})
#   %mul_3 : Tensor "f16[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %arg6_1), kwargs = {})
#   %mul_4 : Tensor "f16[1, 64, 127, 127][1032256, 16129, 127, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 8.0), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clamp_min_convolution_div_expand_linalg_vector_norm_mul_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '5C8C1E15444100DE2F29E26ABEC5DB6FE4DCB7CAEB21D22C1F22959ACFFFDA65', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4129024, 'r0_': 2064768}}
)
@triton.jit
def triton_per_fused_clamp_min_convolution_div_expand_linalg_vector_norm_mul_view_10(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16129
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = tl.sqrt_rn(tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.full([1, 1], 1e-12, tl.float32)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = (tmp2 / tmp12)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full([1, 1], 8.0, tl.float32)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr1 + (x0 + 16129*r0_1), tmp17, xmask)
''', device_str='cuda')

def partition_0(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 64, 127, 127), (1032256, 16129, 127, 1))
    assert_size_stride(arg1_1, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg2_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg3_1, (2, 4, 32, 4), (512, 128, 4, 1))
    assert_size_stride(arg4_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (1, 64, 1, 1), (64, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 64, 127, 127), (1032256, 1, 8128, 64), torch.float16)
        # Topologically Sorted Source Nodes: [normalize, mul, mul_1], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
        # [Provenance debug handles] triton_per_fused_clamp_min_div_expand_linalg_vector_norm_mul_0:1
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_div_expand_linalg_vector_norm_mul_0.run(arg0_1, arg1_1, buf1, 16129, 64, stream=stream0)
        del arg0_1
        del arg1_1
        # Topologically Sorted Source Nodes: [normalize, mul, mul_1, conv2d], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:2
        buf2 = extern_kernels.convolution(buf1, arg2_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (1, 384, 127, 127), (6193536, 1, 48768, 384), 'torch.ops.aten.convolution.default')
        del arg2_1
        del buf1
        buf3 = empty_strided_cuda((1, 4, 32, 1, 127), (16256, 4064, 127, 16256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
        # [Provenance debug handles] triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_1:3
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_1.run(arg3_1, buf2, buf3, 128, 127, 128, stream=stream0)
        buf4 = empty_strided_cuda((1, 4, 32, 1), (128, 32, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
        # [Provenance debug handles] triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_2:4
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_2.run(buf3, buf4, 128, 127, stream=stream0)
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
        # [Provenance debug handles] triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_3:5
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_3.run(arg3_1, buf2, buf4, buf5, 128, 127, 128, stream=stream0)
        buf6 = empty_strided_cuda((1, 4, 32, 1), (128, 32, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, prims.prepare_softmax_online]
        # [Provenance debug handles] triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_4:6
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_cat_expand_prepare_softmax_online_select_split_unsqueeze_view_4.run(buf5, buf6, 128, 127, stream=stream0)
        del buf5
        buf7 = empty_strided_cuda((4, 32, 16136), (518144, 16192, 1), torch.float16)
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, , context], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, aten.sub, aten.exp, aten.bmm]
        # [Provenance debug handles] triton_poi_fused__softmax_bmm_cat_exp_expand_select_split_sub_unsqueeze_view_5:7
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_bmm_cat_exp_expand_select_split_sub_unsqueeze_view_5.run(arg3_1, buf2, buf4, buf6, buf7, 128, 16136, stream=stream0)
        del buf4
        del buf6
        buf8 = empty_strided_cuda((4, 16136, 32), (516352, 32, 1), torch.float16)
        # Topologically Sorted Source Nodes: [chunk, getitem_4, unsqueeze_1, memory_v, v_1, v_2, transpose, context, ], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten.transpose, aten.bmm]
        # [Provenance debug handles] triton_poi_fused_bmm_cat_expand_select_split_transpose_unsqueeze_view_6:8
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_cat_expand_select_split_transpose_unsqueeze_view_6.run(arg3_1, buf2, buf8, 2065408, stream=stream0)
        del arg3_1
        buf9 = empty_strided_cuda((4, 32, 32), (1024, 32, 1), torch.float16)
        # Topologically Sorted Source Nodes: [chunk, getitem_3, unsqueeze, memory_k, k_1, k_2, k_3, , context, getitem_4, unsqueeze_1, memory_v, v_1, v_2, transpose], Original ATen: [aten.split, aten.select, aten.unsqueeze, aten.expand, aten.view, aten.cat, aten._softmax, aten.sub, aten.exp, aten.bmm, aten.transpose]
        # [Provenance debug handles] extern_kernels.bmm:9
        extern_kernels.bmm(buf7, buf8, out=buf9)
        del buf7
        del buf8
        buf10 = empty_strided_cuda((1, 4, 1, 16129), (64516, 1, 64516, 4), torch.float32)
        buf11 = empty_strided_cuda((1, 4, 1, 16129), (64516, 1, 64516, 4), torch.float32)
        # Topologically Sorted Source Nodes: [chunk, q_1, softmax, ], Original ATen: [aten.split, aten.view, aten._softmax, prims.prepare_softmax_online]
        # [Provenance debug handles] triton_per_fused__softmax_prepare_softmax_online_split_view_7:10
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_prepare_softmax_online_split_view_7.run(buf2, buf10, buf11, 64516, 32, stream=stream0)
        buf12 = empty_strided_cuda((1, 4, 32, 16129), (2064512, 516128, 16129, 1), torch.float16)
        # Topologically Sorted Source Nodes: [chunk, q_1, softmax, , q_2], Original ATen: [aten.split, aten.view, aten._softmax, aten.sub, aten.exp, aten.mul]
        # [Provenance debug handles] triton_poi_fused__softmax_exp_mul_split_sub_view_8:11
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_exp_mul_split_sub_view_8.run(buf2, buf10, buf11, buf12, 16129, 128, stream=stream0)
        del buf10
        del buf11
        del buf2
        buf13 = empty_strided_cuda((4, 32, 16129), (516128, 16129, 1), torch.float16)
        # Topologically Sorted Source Nodes: [chunk, context, transpose_1, out, q_1, softmax, , q_2], Original ATen: [aten.split, aten.view, aten.transpose, aten._softmax, aten.sub, aten.exp, aten.mul, aten.bmm]
        # [Provenance debug handles] extern_kernels.bmm:12
        extern_kernels.bmm(reinterpret_tensor(buf9, (4, 32, 32), (1024, 1, 32), 0), reinterpret_tensor(buf12, (4, 32, 16129), (516128, 16129, 1), 0), out=buf13)
        del buf9
        buf14 = reinterpret_tensor(buf12, (1, 128, 127, 127), (2064512, 1, 16256, 128), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [out, out_1, input_1], Original ATen: [aten.view, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_view_9:13
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_view_9.run(buf13, buf14, 128, 16129, stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [out, out_1, input_1], Original ATen: [aten.view, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:14
        buf15 = extern_kernels.convolution(buf14, arg4_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (1, 64, 127, 127), (1032256, 1, 8128, 64), 'torch.ops.aten.convolution.default')
        del arg4_1
        del buf14
        buf17 = empty_strided_cuda((1, 64, 127, 127), (1032256, 16129, 127, 1), torch.float16)
        # Topologically Sorted Source Nodes: [out, out_1, input_1, normalize_1, mul_3, input_2], Original ATen: [aten.view, aten.convolution, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
        # [Provenance debug handles] triton_per_fused_clamp_min_convolution_div_expand_linalg_vector_norm_mul_view_10:15
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_convolution_div_expand_linalg_vector_norm_mul_view_10.run(buf15, arg5_1, arg6_1, buf17, 16129, 64, stream=stream0)
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
    arg0_1 = rand_strided((1, 64, 127, 127), (1032256, 16129, 127, 1), device='cuda:0', dtype=torch.float16)
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
