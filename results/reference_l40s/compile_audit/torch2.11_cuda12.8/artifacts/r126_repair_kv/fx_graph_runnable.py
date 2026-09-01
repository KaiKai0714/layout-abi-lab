
import os
os.environ['PYTORCH_VERSION'] = '2.11.0'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '<AUDIT_CACHE>/r126_repair_kv/inductor'
os.environ['TRITON_CACHE_DIR'] = '<AUDIT_CACHE>/r126_repair_kv/triton'
os.environ['TORCH_COMPILE_DEBUG'] = '1'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.replay_side_effects = True
torch._dynamo.config.side_effect_replay_policy = 'info'
torch._dynamo.config.specialize_int = False
torch._dynamo.config.specialize_float = False
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.capture_scalar_outputs = False
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch._dynamo.config.prefer_deferred_runtime_asserts_over_guards = False
torch._dynamo.config.do_not_emit_runtime_asserts = False
torch._dynamo.config.allow_rnn = False
torch._dynamo.config.debug_dir_root = '<REPO>/results/local_l40s_compile_three_level_audit/artifacts/r126_repair_kv/debug_tmp/torch_compile_debug'
torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.11.0+cu128
# torch cuda version: 12.8
# torch git version: 70d99e998b4955e0049d13a98d77ae1b14db1f45


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Fri_Feb_21_20:23:50_PST_2025 
# Cuda compilation tools, release 12.8, V12.8.93 
# Build cuda_12.8.r12.8/compiler.35583870_0 

# GPU Hardware Info: 
# NVIDIA L40S : 4 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2.0);  convert_element_type = None
        sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(pow_2, torch.float16);  pow_2 = None
        clamp_min = torch.ops.aten.clamp_min.default(convert_element_type_1, 1e-12);  convert_element_type_1 = None
        expand = torch.ops.aten.expand.default(clamp_min, [1, 64, 126, 126]);  clamp_min = None
        div = torch.ops.aten.div.Tensor(arg0_1, expand);  arg0_1 = expand = None
        mul = torch.ops.aten.mul.Tensor(div, arg1_1);  div = arg1_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, 8.0);  mul = None
        convolution = torch.ops.aten.convolution.default(mul_1, arg2_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_1 = arg2_1 = None
        split = torch.ops.aten.split.Tensor(convolution, 128, 1);  convolution = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2];  split = None
        view = torch.ops.aten.view.default(getitem, [1, 4, 32, 15876]);  getitem = None
        view_1 = torch.ops.aten.view.default(getitem_1, [1, 4, 32, 15876]);  getitem_1 = None
        view_2 = torch.ops.aten.view.default(getitem_2, [1, 4, 32, 15876]);  getitem_2 = None
        select = torch.ops.aten.select.int(arg3_1, 0, 0)
        unsqueeze = torch.ops.aten.unsqueeze.default(select, 0);  select = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze, [1, -1, -1, -1]);  unsqueeze = None
        select_1 = torch.ops.aten.select.int(arg3_1, 0, 1);  arg3_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(select_1, 0);  select_1 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_1, [1, -1, -1, -1]);  unsqueeze_1 = None
        cat = torch.ops.aten.cat.default([expand_1, view_1], -1);  expand_1 = view_1 = None
        cat_1 = torch.ops.aten.cat.default([expand_2, view_2], -1);  expand_2 = view_2 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        amax = torch.ops.aten.amax.default(convert_element_type_2, [-2], True)
        sub = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp, [-2], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_2);  exp = sum_2 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.1767766952966369);  convert_element_type_3 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(cat, torch.float32);  cat = None
        amax_1 = torch.ops.aten.amax.default(convert_element_type_4, [-1], True)
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_4, amax_1);  convert_element_type_4 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp_1, sum_3);  exp_1 = sum_3 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(div_2, torch.float16);  div_2 = None
        permute = torch.ops.aten.permute.default(convert_element_type_5, [0, 1, 3, 2]);  convert_element_type_5 = None
        clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        permute_1 = torch.ops.aten.permute.default(clone, [0, 1, 3, 2]);  clone = None
        permute_2 = torch.ops.aten.permute.default(cat_1, [0, 1, 3, 2]);  cat_1 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        expand_3 = torch.ops.aten.expand.default(permute_1, [1, 4, 32, 15880]);  permute_1 = None
        view_3 = torch.ops.aten.view.default(expand_3, [4, 32, 15880]);  expand_3 = None
        expand_4 = torch.ops.aten.expand.default(clone_1, [1, 4, 15880, 32]);  clone_1 = None
        view_4 = torch.ops.aten.view.default(expand_4, [4, 15880, 32]);  expand_4 = None
        bmm = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5 = torch.ops.aten.view.default(bmm, [1, 4, 32, 32]);  bmm = None
        permute_5 = torch.ops.aten.permute.default(view_5, [0, 1, 3, 2]);  view_5 = None
        expand_5 = torch.ops.aten.expand.default(permute_5, [1, 4, 32, 32]);  permute_5 = None
        view_6 = torch.ops.aten.view.default(expand_5, [4, 32, 32]);  expand_5 = None
        expand_6 = torch.ops.aten.expand.default(mul_2, [1, 4, 32, 15876]);  mul_2 = None
        view_7 = torch.ops.aten.view.default(expand_6, [4, 32, 15876]);  expand_6 = None
        bmm_1 = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
        view_8 = torch.ops.aten.view.default(bmm_1, [1, 4, 32, 15876]);  bmm_1 = None
        view_9 = torch.ops.aten.view.default(view_8, [1, 128, 126, 126]);  view_8 = None
        convolution_1 = torch.ops.aten.convolution.default(view_9, arg4_1, arg5_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_9 = arg4_1 = arg5_1 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_10, 2.0);  convert_element_type_10 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(pow_3, [1], True);  pow_3 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(pow_4, torch.float16);  pow_4 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(convert_element_type_11, 1e-12);  convert_element_type_11 = None
        expand_7 = torch.ops.aten.expand.default(clamp_min_1, [1, 64, 126, 126]);  clamp_min_1 = None
        div_3 = torch.ops.aten.div.Tensor(convolution_1, expand_7);  convolution_1 = expand_7 = None
        mul_3 = torch.ops.aten.mul.Tensor(div_3, arg6_1);  div_3 = arg6_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 8.0);  mul_3 = None
        return (mul_4,)

def load_args(reader):
    buf0 = reader.storage(None, 2032128, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (1, 64, 126, 126), dtype=torch.float16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (1, 64, 1, 1), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 49152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (384, 64, 1, 1), dtype=torch.float16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf3, (2, 4, 32, 4), dtype=torch.float16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 16384, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf4, (64, 128, 1, 1), dtype=torch.float16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf5, (64,), dtype=torch.float16, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf6, (1, 64, 1, 1), dtype=torch.float16, is_leaf=True)  # arg6_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)