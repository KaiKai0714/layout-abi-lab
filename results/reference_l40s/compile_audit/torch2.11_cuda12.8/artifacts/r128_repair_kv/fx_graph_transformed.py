class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 64, 128, 128]", arg1_1: "f16[1, 64, 1, 1]", arg2_1: "f16[384, 64, 1, 1]", arg3_1: "f16[2, 4, 32, 4]", arg4_1: "f16[64, 128, 1, 1]", arg5_1: "f16[64]", arg6_1: "f16[1, 64, 1, 1]"):
        # File: <REPO>/layoutabi/workload.py:59 in forward, code: return F.normalize(x, dim=1) * self.g * self.scale
        convert_element_type: "f32[1, 64, 128, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        pow_1: "f32[1, 64, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2.0);  convert_element_type = None
        sum_1: "f32[1, 1, 128, 128]" = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        pow_2: "f32[1, 1, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        convert_element_type_1: "f16[1, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(pow_2, torch.float16);  pow_2 = None
        clamp_min: "f16[1, 1, 128, 128]" = torch.ops.aten.clamp_min.default(convert_element_type_1, 1e-12);  convert_element_type_1 = None
        expand: "f16[1, 64, 128, 128]" = torch.ops.aten.expand.default(clamp_min, [1, 64, 128, 128]);  clamp_min = None
        div: "f16[1, 64, 128, 128]" = torch.ops.aten.div.Tensor(arg0_1, expand);  arg0_1 = expand = None
        mul: "f16[1, 64, 128, 128]" = torch.ops.aten.mul.Tensor(div, arg1_1);  div = arg1_1 = None
        mul_1: "f16[1, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul, 8.0);  mul = None

        # File: <REPO>/layoutabi/workload.py:90 in forward, code: q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        convolution: "f16[1, 384, 128, 128]" = torch.ops.aten.convolution.default(mul_1, arg2_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_1 = arg2_1 = None
        split = torch.ops.aten.split.Tensor(convolution, 128, 1);  convolution = None
        getitem: "f16[1, 128, 128, 128]" = split[0]
        getitem_1: "f16[1, 128, 128, 128]" = split[1]
        getitem_2: "f16[1, 128, 128, 128]" = split[2];  split = None

        # File: <REPO>/layoutabi/workload.py:96 in forward, code: memory_k = self.mem_kv[0].unsqueeze(0).expand(batch, -1, -1, -1)
        select: "f16[4, 32, 4]" = torch.ops.aten.select.int(arg3_1, 0, 0)
        unsqueeze: "f16[1, 4, 32, 4]" = torch.ops.aten.unsqueeze.default(select, 0);  select = None
        expand_1: "f16[1, 4, 32, 4]" = torch.ops.aten.expand.default(unsqueeze, [1, -1, -1, -1]);  unsqueeze = None

        # File: <REPO>/layoutabi/workload.py:93 in forward, code: k = k.reshape(batch, self.heads, self.dim_head, spatial_n)
        view_1: "f16[1, 4, 32, 16384]" = torch.ops.aten.reshape.default(getitem_1, [1, 4, 32, 16384]);  getitem_1 = None

        # File: <REPO>/layoutabi/workload.py:98 in forward, code: k = torch.cat((memory_k, k), dim=-1)
        cat: "f16[1, 4, 32, 16388]" = torch.ops.aten.cat.default([expand_1, view_1], -1);  expand_1 = view_1 = None

        # File: <REPO>/layoutabi/workload.py:102 in forward, code: k = k.softmax(dim=-1)
        convert_element_type_4: "f32[1, 4, 32, 16388]" = torch.ops.prims.convert_element_type.default(cat, torch.float32);  cat = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default_1 = torch.ops.prims.prepare_softmax_online.default(convert_element_type_4, -1)
        getitem_5: "f32[1, 4, 32, 1]" = prepare_softmax_online_default_1[0]
        getitem_6: "f32[1, 4, 32, 1]" = prepare_softmax_online_default_1[1];  prepare_softmax_online_default_1 = None
        sub_tensor_1: "f32[1, 4, 32, 16388]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_5);  convert_element_type_4 = getitem_5 = None
        exp_default_1: "f32[1, 4, 32, 16388]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None

        # File: <REPO>/layoutabi/workload.py:102 in forward, code: k = k.softmax(dim=-1)
        div_2: "f32[1, 4, 32, 16388]" = torch.ops.aten.div.Tensor(exp_default_1, getitem_6);  exp_default_1 = getitem_6 = None
        convert_element_type_5: "f16[1, 4, 32, 16388]" = torch.ops.prims.convert_element_type.default(div_2, torch.float16);  div_2 = None

        # File: <REPO>/layoutabi/workload.py:26 in bhnd_backed_view, code: return x.transpose(-2, -1).contiguous().transpose(-2, -1)
        permute: "f16[1, 4, 16388, 32]" = torch.ops.aten.permute.default(convert_element_type_5, [0, 1, 3, 2]);  convert_element_type_5 = None
        clone: "f16[1, 4, 16388, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        permute_1: "f16[1, 4, 32, 16388]" = torch.ops.aten.permute.default(clone, [0, 1, 3, 2]);  clone = None

        # File: <REPO>/layoutabi/workload.py:38 in context_from_kv, code: return k @ v.transpose(-2, -1)
        expand_3: "f16[1, 4, 32, 16388]" = torch.ops.aten.expand.default(permute_1, [1, 4, 32, 16388]);  permute_1 = None
        view_3: "f16[4, 32, 16388]" = torch.ops.aten.reshape.default(expand_3, [4, 32, 16388]);  expand_3 = None

        # File: <REPO>/layoutabi/workload.py:97 in forward, code: memory_v = self.mem_kv[1].unsqueeze(0).expand(batch, -1, -1, -1)
        select_1: "f16[4, 32, 4]" = torch.ops.aten.select.int(arg3_1, 0, 1);  arg3_1 = None
        unsqueeze_1: "f16[1, 4, 32, 4]" = torch.ops.aten.unsqueeze.default(select_1, 0);  select_1 = None
        expand_2: "f16[1, 4, 32, 4]" = torch.ops.aten.expand.default(unsqueeze_1, [1, -1, -1, -1]);  unsqueeze_1 = None

        # File: <REPO>/layoutabi/workload.py:94 in forward, code: v = v.reshape(batch, self.heads, self.dim_head, spatial_n)
        view_2: "f16[1, 4, 32, 16384]" = torch.ops.aten.reshape.default(getitem_2, [1, 4, 32, 16384]);  getitem_2 = None

        # File: <REPO>/layoutabi/workload.py:99 in forward, code: v = torch.cat((memory_v, v), dim=-1)
        cat_1: "f16[1, 4, 32, 16388]" = torch.ops.aten.cat.default([expand_2, view_2], -1);  expand_2 = view_2 = None

        # File: <REPO>/layoutabi/workload.py:26 in bhnd_backed_view, code: return x.transpose(-2, -1).contiguous().transpose(-2, -1)
        permute_2: "f16[1, 4, 16388, 32]" = torch.ops.aten.permute.default(cat_1, [0, 1, 3, 2]);  cat_1 = None
        clone_1: "f16[1, 4, 16388, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None

        # File: <REPO>/layoutabi/workload.py:38 in context_from_kv, code: return k @ v.transpose(-2, -1)
        expand_4: "f16[1, 4, 16388, 32]" = torch.ops.aten.expand.default(clone_1, [1, 4, 16388, 32]);  clone_1 = None
        view_4: "f16[4, 16388, 32]" = torch.ops.aten.reshape.default(expand_4, [4, 16388, 32]);  expand_4 = None
        bmm: "f16[4, 32, 32]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[1, 4, 32, 32]" = torch.ops.aten.reshape.default(bmm, [1, 4, 32, 32]);  bmm = None

        # File: <REPO>/layoutabi/workload.py:104 in forward, code: out = context.transpose(-2, -1) @ q
        permute_5: "f16[1, 4, 32, 32]" = torch.ops.aten.permute.default(view_5, [0, 1, 3, 2]);  view_5 = None
        expand_5: "f16[1, 4, 32, 32]" = torch.ops.aten.expand.default(permute_5, [1, 4, 32, 32]);  permute_5 = None
        view_6: "f16[4, 32, 32]" = torch.ops.aten.reshape.default(expand_5, [4, 32, 32]);  expand_5 = None

        # File: <REPO>/layoutabi/workload.py:92 in forward, code: q = q.reshape(batch, self.heads, self.dim_head, spatial_n)
        view: "f16[1, 4, 32, 16384]" = torch.ops.aten.reshape.default(getitem, [1, 4, 32, 16384]);  getitem = None

        # File: <REPO>/layoutabi/workload.py:101 in forward, code: q = q.softmax(dim=-2) * self.scale
        convert_element_type_2: "f32[1, 4, 32, 16384]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None

        # No stacktrace found for following nodes
        prepare_softmax_online_default = torch.ops.prims.prepare_softmax_online.default(convert_element_type_2, -2)
        getitem_3: "f32[1, 4, 1, 16384]" = prepare_softmax_online_default[0]
        getitem_4: "f32[1, 4, 1, 16384]" = prepare_softmax_online_default[1];  prepare_softmax_online_default = None
        sub_tensor: "f32[1, 4, 32, 16384]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_3);  convert_element_type_2 = getitem_3 = None
        exp_default: "f32[1, 4, 32, 16384]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None

        # File: <REPO>/layoutabi/workload.py:101 in forward, code: q = q.softmax(dim=-2) * self.scale
        div_1: "f32[1, 4, 32, 16384]" = torch.ops.aten.div.Tensor(exp_default, getitem_4);  exp_default = getitem_4 = None
        convert_element_type_3: "f16[1, 4, 32, 16384]" = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        mul_2: "f16[1, 4, 32, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.1767766952966369);  convert_element_type_3 = None

        # File: <REPO>/layoutabi/workload.py:104 in forward, code: out = context.transpose(-2, -1) @ q
        expand_6: "f16[1, 4, 32, 16384]" = torch.ops.aten.expand.default(mul_2, [1, 4, 32, 16384]);  mul_2 = None
        view_7: "f16[4, 32, 16384]" = torch.ops.aten.reshape.default(expand_6, [4, 32, 16384]);  expand_6 = None
        bmm_1: "f16[4, 32, 16384]" = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
        view_8: "f16[1, 4, 32, 16384]" = torch.ops.aten.reshape.default(bmm_1, [1, 4, 32, 16384]);  bmm_1 = None

        # File: <REPO>/layoutabi/workload.py:105 in forward, code: out = out.reshape(batch, self.heads * self.dim_head, height, width)
        view_9: "f16[1, 128, 128, 128]" = torch.ops.aten.reshape.default(view_8, [1, 128, 128, 128]);  view_8 = None

        # File: <REPO>/layoutabi/workload.py:106 in forward, code: return self.to_out(out)
        convolution_1: "f16[1, 64, 128, 128]" = torch.ops.aten.convolution.default(view_9, arg4_1, arg5_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_9 = arg4_1 = arg5_1 = None

        # File: <REPO>/layoutabi/workload.py:59 in forward, code: return F.normalize(x, dim=1) * self.g * self.scale
        convert_element_type_10: "f32[1, 64, 128, 128]" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        pow_3: "f32[1, 64, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_10, 2.0);  convert_element_type_10 = None
        sum_4: "f32[1, 1, 128, 128]" = torch.ops.aten.sum.dim_IntList(pow_3, [1], True);  pow_3 = None
        pow_4: "f32[1, 1, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None
        convert_element_type_11: "f16[1, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(pow_4, torch.float16);  pow_4 = None
        clamp_min_1: "f16[1, 1, 128, 128]" = torch.ops.aten.clamp_min.default(convert_element_type_11, 1e-12);  convert_element_type_11 = None
        expand_7: "f16[1, 64, 128, 128]" = torch.ops.aten.expand.default(clamp_min_1, [1, 64, 128, 128]);  clamp_min_1 = None
        div_3: "f16[1, 64, 128, 128]" = torch.ops.aten.div.Tensor(convolution_1, expand_7);  convolution_1 = expand_7 = None
        mul_3: "f16[1, 64, 128, 128]" = torch.ops.aten.mul.Tensor(div_3, arg6_1);  div_3 = arg6_1 = None
        mul_4: "f16[1, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_3, 8.0);  mul_3 = None
        return (mul_4,)
