import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from .utils import get_freqs, nablaT_v2

USE_SAGE_ATTENTION = False

try:
    from sageattention import sageattn
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False
    sageattn = None

if torch.cuda.get_device_capability()[0] >= 9:
    try:
        from flash_attn import flash_attn_func as FA
        print("FlashAttention 2 is found")
    except:
        FA = None

    try:
        from flash_attn_interface import flash_attn_func as FA
        print("FlashAttention 3 is found")
    except:
        FA = FA
else:
    try:
        from flash_attn import flash_attn_func as FA
        print("FlashAttention 2 is found")
    except:
        FA = None

#@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def sdpa(q, k, v, attn_mask=None):
    query = q.transpose(1, 2).contiguous()
    key = k.transpose(1, 2).contiguous()
    value = v.transpose(1, 2).contiguous()
    out = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask
        )
        .transpose(1, 2)
        .contiguous()
    )
    return out

if FA is None:
    print("FlashAttention is not found. Using SDPA instead.")
    FA = sdpa

def sage_attn(q, k, v):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    try:
        if q.dtype not in [torch.float16, torch.bfloat16]:
            original_dtype = q.dtype
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            out = sageattn(q, k, v, tensor_layout="NHD")
            out = out.to(original_dtype)
        else:
            out = sageattn(q, k, v, tensor_layout="NHD")
    except Exception:
        out = FA(q=q, k=k, v=v)

    return out

def set_sage_attention(enabled: bool):
    global USE_SAGE_ATTENTION
    if enabled and not SAGE_AVAILABLE:
        USE_SAGE_ATTENTION = False
    else:
        USE_SAGE_ATTENTION = enabled

def apply_scale_shift_norm(norm, x, scale, shift):
    return (norm(x) * (scale + 1.0) + shift).to(x.dtype)

def apply_gate_sum(x, out, gate):
    return (x + gate * out).to(x.dtype)

@torch.autocast(device_type="cuda", enabled=False)
def apply_rotary(x, rope):
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = (rope * x_).sum(dim=-1)
    return x_out.reshape(*x.shape).to(x.dtype) # Use .to(x.dtype) instead of hardcoded bfloat16


class TimeEmbeddings(nn.Module):
    def __init__(self, model_dim, time_dim, max_period=10000.0, operations=None):
        super().__init__()
        if operations is None:
            operations = nn
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer(
            "freqs", get_freqs(model_dim // 2, max_period), persistent=False
        )
        self.in_layer = operations.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = operations.Linear(time_dim, time_dim, bias=True)

    def forward(self, time):
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Don't convert to weight.dtype for quantized weights (would be uint8)
        # Let the layer's comfy_cast_weights handle dtype conversion
        time_embed = self.out_layer(self.activation(self.in_layer(time_embed)))
        return time_embed


class TextEmbeddings(nn.Module):
    def __init__(self, text_dim, model_dim, operations=None):
        super().__init__()
        if operations is None:
            operations = nn
        self.in_layer = operations.Linear(text_dim, model_dim, bias=True)
        self.norm = operations.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed):
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim, model_dim, patch_size, operations=None):
        super().__init__()
        if operations is None:
            operations = nn
        self.patch_size = patch_size
        self.in_layer = operations.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x):
        duration, height, width, dim = x.shape
        x = (
            x.view(
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 2, 4, 1, 3, 5, 6)
            .flatten(3, 6)
        )
        return self.in_layer(x)


class RoPE1D(nn.Module):
    def __init__(self, dim, max_pos=1024, max_period=10000.0):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer(f"args", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, pos):
        args = self.args[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class RoPE3D(nn.Module):
    def __init__(self, axes_dims, max_pos=(128, 128, 128), max_period=10000.0):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, shape, pos, scale_factor=(1.0, 1.0, 1.0)):
        duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(duration, 1, 1, -1).repeat(1, height, width, 1),
                args_h.view(1, height, 1, -1).repeat(duration, 1, width, 1),
                args_w.view(1, 1, width, -1).repeat(duration, height, 1, 1),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Modulation(nn.Module):
    def __init__(self, time_dim, model_dim, num_params, operations=None):
        super().__init__()
        if operations is None:
            operations = nn
        self.activation = nn.SiLU()
        self.out_layer = operations.Linear(time_dim, num_params * model_dim)
        # Only zero weights if they exist (GGMLOps doesn't initialize until load_state_dict)
        if self.out_layer.weight is not None:
            self.out_layer.weight.data.zero_()
        if self.out_layer.bias is not None:
            self.out_layer.bias.data.zero_()

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x):
        return self.out_layer(self.activation(x))

class MultiheadSelfAttentionEnc(nn.Module):
    def __init__(self, num_channels, head_dim, operations=None):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        if operations is None:
            operations = nn

        self.to_query = operations.Linear(num_channels, num_channels, bias=True)
        self.to_key = operations.Linear(num_channels, num_channels, bias=True)
        self.to_value = operations.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = operations.Linear(num_channels, num_channels, bias=True)

    def get_qkv(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None):
        if USE_SAGE_ATTENTION and SAGE_AVAILABLE:
            # SageAttention doesn't support attn_mask, so we ignore it
            out = sage_attn(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0))[0].flatten(-2, -1)
        else:
            # FA will either be flash_attn_func or sdpa
            # flash_attn_func doesn't support attn_mask, sdpa does
            if FA == sdpa and attn_mask is not None:
                out = FA(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0), attn_mask=attn_mask)[0].flatten(-2, -1)
            else:
                out = FA(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0))[0].flatten(-2, -1)
        return out

    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, rope, attn_mask=None):
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope)
        key = apply_rotary(key, rope)

        out = self.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

        out = self.out_l(out)
        return out

class MultiheadSelfAttentionDec(nn.Module):
    def __init__(self, num_channels, head_dim, operations=None):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        if operations is None:
            operations = nn

        self.to_query = operations.Linear(num_channels, num_channels, bias=True)
        self.to_key = operations.Linear(num_channels, num_channels, bias=True)
        self.to_value = operations.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = operations.Linear(num_channels, num_channels, bias=True)

    def get_qkv(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def attention(self, query, key, value, attn_mask=None):
        if USE_SAGE_ATTENTION and SAGE_AVAILABLE:
            # SageAttention doesn't support attn_mask, so we ignore it
            out = sage_attn(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0))[0].flatten(-2, -1)
        else:
            # FA will either be flash_attn_func or sdpa
            # flash_attn_func doesn't support attn_mask, sdpa does
            if FA == sdpa and attn_mask is not None:
                out = FA(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0), attn_mask=attn_mask)[0].flatten(-2, -1)
            else:
                out = FA(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0))[0].flatten(-2, -1)
        return out

    def nabla(self, query, key, value, sparse_params=None):
        query = query.unsqueeze(0).transpose(1, 2).contiguous()
        key = key.unsqueeze(0).transpose(1, 2).contiguous()
        value = value.unsqueeze(0).transpose(1, 2).contiguous()
        block_mask = nablaT_v2(
            query,
            key,
            sparse_params["sta_mask"],
            thr=sparse_params["P"],
        )
        out = (
            flex_attention(
                query,
                key,
                value,
                block_mask=block_mask
            )
            .transpose(1, 2)
            .squeeze(0)
            .contiguous()
        )
        out = out.flatten(-2, -1)
        return out

    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, rope, sparse_params=None, attn_mask=None):
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope)
        key = apply_rotary(key, rope)

        if sparse_params is not None:
            out = self.nabla(query, key, value, sparse_params=sparse_params)
        else:
            out = self.attention(query, key, value, attn_mask=attn_mask)

        out = self.out_l(out)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(self, num_channels, head_dim, operations=None):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        if operations is None:
            operations = nn

        self.to_query = operations.Linear(num_channels, num_channels, bias=True)
        self.to_key = operations.Linear(num_channels, num_channels, bias=True)
        self.to_value = operations.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = operations.Linear(num_channels, num_channels, bias=True)

    def get_qkv(self, x, cond):
        query = self.to_query(x)
        key = self.to_key(cond)
        value = self.to_value(cond)

        shape, cond_shape = query.shape[:-1], key.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*cond_shape, self.num_heads, -1)
        value = value.reshape(*cond_shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def attention(self, query, key, value, attn_mask=None):
        if USE_SAGE_ATTENTION and SAGE_AVAILABLE:
            # SageAttention doesn't support attn_mask, so we ignore it
            out = sage_attn(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0))[0].flatten(-2, -1)
        else:
            # FA will either be flash_attn_func or sdpa
            # flash_attn_func doesn't support attn_mask, sdpa does
            if FA == sdpa and attn_mask is not None:
                out = FA(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0), attn_mask=attn_mask)[0].flatten(-2, -1)
            else:
                out = FA(q=query.unsqueeze(0), k=key.unsqueeze(0), v=value.unsqueeze(0))[0].flatten(-2, -1)
        return out

    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, cond, attn_mask=None):
        query, key, value = self.get_qkv(x, cond)
        query, key = self.norm_qk(query, key)

        out = self.attention(query, key, value, attn_mask=attn_mask)
        out = self.out_l(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, operations=None):
        super().__init__()
        if operations is None:
            operations = nn
        self.in_layer = operations.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = operations.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, visual_dim, patch_size, operations=None):
        super().__init__()
        self.patch_size = patch_size
        if operations is None:
            operations = nn
        self.modulation = Modulation(time_dim, model_dim, 2, operations=operations)
        self.norm = operations.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = operations.Linear(
            model_dim, math.prod(patch_size) * visual_dim, bias=True
        )

    def forward(self, visual_embed, text_embed, time_embed):
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        visual_embed = apply_scale_shift_norm(
            self.norm,
            visual_embed,
            scale[:, None, None],
            shift[:, None, None],
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)

        duration, height, width, _ = x.shape
        x = (
            x.view(
                duration,
                height,
                width,
                -1,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )
        return x