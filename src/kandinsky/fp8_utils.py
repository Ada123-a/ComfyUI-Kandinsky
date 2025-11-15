"""
FP8 Quantization Utilities for Kandinsky Models

This module provides FP8 (8-bit floating point) quantization support for neural network layers,
enabling reduced memory usage and potentially faster inference.
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_fp_maxval(bits=8, mantissa_bit=3, sign_bits=1):
    """
    Calculate the maximum representable value for a given floating-point format.

    Args:
        bits: Total number of bits (default: 8)
        mantissa_bit: Number of mantissa bits (default: 3 for E4M3)
        sign_bits: Number of sign bits (default: 1)

    Returns:
        Maximum representable value for the format
    """
    _bits = torch.tensor(bits)
    _mantissa_bit = torch.tensor(mantissa_bit)
    _sign_bits = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(_mantissa_bit), 1, _bits - _sign_bits)
    E = _bits - _sign_bits - M
    bias = 2 ** (E - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i+1))
    maxval = mantissa * 2 ** (2**E - 1 - bias)
    return maxval


def quantize_to_fp8(x, bits=8, mantissa_bit=3, sign_bits=1):
    """
    Quantize a tensor to FP8 format (default E4M3).

    Args:
        x: Input tensor to quantize
        bits: Total number of bits (default: 8)
        mantissa_bit: Number of mantissa bits (default: 3)
        sign_bits: Number of sign bits (default: 1)

    Returns:
        Tuple of (quantized-dequantized tensor, log scales)
    """
    bits = torch.tensor(bits)
    mantissa_bit = torch.tensor(mantissa_bit)
    sign_bits = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(mantissa_bit), 1, bits - sign_bits)
    E = bits - sign_bits - M
    bias = 2 ** (E - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i+1))
    maxval = mantissa * 2 ** (2**E - 1 - bias)
    minval = - maxval if sign_bits == 1 else torch.zeros_like(maxval)
    input_clamp = torch.min(torch.max(x, minval), maxval)
    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input_clamp)) + bias)).detach(), 1.0)
    log_scales = 2.0 ** (log_scales - M - bias.type(x.dtype))
    # dequant
    qdq_out = torch.round(input_clamp / log_scales) * log_scales
    return qdq_out, log_scales


def fp8_tensor_quant(x, scale, bits=8, mantissa_bit=3, sign_bits=1):
    """
    Quantize a tensor to FP8 with scaling.

    Args:
        x: Input tensor
        scale: Scaling factor
        bits: Total number of bits (default: 8)
        mantissa_bit: Number of mantissa bits (default: 3)
        sign_bits: Number of sign bits (default: 1)

    Returns:
        Tuple of (quantized-dequantized tensor, scale, log_scales)
    """
    for i in range(len(x.shape) - 1):
        scale = scale.unsqueeze(-1)
    new_x = x / scale
    quant_dequant_x, log_scales = quantize_to_fp8(new_x, bits=bits, mantissa_bit=mantissa_bit, sign_bits=sign_bits)
    return quant_dequant_x, scale, log_scales


def fp8_activation_dequant(qdq_out, scale, dtype):
    """
    Dequantize FP8 activations back to the original dtype.

    Args:
        qdq_out: Quantized-dequantized output
        scale: Scaling factor
        dtype: Target dtype

    Returns:
        Dequantized tensor
    """
    qdq_out = qdq_out.type(dtype)
    quant_dequant_x = qdq_out * scale.to(dtype)
    return quant_dequant_x


def fp8_linear_forward(cls, original_dtype, input):
    """
    Forward pass for a linear layer with FP8 quantization.

    Args:
        cls: The linear layer instance
        original_dtype: Original computation dtype
        input: Input tensor

    Returns:
        Output tensor after FP8 quantized linear operation
    """
    weight_dtype = cls.weight.dtype

    # Check if weights need to be quantized to FP8
    if cls.weight.dtype != torch.float8_e4m3fn:
        maxval = get_fp_maxval()
        scale = torch.max(torch.abs(cls.weight.flatten())) / maxval
        linear_weight, scale, log_scales = fp8_tensor_quant(cls.weight, scale)
        linear_weight = linear_weight.to(torch.float8_e4m3fn)
        weight_dtype = linear_weight.dtype
    else:
        # Weights are already in FP8, use stored scale
        scale = cls.fp8_scale.to(cls.weight.device)
        linear_weight = cls.weight

    # Perform FP8 computation if weights are in FP8
    if weight_dtype == torch.float8_e4m3fn:
        # Dequantize weights for computation
        cls_dequant = fp8_activation_dequant(linear_weight, scale, original_dtype)
        if cls.bias is not None:
            output = F.linear(input, cls_dequant, cls.bias)
        else:
            output = F.linear(input, cls_dequant)
        return output
    else:
        # Fall back to original forward pass
        return cls.original_forward(input)


def convert_fp8_linear(module, dit_weight_path, original_dtype, params_to_keep={}):
    """
    Convert linear layers in a module to use FP8 quantization.

    This function finds all Linear layers in visual_transformer_blocks and either:
    1. Uses pre-converted FP8 weights if the model is already in FP8 format
    2. Converts weights to FP8 if they're in higher precision

    Args:
        module: The neural network module to convert
        dit_weight_path: Path to the DiT weights file (used to find FP8 scales)
        original_dtype: Original computation dtype
        params_to_keep: Dictionary of parameters to preserve (unused, for compatibility)

    Returns:
        None (modifies module in-place)

    Raises:
        ValueError: If FP8 scales file is not found
    """
    setattr(module, "fp8_matmul_enabled", True)

    # Loading FP8 scales file
    # For pre-converted models: model_fp8.safetensors → model_fp8_scales.pt
    # For regular models: model.safetensors → model_map.pt (legacy)
    scales_path = dit_weight_path.replace('.safetensors', '_scales.pt')
    if not os.path.exists(scales_path):
        # Try legacy _map.pt naming
        scales_path = dit_weight_path.replace('.safetensors', '_map.pt')
    if not os.path.exists(scales_path):
        # Try .pt extension
        scales_path = dit_weight_path.replace('.pt', '_scales.pt')

    if os.path.exists(scales_path):
        fp8_scales = torch.load(scales_path, map_location=lambda storage, loc: storage)
        print(f"Loaded FP8 scales from: {os.path.basename(scales_path)}")
    else:
        raise ValueError(f"Invalid FP8 scales path: {scales_path}. "
                        f"FP8 quantization requires a scales file (_scales.pt or _map.pt) alongside the model weights.")

    fp8_layers = []
    skipped_layers = []
    already_fp8 = 0

    for key, layer in module.named_modules():
        # Only convert linear layers in transformer blocks
        if isinstance(layer, nn.Linear) and ('visual_transformer_blocks' in key):
            if key in fp8_scales:
                fp8_layers.append(key)
                original_forward = layer.forward

                # Store the FP8 scale factor first
                setattr(layer, "fp8_scale", fp8_scales[key].to(dtype=original_dtype))

                # Check if weights are already in FP8
                if layer.weight.dtype == torch.float8_e4m3fn:
                    # Weights already in FP8, just set up forward pass
                    already_fp8 += 1
                else:
                    # Convert weights to FP8 (ensure on CPU for conversion)
                    weight_device = layer.weight.device
                    layer.weight = torch.nn.Parameter(layer.weight.cpu().to(torch.float8_e4m3fn).to(weight_device))

                # Store original forward method and replace with FP8 version
                setattr(layer, "original_forward", original_forward)
                setattr(layer, "forward", lambda input, m=layer: fp8_linear_forward(m, original_dtype, input))
            else:
                skipped_layers.append(key)

    if len(fp8_layers) > 0:
        print(f"Configured {len(fp8_layers)} layers for FP8 inference")
        if already_fp8 > 0:
            print(f"  - {already_fp8} layers were already in FP8 format (pre-converted)")
            print(f"  - {len(fp8_layers) - already_fp8} layers converted to FP8")
    else:
        print("Warning: No layers were configured for FP8. Check if the model architecture matches expected patterns.")

    if len(skipped_layers) > 0:
        print(f"Note: {len(skipped_layers)} layers skipped (not in FP8 scales)")


def convert_fp8_linear_on_the_fly(module, original_dtype):
    """
    Convert linear layers to FP8 on-the-fly without a pre-computed mapping file.

    This calculates FP8 scales dynamically from the current weights. This is slower
    than using a pre-computed mapping but doesn't require an external file.

    Args:
        module: The neural network module to convert
        original_dtype: Original computation dtype

    Returns:
        None (modifies module in-place)
    """
    setattr(module, "fp8_matmul_enabled", True)

    fp8_layers = []
    total_params_before = 0
    total_params_after = 0

    for key, layer in module.named_modules():
        # Only convert linear layers in transformer blocks
        if isinstance(layer, nn.Linear) and ('visual_transformer_blocks' in key):
            fp8_layers.append(key)
            original_forward = layer.forward

            # Track memory usage
            total_params_before += layer.weight.numel() * layer.weight.element_size()

            # Calculate FP8 scale on-the-fly (on CPU to avoid memory spike)
            weight_device = layer.weight.device
            weight_cpu = layer.weight.cpu()
            maxval = get_fp_maxval()
            scale = torch.max(torch.abs(weight_cpu.flatten())) / maxval

            # Convert weights to FP8
            linear_weight, scale, log_scales = fp8_tensor_quant(weight_cpu, scale)
            layer.weight = torch.nn.Parameter(linear_weight.to(torch.float8_e4m3fn).to(weight_device))

            # Track memory usage
            total_params_after += layer.weight.numel() * layer.weight.element_size()

            # Store the FP8 scale factor
            setattr(layer, "fp8_scale", scale.to(dtype=original_dtype))

            # Store original forward method and replace with FP8 version
            setattr(layer, "original_forward", original_forward)
            setattr(layer, "forward", lambda input, m=layer: fp8_linear_forward(m, original_dtype, input))

    if len(fp8_layers) > 0:
        memory_saved_mb = (total_params_before - total_params_after) / (1024 * 1024)
        print(f"Converted {len(fp8_layers)} layers to FP8 quantization (on-the-fly)")
        print(f"Estimated memory saved: {memory_saved_mb:.1f} MB")
    else:
        print("Warning: No layers were converted to FP8. Check if the model architecture matches expected patterns.")
