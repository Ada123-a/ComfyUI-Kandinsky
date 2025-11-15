# GGUF loader for Kandinsky models
# Adapted from ComfyUI-GGUF (c) City96 || Apache-2.0

import warnings
import logging
import torch
import gguf

from .ops import GGMLTensor

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]])
    else:
        raise TypeError(f"Unknown field type {field_type}")

def load_gguf_state_dict(path):
    """
    Load a GGUF file and return a state dict compatible with Kandinsky models
    """
    reader = gguf.GGUFReader(path)

    arch_str = get_field(reader, "general.architecture", str)
    if arch_str != "wan":
        raise ValueError(f"Expected 'wan' architecture for Kandinsky GGUF, got '{arch_str}'")

    logging.info(f"Loading Kandinsky GGUF model with architecture: {arch_str}")

    state_dict = {}
    qtype_dict = {}

    for tensor in reader.tensors:
        tensor_name = tensor.name

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

            shape_list = list(shape)
            while len(shape_list) > 2 and shape_list[0] == 1:
                shape_list = shape_list[1:]
            while len(shape_list) > 2 and shape_list[-1] == 1:
                shape_list = shape_list[:-1]
            shape = torch.Size(shape_list)

        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
            state_dict[tensor_name] = torch_tensor
        else:
            state_dict[tensor_name] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    logging.info("GGUF qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    return state_dict
