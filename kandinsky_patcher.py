import torch
import gc
import comfy.model_patcher
import comfy.model_management as model_management
import comfy.utils

from .src.kandinsky.models.dit import DiffusionTransformer3D
from .src.kandinsky.fp8_utils import convert_fp8_linear, convert_fp8_linear_on_the_fly

KANDINSKY_CONFIGS = {
    "sft_5s": {"config": "config_5s_sft.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_sft_5s.safetensors"},
    "sft_10s": {"config": "config_10s_sft.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_sft_10s.safetensors"},
    "i2v_5s": {"config": "config_5s_i2v.yaml", "ckpt": "kandinsky/kandinsky5lite_i2v_5s.safetensors"},
    "i2v_pro_20b": {"config": "config_5s_i2v_pro_20b.yaml", "ckpt": "kandinsky/kandinsky5_i2v_pro_sft_5s_20b.safetensors"},
    "t2v_pro_20b": {"config": "config_5s_t2v_pro_20b.yaml", "ckpt": "kandinsky/kandinsky5Pro_t2v_sft_5s.safetensors"},
    "pretrain_5s": {"config": "config_5s_pretrain.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_pretrain_5s.safetensors"},
    "pretrain_10s": {"config": "config_10s_pretrain.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_pretrain_10s.safetensors"},
    "nocfg_5s": {"config": "config_5s_nocfg.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_nocfg_5s.safetensors"},
    "nocfg_10s": {"config": "config_10s_nocfg.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_sft_10s.safetensors"},
    "distil_5s": {"config": "config_5s_distil.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_distilled16steps_5s.safetensors"},
    "distil_10s": {"config": "config_10s_distil.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_distilled16steps_10s.safetensors"},
}

class KandinskyModelHandler(torch.nn.Module):
    """
    A lightweight placeholder for the Kandinsky DiT model.
    """
    def __init__(self, conf, ckpt_path):
        super().__init__()
        self.conf = conf
        self.ckpt_path = ckpt_path
        self.diffusion_model = None
        self.size = int(conf.model.dit_params.model_dim * 12 * 24 * 1.5)

class KandinskyPatcher(comfy.model_patcher.ModelPatcher):
    """
    Custom ModelPatcher to load, patch, and manage the Kandinsky DiT model.
    """
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    @property
    def is_loaded(self) -> bool:
        return hasattr(self, 'model') and self.model is not None and self.model.diffusion_model is not None

    def patch_model(self, device_to=None, *args, **kwargs):
        if self.is_loaded:
            self.model.diffusion_model.to(self.load_device)
            return

        model_dtype = model_management.unet_dtype()

        dit_params = dict(self.model.conf.model.dit_params)

        if hasattr(self.model.conf, 'block_swap') and self.model.conf.block_swap.enabled:
            dit_params['block_swap_enabled'] = True
            dit_params['blocks_in_memory'] = self.model.conf.block_swap.get('blocks_in_memory', 6)
            dit_params['pin_first_n_blocks'] = self.model.conf.block_swap.get('pin_first_n_blocks', 2)
            dit_params['pin_last_n_blocks'] = self.model.conf.block_swap.get('pin_last_n_blocks', 2)

        model = DiffusionTransformer3D(**dit_params)

        model.to(dtype=model_dtype)

        # Ensure fp8 and gguf are mutually exclusive
        use_fp8 = hasattr(self.model.conf, 'use_fp8') and self.model.conf.use_fp8
        use_gguf = hasattr(self.model.conf, 'use_gguf') and self.model.conf.use_gguf

        if use_fp8 and use_gguf:
            raise ValueError("Cannot use both FP8 and GGUF formats simultaneously. Please choose one.")

        # Load model weights - support GGUF or regular formats
        if use_gguf:
            print("Loading GGUF model...")
            from .gguf_loader import load_gguf_state_dict
            sd = load_gguf_state_dict(self.model.ckpt_path)
        else:
            sd = comfy.utils.load_torch_file(self.model.ckpt_path)

        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print("Kandinsky missing keys:", m)
        if len(u) > 0:
            print("Kandinsky unexpected keys:", u)

        # Apply FP8 quantization only if not using GGUF (GGUF models are already quantized)
        if use_fp8 and not use_gguf:
            print(f"Applying FP8 quantization (mode: {self.model.conf.fp8_mode})...")
            try:
                if self.model.conf.fp8_mode == "with_map":
                    convert_fp8_linear(model, self.model.ckpt_path, model_dtype)
                else:
                    convert_fp8_linear_on_the_fly(model, model_dtype)
                print("FP8 quantization applied successfully.")
            except Exception as e:
                print(f"Warning: Failed to apply FP8 quantization: {e}")
                print("Continuing with normal precision weights.")
        elif use_gguf:
            print("GGUF model loaded successfully (already quantized)")

        model.eval()

        if hasattr(model, 'block_swap_enabled') and model.block_swap_enabled:

            pinned_blocks = set()
            for i in range(min(model.pin_first_n_blocks, model.num_visual_blocks)):
                pinned_blocks.add(i)
            for i in range(max(0, model.num_visual_blocks - model.pin_last_n_blocks), model.num_visual_blocks):
                pinned_blocks.add(i)

            model.pinned_blocks = pinned_blocks

            model.time_embeddings.to(self.load_device)
            model.text_embeddings.to(self.load_device)
            model.pooled_text_embeddings.to(self.load_device)
            model.visual_embeddings.to(self.load_device)
            model.text_rope_embeddings.to(self.load_device)
            model.visual_rope_embeddings.to(self.load_device)
            model.out_layer.to(self.load_device)

            for block in model.text_transformer_blocks:
                block.to(self.load_device)

            for i, block in enumerate(model.visual_transformer_blocks):
                if i in pinned_blocks:
                    block.to(self.load_device)
                else:
                    block.to(self.offload_device)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if model_management.force_channels_last():
                pass
        else:
            model.to(self.load_device)

            if model_management.force_channels_last():
                model.to(memory_format=torch.channels_last)

        self.model.diffusion_model = model
        return

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        if self.is_loaded:
            self.model.diffusion_model.to(self.offload_device)

        if unpatch_weights:
             if self.is_loaded:
                del self.model.diffusion_model
                self.model.diffusion_model = None
             gc.collect()
             model_management.soft_empty_cache()
        return
