import torch
import folder_paths
import comfy.model_patcher
import comfy.model_management
import comfy.utils
from omegaconf import OmegaConf
from typing_extensions import override
from comfy_api.latest import io
import os

from .kandinsky_patcher import KandinskyModelHandler, KandinskyPatcher
from .src.kandinsky.magcache_utils import set_magcache_params
from .src.kandinsky.models.text_embedders import Qwen2_5_VLTextEmbedder

class KandinskyLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        from .kandinsky_patcher import KANDINSKY_CONFIGS

        if "diffusion_models" in folder_paths.folder_names_and_paths:
            paths, extensions = folder_paths.folder_names_and_paths["diffusion_models"]
            if '.gguf' not in extensions:
                extensions.add('.gguf')
                folder_paths.folder_names_and_paths["diffusion_models"] = (paths, extensions)

        checkpoint_files = folder_paths.get_filename_list("diffusion_models")

        return io.Schema(
            node_id="KandinskyV5_Loader",
            display_name="Kandinsky 5 Loader",
            category="Kandinsky",
            description="Loads a Kandinsky-5 text-to-video model variant.",
            inputs=[
                io.Combo.Input("variant", options=list(KANDINSKY_CONFIGS.keys()), default="sft_5s"),
                io.Combo.Input("checkpoint_file", options=checkpoint_files,
                              tooltip="Select checkpoint file (.safetensors or .gguf) from diffusion_models folder. Leave at default to use the variant's default checkpoint."),
                io.Boolean.Input("use_magcache", default=False, tooltip="Enable MagCache for faster inference."),
                io.Float.Input("magcache_threshold", default=0.12, min=0.01, max=0.3, step=0.01,
                              tooltip="MagCache quality threshold. Lower = better quality, slower. Higher = faster, more artifacts."),
                io.Int.Input("blocks_in_memory", default=0, min=1, max=60,
                            tooltip="Block swapping. For 20B model: 24GB=1-3, 48GB=3-5."),
                io.Combo.Input("fp8_mode", options=["off", "on_the_fly", "with_map"], default="off",
                              tooltip="FP8 mode: 'off' = disabled, 'on_the_fly' = converts model to fp8, 'with_map' = uses pre-converted FP8 model with _scales.pt file. Only used with .safetensors files, not .gguf."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, variant: str, checkpoint_file: str, use_magcache: bool, magcache_threshold: float, blocks_in_memory: int, fp8_mode: str) -> io.NodeOutput:
        from .kandinsky_patcher import KANDINSKY_CONFIGS
        config_data = KANDINSKY_CONFIGS[variant]

        base_path = os.path.dirname(__file__)
        config_path = os.path.join(base_path, 'src', 'configs', config_data["config"])

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at '{config_path}'.")

        try:
            ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_file}' not found in diffusion_models folder.")

        conf = OmegaConf.load(config_path)

        if blocks_in_memory > 0 and hasattr(conf, 'block_swap') and conf.block_swap.enabled:
            conf.block_swap.blocks_in_memory = blocks_in_memory

        is_gguf = checkpoint_file.lower().endswith('.gguf')
        is_fp8 = not is_gguf and fp8_mode != "off"

        if is_gguf:
            print(f"Loading GGUF model: {checkpoint_file}")
        elif is_fp8:
            print(f"Loading model with FP8 quantization (mode: {fp8_mode})")
        else:
            print(f"Loading model: {checkpoint_file}")

        handler = KandinskyModelHandler(conf, ckpt_path)
        patcher = KandinskyPatcher(
            handler,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device()
        )
        handler.conf.use_magcache = use_magcache
        handler.conf.magcache_threshold = magcache_threshold
        handler.conf.use_fp8 = is_fp8
        handler.conf.use_gguf = is_gguf
        handler.conf.fp8_mode = fp8_mode

        return io.NodeOutput(patcher)


class KandinskyTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyV5_TextEncode",
            display_name="Kandinsky 5 Text Encode",
            category="Kandinsky",
            description="Encodes text using Kandinsky's combined CLIP and Qwen2.5-VL embedding logic.",
            inputs=[
                io.Clip.Input("clip", tooltip="A standard CLIP-L/14 model. Use CLIPLoader with 'stable_diffusion' type."),
                io.Clip.Input("qwen_vl", tooltip="The Qwen2.5-VL model. Use CLIPLoader with 'qwen_image' type."),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_text", multiline=True, dynamic_prompts=True, default="Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"),
                io.Combo.Input("content_type", options=["video", "image"], default="video"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def _get_raw_embeds(cls, text: str, clip_wrapper: comfy.sd.CLIP, qwen_vl_wrapper: comfy.sd.CLIP, content_type: str):
        import comfy.model_management as mm

        load_device = mm.get_torch_device()
        try:
            if hasattr(clip_wrapper, 'cond_stage_model'):
                clip_wrapper.cond_stage_model.to(load_device)
            if hasattr(qwen_vl_wrapper, 'cond_stage_model'):
                qwen_vl_wrapper.cond_stage_model.to(load_device)
        except Exception:
            pass

        clip_tokens = clip_wrapper.tokenize(text)
        _, pooled_embed = clip_wrapper.encode_from_tokens(clip_tokens, return_pooled=True)

        qwen_template_config = Qwen2_5_VLTextEmbedder.PROMPT_TEMPLATE
        prompt_template = "\n".join(qwen_template_config["template"][content_type])
        full_text = prompt_template.format(text)

        qwen_tokens = qwen_vl_wrapper.tokenize(full_text)
        encoded_output = qwen_vl_wrapper.encode_from_tokens(qwen_tokens, return_dict=True)

        text_embeds_padded = encoded_output.get('cond')
        if text_embeds_padded is None:
             raise ValueError("The Qwen CLIP encoder did not return 'cond'. Ensure you are using a CLIPLoader with the 'qwen_image' type.")

        attention_mask = encoded_output.get('attention_mask')
        if attention_mask is not None:
             text_embeds_unpadded = text_embeds_padded[attention_mask.bool()]
        else:
             text_embeds_unpadded = text_embeds_padded.squeeze(0)

        return text_embeds_unpadded, pooled_embed

    @classmethod
    def execute(cls, clip, qwen_vl, text, negative_text, content_type) -> io.NodeOutput:
        import comfy.model_management as mm

        offload_device = mm.unet_offload_device()

        pos_text_embeds, pos_pooled_embed = cls._get_raw_embeds(text.split('\n')[0], clip, qwen_vl, content_type)
        neg_text_embeds, neg_pooled_embed = cls._get_raw_embeds(negative_text.split('\n')[0], clip, qwen_vl, content_type)

        offloaded_models = []
        try:
            if hasattr(clip, 'cond_stage_model'):
                clip.cond_stage_model.to(offload_device)
                offloaded_models.append("CLIP")
            if hasattr(qwen_vl, 'cond_stage_model'):
                qwen_vl.cond_stage_model.to(offload_device)
                offloaded_models.append("Qwen-VL")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if offloaded_models:
                    print(f"CLIP models offloaded to CPU: {', '.join(offloaded_models)}")
        except Exception as e:
            print(f"Warning: CLIP offload failed: {e}")

        max_len = max(pos_text_embeds.shape[0], neg_text_embeds.shape[0])

        if pos_text_embeds.shape[0] < max_len:
            pad_amount = max_len - pos_text_embeds.shape[0]
            padding = torch.zeros((pad_amount, pos_text_embeds.shape[1]), dtype=pos_text_embeds.dtype, device=pos_text_embeds.device)
            pos_text_embeds = torch.cat([pos_text_embeds, padding], dim=0)

        if neg_text_embeds.shape[0] < max_len:
            pad_amount = max_len - neg_text_embeds.shape[0]
            padding = torch.zeros((pad_amount, neg_text_embeds.shape[1]), dtype=neg_text_embeds.dtype, device=neg_text_embeds.device)
            neg_text_embeds = torch.cat([neg_text_embeds, padding], dim=0)

        pos_embeds = {"text_embeds": pos_text_embeds, "pooled_embed": pos_pooled_embed}
        positive = [[torch.zeros(1), {"kandinsky_embeds": pos_embeds}]]

        neg_embeds = {"text_embeds": neg_text_embeds, "pooled_embed": neg_pooled_embed}
        negative = [[torch.zeros(1), {"kandinsky_embeds": neg_embeds}]]

        return io.NodeOutput(positive, negative)


class EmptyKandinskyLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmptyKandinskyV5_Latent",
            display_name="Empty Kandinsky 5 Latent",
            category="Kandinsky",
            description="Creates an empty latent tensor with the correct shape for Kandinsky-5.",
            inputs=[
                io.Int.Input("width", default=768, min=64, max=4096, step=64),
                io.Int.Input("height", default=512, min=64, max=4096, step=64),
                io.Float.Input("time_length", default=5.0, min=0.0, max=30.0, step=0.1, tooltip="Time in seconds. 0 for image generation."),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, time_length, batch_size) -> io.NodeOutput:
        if time_length == 0:
            latent_frames = 1
        else:
            num_frames = int(time_length * 24)
            latent_frames = (num_frames - 1) // 4 + 1

        latent = torch.zeros([batch_size, 16, latent_frames, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent})


class KandinskyImageToVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyV5_ImageToVideoLatent",
            display_name="Kandinsky 5 Image to Video Latent",
            category="Kandinsky",
            description="Encodes an image for image-to-video generation with Kandinsky-5.",
            inputs=[
                io.Vae.Input("vae", tooltip="The Kandinsky VAE from the VAE Loader."),
                io.Image.Input("image", tooltip="Input image to condition the video generation."),
                io.Float.Input("time_length", default=5.0, min=0.1, max=30.0, step=0.1, tooltip="Time in seconds for the output video."),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, vae, image, time_length, batch_size) -> io.NodeOutput:
        from math import floor, sqrt
        import comfy.model_management as mm

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        image = image.permute(0, 3, 1, 2).to(device)

        MAX_AREA = 768 * 512
        B, C, H, W = image.shape
        area = H * W
        k = sqrt(MAX_AREA / area) / 16
        new_h = int(floor(H * k) * 16)
        new_w = int(floor(W * k) * 16)

        if new_h != H or new_w != W:
            import torch.nn.functional as F_torch
            image = F_torch.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

        image = image.unsqueeze(2)

        with torch.no_grad():
            image_scaled = image * 2.0 - 1.0

            vae_model = vae.first_stage_model
            vae_model = vae_model.to(device)

            vae_dtype = next(vae_model.parameters()).dtype
            image_scaled = image_scaled.to(dtype=vae_dtype)

            original_use_tiling = getattr(vae_model, 'use_tiling', None)
            original_use_framewise = getattr(vae_model, 'use_framewise_encoding', None)

            if hasattr(vae_model, 'use_tiling'):
                vae_model.use_tiling = False
            if hasattr(vae_model, 'use_framewise_encoding'):
                vae_model.use_framewise_encoding = False

            try:
                encoded = vae_model.encode(image_scaled, opt_tiling=False)
            except TypeError:
                encoded = vae_model.encode(image_scaled)

            if original_use_tiling is not None:
                vae_model.use_tiling = original_use_tiling
            if original_use_framewise is not None:
                vae_model.use_framewise_encoding = original_use_framewise

            if hasattr(encoded, 'latent_dist'):
                image_latent = encoded.latent_dist.mode()
            elif hasattr(encoded, 'sample'):
                image_latent = encoded.sample
            else:
                image_latent = encoded

            scaling_factor = 0.476986
            image_latent = image_latent * scaling_factor

        try:
            vae_model.to(offload_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("VAE offloaded to CPU after image encoding")
        except Exception as e:
            print(f"Warning: VAE offload failed: {e}")

        num_frames = int(time_length * 24)
        latent_frames = (num_frames - 1) // 4 + 1

        _, C_latent, _, H_latent, W_latent = image_latent.shape

        video_latent = torch.zeros(
            [batch_size, C_latent, latent_frames, H_latent, W_latent],
            device=comfy.model_management.intermediate_device(),
            dtype=image_latent.dtype
        )

        if image_latent.shape[0] == 1 and batch_size > 1:
            image_latent = image_latent.repeat(batch_size, 1, 1, 1, 1)

        visual_cond_mask = torch.zeros(
            [batch_size, latent_frames, H_latent, W_latent, 1],
            device=comfy.model_management.intermediate_device(),
            dtype=image_latent.dtype
        )
        visual_cond_mask[:, 0:1, :, :, :] = 1.0

        image_latent = image_latent.to(comfy.model_management.intermediate_device())

        return io.NodeOutput({
            "samples": video_latent,
            "visual_cond": image_latent,
            "visual_cond_mask": visual_cond_mask
        })


class KandinskyPruneFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyV5_PruneFrames",
            display_name="Kandinsky 5 Prune Frames",
            category="Kandinsky",
            description="Remove frames from the start or end of a video batch.",
            inputs=[
                io.Image.Input("images", tooltip="Video frames to prune."),
                io.Int.Input("prune_start", default=0, min=0, max=100, tooltip="Number of frames to remove from the start."),
                io.Int.Input("prune_end", default=0, min=0, max=100, tooltip="Number of frames to remove from the end."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, images, prune_start, prune_end) -> io.NodeOutput:
        total_frames = images.shape[0]

        if prune_start + prune_end >= total_frames:
            raise ValueError(f"Cannot prune {prune_start + prune_end} frames from a video with {total_frames} frames. Must leave at least 1 frame.")

        start_idx = prune_start
        end_idx = total_frames - prune_end if prune_end > 0 else total_frames

        pruned_images = images[start_idx:end_idx]

        return io.NodeOutput(pruned_images)
