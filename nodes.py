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

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    AutoProcessor = None

class KandinskyQwenWrapper:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.patcher = comfy.model_patcher.ModelPatcher(self.model, load_device=device, offload_device=comfy.model_management.unet_offload_device())
    
    def encode(self, text, content_type):
        qwen_template_config = Qwen2_5_VLTextEmbedder.PROMPT_TEMPLATE
        prompt_template = "\n".join(qwen_template_config["template"][content_type])
        crop_start = qwen_template_config["crop_start"][content_type]
        full_text = prompt_template.format(text)
        
        max_length = 256 + crop_start # Default max length
        
        inputs = self.processor(
            text=[full_text],
            images=None,
            videos=None,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        ).to(self.device)
        
        with torch.no_grad():
            embeds = self.model(
                input_ids=inputs["input_ids"],
                return_dict=True,
                output_hidden_states=True,
            )["hidden_states"][-1][:, crop_start:]
            
        attention_mask = inputs["attention_mask"][:, crop_start:]
        return embeds, attention_mask

class KandinskyQwenLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyV5_QwenLoader",
            display_name="Kandinsky 5 Qwen Loader",
            category="Kandinsky",
            description="Loads the Qwen2.5-VL model directly using Transformers. Requires config.json in the model folder.",
            inputs=[
                io.Combo.Input("checkpoint", options=folder_paths.get_filename_list("text_encoders"), tooltip="Select the Qwen2.5-VL model."),
                io.Combo.Input("precision", options=["bf16", "fp16", "fp32"], default="bf16", tooltip="Precision to load the model in."),
            ],
            outputs=[io.Output("QWEN2_5_VL", type="CLIP")], # Output as CLIP type for compatibility
        )

    @classmethod
    def execute(cls, checkpoint, precision) -> io.NodeOutput:
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError("transformers library is required for KandinskyQwenLoader")
            
        checkpoint_path = folder_paths.get_full_path_or_raise("text_encoders", checkpoint)
        
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else torch.float32
        device = comfy.model_management.get_torch_device()
        
        # Try to load from directory if checkpoint is a file
        load_path = checkpoint_path
        if os.path.isfile(checkpoint_path):
            load_path = os.path.dirname(checkpoint_path)
            
        try:
            # Load to CPU first to avoid OOM/Bus Error during load
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(load_path, torch_dtype=dtype, device_map="cpu")
        except Exception as e:
            # If loading from folder fails, try loading the specific file if possible (transformers usually needs folder)
            # But maybe the user has the file in a folder with config.json
            raise ValueError(f"Failed to load Qwen model from {load_path}. Ensure config.json is present. Error: {e}")

        try:
            processor = AutoProcessor.from_pretrained(load_path, use_fast=True)
        except Exception as e:
             raise ValueError(f"Failed to load AutoProcessor from {load_path}. Error: {e}")
        
        wrapper = KandinskyQwenWrapper(model, processor, device)
        return io.NodeOutput(wrapper)

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
                io.Boolean.Input("use_magcache", default=False, tooltip="Enable MagCache for faster inference with non distilled models. Use 50 steps with it."),
                io.Float.Input("magcache_threshold", default=0.12, min=0.01, max=0.3, step=0.01,
                              tooltip="MagCache quality threshold. Lower = better quality, slower. Higher = faster, more artifacts. Recomended: 0.02-0.06"),
                io.Int.Input("blocks_in_memory", default=0, min=0, max=60,
                            tooltip="Block swapping: Keep N transformer blocks in VRAM, swap others to RAM. 0=disabled (load all to VRAM). 2B: 32 blocks total. 20B: 60 blocks total."),
                io.Combo.Input("quantize_to_fp8", options=["off", "on"], default="off",
                              tooltip="Quantize to FP8: Converts a standard float32/float16 checkpoint to FP8 for faster inference. Not compatible with GGUF."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, variant: str, checkpoint_file: str, use_magcache: bool, magcache_threshold: float, blocks_in_memory: int, quantize_to_fp8: str) -> io.NodeOutput:
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

        conf.blocks_in_memory = blocks_in_memory

        is_gguf = checkpoint_file.lower().endswith('.gguf')
        is_fp8 = not is_gguf and quantize_to_fp8 == "on"

        if is_gguf:
            print(f"Loading GGUF model: {checkpoint_file}")
        elif is_fp8:
            print(f"Quantizing model to FP8 on-the-fly...")
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
        handler.conf.fp8_mode = "on_the_fly" if is_fp8 else "off"

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
                io.Clip.Input("qwen_vl", tooltip="The Qwen2.5-VL model. Use CLIPLoader with 'qwen_image' type OR Kandinsky 5 Qwen Loader."),
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
    @torch.no_grad()
    def _run_clip(cls, text: str, clip_wrapper: comfy.sd.CLIP):
        clip_tokens = clip_wrapper.tokenize(text)
        _, pooled_embed = clip_wrapper.encode_from_tokens(clip_tokens, return_pooled=True)
        return pooled_embed

    @classmethod
    @torch.no_grad()
    def _run_qwen(cls, text: str, qwen_vl_wrapper, content_type: str):
        if isinstance(qwen_vl_wrapper, KandinskyQwenWrapper):
             return qwen_vl_wrapper.encode(text, content_type)

        qwen_template_config = Qwen2_5_VLTextEmbedder.PROMPT_TEMPLATE
        prompt_template = "\n".join(qwen_template_config["template"][content_type])
        full_text = prompt_template.format(text)

        qwen_tokens = qwen_vl_wrapper.tokenize(full_text)
        encoded_output = qwen_vl_wrapper.encode_from_tokens(qwen_tokens, return_dict=True)

        text_embeds_padded = encoded_output.get('cond')
        attention_mask = encoded_output.get('attention_mask')
        if attention_mask is None:
            is_padding = torch.abs(text_embeds_padded).sum(dim=-1) < 1e-6
            attention_mask = (~is_padding).long()
        
        return text_embeds_padded, attention_mask

    @classmethod
    def execute(cls, clip, qwen_vl, text, negative_text, content_type) -> io.NodeOutput:
        import comfy.model_management as mm

        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Process CLIP
        mm.soft_empty_cache()
        mm.load_model_gpu(clip.patcher)
        pos_pooled_embed = cls._run_clip(text.split('\n')[0], clip)
        neg_pooled_embed = cls._run_clip(negative_text.split('\n')[0], clip)

        # Process Qwen-VL
        try:
            mm.soft_empty_cache()
            mm.load_model_gpu(qwen_vl.patcher)
            
            pos_text_embeds, pos_attn_mask = cls._run_qwen(text.split('\n')[0], qwen_vl, content_type)
            neg_text_embeds, neg_attn_mask = cls._run_qwen(negative_text.split('\n')[0], qwen_vl, content_type)
            
        except torch.OutOfMemoryError:
            print("Warning: Qwen-VL OOM on GPU, falling back to CPU. This will be slower.")
            mm.soft_empty_cache()
            try:
                if hasattr(qwen_vl, 'cond_stage_model'):
                    qwen_vl.cond_stage_model.to("cpu")
                
                pos_text_embeds, pos_attn_mask = cls._run_qwen(text.split('\n')[0], qwen_vl, content_type)
                neg_text_embeds, neg_attn_mask = cls._run_qwen(negative_text.split('\n')[0], qwen_vl, content_type)
            except Exception as e:
                raise e

        if pos_text_embeds.shape[1] != neg_text_embeds.shape[1]:
            max_len = max(pos_text_embeds.shape[1], neg_text_embeds.shape[1])
            
            if pos_text_embeds.shape[1] < max_len:
                pad_amount = max_len - pos_text_embeds.shape[1]
                padding = torch.zeros((1, pad_amount, pos_text_embeds.shape[2]), dtype=pos_text_embeds.dtype, device=pos_text_embeds.device)
                pos_text_embeds = torch.cat([pos_text_embeds, padding], dim=1)
                mask_padding = torch.zeros((1, pad_amount), dtype=pos_attn_mask.dtype, device=pos_attn_mask.device)
                pos_attn_mask = torch.cat([pos_attn_mask, mask_padding], dim=1)

            if neg_text_embeds.shape[1] < max_len:
                pad_amount = max_len - neg_text_embeds.shape[1]
                padding = torch.zeros((1, pad_amount, neg_text_embeds.shape[2]), dtype=neg_text_embeds.dtype, device=neg_text_embeds.device)
                neg_text_embeds = torch.cat([neg_text_embeds, padding], dim=1)
                mask_padding = torch.zeros((1, pad_amount), dtype=neg_attn_mask.dtype, device=neg_attn_mask.device)
                neg_attn_mask = torch.cat([neg_attn_mask, mask_padding], dim=1)

        pos_embeds = {"text_embeds": pos_text_embeds, "pooled_embed": pos_pooled_embed, "attention_mask": pos_attn_mask}
        positive = [[torch.zeros(1), {"kandinsky_embeds": pos_embeds}]]

        neg_embeds = {"text_embeds": neg_text_embeds, "pooled_embed": neg_pooled_embed, "attention_mask": neg_attn_mask}
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
            latent_frames = int(time_length * 24 // 4 + 1)

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
                io.Int.Input("alignment", default=16, min=16, max=128, step=16, tooltip="Pixel alignment for resizing. Use 128 for 20B models (NABLA attention), 16 for others."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, vae, image, time_length, batch_size, alignment) -> io.NodeOutput:
        from math import floor, sqrt
        import comfy.model_management as mm

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        image = image.permute(0, 3, 1, 2).to(device)

        MAX_AREA = 768 * 512
        B, C, H, W = image.shape
        area = H * W
        k = sqrt(MAX_AREA / area) / alignment
        new_h = int(floor(H * k) * alignment)
        new_w = int(floor(W * k) * alignment)

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
                image_latent = encoded.latent_dist.sample()
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
        except Exception as e:
            print(f"Warning: VAE offload failed: {e}")

        latent_frames = int(time_length * 24 // 4 + 1)

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


class KandinskyVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        import folder_paths
        vae_files = folder_paths.get_filename_list("vae")

        return io.Schema(
            node_id="KandinskyV5_VAELoader",
            display_name="Kandinsky 5 VAE Loader",
            category="Kandinsky",
            description="Loads the Hunyuan Video VAE for Kandinsky-5. Use hunyuan_video_vae_bf16.safetensors.",
            inputs=[
                io.Combo.Input("vae_name", options=vae_files, tooltip="Select the Hunyuan Video VAE (hunyuan_video_vae_bf16.safetensors)"),
            ],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae_name) -> io.NodeOutput:
        import folder_paths
        import os
        import comfy.utils
        from .src.kandinsky.models.vae import AutoencoderKLHunyuanVideo

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)

        if os.path.isfile(vae_path):
            sd = comfy.utils.load_torch_file(vae_path)
            vae = AutoencoderKLHunyuanVideo(
                in_channels=3,
                out_channels=3,
                latent_channels=16,
                block_out_channels=(128, 256, 512, 512),
                layers_per_block=2,
                act_fn="silu",
                norm_num_groups=32,
                scaling_factor=0.476986,
                spatial_compression_ratio=8,
                temporal_compression_ratio=4,
            )
            vae.load_state_dict(sd, strict=True)
            vae = vae.to(dtype=torch.bfloat16)
        else:
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                vae_path,
                torch_dtype=torch.bfloat16
            )

        vae.eval()

        class VAEWrapper:
            def __init__(self, vae_model):
                self.first_stage_model = vae_model

        return io.NodeOutput(VAEWrapper(vae))


class KandinskyVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyV5_VAEDecode",
            category="Kandinsky",
            description="Decodes Kandinsky-5 5D video latents to frames using the VAE's built-in tiling.",
            inputs=[
                io.Vae.Input("vae", tooltip="The Kandinsky VAE from the VAE Loader."),
                io.Latent.Input("samples", tooltip="Latent samples from Kandinsky 5 Sampler."),
                io.Boolean.Input("enable_tiling", default=True, tooltip="Enable tiling to reduce VRAM usage. Disable only for very small videos."),
                io.Int.Input("tile_min_frames", default=16, min=1, max=64, tooltip="Temporal size. Lower = more VRAM efficient but slower."),
                io.Int.Input("tile_min_height", default=256, min=64, max=1024, step=64, tooltip="Tile height. Lower = more VRAM efficient but slower."),
                io.Int.Input("tile_min_width", default=256, min=64, max=1024, step=64, tooltip="Tile width. Lower = more VRAM efficient but slower."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, vae, samples, enable_tiling, tile_min_frames, tile_min_height, tile_min_width) -> io.NodeOutput:
        import comfy.model_management as mm

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        latent = samples["samples"]
        B, C, F, H, W = latent.shape

        latent = latent.to(device)

        scaling_factor = 0.476986
        latent = latent / scaling_factor
        vae_model = vae.first_stage_model
        vae_model = vae_model.to(device)

        vae_dtype = next(vae_model.parameters()).dtype
        latent = latent.to(dtype=vae_dtype)

        has_tiling = hasattr(vae_model, 'use_tiling')

        if not has_tiling and enable_tiling:
            raise RuntimeError(
                "Please use the 'Kandinsky 5 VAE Loader' node "
            )

        if enable_tiling:
            original_use_tiling = vae_model.use_tiling
            original_use_framewise = vae_model.use_framewise_decoding
            original_tile_min_frames = vae_model.tile_sample_min_num_frames
            original_tile_min_height = vae_model.tile_sample_min_height
            original_tile_min_width = vae_model.tile_sample_min_width

            vae_model.use_tiling = True
            vae_model.use_framewise_decoding = True

            vae_model.tile_sample_min_num_frames = tile_min_frames
            vae_model.tile_sample_min_height = tile_min_height
            vae_model.tile_sample_min_width = tile_min_width

            vae_model.tile_sample_stride_num_frames = max(1, tile_min_frames * 3 // 4)
            vae_model.tile_sample_stride_height = tile_min_height * 3 // 4
            vae_model.tile_sample_stride_width = tile_min_width * 3 // 4

        else:
            original_use_tiling = vae_model.use_tiling if has_tiling else None
            original_use_framewise = vae_model.use_framewise_decoding if has_tiling else None
            if has_tiling:
                vae_model.use_tiling = False
                vae_model.use_framewise_decoding = False

        with torch.no_grad():
            autocast_dtype = torch.bfloat16 if vae_dtype == torch.bfloat16 else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(vae_dtype != torch.float32)):
                decoded = vae_model.decode(latent, return_dict=False)[0]

                B, C, F, H, W = decoded.shape

                decoded = decoded.permute(0, 2, 3, 4, 1)
                decoded = decoded.reshape(B * F, H, W, C)

                decoded = (decoded + 1.0) / 2.0
                decoded = decoded.clamp(0.0, 1.0)

        if enable_tiling and has_tiling:
            vae_model.use_tiling = original_use_tiling
            vae_model.use_framewise_decoding = original_use_framewise
            vae_model.tile_sample_min_num_frames = original_tile_min_frames
            vae_model.tile_sample_min_height = original_tile_min_height
            vae_model.tile_sample_min_width = original_tile_min_width
        elif has_tiling and original_use_tiling is not None:
            vae_model.use_tiling = original_use_tiling
            vae_model.use_framewise_decoding = original_use_framewise

        try:
            vae_model.to(offload_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: VAE offload failed: {e}")

        decoded = decoded.cpu().float()

        return io.NodeOutput(decoded)
