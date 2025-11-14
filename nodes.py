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
        return io.Schema(
            node_id="KandinskyV5_Loader",
            display_name="Kandinsky 5 Loader",
            category="Kandinsky",
            description="Loads a Kandinsky-5 text-to-video model variant.",
            inputs=[
                io.Combo.Input("variant", options=list(KANDINSKY_CONFIGS.keys()), default="sft_5s"),
                io.Boolean.Input("use_magcache", default=False, tooltip="Enable MagCache for faster inference."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, variant: str, use_magcache: bool) -> io.NodeOutput:
        from .kandinsky_patcher import KANDINSKY_CONFIGS
        config_data = KANDINSKY_CONFIGS[variant]
        
        base_path = os.path.dirname(__file__)
        config_path = os.path.join(base_path, 'src', 'configs', config_data["config"])

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at '{config_path}'.")

        try:
            ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", config_data["ckpt"])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Checkpoint not found for '{variant}'. Ensure '{config_data['ckpt']}' is in 'ComfyUI/models/diffusion_models/'.")

        conf = OmegaConf.load(config_path)
        handler = KandinskyModelHandler(conf, ckpt_path)
        patcher = KandinskyPatcher(
            handler, 
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device()
        )
        handler.conf.use_magcache = use_magcache
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
        pos_text_embeds, pos_pooled_embed = cls._get_raw_embeds(text.split('\n')[0], clip, qwen_vl, content_type)
        neg_text_embeds, neg_pooled_embed = cls._get_raw_embeds(negative_text.split('\n')[0], clip, qwen_vl, content_type)
        
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
        device = comfy.model_management.get_torch_device()

        image = image.to(device)

        image = image.permute(0, 3, 1, 2)

        B, C, H, W = image.shape

        image = image.unsqueeze(2)

        with torch.no_grad():
            image_scaled = image * 2.0 - 1.0

            vae_model = vae.first_stage_model
            vae_model = vae_model.to(device)

            vae_dtype = next(vae_model.parameters()).dtype
            image_scaled = image_scaled.to(dtype=vae_dtype)

            encoded = vae_model.encode(image_scaled)

            if hasattr(encoded, 'latent_dist'):
                image_latent = encoded.latent_dist.mode()
            elif hasattr(encoded, 'sample'):
                image_latent = encoded.sample
            else:
                image_latent = encoded

            scaling_factor = 0.476986
            image_latent = image_latent * scaling_factor

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
