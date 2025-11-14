import torch
from tqdm import trange
from typing_extensions import override
from comfy_api.latest import io
import comfy.utils
import comfy.model_management

from .src.kandinsky.magcache_utils import set_magcache_params
from .src.kandinsky.models.utils import fast_sta_nabla
from .src.kandinsky.models.nn import set_sage_attention

@torch.no_grad()
def get_sparse_params(conf, batch_shape, device):
    F_cond, H_cond, W_cond, C_cond = batch_shape
    patch_size = conf.model.dit_params.patch_size
    
    T = F_cond // patch_size[0]
    H = H_cond // patch_size[1]
    W = W_cond // patch_size[2]

    if conf.model.attention.type == "nabla":
        sta_mask = fast_sta_nabla(T, H // 8, W // 8, conf.model.attention.wT,
                                  conf.model.attention.wH, conf.model.attention.wW, device=device)
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
        }
    else:
        sparse_params = None

    return sparse_params

@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    sparse_params=None,
    visual_cond=None,
    visual_cond_mask=None,
):
    model_input = x
    if dit.visual_cond:
        if visual_cond is not None and visual_cond_mask is not None:
            F = x.shape[0]
            visual_cond_input = torch.zeros_like(x)
            visual_cond_input[0:1] = visual_cond.to(dtype=x.dtype)

            visual_cond_mask_input = visual_cond_mask.to(dtype=x.dtype)
        else:
            visual_cond_input = torch.zeros_like(x)
            visual_cond_mask_input = torch.zeros([*x.shape[:-1], 1], dtype=x.dtype, device=x.device)

        model_input = torch.cat([x, visual_cond_input, visual_cond_mask_input], dim=-1)

    pred_velocity = dit(
        model_input,
        text_embeds["text_embeds"],
        text_embeds["pooled_embed"],
        t * 1000,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=conf.metrics.scale_factor,
        sparse_params=sparse_params,
    )

    if abs(guidance_weight - 1.0) > 1e-6:
        uncond_pred_velocity = dit(
            model_input,
            null_text_embeds["text_embeds"],
            null_text_embeds["pooled_embed"],
            t * 1000,
            visual_rope_pos,
            null_text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
        )
        pred_velocity = torch.lerp(uncond_pred_velocity, pred_velocity, guidance_weight)

    return pred_velocity

@torch.no_grad()
def generate(
    diffusion_model,
    device,
    shape,
    steps,
    text_embed,
    null_embed,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    cfg,
    scheduler_scale,
    conf,
    seed,
    pbar,
    visual_cond=None,
    visual_cond_mask=None,
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    model_dtype = next(diffusion_model.parameters()).dtype
    current_latent = torch.randn(shape, generator=g, device=device, dtype=model_dtype)

    if torch.isnan(current_latent).any() or torch.isinf(current_latent).any():
        current_latent = torch.randn(shape, device=device, dtype=model_dtype)

    lock_first_frame = False
    visual_cond_typed = None
    if visual_cond is not None and visual_cond_mask is not None:
        if visual_cond_mask[0].sum() > 0:
            visual_cond_typed = visual_cond.to(dtype=model_dtype)
            current_latent[0:1] = visual_cond_typed
            lock_first_frame = True

    sparse_params = get_sparse_params(conf, shape, device)

    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=model_dtype)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    for i in range(steps):
        t_now = timesteps[i]
        t_next = timesteps[i+1]
        dt = t_next - t_now

        pred_velocity = get_velocity(
            diffusion_model,
            current_latent,
            t_now.unsqueeze(0),
            text_embed,
            null_embed,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            cfg,
            conf,
            sparse_params=sparse_params,
            visual_cond=visual_cond,
            visual_cond_mask=visual_cond_mask,
        )

        if torch.isnan(pred_velocity).any() or torch.isinf(pred_velocity).any():
            pred_velocity = torch.nan_to_num(pred_velocity, nan=0.0, posinf=0.0, neginf=0.0)

        if lock_first_frame:
            pred_velocity[0:1] = 0.0

        current_latent = current_latent + dt * pred_velocity

        max_val = current_latent.abs().max()
        if max_val > 50.0:
            scale = 50.0 / max_val
            current_latent = current_latent * scale

        if torch.isnan(current_latent).any() or torch.isinf(current_latent).any():
            current_latent = torch.nan_to_num(current_latent, nan=0.0, posinf=0.0, neginf=0.0)

        pbar.update(1)

    return current_latent

class KandinskySampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyV5_Sampler",
            display_name="Kandinsky 5 Sampler",
            category="Kandinsky",
            description="Performs the specific Flow Matching sampling loop for Kandinsky-5 models.",
            inputs=[
                io.Model.Input("model", tooltip="The Kandinsky 5 model patcher from the Kandinsky 5 Loader."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=50, min=1, max=200, tooltip="Number of sampling steps."),
                io.Float.Input("cfg", default=5.0, min=1.0, max=20.0, step=0.1),
                io.Float.Input("scheduler_scale", default=5.0, min=1.0, max=20.0, step=0.1),
                io.Boolean.Input("use_sage_attention", default=False, tooltip="Enable SageAttention for faster inference with lower memory usage."),
                io.Conditioning.Input("positive", tooltip="Positive conditioning from Kandinsky 5 Text Encode."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning from Kandinsky 5 Text Encode."),
                io.Latent.Input("latent_image", tooltip="Empty latent from Empty Kandinsky 5 Latent."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, model, seed, steps, cfg, scheduler_scale, use_sage_attention, positive, negative, latent_image) -> io.NodeOutput:
        patcher = model
        set_sage_attention(use_sage_attention)

        comfy.model_management.load_model_gpu(patcher)
        k_handler = patcher.model
        diffusion_model = k_handler.diffusion_model
        conf = k_handler.conf
        device = patcher.load_device
        model_dtype = next(diffusion_model.parameters()).dtype

        if conf.get('use_magcache', False):
            if hasattr(conf, "magcache"):
                set_magcache_params(diffusion_model, conf.magcache.mag_ratios, steps, conf.model.guidance_weight == 1.0)

        latent = latent_image["samples"].to(device)
        B, C, F, H, W = latent.shape

        visual_cond = None
        visual_cond_mask = None
        if "visual_cond" in latent_image and "visual_cond_mask" in latent_image:
            visual_cond = latent_image["visual_cond"].to(device=device, dtype=model_dtype)
            visual_cond_mask = latent_image["visual_cond_mask"].to(device=device, dtype=model_dtype)

        pos_cond = positive[0][1].get("kandinsky_embeds")
        neg_cond = negative[0][1].get("kandinsky_embeds")

        for key in pos_cond:
            pos_cond[key] = pos_cond[key].to(device=device, dtype=model_dtype)
            neg_cond[key] = neg_cond[key].to(device=device, dtype=model_dtype)

        patch_size = conf.model.dit_params.patch_size
        visual_rope_pos = [
            torch.arange(F // patch_size[0], device=device),
            torch.arange(H // patch_size[1], device=device),
            torch.arange(W // patch_size[2], device=device)
        ]

        text_rope_pos = torch.arange(pos_cond["text_embeds"].shape[0], device=device)
        null_text_rope_pos = torch.arange(neg_cond["text_embeds"].shape[0], device=device)

        output_latents = []
        pbar = comfy.utils.ProgressBar(steps * B)
        for i in range(B):
            current_seed = seed + i

            batch_visual_cond = None
            batch_visual_cond_mask = None
            if visual_cond is not None and visual_cond_mask is not None:
                batch_visual_cond = visual_cond[i].permute(1, 2, 3, 0)
                batch_visual_cond_mask = visual_cond_mask[i]

            final_latent_unbatched = generate(
                diffusion_model,
                device,
                (F, H, W, C),
                steps,
                pos_cond,
                neg_cond,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                cfg,
                scheduler_scale,
                conf,
                current_seed,
                pbar,
                visual_cond=batch_visual_cond,
                visual_cond_mask=batch_visual_cond_mask,
            )
            output_latents.append(final_latent_unbatched.permute(3, 0, 1, 2))

        final_latents = torch.stack(output_latents, dim=0)

        if torch.isnan(final_latents).any() or torch.isinf(final_latents).any():
            final_latents = torch.nan_to_num(final_latents, nan=0.0, posinf=0.0, neginf=0.0)

        scaling_factor = 0.476986
        scaled_latents = final_latents / scaling_factor

        if torch.isnan(scaled_latents).any() or torch.isinf(scaled_latents).any():
            scaled_latents = torch.nan_to_num(scaled_latents, nan=0.0, posinf=0.0, neginf=0.0)

        return io.NodeOutput({"samples": scaled_latents.to(comfy.model_management.intermediate_device())})
