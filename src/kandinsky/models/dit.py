import torch
from torch import nn

from .nn import (
    TimeEmbeddings,
    TextEmbeddings,
    VisualEmbeddings,
    RoPE1D,
    RoPE3D,
    Modulation,
    MultiheadSelfAttentionEnc,
    MultiheadSelfAttentionDec,
    MultiheadCrossAttention,
    FeedForward,
    OutLayer,
    apply_scale_shift_norm,
    apply_gate_sum,
)
from .utils import fractal_flatten, fractal_unflatten


class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, operations=None):
        super().__init__()
        if operations is None:
            operations = nn
        self.text_modulation = Modulation(time_dim, model_dim, 6, operations=operations)

        self.self_attention_norm = operations.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionEnc(model_dim, head_dim, operations=operations)

        self.feed_forward_norm = operations.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim, operations=operations)

    def forward(self, x, time_embed, rope, attn_mask=None):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift)
        out = self.self_attention(out, rope, attn_mask=attn_mask)
        x = apply_gate_sum(x, out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)
        out = self.feed_forward(out)
        x = apply_gate_sum(x, out, gate)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, operations=None):
        super().__init__()
        if operations is None:
            operations = nn
        self.visual_modulation = Modulation(time_dim, model_dim, 9, operations=operations)

        self.self_attention_norm = operations.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionDec(model_dim, head_dim, operations=operations)

        self.cross_attention_norm = operations.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim, operations=operations)

        self.feed_forward_norm = operations.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim, operations=operations)

    def forward(self, visual_embed, text_embed, time_embed, rope, sparse_params, attn_mask=None):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed), 3, dim=-1
        )
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.self_attention_norm, visual_embed, scale, shift)
        visual_out = self.self_attention(visual_out, rope, sparse_params, attn_mask=attn_mask)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.cross_attention_norm, visual_embed, scale, shift)
        visual_out = self.cross_attention(visual_out, text_embed, attn_mask=attn_mask)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.feed_forward_norm, visual_embed, scale, shift)
        visual_out = self.feed_forward(visual_out)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)
        return visual_embed


class DiffusionTransformer3D(nn.Module):
    def __init__(
        self,
        in_visual_dim=4,
        in_text_dim=3584,
        in_text_dim2=768,
        time_dim=512,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_text_blocks=2,
        num_visual_blocks=32,
        axes_dims=(16, 24, 24),
        visual_cond=False,
        block_swap_enabled=False,
        blocks_in_memory=6,
        pin_first_n_blocks=2,
        pin_last_n_blocks=2,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = nn

        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond

        # Block swapping configuration
        self.block_swap_enabled = block_swap_enabled
        self.blocks_in_memory = blocks_in_memory
        self.pin_first_n_blocks = pin_first_n_blocks
        self.pin_last_n_blocks = pin_last_n_blocks
        self.num_visual_blocks = num_visual_blocks

        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond else in_visual_dim
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim, operations=operations)
        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim, operations=operations)
        self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim, operations=operations)
        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size, operations=operations)

        self.text_rope_embeddings = RoPE1D(head_dim)
        self.text_transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim, operations=operations)
                for _ in range(num_text_blocks)
            ]
        )

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, operations=operations)
                for _ in range(num_visual_blocks)
            ]
        )

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size, operations=operations)

    def setup_block_swapping(self, device, offload_device):
        """Initialize block swapping by moving most blocks to CPU."""
        if not self.block_swap_enabled:
            return

        print(f"Setting up block swapping: {self.blocks_in_memory} blocks in GPU memory (out of {self.num_visual_blocks})")

        # Determine which blocks stay on GPU
        self.pinned_blocks = set()

        # Pin first N blocks
        for i in range(min(self.pin_first_n_blocks, self.num_visual_blocks)):
            self.pinned_blocks.add(i)

        # Pin last N blocks
        for i in range(max(0, self.num_visual_blocks - self.pin_last_n_blocks), self.num_visual_blocks):
            self.pinned_blocks.add(i)

        # Move unpinned blocks to CPU
        for i, block in enumerate(self.visual_transformer_blocks):
            if i not in self.pinned_blocks:
                block.to(offload_device)

        print(f"Pinned blocks (always on GPU): {sorted(self.pinned_blocks)}")

    def swap_block_to_device(self, block_idx, device):
        """Move a specific block to the target device."""
        if block_idx < len(self.visual_transformer_blocks):
            self.visual_transformer_blocks[block_idx].to(device)

    # @torch.compile() # DISABLED FOR DEBUGGING
    def before_text_transformer_blocks(self, text_embed, time, pooled_text_embed, x,
                                       text_rope_pos):
        text_embed = self.text_embeddings(text_embed)
        time_embed = self.time_embeddings(time)
        time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
        visual_embed = self.visual_embeddings(x)
        text_rope = self.text_rope_embeddings(text_rope_pos)
        return text_embed, time_embed, text_rope, visual_embed

    # @torch.compile() # DISABLED FOR DEBUGGING
    def before_visual_transformer_blocks(self, visual_embed, visual_rope_pos, scale_factor,
                                         sparse_params):
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope = fractal_flatten(visual_embed, visual_rope, visual_shape,
                                                    block_mask=to_fractal)
        return visual_embed, visual_shape, to_fractal, visual_rope

    # @torch.compile() # DISABLED FOR DEBUGGING
    def after_blocks(self, visual_embed, visual_shape, to_fractal, text_embed, time_embed):
        visual_embed = fractal_unflatten(visual_embed, visual_shape, block_mask=to_fractal)
        x = self.out_layer(visual_embed, text_embed, time_embed)
        return x

    def forward(
        self,
        x,
        text_embed,
        pooled_text_embed,
        time,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=(1.0, 1.0, 1.0),
        sparse_params=None,
        attention_mask=None
    ):
        text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
            text_embed, time, pooled_text_embed, x, text_rope_pos)

        for text_transformer_block in self.text_transformer_blocks:
            text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)

        visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
            visual_embed, visual_rope_pos, scale_factor, sparse_params)

        if self.block_swap_enabled:
            # Block swapping enabled - manage GPU memory carefully
            device = visual_embed.device
            offload_device = torch.device('cpu')

            # Track which unpinned blocks are currently on GPU (FIFO queue)
            currently_loaded_blocks = []

            # Process blocks with swapping
            for i, visual_transformer_block in enumerate(self.visual_transformer_blocks):
                # Check if block needs to be loaded to GPU
                if i not in self.pinned_blocks:
                    # BEFORE loading new block, check if we need to offload
                    # We need to stay UNDER the limit, so offload when at capacity
                    while len(currently_loaded_blocks) >= self.blocks_in_memory:
                        # Offload the oldest loaded block (FIFO)
                        block_to_offload = currently_loaded_blocks.pop(0)
                        self.visual_transformer_blocks[block_to_offload].to(offload_device)
                        # Force immediate memory release
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # Now load the current block
                    visual_transformer_block.to(device)
                    currently_loaded_blocks.append(i)

                # Process the block
                visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                        visual_rope, sparse_params, attention_mask)

            # Clean up: offload all unpinned blocks after processing
            for block_idx in currently_loaded_blocks:
                self.visual_transformer_blocks[block_idx].to(offload_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Normal forward pass without block swapping
            for visual_transformer_block in self.visual_transformer_blocks:
                visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                        visual_rope, sparse_params, attention_mask)

        x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)
        return x


def get_dit(conf):
    dit = DiffusionTransformer3D(**conf)
    return dit