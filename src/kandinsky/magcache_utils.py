# This is an adaptation of Magcache from https://github.com/Zehong-Ma/MagCache/
import numpy as np
import torch


def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


def set_magcache_params(dit, mag_ratios, num_steps, no_cfg, threshold=0.12, max_skip_steps=2, start_percent=0.2, end_percent=1.0):
    dit.cnt = 0
    dit.num_steps = num_steps * 2
    dit.magcache_thresh = threshold
    dit.K = max_skip_steps
    dit.accumulated_err = [0.0, 0.0]
    dit.accumulated_steps = [0, 0]
    dit.accumulated_ratio = [1.0, 1.0]
    dit.magcache_start_percent = start_percent
    dit.magcache_end_percent = end_percent
    dit.residual_cache = [None, None]
    dit.mag_ratios = np.array([1.0]*2 + mag_ratios)
    dit.no_cfg = no_cfg
    dit._magcache_enabled = True
    dit._magcache_skipped_count = 0
    dit._magcache_full_count = 0

    # Auto-detect window mode based on end_percent
    use_window = end_percent < 1.0

    print(f'MagCache configured:')
    print(f'  - Threshold: {threshold} (lower=better quality, fewer skips)')
    print(f'  - Max skip steps: {max_skip_steps}')
    magcache_start = int(dit.num_steps * start_percent)
    magcache_end = int(dit.num_steps * end_percent)
    if use_window:
        print(f'  - Window mode: {start_percent*100:.0f}%-{end_percent*100:.0f}% (steps {magcache_start}-{magcache_end} of {dit.num_steps})')
    else:
        print(f'  - Run-to-end mode: starts at {start_percent*100:.0f}% (full compute for first {magcache_start} steps)')
    print(f'  - Total steps: {num_steps}, CFG steps: {dit.num_steps}')
    print(f'  - MagCache eligible steps: {magcache_start}-{magcache_end} (~{((magcache_end - magcache_start)/dit.num_steps)*100:.0f}%)')

    if not hasattr(dit.__class__, '_original_forward'):
        dit.__class__._original_forward = dit.__class__.forward

    dit.__class__.forward = magcache_forward

    if len(dit.mag_ratios) != num_steps * 2:
        print(f'interpolate MAG RATIOS: curr len {len(dit.mag_ratios)}')
        mag_ratio_con = nearest_interp(dit.mag_ratios[0::2], num_steps)
        mag_ratio_ucon = nearest_interp(dit.mag_ratios[1::2], num_steps)
        interpolated_mag_ratios = np.concatenate(
            [mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1).reshape(-1)
        dit.mag_ratios = interpolated_mag_ratios


def disable_magcache(dit):
    """Disables magcache and restores the original forward method."""
    print('disabling Magcache')

    # Restore original forward method
    if hasattr(dit.__class__, '_original_forward'):
        dit.__class__.forward = dit.__class__._original_forward
        delattr(dit.__class__, '_original_forward')

    # Clear torch.compile cache to ensure the old compiled version is not used
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()

    # Clean up magcache attributes
    magcache_attrs = [
        'cnt', 'num_steps', 'magcache_thresh', 'K',
        'accumulated_err', 'accumulated_steps', 'accumulated_ratio',
        'magcache_start_percent', 'magcache_end_percent',
        'residual_cache', 'mag_ratios',
        'no_cfg', '_magcache_enabled', 'pinned_blocks',
        '_magcache_skipped_count', '_magcache_full_count'
    ]

    for attr in magcache_attrs:
        if hasattr(dit, attr):
            delattr(dit, attr)


@torch.compile(mode="max-autotune-no-cudagraphs")
def magcache_forward(
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
    # Safety check: if magcache attributes don't exist, fall back to normal forward
    if not hasattr(self, 'cnt') or not hasattr(self, '_magcache_enabled'):
        # Call the original forward method if it exists
        if hasattr(self.__class__, '_original_forward'):
            return self.__class__._original_forward(self, x, text_embed, pooled_text_embed,
                                                    time, visual_rope_pos, text_rope_pos,
                                                    scale_factor, sparse_params, attention_mask)
        else:
            # Fallback: run without magcache optimization
            text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
                text_embed, time, pooled_text_embed, x, text_rope_pos)
            for text_transformer_block in self.text_transformer_blocks:
                text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)
            visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
                visual_embed, visual_rope_pos, scale_factor, sparse_params)
            for visual_transformer_block in self.visual_transformer_blocks:
                visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                        visual_rope, sparse_params, attention_mask)
            x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)
            return x

    # Reset cache at the start of a new generation to prevent stale data
    if self.cnt == 0:
        self.residual_cache = [None, None]

    text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
        text_embed, time, pooled_text_embed, x, text_rope_pos)

    for text_transformer_block in self.text_transformer_blocks:
        text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)

    visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
        visual_embed, visual_rope_pos, scale_factor, sparse_params)

    skip_forward = False
    ori_visual_embed = visual_embed

    # Check if we're in the MagCache range
    magcache_start = int(self.num_steps * self.magcache_start_percent)
    magcache_end = int(self.num_steps * self.magcache_end_percent)
    in_magcache_range = self.cnt >= magcache_start and self.cnt < magcache_end

    if in_magcache_range:
        cur_mag_ratio = self.mag_ratios[self.cnt]
        self.accumulated_ratio[self.cnt%2] = self.accumulated_ratio[self.cnt%2]*cur_mag_ratio
        self.accumulated_steps[self.cnt%2] += 1
        cur_skip_err = np.abs(1-self.accumulated_ratio[self.cnt%2])
        self.accumulated_err[self.cnt%2] += cur_skip_err

        if self.accumulated_err[self.cnt%2]<self.magcache_thresh and self.accumulated_steps[self.cnt%2]<=self.K:
            # Only skip if we have valid cached residuals
            if self.residual_cache[self.cnt%2] is not None:
                skip_forward = True
                residual_visual_embed = self.residual_cache[self.cnt%2]
        else:
            self.accumulated_err[self.cnt%2] = 0
            self.accumulated_steps[self.cnt%2] = 0
            self.accumulated_ratio[self.cnt%2] = 1.0
    else:
        # Outside MagCache range: reset accumulators
        self.accumulated_err[self.cnt%2] = 0
        self.accumulated_steps[self.cnt%2] = 0
        self.accumulated_ratio[self.cnt%2] = 1.0

    if skip_forward:
        visual_embed =  visual_embed + residual_visual_embed
        self._magcache_skipped_count += 1
    else:
        for visual_transformer_block in self.visual_transformer_blocks:
            visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                    visual_rope, sparse_params, attention_mask)
        residual_visual_embed = visual_embed - ori_visual_embed
        self._magcache_full_count += 1

    self.residual_cache[self.cnt%2] = residual_visual_embed 

    x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)

    if self.no_cfg:
        self.cnt += 2
    else:
        self.cnt += 1

    if self.cnt >= self.num_steps:
        # Print magcache statistics
        total_steps = self._magcache_skipped_count + self._magcache_full_count
        if total_steps > 0:
            skip_percent = (self._magcache_skipped_count / total_steps) * 100
            print(f'MagCache stats: {self._magcache_skipped_count} skipped, {self._magcache_full_count} full ({skip_percent:.1f}% cached)')

        # Reset for next generation
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
        self.residual_cache = [None, None]  # Clear cached residuals between generations
        self._magcache_skipped_count = 0
        self._magcache_full_count = 0
    return x
