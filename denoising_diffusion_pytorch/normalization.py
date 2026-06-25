import numpy as np
import torch

_STATS = np.load(
    '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/norm_stats.npy',
    allow_pickle=True
).item()

INT_LOG_MEAN = float(_STATS['int_log_mean'])   # 6.8979
INT_LOG_STD  = float(_STATS['int_log_std'])    # 0.6816
INT_MEAN     = float(_STATS['int_mean'])       # 1227.93 erg/cm²/s/sr — fallback scale for unconditional generation
INT_MIN      = float(_STATS['int_min'])        # 4.2653 erg/cm²/s/sr — global intensity range for linear [-1,1]
INT_MAX      = float(_STATS['int_max'])         # 7986.07
INT_P001     = float(_STATS['int_p001'])       # 124.69 — 0.1/99.9 percentile range (outlier-robust linear [-1,1])
INT_P999     = float(_STATS['int_p999'])        # 5490.82
MEAS_MIN     = float(_STATS['meas_min'])       # 4.2653 DN — physical floor; clamp noisy meas here before log
VEL_MEAN     = float(_STATS['vel_mean'])       # -1.0849 km/s
VEL_STD      = float(_STATS['vel_std'])        # 9.0853 km/s
VEL_MIN      = float(_STATS['vel_min'])        # -61.674 km/s — global velocity range for linear [-1,1]
VEL_MAX      = float(_STATS['vel_max'])         # 51.965 km/s
VEL_P001     = float(_STATS['vel_p001'])       # -40.258 km/s — 0.1/99.9 percentile range
VEL_P999     = float(_STATS['vel_p999'])        # 23.016 km/s
WIDTH_MEAN   = float(_STATS['width_mean'])     # 0.028477 Å
WIDTH_STD    = float(_STATS['width_std'])      # 0.001394 Å
WIDTH_MIN    = float(_STATS['width_min'])      # 0.019108 Å — global width range for linear [-1,1]
WIDTH_MAX    = float(_STATS['width_max'])        # 0.050955 Å
WIDTH_P001   = float(_STATS['width_p001'])     # 0.024428 Å — 0.1/99.9 percentile range
WIDTH_P999   = float(_STATS['width_p999'])      # 0.036964 Å
LOG_EPS      = float(_STATS['log_eps'])        # 1.0


def _log(x):
    return torch.log(x + LOG_EPS) if torch.is_tensor(x) else np.log(x + LOG_EPS)

def _exp(x):
    return torch.exp(x) if torch.is_tensor(x) else np.exp(x)


class GlobalLogzNorm:
    """Global log-zscore for intensity, z-score for velocity and line width.

    Operates in physical units: intensity in erg/cm²/s/sr, velocity in km/s, width in Å.
    All stats derived from dset_v6 training set.
    """
    name = 'global_logz'
    # x_start clip range during sampling, in this scheme's normalized space.
    # Z-scored channels span several stds, so a wide clip avoids truncating tails.
    clip_denoised = (-5., 5.)
    # std of log-intensity: the per-unit-normalized-space intensity scale (analogous to 1/a_int for linear)
    intensity_slope = INT_LOG_STD

    def __init__(self, rec_mode='all'):
        self.rec_mode = rec_mode

    def forward(self, x):
        """Physical (B, C, H, W) → normalized."""
        x = x.clone()
        if self.rec_mode == 'all':
            x[:, 0] = (_log(x[:, 0]) - INT_LOG_MEAN) / INT_LOG_STD
            x[:, 1] = (x[:, 1] - VEL_MEAN) / VEL_STD
            x[:, 2] = (x[:, 2] - WIDTH_MEAN) / WIDTH_STD
        elif self.rec_mode == 'int':
            x[:, 0] = (_log(x[:, 0]) - INT_LOG_MEAN) / INT_LOG_STD
        elif self.rec_mode == 'vel':
            x[:, 0] = (x[:, 0] - VEL_MEAN) / VEL_STD
        elif self.rec_mode == 'width':
            x[:, 0] = (x[:, 0] - WIDTH_MEAN) / WIDTH_STD
        return x

    def normalize_cond(self, cond):
        """Normalize conditioning measurements (DN) → log-zscore.

        Uses the intensity stats int_log_mean/std: meas_0 is the noisy intensity,
        so this lands the conditioning in the same frame as the intensity target
        channel (meas_log_* differs <1%, so this is also ~consistent with slitless
        meas_transform). Noisy measurements can dip below the physical floor
        (clean meas_min≈4.27) and even below -1, which would make log(cond+1) NaN
        — clamp to MEAS_MIN first. The clamp is a no-op at high SNR / noiseless
        (clean meas ≥ meas_min), so it's compatible with models trained without it.
        cond: (B, n_cond, H, W) raw DN.
        """
        return (_log(cond.clamp(min=MEAS_MIN)) - INT_LOG_MEAN) / INT_LOG_STD

    def inverse(self, x):
        """Normalized → physical (B, C, H, W)."""
        x = x.clone()
        if self.rec_mode == 'all':
            x[:, 0] = _exp(x[:, 0] * INT_LOG_STD + INT_LOG_MEAN) - LOG_EPS
            x[:, 1] = x[:, 1] * VEL_STD + VEL_MEAN
            x[:, 2] = x[:, 2] * WIDTH_STD + WIDTH_MEAN
        elif self.rec_mode == 'int':
            x[:, 0] = _exp(x[:, 0] * INT_LOG_STD + INT_LOG_MEAN) - LOG_EPS
        elif self.rec_mode == 'vel':
            x[:, 0] = x[:, 0] * VEL_STD + VEL_MEAN
        elif self.rec_mode == 'width':
            x[:, 0] = x[:, 0] * WIDTH_STD + WIDTH_MEAN
        return x


class PersampleLinearNorm:
    """Per-sample intensity normalization (x / max(x) → [-1,1]), z-score for vel/width.

    Intensity scale is unknown at DPS inference time — call set_infer_scale()
    with an estimate (e.g. meas[:, 0].max()) before running the sampling loop.
    """
    name = 'persample_linear'
    # Intensity is in [-1,1] but vel/width stay z-scored (several stds), so keep
    # a wide clip to avoid truncating those channels.
    clip_denoised = (-5., 5.)
    # Fixed approximation: dataset-mean intensity / 2 (the per-sample slope is unknown at class level)
    intensity_slope = INT_MEAN / 2.0

    def __init__(self, rec_mode='all'):
        self.rec_mode = rec_mode
        self._scale = None  # set by forward() during training or set_infer_scale() at inference

    def set_infer_scale(self, scale):
        """Fix the intensity scale for DPS inference. scale: scalar or (B,1,1,1) tensor."""
        self._scale = scale

    def forward(self, x):
        """Physical (B, C, H, W) → normalized. Records per-sample scale."""
        x = x.clone()
        if self.rec_mode in ('all', 'int'):
            scale = x[:, [0]].amax(dim=(-1, -2), keepdim=True).clamp(min=1.0)
            self._scale = scale
            x[:, 0] = x[:, 0] / scale.squeeze(1) * 2 - 1  # [0,1] → [-1,1]
        if self.rec_mode == 'all':
            x[:, 1] = (x[:, 1] - VEL_MEAN) / VEL_STD
            x[:, 2] = (x[:, 2] - WIDTH_MEAN) / WIDTH_STD
        elif self.rec_mode == 'vel':
            x[:, 0] = (x[:, 0] - VEL_MEAN) / VEL_STD
        elif self.rec_mode == 'width':
            x[:, 0] = (x[:, 0] - WIDTH_MEAN) / WIDTH_STD
        return x

    def normalize_cond(self, cond):
        """Normalize conditioning measurements using the param intensity's per-sample scale.

        Measurements are intensity-like (DN) and share the intensity channel's scale:
        - Training: self._scale set by forward() from the param intensity max.
        - Inference: self._scale is None (forward never called on EMA model) — fall back
          to the zeroth-order measurement cond[:, 0], which is the direct intensity proxy.
        cond: (B, n_cond, H, W) raw DN → [-1, 1].
        """
        scale = self._scale
        if not torch.is_tensor(scale):
            scale = cond[:, [0]].amax(dim=(-1, -2), keepdim=True).clamp(min=1.0)
        return cond / scale * 2 - 1

    def inverse(self, x):
        """Normalized → physical (B, C, H, W). Uses _scale set by forward() or set_infer_scale().

        Falls back to INT_MEAN when _scale is None (e.g. EMA model unconditional generation
        during training, where forward() was never called on this normalization instance).
        """
        x = x.clone()
        if self.rec_mode in ('all', 'int'):
            scale = self._scale
            if not torch.is_tensor(scale):
                # None (EMA model path — forward() never called) or plain Python scalar
                scale = INT_MEAN if scale is None else float(scale)
            elif scale.dim() == 4:
                # (B_fwd, 1, 1, 1) from forward() — squeeze channel dim
                scale = scale[:, 0]  # → (B_fwd, 1, 1)
                if scale.shape[0] != x.shape[0]:
                    # Batch size mismatch — global mean is the safe fallback
                    scale = INT_MEAN
            x[:, 0] = (x[:, 0] + 1) / 2 * scale  # [-1,1] → [0,1] → erg/cm²/s/sr
        if self.rec_mode == 'all':
            x[:, 1] = x[:, 1] * VEL_STD + VEL_MEAN
            x[:, 2] = x[:, 2] * WIDTH_STD + WIDTH_MEAN
        elif self.rec_mode == 'vel':
            x[:, 0] = x[:, 0] * VEL_STD + VEL_MEAN
        elif self.rec_mode == 'width':
            x[:, 0] = x[:, 0] * WIDTH_STD + WIDTH_MEAN
        return x


class GlobalLinearNorm:
    """Global fixed-range linear map of all three channels to [-1, 1].

    Each channel is mapped with its global training min/max:
        x -> 2*(x - min)/(max - min) - 1
    Unlike PersampleLinearNorm (per-sample intensity scale, z-scored vel/width),
    every channel uses a fixed dataset-wide range, so the scale is identical
    across samples and no per-sample state is needed. Operates in physical units
    (intensity erg/cm²/s/sr, velocity km/s, width Å); raw intensity (no log).
    All ranges derived from dset_v6 training set.
    """
    name = 'global_linear'
    # All channels are bounded to [-1,1], so clip x_start to the valid data range.
    clip_denoised = (-1., 1.)

    _RANGES = {0: (INT_MIN, INT_MAX), 1: (VEL_MIN, VEL_MAX), 2: (WIDTH_MIN, WIDTH_MAX)}

    def __init__(self, rec_mode='all'):
        self.rec_mode = rec_mode
        lo, hi = self._RANGES[0]
        self.intensity_slope = (hi - lo) / 2.0

    @staticmethod
    def _fwd(x, lo, hi):
        return 2 * (x - lo) / (hi - lo) - 1

    @staticmethod
    def _inv(x, lo, hi):
        return (x + 1) / 2 * (hi - lo) + lo

    def _channels(self):
        """Map output channel index -> physical channel index for the rec_mode."""
        if self.rec_mode == 'all':
            return {0: 0, 1: 1, 2: 2}
        return {0: {'int': 0, 'vel': 1, 'width': 2}[self.rec_mode]}

    def forward(self, x):
        """Physical (B, C, H, W) → normalized [-1, 1]."""
        x = x.clone()
        for out_c, phys_c in self._channels().items():
            lo, hi = self._RANGES[phys_c]
            x[:, out_c] = self._fwd(x[:, out_c], lo, hi)
        return x

    def normalize_cond(self, cond):
        """Normalize conditioning measurements (DN) → [-1, 1].

        Measurements are intensity-like (DN), so they share the intensity
        channel's global range. Noisy measurements can dip below the physical
        floor (clean meas_min≈4.27), so clamp to MEAS_MIN first; the clamp is a
        no-op at high SNR / noiseless. cond: (B, n_cond, H, W) raw DN.
        """
        return self._fwd(cond.clamp(min=MEAS_MIN), INT_MIN, INT_MAX)

    def inverse(self, x):
        """Normalized [-1, 1] → physical (B, C, H, W)."""
        x = x.clone()
        for out_c, phys_c in self._channels().items():
            lo, hi = self._RANGES[phys_c]
            x[:, out_c] = self._inv(x[:, out_c], lo, hi)
        return x


class GlobalLinearPctNorm(GlobalLinearNorm):
    """Like GlobalLinearNorm but uses outlier-robust 0.1/99.9 percentile ranges.

    The bulk of each channel fills [-1, 1] more fully than with true min/max
    (whose extreme tails otherwise compress the bulk). The ~0.2% of values
    outside the percentile range map beyond [-1, 1], so forward() clamps to
    [-1, 1]; inverse() of a clamped value returns the percentile bound, not the
    original outlier (expected for percentile normalization). All ranges from
    the dset_v6 training set.
    """
    name = 'global_linear_pct'

    _RANGES = {0: (INT_P001, INT_P999), 1: (VEL_P001, VEL_P999), 2: (WIDTH_P001, WIDTH_P999)}

    def forward(self, x):
        return super().forward(x).clamp(-1, 1)

    def normalize_cond(self, cond):
        return self._fwd(cond.clamp(min=MEAS_MIN), INT_P001, INT_P999).clamp(-1, 1)


_REGISTRY = {
    'global_logz': GlobalLogzNorm,
    'persample_linear': PersampleLinearNorm,
    'global_linear': GlobalLinearNorm,
    'global_linear_pct': GlobalLinearPctNorm,
}


def make_normalization(norm_mode: str, rec_mode: str = 'all'):
    if norm_mode not in _REGISTRY:
        raise ValueError(f"Unknown norm_mode {norm_mode!r}. Choose from {list(_REGISTRY)}")
    return _REGISTRY[norm_mode](rec_mode=rec_mode)
