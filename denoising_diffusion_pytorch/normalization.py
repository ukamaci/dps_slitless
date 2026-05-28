import numpy as np
import torch

_STATS = np.load(
    '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/norm_stats.npy',
    allow_pickle=True
).item()

INT_LOG_MEAN = float(_STATS['int_log_mean'])   # 6.8979
INT_LOG_STD  = float(_STATS['int_log_std'])    # 0.6816
INT_MEAN     = float(_STATS['int_mean'])       # 1227.93 DN  — fallback scale for unconditional generation
VEL_MEAN     = float(_STATS['vel_mean'])       # -1.0849 km/s
VEL_STD      = float(_STATS['vel_std'])        # 9.0853 km/s
WIDTH_MEAN   = float(_STATS['width_mean'])     # 0.028477 Å
WIDTH_STD    = float(_STATS['width_std'])      # 0.001394 Å
LOG_EPS      = float(_STATS['log_eps'])        # 1.0


def _log(x):
    return torch.log(x + LOG_EPS) if torch.is_tensor(x) else np.log(x + LOG_EPS)

def _exp(x):
    return torch.exp(x) if torch.is_tensor(x) else np.exp(x)


class GlobalLogzNorm:
    """Global log-zscore for intensity, z-score for velocity and line width.

    Operates in physical units: intensity in DN, velocity in km/s, width in Å.
    All stats derived from dset_v6 training set.
    """
    name = 'global_logz'

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
            x[:, 0] = (x[:, 0] + 1) / 2 * scale  # [-1,1] → [0,1] → DN
        if self.rec_mode == 'all':
            x[:, 1] = x[:, 1] * VEL_STD + VEL_MEAN
            x[:, 2] = x[:, 2] * WIDTH_STD + WIDTH_MEAN
        elif self.rec_mode == 'vel':
            x[:, 0] = x[:, 0] * VEL_STD + VEL_MEAN
        elif self.rec_mode == 'width':
            x[:, 0] = x[:, 0] * WIDTH_STD + WIDTH_MEAN
        return x


_REGISTRY = {
    'global_logz': GlobalLogzNorm,
    'persample_linear': PersampleLinearNorm,
}


def make_normalization(norm_mode: str, rec_mode: str = 'all'):
    if norm_mode not in _REGISTRY:
        raise ValueError(f"Unknown norm_mode {norm_mode!r}. Choose from {list(_REGISTRY)}")
    return _REGISTRY[norm_mode](rec_mode=rec_mode)
