import math, glob, json, subprocess, os
from slitless.forward import Source, add_noise
import matplotlib.pyplot as plt
import numpy as np
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple, defaultdict
from multiprocessing import cpu_count
from datetime import datetime
from denoising_diffusion_pytorch.plotting import plotgrid

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

from denoising_diffusion_pytorch.normalization import GlobalLogzNorm

# Simple linear [−1,1] ↔ [0,1] helpers kept for upstream modules (learned_gaussian_diffusion).
# Not used for EIS data — EIS normalization lives in normalization.py.
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(img):
    return (img + 1) / 2

# def unnormalize_to_zero_to_one(img, mode='all'):  # legacy — kept for reference only
#     if mode=='all':
#         img[:,0] = (img[:,0] + 1) / 2
#         img[:,1] = img[:,1] / 2
#         img[:,2] = (img[:,2] + 1) / 2 * 0.8 + 1
#     elif mode=='int':
#         img = (img + 1) / 2
#     elif mode=='vel':
#         img = img / 2
#     elif mode=='width':
#         img = (img + 1) / 2 * 0.8 + 1
#     return img

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        cond_channels = 0,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.cond_channels = cond_channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1) + cond_channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None, cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        if cond is not None:
            x = torch.cat((x, cond), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        mode='all',
        image_size,
        timesteps = 1000,
        recon = False,
        measurement = None,
        grad_scale = 1.,
        forward_op=None,
        true = None,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'cosine',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        clip_denoised = True,
        normalization = None,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False,
        device=None
    ):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.true = true
        self.mode = mode

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.recon = recon
        if recon:
            self.measurement = measurement
            self.grad_scale = grad_scale
            self.forward_op = forward_op
            self.meas_scale = measurement.pow(2).mean().sqrt().item() if measurement is not None else 1.0

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32).to(device))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # immiscible diffusion

        self.immiscible = immiscible

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        self.clip_denoised = clip_denoised

        if normalization is None:
            normalization = GlobalLogzNorm(rec_mode=mode)
        self.normalization = normalization
        self.normalize = normalization.forward
        self.unnormalize = normalization.inverse
        if recon:
            self.measurement = self.measurement.to(self.device)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _clip(self, x):
        """Clip x_start predictions. self.clip_denoised can be False (no clip)
        or a (min, max) tuple. True is treated as (-1., 1.) for backward compat."""
        if self.clip_denoised is False:
            return x
        lo, hi = (-1., 1.) if self.clip_denoised is True else self.clip_denoised
        return x.clamp(lo, hi)

    def model_predictions(self, x, t, x_self_cond = None, cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond, cond = cond)
        maybe_clip = self._clip if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond, cond = cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start = self._clip(x_start)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None, cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        if not self.recon:
            with torch.no_grad():
                model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, cond = cond, clip_denoised = self.clip_denoised)
        else:
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, cond = cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def grad_and_value(self, x_prev, x_0_hat, measurement):
        x_0_hat = self.unnormalize(x_0_hat)
        difference = (measurement - self.forward_op(x_0_hat, device=self.device)) / self.meas_scale
        norm = torch.sqrt(torch.sum(difference**2, dim=(1,2,3))).sum()
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad, norm

    # @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False, cond = None):
        batch, device = shape[0], self.device

        if self.recon:
            img = torch.randn(shape, device = device, requires_grad = True)
        else:
            img = torch.randn(shape, device = device, requires_grad = False)
        imgs = [img.clone()]

        # Source(param3d=self.unnormalize(img.detach().cpu().numpy())[0], pix=True).plot('Recon - INIT')
        # plt.pause(0.1)

        x_start = None
        norms = []
        grad_norms = []
        ch_grad_norms = []   # per-channel gradient norms: list of (C,) arrays
        rmses = []

        if self.recon:
            print(f'[DPS] meas_scale={self.meas_scale:.3f}  grad_scale={self.grad_scale.tolist()}')

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            if self.recon:
                img = img.requires_grad_()
                pred_img, x_start = self.p_sample(img, t, self_cond, cond = cond)

                norm_grad, norm = self.grad_and_value(
                    x_prev=img,
                    x_0_hat=x_start,
                    measurement=self.measurement
                )
                img = pred_img - norm_grad * self.grad_scale [None,:,None,None]
                img = img.detach_()
                pred_img = pred_img.detach_()
                x_start = x_start.detach_()

                if self.true is not None:
                    rmses.append(torch.sqrt(torch.mean((self.unnormalize(img.detach().clone()) - self.true)**2, dim=(-1,-2))).cpu().numpy())
                norms.append(norm.detach().cpu().numpy())
                grad_norms.append(torch.linalg.norm(norm_grad.detach()).cpu().numpy())
                # per-channel: spatial L2 norm, averaged over batch
                ch_grad_norms.append(norm_grad.detach().norm(dim=(-1, -2)).mean(dim=0).cpu().numpy())
            else:
                img, x_start = self.p_sample(img, t, self_cond, cond = cond)

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        # ret = img

        ret = self.unnormalize(ret)
        if self.recon:
            return ret, norms, grad_norms, rmses, ch_grad_norms
        else:
            return ret

    # @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False, cond = None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if self.recon:
            img = torch.randn(shape, device = device, requires_grad = True)
        else:
            img = torch.randn(shape, device = device)
        imgs = [img.clone()]

        x_start = None
        norms = []
        grad_norms = []
        ch_grad_norms = []
        rmses = []

        if self.recon:
            print(f'[DPS] meas_scale={self.meas_scale:.3f}  grad_scale={self.grad_scale.tolist()}')

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            if self.recon:
                img = img.requires_grad_()
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, cond = cond, clip_x_start = self.clip_denoised, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            if self.recon:
                pred_img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
                norm_grad, norm = self.grad_and_value(
                    x_prev=img,
                    x_0_hat=x_start,
                    measurement=self.measurement
                )
                pred_img -= norm_grad * self.grad_scale[None,:,None,None]
                img = pred_img.detach_()
                norms.append(norm.detach().cpu().numpy())
                grad_norms.append(torch.linalg.norm(norm_grad.detach()).cpu().numpy())
                ch_grad_norms.append(norm_grad.detach().norm(dim=(-1, -2)).mean(dim=0).cpu().numpy())
                rmses.append(torch.sqrt(torch.mean((self.unnormalize(img.detach().clone()) - self.true)**2, dim=(-1,-2))).cpu().numpy())

            else:
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

            # imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        if self.recon:
            return ret, norms, grad_norms, rmses, ch_grad_norms
        else:
            return ret

    # @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False, cond = None):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if cond is not None:
            cond = self.normalization.normalize_cond(cond)   # match training; uses meas[:,0] as intensity scale
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps, cond = cond)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None, cond = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, cond = cond).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond, cond = cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)   # sets per-sample intensity scale (persample_linear)
        cond = kwargs.get('cond', None)
        if cond is not None:
            kwargs['cond'] = self.normalization.normalize_cond(cond)   # share intensity scale
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes

_DIFFRACTION_ORDERS = [0, -1, 1, -2, 2]

# Fixed seed for the partition (dataset-size / no-leakage) ablation. Shared with
# the slitless repo so a given (partno, partnum) selects the identical scans in
# both codebases.
PARTITION_SEED = 42


def _scan_id(path):
    """Scan id from a 'data_<date>_<time>_<patchno>.npy' filename, i.e. the name
    with the trailing '_<patchno>' (and '.npy') stripped. dset_v6 patches are
    64x64 crops of larger EIS scans; grouping by scan id keeps every patch of a
    scan in one partition (no data leakage across partitions)."""
    return os.path.basename(path)[:-4].rsplit('_', 1)[0]


def partition_files(files, partno, partnum, seed=PARTITION_SEED):
    """Return the subset of `files` belonging to partition `partno` of `partnum`.

    Patches are grouped by scan id so a scan never straddles two partitions.
    Scan ids are sorted (canonical, repo/OS-independent order) then seeded-
    shuffled — the shuffle breaks the temporal ordering encoded in the scan
    date, spreading scan times homogeneously across partitions. Whole scans are
    then greedily assigned to the currently-smallest partition (by patch count)
    so partition sizes stay tightly balanced. Reproducible and identical across
    repos given the same files, seed, partno and partnum.
    """
    assert isinstance(partno, int) and isinstance(partnum, int), 'partno/partnum must be ints'
    assert partnum >= 1, f'partnum must be >= 1, got {partnum}'
    assert 1 <= partno <= partnum, f'partno must be in 1..{partnum}, got {partno}'
    if partnum == 1:
        return list(files)

    groups = defaultdict(list)
    for f in files:
        groups[_scan_id(f)].append(f)

    scan_ids = sorted(groups)
    np.random.default_rng(seed).shuffle(scan_ids)

    loads = [0] * partnum
    keep = []
    for sid in scan_ids:
        i = min(range(partnum), key=lambda j: loads[j])   # currently-smallest partition
        loads[i] += len(groups[sid])
        if i == partno - 1:
            keep.extend(groups[sid])
    return sorted(keep)


class EISDataset(Dataset):
    def __init__(
        self,
        folder,
        mode='all',
        numdetectors=0,     # 0 = unconditional; >0 = conditional with _DIFFRACTION_ORDERS[:n]
        dbsnr=None,         # dB SNR for measurement noise; None = no noise
        noise_model='Gaussian',
        partno=1,           # which partition to keep (1..partnum); dataset-size / no-leakage ablation
        partnum=1,          # split the training set into this many leakage-free partitions
        # image_size,
        # exts = ['jpg', 'jpeg', 'png', 'tiff'],
        # augment_horizontal_flip = False,
        # convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        # Split into partnum leakage-free partitions (whole scans never straddle
        # partitions) and keep partition partno; partnum=1 returns the full set.
        self.paths = partition_files(glob.glob(self.folder+'/data*.npy'), partno, partnum)
        self.mode = mode
        self.cond_orders = _DIFFRACTION_ORDERS[:numdetectors] if numdetectors > 0 else None
        self.dbsnr = dbsnr
        self.noise_model = noise_model

        # maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            # T.Lambda(maybe_convert_fn),
            # T.Resize(image_size),
            # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        data = np.load(path, allow_pickle=True).item()
        if self.mode == 'all':
            params = np.stack([data['int'], data['vel'], data['width']])
        elif self.mode == 'int':
            params = data['int'][None,:]
        elif self.mode == 'vel':
            params = data['vel'][None,:]
        elif self.mode == 'width':
            params = data['width'][None,:]

        params = torch.tensor(params, dtype=torch.float32)

        if self.cond_orders is not None:
            meas = np.stack([data[f'meas_{o}'] for o in self.cond_orders])
            meas = add_noise(meas, dbsnr=self.dbsnr, no_noise=self.dbsnr is None,
                             noise_model=self.noise_model)
            meas = torch.tensor(meas, dtype=torch.float32)
            return params, meas

        return params

# trainer class

class Trainer:
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        mode = 'all',
        numdetectors = 0,     # 0 = unconditional; >0 = conditional with _DIFFRACTION_ORDERS[:n]
        dbsnr = None,         # dB SNR for measurement noise; None = no noise
        noise_model = 'Gaussian',
        partno = 1,           # which partition to keep (1..partnum); dataset-size / no-leakage ablation
        partnum = 1,          # split the training set into this many leakage-free partitions
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results2',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False  # "False" == "Save All"
    ):
        super().__init__()

        # accelerator

        self.mode = mode
        self.numdetectors = numdetectors
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        # if not exists(convert_image_to):
        #     convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        # self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        self.ds = EISDataset(folder, self.mode, numdetectors=numdetectors,
                             dbsnr=dbsnr, noise_model=noise_model,
                             partno=partno, partnum=partnum)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        # store a fixed validation cond batch for periodic conditional sampling
        if numdetectors > 0:
            val_items = [self.ds[i] for i in range(min(num_samples, len(self.ds)))]
            self.val_cond = torch.stack([item[1] for item in val_items])
        else:
            self.val_cond = None

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = min(8, cpu_count()))

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents = True, exist_ok = True)
        self.periodic_samples_folder = self.results_folder / 'periodic_samples'
        self.periodic_samples_folder.mkdir(parents = True, exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only
    
    @property
    def device(self):
        return self.accelerator.device

    def save_config(self, config: dict):
        if not self.accelerator.is_local_main_process:
            return
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_commit = 'unknown'
        config_out = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'git_commit': git_commit,
            **config,
        }
        with open(self.results_folder / 'config.json', 'w') as f:
            json.dump(config_out, f, indent=2)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def _save_training_artifacts(self, loss_tracker):
        if not self.accelerator.is_main_process or len(loss_tracker) == 0:
            return
        np.save(str(self.results_folder / 'train_loss.npy'), np.array(loss_tracker))
        train_fig, train_ax = plt.subplots()
        train_ax.plot(np.arange(len(loss_tracker)), loss_tracker, label='total_loss')
        train_ax.set_xlabel('Step')
        train_ax.set_ylabel('Loss')
        train_ax.set_title('Training loss')
        train_ax.legend()
        train_fig.savefig(str(self.results_folder / 'train_total_loss_plot.png'))
        plt.close(train_fig)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        train_total_loss_tracker = []
        try:
            with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

                while self.step < self.train_num_steps:
                    self.model.train()

                    total_loss = 0.

                    for _ in range(self.gradient_accumulate_every):
                        batch = next(self.dl)
                        if self.numdetectors > 0:
                            data, cond = batch[0].to(device), batch[1].to(device)
                        else:
                            data, cond = batch.to(device), None

                        with self.accelerator.autocast():
                            loss = self.model(data, cond=cond)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                        self.accelerator.backward(loss)

                    train_total_loss_tracker.append(total_loss)
                    pbar.set_description(f'loss: {total_loss:.4f}')

                    accelerator.wait_for_everyone()
                    accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()

                    self.step += 1
                    if accelerator.is_main_process:
                        self.ema.update()

                        if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                            self.ema.ema_model.eval()

                            with torch.inference_mode():
                                milestone = self.step // self.save_and_sample_every
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                if self.val_cond is not None:
                                    val_cond_dev = self.val_cond.to(self.device)
                                    offset, all_images_list = 0, []
                                    for n in batches:
                                        all_images_list.append(self.ema.ema_model.sample(batch_size=n, cond=val_cond_dev[offset:offset+n]))
                                        offset += n
                                else:
                                    all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                            all_images = torch.cat(all_images_list, dim = 0)

                            fig, ax = plotgrid(all_images, mode=self.mode)
                            fig.savefig(str(self.periodic_samples_folder / f'sample-{milestone}.png'))
                            plt.close(fig)
                            # utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                            # whether to calculate fid

                            if self.calculate_fid:
                                fid_score = self.fid_scorer.fid_score()
                                accelerator.print(f'fid_score: {fid_score}')

                            if self.save_best_and_latest_only:
                                if self.best_fid > fid_score:
                                    self.best_fid = fid_score
                                    self.save("best")
                                self.save("latest")
                            else:
                                self.save(milestone)

                    pbar.update(1)

            accelerator.print('training complete')
        finally:
            self._save_training_artifacts(train_total_loss_tracker)
