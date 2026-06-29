"""
Generic generator quality evaluation — gaussmatch + intensity shape histogram.

Edit the config block below, then: python evaluate.py
Results are displayed only (not saved). For a saved, experiment-specific run
see experiments/generator_quality/evaluate.py.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.stats import wasserstein_distance
from statistics import NormalDist

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization

# ── config ────────────────────────────────────────────────────────────────────
# RUN_FOLDER  = 'training_results/exp_norm_logz_dset6_lr5e-6'
# RUN_FOLDER  = 'training_results/exp_norm_persample_dset6_lr5e-6'
# RUN_FOLDER  = 'training_results/run_all_lr_1e-4_cosine_b32_logz'
RUN_FOLDER  = 'training_results/2026_06_23__15_52_06_all_lr_1e-4_cosine_b32_global_linear_unconditional'
# RUN_FOLDER  = 'training_results/2026_06_23__18_18_11_all_lr_1e-4_cosine_b32_global_linear_pct_unconditional'
# MODELLIST   = [1, 5, 10, 15, 20, 25, 30, 40, 50]
MODELLIST   = [1, 2, 3, 4, 5]
# MODELLIST   = [1,10]
# MODELLIST        = [1, 10, 50]
NUM_SAMPLES      = 200
PLOT_MS          = [1, 5]   # milestones shown in static intensity histogram
CALC_GAUSSMATCH  = False
CALC_W1          = False
CALC_TVD         = True
CALC_MMD         = False
SLIDER_HIST      = True    # save interactive slider histogram as HTML
SHOW_STATIC_PLOTS = False  # show matplotlib metric + histogram plots

DATA_DIR = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data/train'
# ─────────────────────────────────────────────────────────────────────────────

SPEEDOFLIGHT = 299792.458
WAVELENGTH   = 195.117937907451
W_FAC        = SPEEDOFLIGHT / WAVELENGTH

with open(f'{RUN_FOLDER}/config.json') as f:
    _cfg = json.load(f)
    NORM_MODE  = _cfg['norm_mode']
    SAVE_EVERY = _cfg.get('save_and_sample_every', 1000)
    CLIP_DENOISED = _cfg.get('clip_denoised')   # None for runs pre-dating this key

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gaussmatch_score(means1, stds1, means2, stds2):
    scores = np.zeros(len(means1))
    for i in range(len(means1)):
        scores[i] = NormalDist(mu=means1[i], sigma=stds1[i]).overlap(
                    NormalDist(mu=means2[i], sigma=stds2[i]))
    return scores


def wasserstein_score(channels_ref, channels_gen):
    """W1 distance per channel. Lower = better. Units match the channel."""
    return np.array([wasserstein_distance(r, g) for r, g in zip(channels_ref, channels_gen)])


def tvd_score(channels_ref, channels_gen, n_bins=60):
    """1 - TVD per channel, in [0, 1]. Higher = better. Bins set by ref percentiles."""
    scores = np.zeros(len(channels_ref))
    for i, (r, g) in enumerate(zip(channels_ref, channels_gen)):
        edges = np.linspace(np.percentile(r, 0.5), np.percentile(r, 99.5), n_bins + 1)
        p, _ = np.histogram(r, bins=edges, density=True)
        q, _ = np.histogram(g, bins=edges, density=True)
        width = np.diff(edges)
        p, q = p * width, q * width   # bin probabilities
        scores[i] = 1.0 - 0.5 * np.sum(np.abs(p - q))
    return scores


def mmd_score(channels_ref, channels_gen):
    """MMD² on the joint 3-channel distribution with RBF kernel (median bandwidth).
    Channels are standardised by their reference std before the kernel so no single
    channel dominates. Lower = better. A single scalar, not per-channel."""
    ref_stds = np.array([np.std(r) for r in channels_ref], dtype=np.float32)
    X = np.stack([r / s for r, s in zip(channels_ref, ref_stds)], axis=1).astype(np.float32)
    Y = np.stack([g / s for g, s in zip(channels_gen,  ref_stds)], axis=1).astype(np.float32)
    # subsample to keep cost manageable
    rng = np.random.default_rng(0)
    n = min(len(X), len(Y), 5000)
    X = X[rng.choice(len(X), n, replace=False)]
    Y = Y[rng.choice(len(Y), n, replace=False)]
    # median bandwidth heuristic on pooled data
    Z = np.concatenate([X, Y], axis=0)
    dists = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
    bw = np.median(dists[dists > 0])
    def rbf(A, B): return np.exp(-np.sum((A[:, None] - B[None, :]) ** 2, axis=-1) / bw)
    return rbf(X, X).mean() - 2 * rbf(X, Y).mean() + rbf(Y, Y).mean()


def persample_norm_int(imgs):
    if imgs.ndim == 4:
        imgs = imgs[:, 0]
    mx = imgs.reshape(len(imgs), -1).max(axis=1, keepdims=True)[:, :, None]
    return (imgs / np.maximum(mx, 1.0)).ravel()


# ── EIS reference ─────────────────────────────────────────────────────────────
files = sorted(glob.glob(DATA_DIR + '/data*.npy'))
eis_int, eis_vel, eis_wid, eis_int_norm = [], [], [], []
for f in files:
    d = np.load(f, allow_pickle=True).item()
    eis_int.append(d['int'].ravel())
    eis_vel.append(d['vel'].ravel())
    eis_wid.append(d['width'].ravel() * W_FAC)
    eis_int_norm.append(persample_norm_int(d['int'][None]))

eis_int      = np.concatenate(eis_int)
eis_vel      = np.concatenate(eis_vel)
eis_wid      = np.concatenate(eis_wid)
eis_int_norm = np.concatenate(eis_int_norm)
means_eis    = [eis_int.mean(), eis_vel.mean(), eis_wid.mean()]
stds_eis     = [eis_int.std(),  eis_vel.std(),  eis_wid.std()]

print(f'EIS: int={means_eis[0]:.1f}±{stds_eis[0]:.1f} erg/cm²/s/sr  '
      f'vel={means_eis[1]:.2f}±{stds_eis[1]:.2f} km/s  '
      f'width={means_eis[2]:.2f}±{stds_eis[2]:.2f} km/s')

# ── model ─────────────────────────────────────────────────────────────────────
model = Unet(channels=3, dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(device)

normalization = make_normalization(NORM_MODE, rec_mode='all')
clip_denoised = tuple(CLIP_DENOISED) if CLIP_DENOISED is not None else normalization.clip_denoised
diffusion = GaussianDiffusion(
    model,
    mode               = 'all',
    image_size         = 64,
    timesteps          = 1000,
    sampling_timesteps = 250,
    beta_schedule      = 'cosine',
    clip_denoised      = clip_denoised,
    normalization      = normalization,
    device             = device,
)

# ── sweep ─────────────────────────────────────────────────────────────────────
gaussmatch_scores = []
w1_scores         = []
tvd_scores        = []
mmd_scores        = []
int_norm_by_ms    = {}
vel_by_ms         = {}
wid_by_ms         = {}

for ms in MODELLIST:
    ckpt_path = f'{RUN_FOLDER}/model-{ms}.pt'
    if not os.path.exists(ckpt_path):
        print(f'  [skip] ms={ms} not found')
        if CALC_GAUSSMATCH: gaussmatch_scores.append([np.nan]*3)
        if CALC_W1:         w1_scores.append([np.nan]*3)
        if CALC_TVD:        tvd_scores.append([np.nan]*3)
        if CALC_MMD:        mmd_scores.append(np.nan)
        continue

    data = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = {k[6:]: v for k, v in data['model'].items() if k.startswith('model.')}
    model.load_state_dict(state)
    model.eval()

    with torch.inference_mode():
        s = diffusion.sample(batch_size=NUM_SAMPLES).cpu().numpy()

    s[:, 2] *= W_FAC
    int_norm_by_ms[ms] = persample_norm_int(s)
    vel_by_ms[ms]      = s[:, 1].ravel()
    wid_by_ms[ms]      = s[:, 2].ravel()

    # int: use per-sample normalised [0,1] — raw DN is meaningless for persample_linear
    # (PersampleLinearNorm.inverse() falls back to INT_MEAN for unconditional generation)
    gen_ch = [int_norm_by_ms[ms], s[:, 1].ravel(), s[:, 2].ravel()]
    ref_ch = [eis_int_norm,        eis_vel,          eis_wid]

    means_ddpm = [c.mean() for c in gen_ch]
    stds_ddpm  = [c.std()  for c in gen_ch]
    means_ref  = [eis_int_norm.mean(), means_eis[1], means_eis[2]]
    stds_ref   = [eis_int_norm.std(),  stds_eis[1],  stds_eis[2]]

    msg = f'  ms={ms:>3}'
    if CALC_GAUSSMATCH:
        gm = gaussmatch_score(means_ref, stds_ref, means_ddpm, stds_ddpm)
        gaussmatch_scores.append(gm)
        msg += f'  gm=({gm[0]:.3f},{gm[1]:.3f},{gm[2]:.3f})'
    if CALC_W1:
        w1 = wasserstein_score(ref_ch, gen_ch)
        w1_scores.append(w1)
        msg += f'  w1=({w1[0]:.2f},{w1[1]:.2f},{w1[2]:.2f})'
    if CALC_TVD:
        tvd = tvd_score(ref_ch, gen_ch)
        tvd_scores.append(tvd)
        msg += f'  tvd=({tvd[0]:.3f},{tvd[1]:.3f},{tvd[2]:.3f})'
    if CALC_MMD:
        mmd = mmd_score(ref_ch, gen_ch)
        mmd_scores.append(mmd)
        msg += f'  mmd={mmd:.4f}'
    print(msg)

if CALC_GAUSSMATCH: gaussmatch_scores = np.array(gaussmatch_scores)
if CALC_W1:         w1_scores         = np.array(w1_scores)
if CALC_TVD:        tvd_scores        = np.array(tvd_scores)
if CALC_MMD:        mmd_scores        = np.array(mmd_scores)

# keep old name for backward compat with any downstream code
scores = gaussmatch_scores if CALC_GAUSSMATCH else None

# ── plot: all metrics vs milestone ────────────────────────────────────────────
active_metrics = [CALC_GAUSSMATCH, CALC_TVD, CALC_W1, CALC_MMD]
if SHOW_STATIC_PLOTS and any(active_metrics):
    steps     = [ms * 1000 for ms in MODELLIST]
    ch_labels = ['int', 'vel (km/s)', 'width (km/s)']
    colors    = ['#e6194b', '#3cb44b', '#4363d8']

    n_panels = sum(active_metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    ax_iter = iter(axes)

    if CALC_GAUSSMATCH:
        ax = next(ax_iter)
        for c in range(3):
            ax.plot(steps, gaussmatch_scores[:, c], '-o', color=colors[c], label=ch_labels[c])
        ax.set_ylabel('Gaussmatch ↑'); ax.set_ylim(0, 1)
        ax.set_xlabel('Training step'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    if CALC_TVD:
        ax = next(ax_iter)
        for c in range(3):
            ax.plot(steps, tvd_scores[:, c], '-o', color=colors[c], label=ch_labels[c])
        ax.set_ylabel('1 − TVD ↑'); ax.set_ylim(0, 1)
        ax.set_xlabel('Training step'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    if CALC_W1:
        ax = next(ax_iter)
        ref_stds_plot = np.array([eis_int_norm.std(), stds_eis[1], stds_eis[2]])
        for c in range(3):
            ax.plot(steps, w1_scores[:, c] / ref_stds_plot[c], '-o', color=colors[c], label=ch_labels[c])
        ax.set_ylabel('Wasserstein-1 / σ_ref ↓')
        ax.set_xlabel('Training step'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    if CALC_MMD:
        ax = next(ax_iter)
        ax.plot(steps, mmd_scores, '-o', color='gray')
        ax.set_ylabel('MMD² (joint) ↓')
        ax.set_xlabel('Training step'); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Generator quality vs checkpoint — {NORM_MODE}', fontsize=11)
    fig.tight_layout()
    plt.show()

if SHOW_STATIC_PLOTS:
    # ── plot: per-channel histograms, one subplot per milestone ───────────────
    valid_ms_static = [ms for ms in PLOT_MS if ms in int_norm_by_ms]
    n_ms = len(valid_ms_static)
    ch_data_eis = [eis_int_norm, eis_vel, eis_wid]
    ch_data_ms  = [int_norm_by_ms, vel_by_ms, wid_by_ms]
    ch_xlabels  = ['int  [x / max(x)]', 'vel (km/s)', 'width (km/s)']
    ch_titles   = ['Intensity (per-sample norm)', 'Velocity', 'Line width']

    fig, axes = plt.subplots(3, n_ms, figsize=(4 * n_ms, 9))
    if n_ms == 1:
        axes = axes[:, None]
    for row, (eis_vals, ms_dict, xlabel, title) in enumerate(
            zip(ch_data_eis, ch_data_ms, ch_xlabels, ch_titles)):
        bins = np.linspace(np.percentile(eis_vals, 0.5), np.percentile(eis_vals, 99.5), 60)
        for col, ms in enumerate(valid_ms_static):
            ax = axes[row, col]
            ax.hist(eis_vals,    bins=bins, density=True, alpha=0.5, color='black', label='EIS train')
            ax.hist(ms_dict[ms], bins=bins, density=True, alpha=0.5, color='steelblue', label=f'ms={ms}')
            ax.set_xlabel(xlabel, fontsize=8); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
            ax.set_ylabel('density') if col == 0 else None
            ax.set_title(title if col == 0 else f'ms={ms}', fontsize=9)
    fig.suptitle(f'EIS vs DDPM marginals — {NORM_MODE}', fontsize=11)
    fig.tight_layout(); plt.show()

# ── interactive slider histogram + metric plots (plotly → HTML) ───────────────
if SLIDER_HIST:
    valid_ms    = [ms for ms in MODELLIST if ms in int_norm_by_ms]
    ch_data_eis = [eis_int_norm, eis_vel, eis_wid]
    ch_data_ms  = [int_norm_by_ms, vel_by_ms, wid_by_ms]
    ch_titles   = ['Intensity (per-sample norm)', 'Velocity (km/s)', 'Line width (km/s)']
    ch_xlabels  = ['int [x/max(x)]', 'vel (km/s)', 'width (km/s)']
    ch_colors   = ['#e6194b', '#3cb44b', '#4363d8']
    N_BINS      = 60

    # precompute bin edges and EIS densities
    bin_edges, eis_densities = [], []
    for eis_vals in ch_data_eis:
        edges = np.linspace(np.percentile(eis_vals, 0.5), np.percentile(eis_vals, 99.5), N_BINS + 1)
        eis_densities.append(np.histogram(eis_vals, bins=edges, density=True)[0])
        bin_edges.append(edges)

    # collect enabled per-channel metrics for linked plots
    ref_stds_plot = np.array([eis_int_norm.std(), stds_eis[1], stds_eis[2]])
    interactive_metrics = []
    if CALC_GAUSSMATCH and isinstance(gaussmatch_scores, np.ndarray):
        interactive_metrics.append(('Gaussmatch ↑', gaussmatch_scores))
    if CALC_W1 and isinstance(w1_scores, np.ndarray):
        interactive_metrics.append(('W1/σ ↓', w1_scores / ref_stds_plot[None, :]))
    if CALC_TVD and isinstance(tvd_scores, np.ndarray):
        interactive_metrics.append(('1-TVD ↑', tvd_scores))

    n_metric_rows = len(interactive_metrics)
    n_rows        = 1 + n_metric_rows
    modellist_to_idx = {ms: i for i, ms in enumerate(MODELLIST)}
    steps_all = [ms * SAVE_EVERY for ms in MODELLIST]

    # row heights: histograms take 50%, remainder split equally among metric rows
    if n_metric_rows == 0:
        row_heights = [1.0]
    else:
        row_heights = [0.5] + [0.5 / n_metric_rows] * n_metric_rows

    # subplot titles
    subplot_titles = ch_titles.copy()
    for metric_name, _ in interactive_metrics:
        subplot_titles += [f'{metric_name}  int', f'{metric_name}  vel', f'{metric_name}  wid']

    pfig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        vertical_spacing=0.06,
    )

    # ── histogram traces (indices 0-5) ────────────────────────────────────────
    ms0 = valid_ms[0]
    for ch, (ms_dict, edges, eis_dens) in enumerate(zip(ch_data_ms, bin_edges, eis_densities)):
        centers  = 0.5 * (edges[:-1] + edges[1:])
        gen_dens = np.histogram(ms_dict[ms0], bins=edges, density=True)[0]
        pfig.add_trace(go.Bar(x=centers, y=eis_dens, name='EIS',
                              marker_color='rgba(40,40,40,0.5)', showlegend=(ch == 0),
                              width=np.diff(edges)), row=1, col=ch + 1)
        pfig.add_trace(go.Bar(x=centers, y=gen_dens, name=f'DDPM ms={ms0}',
                              marker_color='rgba(31,119,180,0.6)', showlegend=(ch == 0),
                              width=np.diff(edges)), row=1, col=ch + 1)

    # ── metric traces ─────────────────────────────────────────────────────────
    # trace layout per metric i, channel c (interleaved addition order):
    #   6 + i*6 + c*2      → static line (not in frames)
    #   6 + i*6 + c*2 + 1  → highlight marker (updated by frames)
    ms0_idx = modellist_to_idx[ms0]
    for i, (metric_name, metric_data) in enumerate(interactive_metrics):
        row = 2 + i
        for c in range(3):
            y_vals = metric_data[:, c]
            # static line with small markers
            pfig.add_trace(go.Scatter(
                x=steps_all, y=y_vals,
                mode='lines+markers',
                line=dict(color=ch_colors[c], width=2),
                marker=dict(size=6, color=ch_colors[c]),
                showlegend=False,
            ), row=row, col=c + 1)
            # large highlight marker at current milestone (updated per frame)
            pfig.add_trace(go.Scatter(
                x=[steps_all[ms0_idx]], y=[y_vals[ms0_idx]],
                mode='markers',
                marker=dict(size=16, color=ch_colors[c],
                            line=dict(color='white', width=2)),
                showlegend=False,
                hoverinfo='skip',
            ), row=row, col=c + 1)
        pfig.update_yaxes(title_text=metric_name, row=row, col=1)
        pfig.update_xaxes(title_text='Training step', row=row, col=2)

    # trace indices updated by each frame (interleaved layout: line, hl, line, hl, line, hl)
    hist_idx        = list(range(6))
    highlight_idx   = [6 + i*6 + c*2 + 1 for i in range(n_metric_rows) for c in range(3)]
    frame_trace_idx = hist_idx + highlight_idx

    # ── frames ────────────────────────────────────────────────────────────────
    frames = []
    for ms in valid_ms:
        frame_data = []
        # histogram bars
        for ch, (ms_dict, edges, eis_dens) in enumerate(zip(ch_data_ms, bin_edges, eis_densities)):
            centers  = 0.5 * (edges[:-1] + edges[1:])
            gen_dens = np.histogram(ms_dict[ms], bins=edges, density=True)[0]
            frame_data.append(go.Bar(x=centers, y=eis_dens,
                                     marker_color='rgba(40,40,40,0.5)', width=np.diff(edges)))
            frame_data.append(go.Bar(x=centers, y=gen_dens,
                                     marker_color='rgba(31,119,180,0.6)', width=np.diff(edges)))
        # highlight markers
        ms_idx = modellist_to_idx[ms]
        for i, (_, metric_data) in enumerate(interactive_metrics):
            for c in range(3):
                frame_data.append(go.Scatter(
                    x=[ms * SAVE_EVERY], y=[metric_data[ms_idx, c]],
                    mode='markers',
                    marker=dict(size=16, color=ch_colors[c],
                                line=dict(color='white', width=2)),
                ))
        frames.append(go.Frame(data=frame_data, traces=frame_trace_idx, name=str(ms)))

    pfig.frames = frames
    pfig.update_layout(
        barmode='overlay',
        title=f'EIS vs DDPM marginals — {NORM_MODE}',
        height=400 + 220 * n_metric_rows,
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='milestone: ', font=dict(size=14)),
            steps=[dict(method='animate', label=str(ms),
                        args=[[str(ms)], dict(mode='immediate',
                                              frame=dict(duration=0),
                                              transition=dict(duration=0))])
                   for ms in valid_ms],
        )],
        updatemenus=[dict(type='buttons', showactive=False, y=1.05, x=0,
                          buttons=[dict(label='▶ Play', method='animate',
                                        args=[None, dict(frame=dict(duration=600),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)])])],
    )
    for ch, xlabel in enumerate(ch_xlabels):
        pfig.update_xaxes(title_text=xlabel, row=1, col=ch + 1)
    pfig.update_yaxes(title_text='density', row=1, col=1)

    html_path = f'{RUN_FOLDER}/hist_slider.html'
    pfig.write_html(html_path, include_plotlyjs='cdn')
    print(f'Saved interactive histogram → {html_path}')