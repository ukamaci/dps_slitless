"""
Out-of-distribution experiment — analysis.

Reads outputs/results.npy (written by runner.py) and produces:
  - metrics_table.txt / .tex : RMSE & bias per (phantom, noise level, method, channel)
  - recon_comp_<kind>.png    : reconstruction grid (true vs. DPS/CondDiff/U-Net,
                                across noise levels) -- the three-way extension
                                of Fig. dps_ood_i
  - bar_charts_<kind>.png    : RMSE/bias bar charts per channel, grouped by
                                method and noise level
  - posterior_std_<kind>.png : per-pixel posterior std (DPS vs. CondDiff) across
                                noise levels -- tests whether DPS's unconditional
                                prior collapses to a lower-variance, more
                                in-distribution-looking estimate under OOD input

Run:
    python experiments/ood_experiment/analyze.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
RESULTS_PATH = os.path.join(OUT_DIR, 'results.npy')

CH_LABELS = ['Intensity', 'Velocity', 'Line Width']
CH_UNITS  = ['erg/cm$^2$/s/sr', 'km/s', 'km/s']
CMAPS     = ['hot', 'seismic', 'plasma']
METHODS   = ['dps', 'cond', 'unet']
METHOD_LABELS = {'dps': 'DPS', 'cond': 'CondDiff', 'unet': 'U-Net'}

SNR_LABELS = {
    (None, None):     r'$\infty$',
    (30, 'gaussian'): '30',
    (20, 'gaussian'): '20',
}


def cfg_key(dbsnr, noise_model):
    return f'dbsnr_{dbsnr}_{noise_model}'


def rmse_ch(true, est):
    return np.sqrt(np.mean((true - est) ** 2, axis=(-1, -2)))


def bias_ch(true, est):
    return np.mean(est - true, axis=(-1, -2))


def main():
    results = np.load(RESULTS_PATH, allow_pickle=True).item()
    cfg = results['config']
    phantom_kinds = cfg['phantom_kinds']
    configs = [tuple(c) for c in cfg['configs']]

    # ── metrics table ────────────────────────────────────────────────────────
    lines_txt = []
    lines_tex = []
    lines_tex.append(r'\begin{tabular}{c c c c c c c c}')
    lines_tex.append(r'\hline')
    lines_tex.append(
        r'\textbf{$\gamma$ (dB)} & \textbf{Method} & '
        r'\multicolumn{2}{c}{\textbf{Intensity}} & '
        r'\multicolumn{2}{c}{\textbf{Velocity (km/s)}} & '
        r'\multicolumn{2}{c}{\textbf{Line Width (km/s)}} \\'
    )
    lines_tex.append(r' & & RMSE & Bias & RMSE & Bias & RMSE & Bias \\')

    metrics = {}  # metrics[kind][cfg][method] = {'rmse': (3,), 'bias': (3,)}
    for kind in phantom_kinds:
        metrics[kind] = {}
        true = results['phantoms'][kind]['true']     # (3,H,W)
        lines_txt.append(f'\n=== phantom: {kind} ===')

        for dbsnr, noise_model in configs:
            key = cfg_key(dbsnr, noise_model)
            c = results['phantoms'][kind]['configs'][key]
            metrics[kind][key] = {}

            lines_txt.append(f'  -- {key} --')
            lines_tex.append(r'\hline')
            for mi, method in enumerate(METHODS):
                if method in ('dps', 'cond'):
                    est = c[f'{method}_samples'].mean(axis=0)   # posterior mean (3,H,W)
                else:
                    est = c['unet']                              # (3,H,W)

                rmse = rmse_ch(true, est)
                bias = bias_ch(true, est)
                metrics[kind][key][method] = {'rmse': rmse, 'bias': bias}

                lines_txt.append(
                    f'    {METHOD_LABELS[method]:>8s}  '
                    f'int rmse={rmse[0]:7.1f} bias={bias[0]:7.1f}  '
                    f'vel rmse={rmse[1]:5.2f} bias={bias[1]:5.2f}  '
                    f'width rmse={rmse[2]:5.2f} bias={bias[2]:5.2f}'
                )

                snr_lbl = SNR_LABELS[(dbsnr, noise_model)] if mi == 0 else ''
                multirow = rf'\multirow{{3}}{{*}}{{{SNR_LABELS[(dbsnr, noise_model)]}}}' if mi == 0 else ''
                lines_tex.append(
                    f'{multirow} & {METHOD_LABELS[method]} & '
                    f'{rmse[0]:.1f} & {bias[0]:.1f} & '
                    f'{rmse[1]:.3f} & {bias[1]:.3f} & '
                    f'{rmse[2]:.3f} & {bias[2]:.3f} \\\\'
                )
    lines_tex.append(r'\hline')
    lines_tex.append(r'\end{tabular}')

    txt_path = os.path.join(OUT_DIR, 'metrics_table.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines_txt))
    print(f'Saved -> {txt_path}')

    tex_path = os.path.join(OUT_DIR, 'metrics_table.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines_tex))
    print(f'Saved -> {tex_path}')

    print('\n'.join(lines_txt))

    # ── reconstruction grids ────────────────────────────────────────────────
    for kind in phantom_kinds:
        true = results['phantoms'][kind]['true']    # (3,H,W) [int, vel(km/s), width(km/s)]
        n_cfg = len(configs)
        n_cols = 1 + 3 * n_cfg
        n_rows = 4   # int, vel, width, meas (order 0)

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))

        vmins = [true[c].min() for c in range(3)]
        vmaxs = [true[c].max() for c in range(3)]

        # column 0: ground truth
        for row in range(3):
            ax[row, 0].imshow(true[row], cmap=CMAPS[row], vmin=vmins[row], vmax=vmaxs[row])
            ax[row, 0].set_ylabel(CH_LABELS[row], fontsize=10)
        ax[3, 0].axis('off')
        ax[0, 0].set_title('Ground Truth')

        col = 1
        for dbsnr, noise_model in configs:
            key = cfg_key(dbsnr, noise_model)
            c = results['phantoms'][kind]['configs'][key]
            meas0 = c['meas'][0]   # order-0 measurement

            for method in METHODS:
                if method in ('dps', 'cond'):
                    est = c[f'{method}_samples'].mean(axis=0)
                else:
                    est = c['unet']

                for row in range(3):
                    ax[row, col].imshow(est[row], cmap=CMAPS[row], vmin=vmins[row], vmax=vmaxs[row])

                ax[3, col].imshow(meas0, cmap='hot')
                snr_lbl = SNR_LABELS[(dbsnr, noise_model)]
                ax[0, col].set_title(f'{METHOD_LABELS[method]}\n$\\gamma$={snr_lbl} dB')
                col += 1

        for row in range(n_rows):
            for c_ in range(n_cols):
                ax[row, c_].set_xticks([]); ax[row, c_].set_yticks([])
        ax[3, 0].set_ylabel('Meas (order 0)', fontsize=10)
        for spine in ax[3, 0].spines.values():
            spine.set_visible(True)

        fig.suptitle(f'OOD reconstruction comparison -- phantom: {kind}', y=1.0)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f'recon_comp_{kind}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved -> {out_path}')

    # ── bar charts ───────────────────────────────────────────────────────────
    for kind in phantom_kinds:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        x = np.arange(len(configs))
        width = 0.25

        for ch in range(3):
            ax_rmse, ax_bias = axes[0, ch], axes[1, ch]
            for mi, method in enumerate(METHODS):
                rmses = [metrics[kind][cfg_key(*c)][method]['rmse'][ch] for c in configs]
                biases = [metrics[kind][cfg_key(*c)][method]['bias'][ch] for c in configs]
                ax_rmse.bar(x + (mi - 1) * width, rmses, width, label=METHOD_LABELS[method])
                ax_bias.bar(x + (mi - 1) * width, biases, width, label=METHOD_LABELS[method])

            for ax in (ax_rmse, ax_bias):
                ax.set_xticks(x)
                ax.set_xticklabels([SNR_LABELS[c] for c in configs])
                ax.set_xlabel(r'$\gamma$ (dB)')
                ax.grid(axis='y', alpha=0.3)
            ax_rmse.set_title(f'{CH_LABELS[ch]} RMSE ({CH_UNITS[ch]})')
            ax_bias.set_title(f'{CH_LABELS[ch]} Bias ({CH_UNITS[ch]})')

        axes[0, 0].legend()
        fig.suptitle(f'OOD RMSE / Bias -- phantom: {kind}')
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f'bar_charts_{kind}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved -> {out_path}')

    # ── posterior std (DPS vs CondDiff) ─────────────────────────────────────
    for kind in phantom_kinds:
        n_cfg = len(configs)
        fig, ax = plt.subplots(3, 2 * n_cfg, figsize=(2.0 * 2 * n_cfg, 2.0 * 3))

        col = 0
        for dbsnr, noise_model in configs:
            key = cfg_key(dbsnr, noise_model)
            c = results['phantoms'][kind]['configs'][key]
            snr_lbl = SNR_LABELS[(dbsnr, noise_model)]

            for method in ('dps', 'cond'):
                std = c[f'{method}_samples'].std(axis=0)   # (3,H,W)
                for row in range(3):
                    im = ax[row, col].imshow(std[row], cmap='viridis')
                    ax[row, col].set_xticks([]); ax[row, col].set_yticks([])
                ax[0, col].set_title(f'{METHOD_LABELS[method]}\n$\\gamma$={snr_lbl} dB')
                col += 1

        for row in range(3):
            ax[row, 0].set_ylabel(CH_LABELS[row], fontsize=10)

        fig.suptitle(f'Posterior std (uncertainty) -- phantom: {kind}', y=1.0)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f'posterior_std_{kind}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
