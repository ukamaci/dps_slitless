import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from slitless.plotting import scatter_hexbin

SPEEDOFLIGHT = 299792.458
WAVELENGTH   = 195.117937907451
W_FAC        = SPEEDOFLIGHT / WAVELENGTH

OUTPUT_DIR  = 'experiments/norm_comparison/outputs'
SWEEP_PATH  = os.path.join(OUTPUT_DIR, 'val25_sweep.npy')
TEST50_PATH = os.path.join(OUTPUT_DIR, 'test50_results.npy')

CH_LABELS_UNITS = ['Intensity (erg/cm²/s/sr)', 'Velocity (km/s)', 'Line Width (km/s)']
CH_LABELS_SHORT = ['int', 'vel', 'width']

# ── load results ───────────────────────────────────────────────────────────────
sweep  = np.load(SWEEP_PATH,  allow_pickle=True).item()
test50 = np.load(TEST50_PATH, allow_pickle=True).item()

sweep_cfg  = sweep['config']
sweep_res  = sweep['results']
t50_res    = test50['results']
truths_np  = test50['truths']   # (50,3,64,64) — int(erg), vel(km/s), width(Å)

grad_scales  = sweep_cfg['grad_scales']
best_gs      = sweep_cfg['best_grad_scales']
model_names  = ['global_linear', 'global_linear_pct']


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — val-25 grad_scale sweep
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print(' TABLE 1: val-25 grad_scale sweep — mean RMSE (width in km/s)')
print('='*70)

header  = f"{'Model':<24} {'gs':>6}  {'Int RMSE':>12}  {'Vel RMSE':>12}  {'Width RMSE':>12}"
divider = '-' * len(header)
print(header)
print(divider)

table1_rows = []
for model_name in model_names:
    for gs in grad_scales:
        rmses = sweep_res[model_name][gs]   # (25,3) — width in km/s (already converted)
        mu    = rmses.mean(axis=0)
        marker = ' *' if gs == best_gs[model_name] else '  '
        row = f"{model_name:<24} {gs:>6.2f}  {mu[0]:>12.1f}  {mu[1]:>12.3f}  {mu[2]:>12.3f}{marker}"
        print(row)
        table1_rows.append({'model': model_name, 'gs': gs,
                             'int': mu[0], 'vel': mu[1], 'width': mu[2]})
    print(divider)

print('  (* best grad_scale per model by vel RMSE)\n')

# LaTeX version
print('LaTeX TABLE 1:')
print(r'\begin{tabular}{llrrr}')
print(r'\toprule')
print(r'Model & $\alpha$ & Int RMSE & Vel RMSE & Width RMSE \\')
print(r'\midrule')
for model_name in model_names:
    for gs in grad_scales:
        rmses = sweep_res[model_name][gs]
        mu    = rmses.mean(axis=0)
        bold  = gs == best_gs[model_name]
        vals  = [f'{mu[0]:.1f}', f'{mu[1]:.3f}', f'{mu[2]:.3f}']
        if bold:
            vals = [r'\mathbf{' + v + '}' for v in vals]
        mn_str = model_name.replace('_', '\\_')
        print(rf'{mn_str} & {gs} & ${vals[0]}$ & ${vals[1]}$ & ${vals[2]}$ \\')
    print(r'\midrule')
print(r'\bottomrule')
print(r'\end{tabular}')
print()


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — test-50 comparison
# ══════════════════════════════════════════════════════════════════════════════
print('='*70)
print(' TABLE 2: test-50 comparison — mean ± std RMSE (width in km/s)')
print('='*70)

method_display = {
    'global_linear':     'global\_linear',
    'global_linear_pct': 'global\_linear\_pct',
    'logz':              'global\_logz (baseline)',
}

header2  = f"{'Method':<28} {'gs':>6}  {'Int RMSE':>14}  {'Vel RMSE':>14}  {'Width RMSE':>14}"
divider2 = '-' * len(header2)
print(header2)
print(divider2)

for method_name, res in t50_res.items():
    rmses = res['rmses']   # (50,3) — width in km/s
    mu    = rmses.mean(axis=0)
    sd    = rmses.std(axis=0)
    gs    = res['grad_scale']
    display = method_name.replace('_', ' ')
    print(f"{display:<28} {gs:>6.2f}  "
          f"{mu[0]:>8.1f}±{sd[0]:<4.1f}  "
          f"{mu[1]:>8.3f}±{sd[1]:<5.3f}  "
          f"{mu[2]:>8.3f}±{sd[2]:<5.3f}")
print(divider2)

print('\nLaTeX TABLE 2:')
print(r'\begin{tabular}{llrrr}')
print(r'\toprule')
print(r'Method & $\alpha$ & Int RMSE & Vel RMSE & Width RMSE \\')
print(r'\midrule')
for method_name, res in t50_res.items():
    rmses = res['rmses']
    mu    = rmses.mean(axis=0)
    sd    = rmses.std(axis=0)
    gs    = res['grad_scale']
    mn_str = method_name.replace('_', '\\_')
    print(rf'{mn_str} & {gs} & '
          rf'${mu[0]:.1f} \pm {sd[0]:.1f}$ & '
          rf'${mu[1]:.3f} \pm {sd[1]:.3f}$ & '
          rf'${mu[2]:.3f} \pm {sd[2]:.3f}$ \\')
print(r'\bottomrule')
print(r'\end{tabular}')
print()


# ══════════════════════════════════════════════════════════════════════════════
# SCATTER PLOTS — test-50: 3 methods × 3 channels
# ══════════════════════════════════════════════════════════════════════════════
print('='*70)
print(' Generating scatter plots...')
print('='*70)

# truths: (50,3,64,64) — width in Å; convert to km/s for scatter
truths_phy = truths_np.copy()
truths_phy[:, 2] *= W_FAC   # width: Å → km/s

for method_name, res in t50_res.items():
    recons = res['recons_mean'].copy()   # (50,3,64,64) — width in Å from unnormalize
    recons[:, 2] *= W_FAC                # Å → km/s

    # flatten over all images: (3, 50*H*W)
    true_flat = truths_phy.transpose(1, 0, 2, 3).reshape(3, -1)   # (3, 50*64*64)
    rec_flat  = recons.transpose(1, 0, 2, 3).reshape(3, -1)

    fig_path = os.path.join(OUTPUT_DIR, f'scatter_{method_name}.png')
    scatter_hexbin(
        true_flat,
        rec_flat,
        method_name=method_name.replace('_', ' '),
        save=True,
        savepath=fig_path,
        show=False,
    )
    print(f'  saved -> {fig_path}')

print('\nDone.')
