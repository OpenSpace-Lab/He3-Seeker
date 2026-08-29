"""
Visualization module for the RRT vs RRT* IPP comparison.
Produces: trajectory plots, GP reconstruction plots, 2x2 comparison plots,
and RMSE convergence curves.
"""
import os
from pathlib import Path
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent.parent / '.mpl_cache'))

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects


def _obstacle_grid(env):
    h, w = env.obstacles.shape
    Y_idx, X_idx = np.mgrid[0:h, 0:w]
    return X_idx, Y_idx


def _draw_obstacle_contour(ax, env, linewidth=2.2):
    """Red obstacle boundary with a white halo so it reads clearly against
    any background color in the busy He-3 heatmap."""
    X_idx, Y_idx = _obstacle_grid(env)
    if not env.obstacles.any():
        return
    cs = ax.contour(X_idx, Y_idx, env.obstacles, levels=[0.5], colors='red',
                    linewidths=linewidth, zorder=6)
    cs.set_path_effects([patheffects.withStroke(linewidth=linewidth + 1.8, foreground='white')])


def _draw_trajectory_on_axes(ax, env, round_paths, round_samples, start, vmin, vmax):
    h, w = env.truth.shape
    im = ax.imshow(env.truth, cmap='plasma', origin='lower', extent=[0, w, 0, h],
                   vmin=vmin, vmax=vmax)
    _draw_obstacle_contour(ax, env)

    n_rounds = len(round_paths)
    cmap_blue = plt.get_cmap('Blues')
    for ridx, path in enumerate(round_paths):
        if len(path) < 2:
            continue
        shade = 0.35 + 0.6 * (ridx / max(1, n_rounds - 1))
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, '-', color=cmap_blue(shade), linewidth=1.4, zorder=3)

    for ridx, samples in enumerate(round_samples):
        if not samples:
            continue
        shade = 0.35 + 0.6 * (ridx / max(1, n_rounds - 1))
        xs = [p[0] for p in samples]
        ys = [p[1] for p in samples]
        ax.scatter(xs, ys, s=22, facecolor=[cmap_blue(shade)], edgecolor='black',
                  linewidth=0.4, zorder=4)

    ax.plot(start[0], start[1], marker='*', color='lime', markersize=18,
           markeredgecolor='black', markeredgewidth=0.8, zorder=5, linestyle='None')
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    return im


def plot_trajectory(env, round_paths, round_samples, start, region_label, algo_label,
                    total_length, n_samples, out_path):
    vmin, vmax = float(np.nanmin(env.truth)), float(np.nanmax(env.truth))
    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = _draw_trajectory_on_axes(ax, env, round_paths, round_samples, start, vmin, vmax)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(region_label + " - " + algo_label
               + "\nSamples: " + str(n_samples) + ", Path length: " + str(round(total_length, 1)) + " m")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('He-3 (ppb)')
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()


def plot_reconstruction(env, mean_grid, std_grid, region_label, algo_label, out_path):
    vmin, vmax = float(np.nanmin(env.truth)), float(np.nanmax(env.truth))
    h, w = env.truth.shape

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    im0 = axes[0].imshow(mean_grid, cmap='plasma', origin='lower', extent=[0, w, 0, h],
                         vmin=vmin, vmax=vmax)
    axes[0].set_title('GP Predicted Mean')
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label='He-3 (ppb)')

    im1 = axes[1].imshow(std_grid, cmap='viridis', origin='lower', extent=[0, w, 0, h])
    axes[1].set_title('GP Predictive Std (Uncertainty)')
    axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Y (m)')
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label='Std (ppb)')

    fig.suptitle(region_label + " - " + algo_label + " - GP Reconstruction")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()


def plot_comparison(env, results, region_label, out_path):
    """results: {'RRT': {...}, 'RRT*': {...}} each with round_paths, round_samples,
    start, mean_grid, rmse, path_length, n_samples."""
    vmin, vmax = float(np.nanmin(env.truth)), float(np.nanmax(env.truth))
    h, w = env.truth.shape

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    im_shared = None
    for col, algo in enumerate(['RRT', 'RRT*']):
        r = results[algo]
        ax_traj = axes[0, col]
        im_shared = _draw_trajectory_on_axes(ax_traj, env, r['round_paths'], r['round_samples'],
                                             r['start'], vmin, vmax)
        ax_traj.set_title(algo + ' Trajectory')
        ax_traj.set_xlabel('X (m)'); ax_traj.set_ylabel('Y (m)')

        ax_recon = axes[1, col]
        ax_recon.imshow(r['mean_grid'], cmap='plasma', origin='lower',
                        extent=[0, w, 0, h], vmin=vmin, vmax=vmax)
        _draw_obstacle_contour(ax_recon, env, linewidth=1.6)
        ax_recon.set_title(algo + ' GP Mean Reconstruction')
        ax_recon.set_xlabel('X (m)'); ax_recon.set_ylabel('Y (m)')
        ax_recon.set_xlim(0, w); ax_recon.set_ylim(0, h)

    # Trajectory and reconstruction share the same He-3 color scale, so one
    # colorbar (reserved in a fixed right-margin slot) covers both rows.
    fig.subplots_adjust(left=0.07, right=0.86, top=0.92, bottom=0.10, hspace=0.28, wspace=0.28)
    cbar_ax = fig.add_axes([0.89, 0.12, 0.025, 0.76])
    fig.colorbar(im_shared, cax=cbar_ax, label='He-3 (ppb)')

    table_text = (
        "Metric              RRT          RRT*\n"
        "RMSE (ppb)          %.4f       %.4f\n"
        "Path length (m)     %.1f        %.1f\n"
        "N samples           %d            %d"
    ) % (results['RRT']['rmse'], results['RRT*']['rmse'],
        results['RRT']['path_length'], results['RRT*']['path_length'],
        results['RRT']['n_samples'], results['RRT*']['n_samples'])

    fig.suptitle(region_label + ' - RRT vs RRT* Comparison', fontsize=14)
    fig.text(0.46, 0.01, table_text, ha='center', va='bottom', fontsize=10, family='monospace')
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()


def plot_rmse_curve(history_rrt, history_rrtstar, region_label, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    rounds_a = [h['round'] for h in history_rrt]
    rmse_a = [h['rmse'] for h in history_rrt]
    rounds_b = [h['round'] for h in history_rrtstar]
    rmse_b = [h['rmse'] for h in history_rrtstar]
    ax.plot(rounds_a, rmse_a, '-o', color='blue', markersize=3, linewidth=1.5, label='RRT')
    ax.plot(rounds_b, rmse_b, '-o', color='red', markersize=3, linewidth=1.5, label='RRT*')
    ax.set_xlabel('Round')
    ax.set_ylabel('RMSE (ppb)')
    ax.set_title(region_label + ' - RMSE Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()
