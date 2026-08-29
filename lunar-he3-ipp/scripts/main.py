"""
Main IPP active-loop: RRT vs RRT* comparison on T1 (Taurus-Littrow) and
T2 (Hadley Rille). Implements the Viseras et al. 2019 (Sensors) loop:
GP state -> max-entropy target -> path plan -> arc-length-resampled
measurements -> GP update, 30 rounds per (region, algorithm).
"""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('MPLCONFIGDIR', str(_ROOT / '.mpl_cache'))

import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from environment import Environment
from belief_model import BeliefModel
import planner
import info_metric
from evaluator import Evaluator
import visualizer

ROOT = _ROOT
DIR_OUT = ROOT / "outputs"
DIR_FIG = ROOT / "figures"

N_ROUNDS = 30
START_POS = (25.0, 25.0)
MAX_SAMPLES_PER_ROUND = 5
EXCLUDE_RADIUS = 15.0
SENSOR_SIGMA = 0.1
GP_LENGTH_SCALE = 8.0
GP_NU = 2.5
STEP_SIZE = 10.0
MAX_ITER = 5000
GOAL_THRESH = 10.0
GAMMA = 150.0  # RRT* rewire radius scale. gamma=50 (literature default) collapses the
                # radius to ~2m once the tree has thousands of nodes in this 500m domain,
                # making rewire nearly inert. gamma=150 keeps it useful (empirically:
                # 14% shorter paths, straightness 1.38->1.18 vs RRT) without being extreme.
COVERAGE_RADIUS = 15.0
SEED = 42

REGIONS = [
    ("T1", "Region T1: Taurus-Littrow (Apollo 17)"),
    ("T2", "Region T2: Hadley Rille (Apollo 15)"),
]
ALGORITHMS = ["RRT", "RRT*"]


def arc_length_resample(path, max_points=5, spacing=STEP_SIZE):
    """Resample the path at equal arc-length intervals (excluding the start,
    including the endpoint), capped at max_points."""
    if len(path) == 0:
        return []
    if len(path) == 1:
        return [path[0]]
    pts = np.array(path)
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-6:
        return [tuple(pts[-1])]

    n = min(max_points, max(1, int(np.ceil(total / spacing))))
    targets = np.linspace(total / n, total, n)
    out = []
    for t in targets:
        idx = int(np.searchsorted(cum, t))
        idx = min(idx, len(pts) - 1)
        if idx == 0:
            out.append((float(pts[0, 0]), float(pts[0, 1])))
            continue
        t0, t1 = cum[idx - 1], cum[idx]
        frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        p = pts[idx - 1] + frac * (pts[idx] - pts[idx - 1])
        out.append((float(p[0]), float(p[1])))
    return out


def path_length(path):
    if len(path) < 2:
        return 0.0
    pts = np.array(path)
    seg = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))


def random_traversable_point(env):
    for _ in range(2000):
        x = np.random.uniform(0, env.grid_size - 1)
        y = np.random.uniform(0, env.grid_size - 1)
        if env.is_traversable(x, y):
            return (float(x), float(y))
    return START_POS


def run_region_algorithm(region_name, region_label, algorithm, env):
    sys.stdout.write("\n" + "=" * 60 + "\n")
    sys.stdout.write(region_label + " - " + algorithm + "\n")
    sys.stdout.write("=" * 60 + "\n"); sys.stdout.flush()

    np.random.seed(SEED)

    gp = BeliefModel(length_scale=GP_LENGTH_SCALE, nu=GP_NU, grid_size=env.grid_size)
    evaluator = Evaluator(env, coverage_radius=COVERAGE_RADIUS)

    pos = START_POS
    if not env.is_traversable(*pos):
        raise RuntimeError("Start position " + str(pos) + " is not traversable for " + region_name)

    visited_points = []
    round_paths = []
    round_samples = []
    total_path_length = 0.0
    total_samples = 0
    entry = None

    for round_idx in range(1, N_ROUNDS + 1):
        t0 = time.time()

        if gp.fitted:
            target = info_metric.select_max_entropy_target(gp, env, visited_points, EXCLUDE_RADIUS)
        else:
            target = random_traversable_point(env)

        path = planner.plan(pos, target, env, algorithm=algorithm,
                            step_size=STEP_SIZE, max_iter=MAX_ITER,
                            goal_thresh=GOAL_THRESH, gamma=GAMMA)

        seg_len = path_length(path)
        total_path_length += seg_len

        samples = arc_length_resample(path, max_points=MAX_SAMPLES_PER_ROUND, spacing=STEP_SIZE)
        for p in samples:
            z = env.query(p[0], p[1])
            gp.add_observation(p, z)
            visited_points.append(p)
        total_samples += len(samples)

        gp.fit()

        if path:
            pos = path[-1]

        entry = evaluator.log(round_idx, gp, total_path_length, total_samples, samples)
        round_paths.append(path)
        round_samples.append(samples)

        dt = time.time() - t0
        sys.stdout.write(
            "Round %d/%d: RMSE=%.3f, path_length=%.1fm, n_samples=%d, coverage=%.1f%%, length_scale=%.2f, dt=%.2fs\n"
            % (round_idx, N_ROUNDS, entry['rmse'], total_path_length, total_samples,
               entry['coverage_pct'], entry['length_scale'], dt))
        sys.stdout.flush()

    sys.stdout.write("  Final full-grid GP prediction...\n"); sys.stdout.flush()
    mean_grid, std_grid = gp.predict_full_grid()

    result = {
        'region': region_name, 'algorithm': algorithm,
        'round_paths': round_paths, 'round_samples': round_samples,
        'start': START_POS, 'history': evaluator.history,
        'mean_grid': mean_grid, 'std_grid': std_grid,
        'rmse': entry['rmse'],
        'path_length': total_path_length, 'n_samples': total_samples,
        'X_obs': np.array(gp.X_obs), 'y_obs': np.array(gp.y_obs),
        'length_scale_history': gp.length_scale_history,
    }
    return result


def save_results(region_name, algorithm, result):
    safe_algo = 'RRTstar' if algorithm == 'RRT*' else 'RRT'
    out_path = DIR_OUT / (region_name + "_results_" + safe_algo + ".npz")
    np.savez(
        str(out_path),
        round_paths=np.array(result['round_paths'], dtype=object),
        round_samples=np.array(result['round_samples'], dtype=object),
        start=np.array(result['start']),
        mean_grid=result['mean_grid'], std_grid=result['std_grid'],
        X_obs=result['X_obs'], y_obs=result['y_obs'],
        rmse_history=np.array([h['rmse'] for h in result['history']]),
        coverage_history=np.array([h['coverage_pct'] for h in result['history']]),
        path_length_history=np.array([h['path_length_total'] for h in result['history']]),
        n_samples_history=np.array([h['n_samples_total'] for h in result['history']]),
        length_scale_history=np.array(result['length_scale_history']),
        final_rmse=result['rmse'], final_path_length=result['path_length'],
        final_n_samples=result['n_samples'],
    )
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()


def main():
    t_start = time.time()
    for region_name, region_label in REGIONS:
        he3_path = DIR_OUT / (region_name + "_he3.npy")
        obstacle_path = DIR_OUT / (region_name + "_nontraversable.npy")

        env = Environment(he3_path, obstacle_path, sensor_sigma=SENSOR_SIGMA)
        sys.stdout.write("\nLoaded " + region_name + ": he3 range=[%.2f, %.2f], obstacles=%.1f%%\n"
                         % (float(env.truth.min()), float(env.truth.max()),
                            100.0 * (1 - env.traversable_count / env.obstacles.size)))
        sys.stdout.flush()

        region_results = {}
        for algorithm in ALGORITHMS:
            result = run_region_algorithm(region_name, region_label, algorithm, env)
            region_results[algorithm] = result
            save_results(region_name, algorithm, result)

            algo_tag = 'RRTstar' if algorithm == 'RRT*' else 'RRT'
            visualizer.plot_trajectory(env, result['round_paths'], result['round_samples'],
                                       result['start'], region_label, algorithm,
                                       result['path_length'], result['n_samples'],
                                       DIR_FIG / (region_name + "_trajectory_" + algo_tag + ".png"))
            visualizer.plot_reconstruction(env, result['mean_grid'], result['std_grid'],
                                           region_label, algorithm,
                                           DIR_FIG / (region_name + "_reconstruction_" + algo_tag + ".png"))

        visualizer.plot_comparison(env, region_results, region_label,
                                   DIR_FIG / (region_name + "_comparison.png"))
        visualizer.plot_rmse_curve(region_results['RRT']['history'], region_results['RRT*']['history'],
                                   region_label, DIR_FIG / (region_name + "_rmse_curve.png"))

        sys.stdout.write("\n" + region_name + " complete - all figures saved.\n"); sys.stdout.flush()

    sys.stdout.write("\nAll done! Total time: %.1f s\n" % (time.time() - t_start)); sys.stdout.flush()


if __name__ == "__main__":
    main()
