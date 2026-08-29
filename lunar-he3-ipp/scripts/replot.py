import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('MPLCONFIGDIR', str(_ROOT / '.mpl_cache'))

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from environment import Environment
import visualizer

ROOT = _ROOT
DIR_OUT = ROOT / "outputs"
DIR_FIG = ROOT / "figures"

REGIONS = [
    ("T1", "Region T1: Taurus-Littrow (Apollo 17)"),
    ("T2", "Region T2: Hadley Rille (Apollo 15)"),
]
ALGO_TAGS = {"RRT": "RRT", "RRT*": "RRTstar"}


def load_result(region_name, algorithm):
    safe_algo = ALGO_TAGS[algorithm]
    npz_path = DIR_OUT / (region_name + "_results_" + safe_algo + ".npz")
    data = np.load(str(npz_path), allow_pickle=True)
    round_paths = [list(p) for p in data['round_paths']]
    round_samples = [list(s) for s in data['round_samples']]
    return {
        'round_paths': round_paths,
        'round_samples': round_samples,
        'start': tuple(data['start']),
        'mean_grid': data['mean_grid'],
        'std_grid': data['std_grid'],
        'rmse': float(data['final_rmse']),
        'path_length': float(data['final_path_length']),
        'n_samples': int(data['final_n_samples']),
        'history': [
            {'round': i + 1, 'rmse': r, 'coverage_pct': c, 'path_length_total': pl,
             'n_samples_total': ns, 'length_scale': ls}
            for i, (r, c, pl, ns, ls) in enumerate(zip(
                data['rmse_history'], data['coverage_history'],
                data['path_length_history'], data['n_samples_history'],
                data['length_scale_history']))
        ],
    }


for region_name, region_label in REGIONS:
    he3_path = DIR_OUT / (region_name + "_he3.npy")
    obstacle_path = DIR_OUT / (region_name + "_nontraversable.npy")
    env = Environment(he3_path, obstacle_path)

    region_results = {}
    for algorithm in ["RRT", "RRT*"]:
        result = load_result(region_name, algorithm)
        region_results[algorithm] = result
        algo_tag = ALGO_TAGS[algorithm]
        visualizer.plot_trajectory(env, result['round_paths'], result['round_samples'],
                                   result['start'], region_label, algorithm,
                                   result['path_length'], result['n_samples'],
                                   DIR_FIG / (region_name + "_trajectory_" + algo_tag + ".png"))

    visualizer.plot_comparison(env, region_results, region_label,
                               DIR_FIG / (region_name + "_comparison.png"))
    sys.stdout.write(region_name + " replotted.\n"); sys.stdout.flush()

sys.stdout.write("Done.\n"); sys.stdout.flush()
