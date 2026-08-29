"""
Evaluator: RMSE (GP mean vs truth), path length, sample count, coverage %,
and GP length-scale trajectory, logged once per round.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np


class Evaluator:
    def __init__(self, env, coverage_radius=15.0):
        self.env = env
        self.coverage_radius = coverage_radius
        self.history = []
        self._covered_mask = np.zeros((env.grid_size, env.grid_size), dtype=bool)
        rows, cols = np.indices((env.grid_size, env.grid_size))
        self._grid_rows = rows
        self._grid_cols = cols

    def update_coverage(self, new_points):
        for (x, y) in new_points:
            d2 = (self._grid_cols - x) ** 2 + (self._grid_rows - y) ** 2
            self._covered_mask |= (d2 <= self.coverage_radius ** 2)

    def coverage_pct(self):
        trav = (self.env.obstacles == 0)
        covered_trav = np.sum(self._covered_mask & trav)
        return float(covered_trav) / float(self.env.traversable_count) * 100.0

    def compute_rmse(self, belief_model):
        mean_grid, std_grid, xs, ys = belief_model.get_entropy_map()
        gx, gy = np.meshgrid(xs, ys)
        rows = np.clip(np.round(gy).astype(int), 0, self.env.grid_size - 1)
        cols = np.clip(np.round(gx).astype(int), 0, self.env.grid_size - 1)
        truth_sampled = self.env.truth[rows, cols]
        return float(np.sqrt(np.mean((mean_grid - truth_sampled) ** 2)))

    def log(self, round_idx, belief_model, path_length_total, n_samples_total, new_points):
        self.update_coverage(new_points)
        rmse = self.compute_rmse(belief_model)
        coverage = self.coverage_pct()
        ls = belief_model.length_scale_history[-1] if belief_model.length_scale_history \
            else belief_model.length_scale_init
        entry = {
            'round': round_idx,
            'rmse': rmse,
            'path_length_total': path_length_total,
            'n_samples_total': n_samples_total,
            'coverage_pct': coverage,
            'length_scale': ls,
        }
        self.history.append(entry)
        return entry
