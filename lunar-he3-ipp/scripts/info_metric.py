"""
Information metric: select the next sampling target as the traversable cell
(on the GP's eval-resolution grid) with maximum predictive std (entropy proxy
for a Gaussian belief), excluding cells within `exclude_radius` meters of any
previously visited point.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np


def select_max_entropy_target(belief_model, env, visited_points, exclude_radius=15.0):
    mean_grid, std_grid, xs, ys = belief_model.get_entropy_map()
    gx, gy = np.meshgrid(xs, ys)

    trav_mask = env.is_traversable_xy_array(gx.ravel(), gy.ravel()).reshape(gx.shape)

    if visited_points:
        vx = np.array([p[0] for p in visited_points])
        vy = np.array([p[1] for p in visited_points])
        dx = gx[:, :, None] - vx[None, None, :]
        dy = gy[:, :, None] - vy[None, None, :]
        d2 = dx ** 2 + dy ** 2
        near_visited = np.any(d2 <= exclude_radius ** 2, axis=-1)
        trav_mask = trav_mask & (~near_visited)

    candidate_std = np.where(trav_mask, std_grid, -np.inf)

    if not np.any(np.isfinite(candidate_std)):
        return _random_traversable_fallback(env)

    idx = np.unravel_index(np.argmax(candidate_std), candidate_std.shape)
    return (float(gx[idx]), float(gy[idx]))


def _random_traversable_fallback(env):
    for _ in range(2000):
        x = np.random.uniform(0, env.grid_size - 1)
        y = np.random.uniform(0, env.grid_size - 1)
        if env.is_traversable(x, y):
            return (float(x), float(y))
    return (0.0, 0.0)
