"""
Environment module: loads the He-3 ground-truth field and the non-traversable
mask, exposes a noisy query(x, y) interface and traversability checks.

Coordinate convention: physical (x, y) in [0, grid_size-1] meters, (0,0) at
the bottom-left, x increasing right, y increasing up. Grid arrays are indexed
[row, col] with row = round(y), col = round(x) (i.e. row 0 = y=0 = bottom row).
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np
from sensor import Sensor


class Environment:
    def __init__(self, he3_path, obstacle_path, sensor_sigma=0.1, grid_size=500):
        self.truth = np.load(str(he3_path)).astype(np.float64)
        self.obstacles = np.load(str(obstacle_path)).astype(np.uint8)
        if self.truth.shape != (grid_size, grid_size):
            raise ValueError("he3 field shape %s != (%d,%d)" % (self.truth.shape, grid_size, grid_size))
        if self.obstacles.shape != (grid_size, grid_size):
            raise ValueError("obstacle mask shape %s != (%d,%d)" % (self.obstacles.shape, grid_size, grid_size))
        self.grid_size = grid_size
        self.sensor = Sensor(sigma=sensor_sigma)
        self.traversable_count = int(np.sum(self.obstacles == 0))

    def _to_index(self, x, y):
        col = int(np.clip(round(x), 0, self.grid_size - 1))
        row = int(np.clip(round(y), 0, self.grid_size - 1))
        return row, col

    def query_truth(self, x, y):
        row, col = self._to_index(x, y)
        return float(self.truth[row, col])

    def query(self, x, y):
        """Returns a noisy measurement: truth(x,y) + sensor noise."""
        return self.sensor.measure(self.query_truth(x, y))

    def is_traversable(self, x, y):
        if x < 0 or x > self.grid_size - 1 or y < 0 or y > self.grid_size - 1:
            return False
        row, col = self._to_index(x, y)
        return bool(self.obstacles[row, col] == 0)

    def is_traversable_xy_array(self, xs, ys):
        """Vectorized traversability check; returns a boolean array."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        in_bounds = (xs >= 0) & (xs <= self.grid_size - 1) & (ys >= 0) & (ys <= self.grid_size - 1)
        cols = np.clip(np.round(xs).astype(int), 0, self.grid_size - 1)
        rows = np.clip(np.round(ys).astype(int), 0, self.grid_size - 1)
        trav = self.obstacles[rows, cols] == 0
        return trav & in_bounds

    def is_segment_traversable(self, x0, y0, x1, y1, step=1.0):
        """Pixel-by-pixel collision check along the straight segment (x0,y0)->(x1,y1)."""
        dist = float(np.hypot(x1 - x0, y1 - y0))
        n = max(2, int(np.ceil(dist / step)) + 1)
        ts = np.linspace(0.0, 1.0, n)
        xs = x0 + ts * (x1 - x0)
        ys = y0 + ts * (y1 - y0)
        return bool(np.all(self.is_traversable_xy_array(xs, ys)))
