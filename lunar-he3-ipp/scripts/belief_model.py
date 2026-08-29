"""
GP belief model over the He-3 field.
Kernel: ConstantKernel * Matern(nu=2.5, initial length_scale=8.0) + WhiteKernel.
Hyperparameters are re-optimized (fmin_l_bfgs_b) on every fit() call.

Per-round predictions use a coarse eval_res x eval_res grid (default 100x100)
to keep GP predict() cheap inside the 30-round active loop (benchmarked
~0.27s/predict at 100x100 vs ~5.7s/predict at 500x500). predict_full_grid()
does a one-time full-resolution prediction for final reconstruction figures.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


class BeliefModel:
    def __init__(self, length_scale=8.0, nu=2.5, grid_size=500, eval_res=100, random_state=42):
        self.grid_size = grid_size
        self.eval_res = eval_res
        self.length_scale_init = length_scale
        self.random_state = random_state
        self._build_gp(length_scale, nu)

        self.X_obs = []
        self.y_obs = []
        self.fitted = False
        self.length_scale_history = []

        self.eval_xs = np.linspace(0, grid_size - 1, eval_res)
        self.eval_ys = np.linspace(0, grid_size - 1, eval_res)
        gx, gy = np.meshgrid(self.eval_xs, self.eval_ys)
        self.eval_gx = gx
        self.eval_gy = gy
        self.eval_points = np.column_stack([gx.ravel(), gy.ravel()])

    def _build_gp(self, length_scale, nu):
        kernel = (ConstantKernel(1.0, (1e-2, 1e3))
                  * Matern(length_scale=length_scale, nu=nu, length_scale_bounds=(1.0, 200.0))
                  + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0)))
        self.gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',
                                            n_restarts_optimizer=2, normalize_y=True,
                                            random_state=self.random_state)

    def add_observation(self, point, z):
        self.X_obs.append([float(point[0]), float(point[1])])
        self.y_obs.append(float(z))

    def fit(self):
        if len(self.X_obs) < 2:
            return
        X = np.array(self.X_obs)
        y = np.array(self.y_obs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(X, y)
        self.fitted = True
        self.length_scale_history.append(self._extract_length_scale())

    def _extract_length_scale(self):
        params = self.gp.kernel_.get_params()
        for k, v in params.items():
            if k.endswith('length_scale'):
                arr = np.ravel(v)
                return float(arr[0])
        return float(self.length_scale_init)

    def predict(self, X):
        X = np.atleast_2d(X)
        if not self.fitted:
            prior_mean = float(np.mean(self.y_obs)) if self.y_obs else 0.0
            mean = np.full(len(X), prior_mean)
            std = np.full(len(X), 1.0)
            return mean, std
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self.gp.predict(X, return_std=True)
        return mean, std

    def get_entropy_map(self):
        """Returns (mean_grid, std_grid, xs, ys) at eval_res x eval_res."""
        mean, std = self.predict(self.eval_points)
        shape = (self.eval_res, self.eval_res)
        return mean.reshape(shape), std.reshape(shape), self.eval_xs, self.eval_ys

    def predict_full_grid(self):
        """One-time full grid_size x grid_size prediction (expensive - end-of-run only)."""
        xs = np.arange(self.grid_size, dtype=np.float64)
        ys = np.arange(self.grid_size, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        mean, std = self.predict(pts)
        shape = (self.grid_size, self.grid_size)
        return mean.reshape(shape), std.reshape(shape)
