"""
Unified RRT / RRT* path planner.

plan(start, goal, env, algorithm='RRT'|'RRT*', ...) -> list of (x,y) waypoints.

RRT* differs from RRT ONLY by adding a near-neighbor search + rewire step;
sampling, steering, and goal-check logic are otherwise identical so that
running both algorithms with the same random seed isolates the effect of
rewiring. Random draws happen once per iteration regardless of downstream
branching, so RRT and RRT* consume an identical sample sequence up until
whichever one reaches the goal threshold first.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np


def _reconstruct_path(positions, parents, goal_idx):
    path = []
    idx = goal_idx
    while idx != -1:
        path.append((float(positions[idx, 0]), float(positions[idx, 1])))
        idx = int(parents[idx])
    path.reverse()
    return path


def plan(start, goal, env, algorithm='RRT', step_size=10.0, max_iter=5000,
         goal_thresh=10.0, gamma=150.0, dim=2, collision_step=1.0):
    rewire = (algorithm == 'RRT*')
    grid_size = env.grid_size

    max_nodes = max_iter + 1
    positions = np.zeros((max_nodes, 2), dtype=np.float64)
    parents = np.full(max_nodes, -1, dtype=np.int64)
    costs = np.zeros(max_nodes, dtype=np.float64)

    positions[0] = (float(start[0]), float(start[1]))
    n_nodes = 1
    goal_idx = None
    goal_arr = np.array([float(goal[0]), float(goal[1])])

    for _ in range(max_iter):
        rx = np.random.uniform(0, grid_size - 1)
        ry = np.random.uniform(0, grid_size - 1)

        diffs = positions[:n_nodes] - np.array([rx, ry])
        dists2 = np.einsum('ij,ij->i', diffs, diffs)
        nearest_idx = int(np.argmin(dists2))
        nx0, ny0 = positions[nearest_idx]

        theta = np.arctan2(ry - ny0, rx - nx0)
        nx_ = float(np.clip(nx0 + step_size * np.cos(theta), 0, grid_size - 1))
        ny_ = float(np.clip(ny0 + step_size * np.sin(theta), 0, grid_size - 1))

        if not env.is_segment_traversable(nx0, ny0, nx_, ny_, step=collision_step):
            continue

        parent_idx = nearest_idx
        new_cost = costs[nearest_idx] + float(np.hypot(nx_ - nx0, ny_ - ny0))
        near_idxs = None

        if rewire:
            n_for_radius = n_nodes + 1
            radius = min(50.0, gamma * (np.log(n_for_radius) / n_for_radius) ** (1.0 / dim))
            diffs_new = positions[:n_nodes] - np.array([nx_, ny_])
            d2_new = np.einsum('ij,ij->i', diffs_new, diffs_new)
            near_idxs = np.where(d2_new <= radius * radius)[0]

            best_idx = nearest_idx
            best_cost = new_cost
            for cand in near_idxs:
                cx, cy = positions[cand]
                d = float(np.hypot(nx_ - cx, ny_ - cy))
                cand_cost = costs[cand] + d
                if cand_cost < best_cost and env.is_segment_traversable(cx, cy, nx_, ny_, step=collision_step):
                    best_idx = int(cand)
                    best_cost = cand_cost
            parent_idx = best_idx
            new_cost = best_cost

        positions[n_nodes] = (nx_, ny_)
        parents[n_nodes] = parent_idx
        costs[n_nodes] = new_cost
        new_idx = n_nodes
        n_nodes += 1

        if rewire and near_idxs is not None:
            for cand in near_idxs:
                if cand == parent_idx:
                    continue
                cx, cy = positions[cand]
                d = float(np.hypot(nx_ - cx, ny_ - cy))
                via_new_cost = new_cost + d
                if via_new_cost < costs[cand] and env.is_segment_traversable(nx_, ny_, cx, cy, step=collision_step):
                    parents[cand] = new_idx
                    costs[cand] = via_new_cost

        if np.hypot(nx_ - goal_arr[0], ny_ - goal_arr[1]) <= goal_thresh:
            goal_idx = new_idx
            break

    if goal_idx is None:
        diffs_goal = positions[:n_nodes] - goal_arr
        d2_goal = np.einsum('ij,ij->i', diffs_goal, diffs_goal)
        goal_idx = int(np.argmin(d2_goal))

    return _reconstruct_path(positions, parents, goal_idx)
