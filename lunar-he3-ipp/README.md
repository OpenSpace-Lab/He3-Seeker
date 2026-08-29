# Lunar He-3 Active Path Planning: RRT vs RRT* (Phase 2)

Code and simulation environment for an active information-gathering (Informative
Path Planning) loop that compares standard RRT and RRT* as the path-planning
component, evaluated over two lunar test regions, supplementary to the
accompanying paper. Builds on the He-3 ground truth fields produced by the
companion `lunar-he3-groundtruth` repository.

Each round of the loop: a Gaussian Process belief model picks the most
uncertain reachable point as the next target, a path planner (RRT or RRT*)
generates an obstacle-avoiding path to it, the path is resampled at fixed
arc-length intervals to get measurement points, and the GP is updated with
the new observations. 30 rounds per (region, planner) pair, same random seed
across planners so the only varying factor is whether the planner performs
near-neighbor rewiring.

## Regions

| Region | Center | Setting |
|---|---|---|
| T1 | 20.190N, 30.772E | Taurus-Littrow (Apollo 17), mare/highland contact, steep compositional gradient |
| T2 | 26.132N, 3.633E | Hadley Rille (Apollo 15), sinuous rille / mare contact, higher terrain obstruction |

Non-traversable terrain is derived from the SLDEM2015 DEM using a combined
slope + local-relief criterion (see `docs/task2_METHOD.md` for the full
derivation, including why slope alone was insufficient at this DEM
resolution).

## Data sources

- Kaguya MI mineral maps (FeO, OMAT), USGS Astrogeology Science Center
- SLDEM2015 DEM (LRO LOLA + Kaguya TC fusion, ~59 m/pixel), MIT LOLA Science
  Team: `https://imbrium.mit.edu/DATA/SLDEM2015/TILES/FLOAT_IMG/`

## References

- Karaman, S.; Frazzoli, E. Sampling-based Algorithms for Optimal Motion
  Planning. *The International Journal of Robotics Research* **2011**, *30*,
  846-894.
- LaValle, S. M. Rapidly-Exploring Random Trees: A New Tool for Path
  Planning. Technical Report, Iowa State University, 1998.
- Viseras, A.; Shutin, D.; Merino, L. Robotic Active Information Gathering
  for Spatial Field Reconstruction with Rapidly-Exploring Random Trees and
  Online Learning of Gaussian Processes. *Sensors* **2019**, *19*, 1016.
- Fa, W.; Jin, Y.-Q. Quantitative estimation of helium-3 in lunar regolith.
  *Icarus* **2007**, *190*(1), 15-23.
- Barker, M.K., et al. A new lunar digital elevation model from the Lunar
  Orbiter Laser Altimeter and SELENE Terrain Camera. *Icarus* **2016**, *273*,
  346-355.

## Repository layout

```
scripts/
  generate_task2.py        - fetches He-3 field + DEM/slope for T1/T2 from remote sources
  add_relief_obstacles.py  - adds local-relief obstacle criterion on top of slope mask
  environment.py            - environment queries and path collision checks
  sensor.py                 - additive Gaussian sensor noise model
  belief_model.py           - Gaussian Process belief model (scikit-learn)
  planner.py                 - shared RRT / RRT* implementation
  info_metric.py             - max-uncertainty target selection
  evaluator.py                - RMSE / coverage / path-length tracking
  visualizer.py               - trajectory, reconstruction, and comparison plots
  main.py                      - main active-loop entry point
  replot.py                    - regenerates figures from saved results, no re-planning
outputs/    saved per-round results (.npz) and intermediate fields (.npy)
figures/    trajectory, reconstruction, comparison, and RMSE-curve plots
docs/       task2_METHOD.md (ground truth + obstacle derivation), task3_RRT_REPORT.md (full report)
```

All paths are resolved relative to each script's own location, so the
repository can be cloned and run from any location without modification
(aside from `GDAL_DATA`, see Setup).

## Setup

```
conda env create -f environment.yml
conda activate he3
```

Alternatively, on a system with the GDAL C library already installed,
`pip install -r requirements.txt` can be used instead.

If GDAL can't locate its data directory automatically when running
`generate_task2.py`, set `GDAL_DATA` yourself before running (conda:
`<env>/Library/share/gdal` on Windows, `<env>/share/gdal` on Linux/Mac).

## Running

```
python scripts/generate_task2.py        # He-3 field + DEM + slope for T1/T2
python scripts/add_relief_obstacles.py  # adds relief-based obstacle mask
python scripts/main.py                  # full RRT vs RRT* active loop (30 rounds x 2 regions x 2 planners)
python scripts/replot.py                # optional: regenerate figures from saved results
```

`main.py` takes a few minutes per region/planner combination; it writes
results to `outputs/T{1,2}_results_{RRT,RRTstar}.npz` and figures to
`figures/`.

## Outputs

For each region and planner, the loop produces:
- `outputs/T{1,2}_results_{RRT,RRTstar}.npz` - per-round paths, sample
  points, final GP mean/std grids, all observations, and the RMSE /
  coverage / path-length / length-scale history
- `figures/T{1,2}_trajectory_{RRT,RRTstar}.png` - He-3 heatmap with
  trajectory and sample points overlaid
- `figures/T{1,2}_reconstruction_{RRT,RRTstar}.png` - GP predicted mean
  and predictive uncertainty
- `figures/T{1,2}_comparison.png` - side-by-side RRT vs RRT* comparison
  with summary metrics
- `figures/T{1,2}_rmse_curve.png` - RMSE convergence over rounds
- `figures/T{1,2}_sample_value_distribution.png` - sample value
  distribution vs. full-field distribution, and percentile-rank histogram

See `docs/task3_RRT_REPORT.md` for the full methodology, results, and
discussion.
