# He3-Seeker

Code and simulation environment for the lunar He-3 mapping and informative
path-planning experiments described in the accompanying paper.

## Contents

- `lunar-he3-groundtruth/`: generates 500 x 500 m He-3 ground-truth fields
  for two lunar test regions from Kaguya MI mineral maps.
- `lunar-he3-ipp/`: compares RRT and RRT* informative path planning using
  the generated He-3 fields and lunar terrain constraints.

Each subdirectory contains its own README, dependency files, scripts, data
products, figures, and method documentation. Run the scripts from the
corresponding subdirectory as described in its README.
