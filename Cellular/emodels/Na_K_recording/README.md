# Analysis of recorded Na⁺ and K⁺ currents during an action potential
Copyright (c) 2025 Open Brain Institute

Authors: Ilkan Kilic

last modified: 08.2026

## Summary
This is a load-only analysis notebook for precomputed `SingleNeuronSimulation` result entities from the legacy single-cell workflow.

The notebook loads recorded membrane potential (`v`), sodium current (`ina`), and potassium current (`ik`) traces. It catalogs the available trace metadata, selects one matching recording location and stimulus condition, and plots the complete recording and a zoomed action potential.

## Inputs
The notebook uses existing platform results. Set the following values in the notebook:

- `virtual_lab_id`: the VLab containing the results;
- `project_id`: the project containing the results;
- `simulation_ids`: one or more `SingleNeuronSimulation` IDs, or an empty list to select results from the project.

Each result asset must contain the expected `simulation` and `stimulus` sections.

## Analysis
The notebook exposes the analysis choices directly:

1. inspect the result summary and trace catalog;
2. select the recording location, stimulus, and variable names explicitly;
3. plot voltage, Na⁺ current, and K⁺ current over the full recording;
4. inspect one action potential with a dual-axis voltage/current plot.

## Use
Install the dependencies listed in `analysis_info.json`. Open `analysis_notebook.ipynb`, set the VLab and project identifiers, provide or select the precomputed result IDs, and execute the analysis cells. Authentication and asset download use the configured platform project context; no new simulation is run.
