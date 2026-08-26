# Jaxley L5PC Simulation

Copyright (c) 2026 Open Brain Institute

## Summary

This notebook demonstrates how to run a native [Jaxley](https://jaxley.readthedocs.io/) L5PC model in an interactive notebook.

The notebook loads the L5PC model from `jaxley-models`, applies a simple current-clamp stimulus, runs the simulation, plots the soma voltage response, and extracts electrophysiological features with eFEL.

The example is intended to be simple and directly runnable. Users can adjust the stimulus parameters and rerun the notebook to see how the voltage trace and extracted features change.

## What the notebook does

- Loads a native Jaxley L5PC model.
- Lets the user configure a current-clamp stimulus.
- Runs a Jaxley simulation.
- Plots the soma voltage response.
- Extracts basic electrophysiological features using eFEL.
- Displays the feature summary directly in the notebook.

## What the notebook does not do

This notebook does not convert NEURON/MOD/HOC models to Jaxley and does not compare the Jaxley model against an OBI/NEURON MEModel.

## Use

Install the dependencies listed in `analysis_info.json`, then run the notebook from top to bottom.

The `jaxley-models` repository should be available next to the notebook or installed in the notebook environment.