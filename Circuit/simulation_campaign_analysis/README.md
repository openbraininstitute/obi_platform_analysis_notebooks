# Explore a circuit simulation campaign
Copyright (c) 2026 Open Brain Institute

Authors: James B. Isbister

last modified: 10.2026

## Summary
Loads the spikes of a circuit simulation campaign, or of a single simulation, and explores them. It works for any circuit simulation: point neurons (Brian2) or detailed neurons (NEURON), one or several node populations (simulated or virtual), any stimuli and recordings, and a scan over any number of parameters. On the platform, the files of the project's simulations and the public ones are mounted, and the notebook links to them rather than downloading them.

The notebook has four sections:
1. What was simulated: the campaign, its circuit and populations, the stimuli and recordings.
2. The neurons and their connectivity: a neuron selector, the node table as a pandas DataFrame, where the neurons are, and the circuit's connectivity between groups of neurons.
3. The spikes: a raster of every simulation, an interactive raster with the stimuli, recorded traces and firing rate, statistics of any selection of neurons across the scan, and the spikes as a pandas table, with statistics, a raster and histograms by group.
4. Saving the notebook to the project as a result notebook.

## Use
Run the first cell, log in if a link appears under it, and pick a project in its menu. Then set `CAMPAIGN` to a campaign id or platform link, or leave it empty and pick a campaign in the table, and run the rest of the cells. The **Group by** menu (section 2) and the **Statistic** menu (section 3d) set what the cells below them show: after changing one, run those cells again.

## Files
- `analysis_notebook.ipynb`: the notebook.
- `sim_analysis.py`: the code behind the plots and widgets, imported by the notebook. It must be next to the notebook, or in a folder beside it.

pandas is capped below 3, whose string columns the helper does not expect yet.
