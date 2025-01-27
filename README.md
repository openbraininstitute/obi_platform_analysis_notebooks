# obi_platform_analysis_notebooks
OBI Platform Analysis Notebooks

## Notebook Folder Structure

The folder has been divided into 3 main subfolders, each corresponding to the level of analysis: 

- **Cellular**
  - [**display_morphology_population_features**](Cellular/display_morphology_population_features/analysis_notebook.ipynb): Plot morphological features for a population of morphologies
  - **efeature_extraction**
    - [**current_clamp_ephys_extraction**](Cellular/efeature_extraction/current_clamp_ephys_extraction/analysis_notebook.ipynb): Extraction e-features from voltage clamp experiments for analysing the firing behaviour of neurons, AP properties, subthreshold and suprathreshold voltage properties
    - [**voltage_clamp_ephys_extraction**](Cellular/efeature_extraction/voltage_clamp_ephys_extraction/analysis_notebook.ipynb): Extraction e-features from whole cell patch voltage clamp data. Use for ion channel model building.
  - **emodels**
      - [**parameter_plots**](Cellular/emodels/parameters_plot/analysis_notebook.ipynb): Compare parameters of an e-model across sections on Open Brain Institute Platform (OBP)
      - **lfpy_simulations**
        - [**LFPy_active_iclamp**](Cellular/emodels/lfpy_simulations/active_emodel_Iclamp/analysis_notebook.ipynb): Running an OBP e-model with current clamp: record and plot LFP on a microelectrode array
        - [**LFPy_active_synapse**](Cellular/emodels/lfpy_simulations/active_emodel_synapses/analysis_notebook.ipynb): Running an OBP active e-model with a synapse activation
        - [**LFPy_passive_synapse_1**](Cellular/emodels/lfpy_simulations/passive_emodel_synapses/analysis_notebook_1.ipynb): Running a passive OBP e-model with a synapse activation: Plot LFP Heatmap
        - [**LFPy_passive_synapse_2**](Cellular/emodels/lfpy_simulations/passive_emodel_synapses/analysis_notebook_2.ipynb): Running a passive OBP e-model with a synapse activation: Plot Local Field
    - [**single_cell_currentscape_analysis**](Cellular/single_cell_currentscape_analysis/analysis_notebook.ipynb): Currentscape analysis of single cells
    - [**single_cell_impedance_analysis**](Cellular/single_cell_impedance_analysis/analysis_notebook.ipynb): Impedance analysis of single cells
        
- **Circuit**
  - [**adjacency_matrix**](Circuit/adjacency_matrix/analysis_notebook.ipynb): extracts and visualize the connectivity between all pairs of pre- and post-synaptic neurons
  - [**circuit_composition**](Circuit/circuit_composition/analysis_notebook.ipynb): displays the composition of a circuit model with respect to user-selected neuron properties, such as morphological types, electrical types, etc.
  - [**connectivity_matrix**](Circuit/connectivity_matrix/analysis_notebook.ipynb):  extracts and visualizes a matrix of connection probabilities or #synapses per connection (mean/std/...), grouped by a selected neuron property (layer, m-type, ...).
 
- **System**
