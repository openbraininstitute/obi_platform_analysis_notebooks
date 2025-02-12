# cADpyr E-model Showcase
Copyright (c) 2025 Open Brain Institute

Author: Darshan Mandge, Open Brain Institute

Last Modified: 02.2025

## Summary
The notebook demonstrates various properties detailed morphology cADpyr (continuos adapting type pyramidal neuron) e-model of the Open Brain Institute(OBI). 
It plots: 
- optimisation protocol responses
- [currentscape](https://github.com/openbraininstitute/Currentscape) analysis
- frequency-current (F-I) curve
- current-voltage (V-I) curve
- recordings in different sections of the neuron using [BlueCellulab](https://github.com/openbraininstitute/BlueCelluLab)
- dendritic potential decay 
- backpropagating action potential (bAP) recordings
- excitatory postsynaptic potential (EPSP) recordings and EPSP ratio.

## Use
The software requirements of the notebook are mentioned in the `analysis_info.json`. Steps to run the notebook are given below:

1. The cADpyr e-model model `hoc` file,  `.asc` morphology file, mechanism (`.mod`) and `EM_*.json` file (containing the e-model parameters) files should be obtained from OBI. These could be downloaded from 
  - the [OBI Platform](https://openbraininstitute.org) from `Explore` -->`E-model` section --> Click on an e-model from the list --> `Download`. This final parameters file `EM_*.json` of the e-model.
  - [Blue Brain Open Data](https://registry.opendata.aws/bluebrain_opendata/) using [Amazon CLI](https://aws.amazon.com/cli/) 
  - (temporary for testing purpose) the data for below notebooks is also saved [here](https://openbraininstitute.sharepoint.com/:f:/s/OBI-Scientificstaff/EpqQOMfkUoRIv5mkPmaTdWEBuVeg6qEi93fJbmy-FSsgRA?e=SwGY4V) in a private folder.
2. After obtaining the files mentioned above you need to update the 
`emodel_folder_path`. The `emodel_name` (optional) can also be changed. Currently it is set to `cadpyr_emodel`. The detailed instructions are also available in the notebook.
3. Now, you should be able to run the notebook. Follow the text and comments in notebook to learn more about the model and results.

You can also test this notebook other pyramidal neuron e-models of OBI. Follow the steps above replacing the models files botained from OBI platform