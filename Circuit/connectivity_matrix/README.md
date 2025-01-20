# Extract and display a connectivity matrix from a SONATA circuit
Copyright (c) 2025 Open Brain Institute

Authors: Christoph Pokorny

Last modified: 01.2025

## Summary
This analysis extracts and visualizes a matrix of connection probabilities or #synapses per connection (mean/SD/...), grouped by a selected neuron property (layer, m-type, ...).

## Use
After loading a [SONATA](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md) circuit, the user can select one [_edge population_](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#representing-edges) (containing connections, formed by synapses) to extract the adjacency matrix from. In general, a SONATA circuit may contain more than one edge population in which case only one can be used at a time.

As a next step, the user can optionally select a pre- and post-synaptic _node set_ which is a conceptual group of neurons defined in the circuit's [_node sets file_](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#node-sets-file). 

Finally, the user can select a group-by property (e.g., layer, m-type, etc.) and the respective connectivity matrix will be extracted. For visualization, the type of connectivity matrix (connection probability or mean/SD/... #synapses per connection),  can be selected interactively.