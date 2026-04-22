# nbS1 slice neuron sets based on a hexgrid parcellation
Copyright (c) 2026 Open Brain Institute

Authors: Christoph Pokorny

Last modified: 04.2026

## Summary
This notebook allows a user to interactively select and visualize slice neuron sets based on a pre-computed hexagonal parcellation of the `nbS1` circuit. The pre-computed parcellation from the [ConnectomeUtilities](https://github.com/openbraininstitute/ConnectomeUtilities/blob/main/examples/data) repo is used.

## Use
After loading the `nbS1` circuit together with the corresponding hexgrid parcellation, the user can choose how many connected hex columns to select in a long slice spanning the diagonal of the full circuit model (from 1 to 18 columns).

The selected slice can then be interactively visualized in 2D and 3D. The actual neuron IDs belonging to the selected hex columns can be obtained by writing them to a [SONATA](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md)-compatible .json [node sets file](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#node-sets-file), or optionally by copy-pasting them directly from the notebook. These neuron IDs can then be used in other workflows, such as circuit extraction, simulation, or analysis.

Finally, a plot will be generated showing an overview of the neuron counts per hex column.