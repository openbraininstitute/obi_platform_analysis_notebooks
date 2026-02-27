import time
import os
import pandas

import bluepysnap as snap
import plotly.graph_objects as go

from datetime import datetime
from entitysdk import models
from ipywidgets import widgets

def stage_selected_circuit_locally(client, circuit_id):
    # Fetch circuit
    fetched = client.get_entity(entity_id=circuit_id, entity_type=models.Circuit)
    print(f"Circuit fetched: {fetched.name} (ID {fetched.id})\n")
    print(f"#Neurons: {fetched.number_neurons}, #Synapses: {fetched.number_synapses}, #Connections: {fetched.number_connections}\n")
    print(f"{fetched.description}\n")

    # Download SONATA circuit files
    asset = [asset for asset in fetched.assets if asset.label=="sonata_circuit"][0]
    asset_dir = asset.path 
    circuit_dir = "analysis_circuit_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')

    t0 = time.time()
    client.download_directory(
        entity_id=fetched.id,
        entity_type=models.Circuit,
        asset_id=asset.id,
        output_path=circuit_dir,
        max_concurrent=4,  # Parallel file download
    )
    t = time.time() - t0
    print(f"Circuit files downloaded to '{os.path.join(circuit_dir, asset_dir)}' in {t:.1f}s")
    circuit_asset_dir = os.path.join(circuit_dir, asset_dir)
    circuit_config = os.path.join(circuit_asset_dir, "circuit_config.json")
    assert os.path.exists(circuit_config), f"ERROR: Circuit config '{os.path.split(circuit_config)[1]}' not found!"

    circ = snap.Circuit(circuit_config)
    return circ

def nodeset_selector(circ):
    nodepop = widgets.Dropdown(
        options=
        list(circ.nodes.keys()),
        description="Node population"
    )
    nodeset = widgets.Dropdown(
        options=
        list(circ.node_sets.content.keys()),
        description='Node set')
    return nodepop, nodeset

def properties_from_dataframe_selector(val_df):
    # This type of display only works for categorical properties. In the future, numerical properties could be binned...
    max_num_unique_vals = 25
    is_categorical = val_df.dtypes.apply(lambda _x: isinstance(_x, pandas.CategoricalDtype))
    has_few_vals = val_df.apply(lambda _x: len(_x.drop_duplicates()) <= max_num_unique_vals, axis=0)
    valid_props = is_categorical[is_categorical | has_few_vals].index.values

    to_display = widgets.SelectMultiple(options=valid_props,
                                        index=tuple(range(len(valid_props)))[:8],
                                        description="Properties") # 8 is the arbitrarily decided maximum
    return to_display

def display_composition_for_selected_properties(neuron_properties, selected_properties, title):
    # Test of user selection
    assert len(selected_properties) >= 2, "Please select AT LEAST 2 properties"
    assert len(selected_properties) <= 8, "Please select AT MOST 8 properties"
    # Dataframe of only the selected properties
    use_df = neuron_properties[selected_properties].apply(pandas.Categorical, axis=0)

    # Create a dataframe for a lookup of every possible (categorical) value of the selected properties to a unique index.
    # Index: level 0: Name of the property, level 1: value of the property; values: unique index.
    label_idx_lo = pandas.concat([pandas.Series(use_df[col].values.categories.values, name="value")
                                for col in use_df.columns], keys=use_df.columns,
                                names=["column"], axis=0).reset_index(level="column")
    label_idx_lo["index"] = range(len(label_idx_lo))
    label_idx_lo = label_idx_lo.set_index(["column", "value"])["index"]

    # The sankey links are built by iterating over pairs of adjacent columns.
    lnk_src = []; lnk_tgt = []; lnk_sz = []

    for c1, c2 in zip(use_df.columns[:-1], use_df.columns[1:]):
        # Size of a link: Number of overlapping values.
        counts = use_df[[c1, c2]].value_counts()
        for row_idx, row_val in counts.items():
            lnk_src.append(label_idx_lo[c1][row_idx[0]])
            lnk_tgt.append(label_idx_lo[c2][row_idx[1]])
            lnk_sz.append(row_val)

    # Create sankey
    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = label_idx_lo.index.to_frame()["value"],
        color = "blue"
        ),
        link = dict(
        source = lnk_src, # indices correspond to labels, eg A1, A2, A1, B1, ...
        target = lnk_tgt,
        value = lnk_sz
    ))])

    fig.update_layout(title_text=title, font_size=10)
    fig.show()
