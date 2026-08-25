from morph_spines import load_morphology_with_spines

import obi_auth
import pylmesh

import os
from entitysdk import Client, types, models
from obi_one import CellMorphologyFromID
from entitysdk.models import EMCellMesh, EMDenseReconstructionDataset, CellMorphology
from obi_notebook. get_projects import get_projects
from obi_notebook.get_environment import get_environment
from ipywidgets import widgets

def authenticate():
    import logging
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.ERROR)

    env_ = get_environment()
    token = obi_auth.get_token(environment=env_, auth_mode="daf")
    project_context = get_projects(token, env=env_)

    return env_, token, project_context

def display_datasets(env, token, project_context):
    client = Client(project_context=project_context, token_manager=token, environment=env)
    em_datasets = client.search_entity(entity_type=models.EMDenseReconstructionDataset).all()
    sel_em = widgets.Dropdown(options={dataset.name: dataset for dataset in em_datasets})
    display(sel_em)
    return {client, sel_em}

def display_neurons(client, sel_em, neuron_id="PASTE ID IN HERE"):

    sel_nrn = None
    if neuron_id == "PASTE ID IN HERE":
        # Find all CellMorphologies from MICrONS
        derivations = client.search_entity(entity_type=models.Derivation, query={
            "derivation_type": types.DerivationType.em_dense_reconstruction_dataset_cell_morphology,
            "used__id": sel_em.value.id
        })
        morphologies = [client.get_entity(entity_id=derivation.generated.id, entity_type=CellMorphology)
                        for derivation in derivations]
        sel_nrn = widgets.Dropdown(options={m.description: m for m in morphologies})
        display(sel_nrn)
    return sel_nrn

from contextlib import contextmanager
@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)

        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
                        
def download_data(client, sel_nrn):

    morphology_id = None
    neuron_pt_root_id = None

    if sel_nrn is not None:
        selected_morphology = sel_nrn.value
        morphology = CellMorphologyFromID(id_str=str(selected_morphology.id))
        morphology_id = selected_morphology.assets[0].id
        morphology_with_pt_root_it = selected_morphology.assets[0].path 
        neuron_pt_root_id = morphology_with_pt_root_it.split("_")[0]
    else:
        morphology = CellMorphologyFromID(id_str=neuron_id)
        morphology_id = neuron_id
        neuron_pt_root_id = neuron_id
        
    # Where to place the neuron and mesh
    mesh_path = f"{neuron_pt_root_id}.glb"
    neuron_path = f"{neuron_pt_root_id}.h5"

    # Load spiny neuron
    morphology.write_spiny_neuron_h5(path_to=neuron_path, db_client=client)

    with suppress_output():
        m = load_morphology_with_spines(neuron_path, load_meshes=True)

    # Download and load mesh
    mesh = morphology.source_mesh_entity(db_client=client)
    client.download_file(entity_id=mesh.id, entity_type=models.EMCellMesh, asset_id=mesh.assets[0].id, output_path=mesh_path)

    return neuron_path, mesh_path