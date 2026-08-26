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
    """Authenticate with OBI and return the environment, token, and project context.

    This function is used at the start of the notebook workflow. It selects the
    configured OBI environment, requests a DAF authentication token, and uses
    that token to determine the projects available to the current user.
    """
    # Import logging locally because it is only needed when authentication runs.
    import logging

    # Find every logger currently registered with Python's root logger.
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]

    # Iterate over the registered loggers so SDK/authentication noise is reduced.
    for logger in loggers:
        # Keep only error messages from each registered logger for this process.
        logger.setLevel(logging.ERROR)

    # Resolve the OBI environment configured for the current notebook session.
    env_ = get_environment()

    # Request an authentication token using the DAF authentication mode.
    token = obi_auth.get_token(environment=env_, auth_mode="daf")

    # Resolve the project context available with this token and environment.
    project_context = get_projects(token, env=env_)

    # Return values in the order expected by the notebook caller.
    return env_, token, project_context


def display_datasets(env, token, project_context):
    """Display an interactive dropdown containing accessible EM datasets.

    The selected dataset is returned indirectly through the widget. The client
    is returned alongside the widget so later functions can query OBI using the
    same authenticated session.
    """
    # Create an entitysdk client for the selected environment and project.
    client = Client(project_context=project_context, token_manager=token, environment=env)

    # Query OBI for all electron-microscopy dense-reconstruction datasets.
    em_datasets = client.search_entity(entity_type=models.EMDenseReconstructionDataset).all()

    # Build a dropdown whose labels are dataset names and values are dataset objects.
    sel_em = widgets.Dropdown(options={dataset.name: dataset for dataset in em_datasets})

    # Render the dropdown in the active Jupyter/IPython display context.
    display(sel_em)

    # Preserve the original unordered set return used by the notebook workflow.
    return {client, sel_em}


def display_neurons(client, sel_em, neuron_id="PASTE ID IN HERE"):
    """Display morphologies derived from the selected EM dataset when requested.

    If ``neuron_id`` keeps its placeholder value, this function queries OBI for
    derived cell morphologies and displays a second dropdown. If a real ID is
    supplied, the caller uses manual-ID mode and this function returns ``None``.
    """
    # Start with no morphology widget; this remains the result in manual-ID mode.
    sel_nrn = None

    # Use the interactive morphology lookup only when no manual ID was supplied.
    if neuron_id == "PASTE ID IN HERE":
        # Search for derivations that produce cell morphologies from the selected dataset.
        derivations = client.search_entity(entity_type=models.Derivation, query={
            # Restrict results to the EM dense-reconstruction morphology derivation type.
            "derivation_type": types.DerivationType.em_dense_reconstruction_dataset_cell_morphology,
            # Restrict results to derivations that used the selected dataset.
            "used__id": sel_em.value.id
        })

        # Resolve each derivation's generated entity as a CellMorphology object.
        morphologies = [
            # Fetch the generated morphology by its entity ID.
            client.get_entity(entity_id=derivation.generated.id, entity_type=CellMorphology)
            # Repeat the fetch for every matching derivation.
            for derivation in derivations
        ]

        # Build a dropdown whose labels are descriptions and values are morphology objects.
        sel_nrn = widgets.Dropdown(options={m.description: m for m in morphologies})

        # Render the morphology dropdown in the active notebook display context.
        display(sel_nrn)

    # Return the widget for selection mode or None for manual-ID mode.
    return sel_nrn


from contextlib import contextmanager
@contextmanager
def suppress_output():
    """Temporarily redirect process stdout and stderr to the null device.

    The context manager is used around the morphology loader because that loader
    may emit low-level output. Original file descriptors are restored even when
    the wrapped operation raises an exception.
    """
    # Open the operating system's null device as a writable file destination.
    with open(os.devnull, "w") as devnull:
        # Duplicate the current stdout file descriptor for later restoration.
        old_stdout = os.dup(1)

        # Duplicate the current stderr file descriptor for later restoration.
        old_stderr = os.dup(2)

        try:
            # Redirect stdout writes to the null device.
            os.dup2(devnull.fileno(), 1)

            # Redirect stderr writes to the null device.
            os.dup2(devnull.fileno(), 2)

            # Run the caller's block while both standard streams are redirected.
            yield
        finally:
            # Restore the original stdout file descriptor.
            os.dup2(old_stdout, 1)

            # Restore the original stderr file descriptor.
            os.dup2(old_stderr, 2)

            # Close the duplicated stdout descriptor after restoration.
            os.close(old_stdout)

            # Close the duplicated stderr descriptor after restoration.
            os.close(old_stderr)


def download_data(client, sel_nrn):
    """Export the selected morphology and download its source mesh.

    ``sel_nrn`` is the morphology dropdown returned by ``display_neurons``.
    When it is ``None``, this function intentionally reads the notebook-level
    ``neuron_id`` variable for manual-ID mode. The returned paths point to the
    local HDF5 morphology and GLB mesh files used by the validation viewer.
    """
    # Keep the selected morphology asset ID available for the original workflow.
    morphology_id = None

    # Initialize the identifier used to name the local output files.
    neuron_pt_root_id = None

    # Use the selected dropdown morphology when interactive selection is enabled.
    if sel_nrn is not None:
        # Read the currently selected CellMorphology entity from the widget.
        selected_morphology = sel_nrn.value

        # Create an OBI-One morphology wrapper from the platform entity ID.
        morphology = CellMorphologyFromID(id_str=str(selected_morphology.id))

        # Read the first morphology asset ID; this variable is retained for compatibility.
        morphology_id = selected_morphology.assets[0].id

        # Read the first asset path, which contains the neuron root ID in its filename.
        morphology_with_pt_root_it = selected_morphology.assets[0].path

        # Use the portion before the first underscore as the local file-name root.
        neuron_pt_root_id = morphology_with_pt_root_it.split("_")[0]
    else:
        # Create an OBI-One morphology wrapper from the notebook-level manual ID.
        morphology = CellMorphologyFromID(id_str=neuron_id)

        # Use the manual ID as the morphology identifier for the original workflow.
        morphology_id = neuron_id

        # Use the manual ID as the local file-name root.
        neuron_pt_root_id = neuron_id

    # Construct the local path for the downloaded neuron mesh.
    mesh_path = f"{neuron_pt_root_id}.glb"

    # Construct the local path for the exported spiny-neuron morphology.
    neuron_path = f"{neuron_pt_root_id}.h5"

    # Export the platform morphology into the local HDF5 file.
    morphology.write_spiny_neuron_h5(path_to=neuron_path, db_client=client)

    # Suppress low-level loader output while validating/loading the exported morphology.
    with suppress_output():
        # Load the HDF5 morphology and its associated meshes; the result is intentionally unused.
        m = load_morphology_with_spines(neuron_path, load_meshes=True)

    # Resolve the source mesh entity associated with the selected morphology.
    mesh = morphology.source_mesh_entity(db_client=client)

    # Download the first mesh asset to the local GLB path.
    client.download_file(entity_id=mesh.id, entity_type=models.EMCellMesh, asset_id=mesh.assets[0].id, output_path=mesh_path)

    # Return the local morphology and mesh paths for the validation viewer.
    return neuron_path, mesh_path
