import json

from tqdm import tqdm
from pathlib import Path
from entitysdk.models import SimulationCampaign, SimulationResult, SimulationExecution, Circuit
from entitysdk.types import AssetLabel
from entitysdk.exception import EntitySDKError


def download_circuit_connectivity(client, simulation_campaign, output_dir=None):
    import json

    from tqdm import tqdm
    from pathlib import Path
    from entitysdk.models import SimulationCampaign, SimulationResult, SimulationExecution, Circuit
    from entitysdk.types import AssetLabel
    from entitysdk.exception import EntitySDKError
    if output_dir is None:
        output_dir = "."
    output_dir = Path(output_dir)

    cfg_path = client.download_file(
        entity_id=simulation_campaign.id,
        entity_type=SimulationCampaign,
        asset_id=simulation_campaign.assets[0].id,
        output_path=output_dir/"campaign_config.json"
    )
    with open(cfg_path, "r") as fid:
        campaign_cfg = json.load(fid)

    try:
        circ_entity = client.get_entity(
            entity_id=campaign_cfg["form"]["initialize"]["circuit"]["id_str"],
            entity_type=Circuit
        )
    except EntitySDKError:
        raise ValueError("Campaign circuit cannot be accessed.")

    assets_dict = {asset.label: asset.id for asset in circ_entity.assets}
    if AssetLabel.circuit_connectivity_matrices not in assets_dict:
        raise ValueError("No connectivity matrix available for the campaign circuit.")
    matrices_id_str = assets_dict[AssetLabel.circuit_connectivity_matrices]

    connectivity_root = output_dir/"connectivity"
    connectivity_files = client.fetch_directory(
        entity_id=circ_entity.id,
        entity_type=Circuit,
        asset_id=matrices_id_str,
        output_path=connectivity_root
    )

    connectivity_file_dict = {
        file.name: file
        for file in connectivity_files
    }
    assert "matrix_config.json" in connectivity_file_dict
    with open(connectivity_file_dict["matrix_config.json"], "r") as fid:
        matrix_file_dict = json.load(fid)
    matrix_files_out = {}
    for _, matrix_files in matrix_file_dict.items():
        for k, v in matrix_files.items():
            matrix_files_out[v["description"]] = connectivity_file_dict["matrix_config.json"].parent / v["path"]

    return matrix_files_out


def download_simulation_spikes(client=None, model=None, output_dir=None):
    if output_dir is None:
        output_dir = "."
    output_dir = Path(output_dir)

    asset_dict = {
        asset.label: asset
        for asset in model.assets
    }
    if AssetLabel.spike_report in asset_dict:
        spike_asset = asset_dict[AssetLabel.spike_report]

        fn = client.download_file(
            entity_id=model.id,
            entity_type=SimulationResult,
            asset_id=spike_asset.id,
            output_path=output_dir/"spikes.h5"
        )
        return fn
    else:
        return None

def stage_sim_campaign_spikes(client, simulation_campaign_id, output_dir):
    simulation_campaign = client.get_entity(entity_type=SimulationCampaign, entity_id=simulation_campaign_id)

    simulation_spikes_paths = []

    for id, simulation in tqdm(list(enumerate(simulation_campaign.simulations))):
        
        simulation_executions = client.search_entity(
                    entity_type=SimulationExecution,
                    query={"used__id": simulation.id}
                ).all()
        
        if len(simulation_executions) == 0:
            print(f"Warning: No SimulationExecution found for Simulation ID: {simulation.id}")
            return simulation_spikes_paths
        simulation_execution = simulation_executions[0]
        
        simulation_result_id = simulation_execution.generated[0].id
        simulation_result = client.get_entity(entity_type=SimulationResult, entity_id=simulation_result_id)
        
        simulation_spikes_path = download_simulation_spikes(
            client=client,
            model=simulation_result,
            output_dir=output_dir + f"/simulation_{id}",
        )
        if simulation_spikes_path is not None:
            simulation_spikes_paths.append(simulation_spikes_path)

    return simulation_spikes_paths
