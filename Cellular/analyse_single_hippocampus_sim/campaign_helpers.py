import json
import bluepysnap
import numpy as np
import efel

from entitysdk.staging.simulation import stage_simulation
from entitysdk.staging.simulation_result import stage_simulation_result

from entitysdk.models import SimulationCampaign, SimulationExecution, SimulationResult

from matplotlib import pyplot as plt

# Set eFEL settings for proper AP feature extraction
efel.set_setting('Threshold', -20.0)
efel.set_setting('interp_step', 0.025)
efel.set_setting('strict_stiminterval', True)

def stage_sim_campaign(client, simulation_campaign_id, output_dir):
    simulation_campaign = client.get_entity(entity_type=SimulationCampaign, entity_id=simulation_campaign_id)

    simulation_config_paths = []

    for id, simulation in enumerate(simulation_campaign.simulations):
        
        simulation_executions = client.search_entity(
                    entity_type=SimulationExecution,
                    query={"used__id": simulation.id}
                ).all()
        
        if len(simulation_executions) == 0:
            print(f"Warning: No SimulationExecution found for Simulation ID: {simulation.id}")
        simulation_execution = simulation_executions[0]
        
        simulation_result_id = simulation_execution.generated[0].id
        simulation_result = client.get_entity(entity_type=SimulationResult, entity_id=simulation_result_id)
        
        simulation_config_path = stage_simulation_result(
            client=client,
            model=simulation_result,
            output_dir=output_dir + f"/simulation_{id}",
            simulation_config_file=None,
        )
        simulation_config_paths.append(simulation_config_path)

    return simulation_config_paths

# Extract percent_start values from simulation configs
def get_percent_start(sim_path):
    """Read the percent_start value from simulation config"""
    config_file = sim_path.parent / "simulation_config.json"
    with open(config_file) as f:
        config = json.load(f)
        # Get the first stimulus input
        first_input = list(config['inputs'].values())[1]
        return first_input['percent_start']

def detect_first_spike_time(voltage_trace, time_points, threshold=-20):
    """Find the time of the first spike peak"""
    above_threshold = voltage_trace > threshold
    crossings = np.where(np.diff(above_threshold.astype(int)) > 0)[0]
    
    if len(crossings) == 0:
        return None
    
    # Find peak after first crossing
    first_crossing = crossings[0]
    search_end = min(first_crossing + 50, len(voltage_trace))
    peak_idx = first_crossing + np.argmax(voltage_trace[first_crossing:search_end])
    
    return time_points[peak_idx]

def extract_first_ap(voltage_trace, time_points, spike_time, before_ms=10, after_ms=30):
    """Extract voltage window around first AP"""
    mask = (time_points >= spike_time - before_ms) & (time_points <= spike_time + after_ms)
    return time_points[mask] - spike_time, voltage_trace[mask]

def extract_features_from_simulation(sim_path):
    """
    Extract eFEL features from a single simulation.
    
    Parameters:
    - sim_path: Path to simulation config
    
    Returns:
    - Dictionary with features and metadata
    """
    # Load simulation
    sim = bluepysnap.Simulation(str(sim_path))
    
    # Get voltage data (first cell)
    voltage_report = sim.reports['Recording 0'].filter().report
    time = voltage_report.index.to_numpy()
    voltage = voltage_report.to_numpy()[:, 0] if voltage_report.to_numpy().ndim > 1 else voltage_report.to_numpy()
    
    # Read stimulus timing from config
    config_file = sim_path.parent / "simulation_config.json"
    with open(config_file) as f:
        config = json.load(f)
    
    # Get stimulus parameters
    first_input = list(config['inputs'].values())[1]
    stim_delay = first_input['delay']
    stim_duration = first_input['duration']
    percent_start = first_input['percent_start']
    
    # Calculate stim_start and stim_end
    stim_start = stim_delay
    stim_end = stim_delay + stim_duration
    
    # Prepare trace for eFEL
    trace = {
        'T': time,
        'V': voltage,
        'stim_start': [stim_start],
        'stim_end': [stim_end]
    }
    
    # Features to extract
    feature_names = [
        'Spikecount',                    # Number of spikes
        # 'AP_duration_half_width',        # AP half-width duration
        # 'AHP_depth_abs',                 # Fast AHP depth (absolute)
        # 'AHP_depth_abs_slow',            # Slow AHP depth (absolute)
        # 'fast_AHP'                       # Fast AHP amplitude
    ]
    
    # Extract features
    result = efel.get_feature_values([trace], feature_names)[0]
    
    # Organize results
    features = {
        'percent_start': percent_start,
        'stim_start': stim_start,
        'stim_end': stim_end,
        'spike_count': result['Spikecount'][0] if result['Spikecount'] is not None else 0,
        # 'AP_half_width_mean': np.mean(result['AP_duration_half_width']) if result['AP_duration_half_width'] is not None else None,
        # 'AHP_depth_abs_mean': np.mean(result['AHP_depth_abs']) if result['AHP_depth_abs'] is not None else None,
        # 'fast_AHP_mean': np.mean(result['fast_AHP']) if result['fast_AHP'] is not None else None,
        # 'AHP_depth_abs_slow_mean': np.mean(result['AHP_depth_abs_slow']) if result['AHP_depth_abs_slow'] is not None else None,
    }
    
    return features
