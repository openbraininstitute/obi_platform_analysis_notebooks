import numpy as np
import bluepysnap

from campaign_helpers import detect_first_spike_time, extract_first_ap
from matplotlib import pyplot as plt


def plot_first_ap_comparison(control_map, mutation_map):
    """
    Plot first AP comparison between control and mutation.
    
    Parameters:
    - percent_start: Stimulus step to analyze
    """
    def plot_func_(percent_start):
        # Load simulations
        control_sim = bluepysnap.Simulation(str(control_map[percent_start]))
        mutation_sim = bluepysnap.Simulation(str(mutation_map[percent_start]))
        
        # Get voltage data (first cell)
        control_voltage = control_sim.reports['Recording 0'].filter().report
        control_time = control_voltage.index.to_numpy()
        control_v = control_voltage.to_numpy()[:, 0] if control_voltage.to_numpy().ndim > 1 else control_voltage.to_numpy()
        
        mutation_voltage = mutation_sim.reports['Recording 0'].filter().report
        mutation_time = mutation_voltage.index.to_numpy()
        mutation_v = mutation_voltage.to_numpy()[:, 0] if mutation_voltage.to_numpy().ndim > 1 else mutation_voltage.to_numpy()
        
        # Detect first spike
        control_spike_time = detect_first_spike_time(control_v, control_time)
        mutation_spike_time = detect_first_spike_time(mutation_v, mutation_time)
        
        if control_spike_time is None or mutation_spike_time is None:
            print("Warning: Could not detect spikes")
            return
        
        # Extract first AP windows
        control_t, control_ap = extract_first_ap(control_v, control_time, control_spike_time)
        mutation_t, mutation_ap = extract_first_ap(mutation_v, mutation_time, mutation_spike_time)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot traces
        ax.plot(control_t, control_ap, color='blue', linewidth=2.5, label='Control', alpha=0.8)
        ax.plot(mutation_t, mutation_ap, color='red', linewidth=2.5, label='Mutation', alpha=0.8)
        
        # Add vertical line at spike peak
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Format plot
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Voltage (mV)', fontsize=11)
        ax.set_title(f'First AP Waveform - Step_{int(percent_start)}', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # # Add scale bar (5 ms)
        # ax.plot([15, 20], [-80, -80], 'k-', linewidth=2)
        # ax.text(17.5, -85, '5 ms', ha='center', fontsize=9)
        
        # # Add voltage scale bar (20 mV)
        # ax.plot([22, 22], [-80, -60], 'k-', linewidth=2)
        # ax.text(24, -70, '20 mV', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        control_amp = np.max(control_ap) - np.min(control_ap[:len(control_ap)//3])
        mutation_amp = np.max(mutation_ap) - np.min(mutation_ap[:len(mutation_ap)//3])
        
        print(f"\nFirst AP Summary for Step_{int(percent_start)}:")
        print(f"   Control amplitude:  {control_amp:.1f} mV")
        print(f"   Mutation amplitude: {mutation_amp:.1f} mV")
        print(f"   Difference:         {mutation_amp - control_amp:+.1f} mV")

    return plot_func_
