import numpy as np
import bluepysnap

from matplotlib import pyplot as plt

def plot_comparison(control_map, mutation_map):
    """
    Plot side-by-side comparison of control and mutation for the selected stimulus step.
    
    Parameters:
    - percent_start: The stimulus amplitude (percent_start) to compare
    """

    def plot_func_(percent_start):    
        # Create figure with 2 subplots (side by side)
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ========== LEFT PLOT: CONTROL SIMULATION ==========
        # Load the control simulation
        control_sim = bluepysnap.Simulation(str(control_map[percent_start]))
        control_voltage = control_sim.reports['Recording 0'].filter().report
        
        # Extract time and voltage data
        time = control_voltage.index.to_numpy()
        voltage = control_voltage.to_numpy()
        
        # Plot voltage traces (up to 5 cells)
        num_cells = min(5, voltage.shape[1] if voltage.ndim > 1 else 1)
        for i in range(num_cells):
            cell_voltage = voltage[:, i] if voltage.ndim > 1 else voltage
            ax_left.plot(time, cell_voltage, alpha=0.7, linewidth=1.2)
        
        # Count spikes
        control_spikes = control_sim.spikes[control_sim.spikes.population_names[0]].get()
        num_control_spikes = len(control_spikes)
        
        # Format the plot
        ax_left.set_title(f'CONTROL - Step_{int(percent_start)}\nTotal Spikes: {num_control_spikes}', 
                        fontweight='bold', fontsize=12)
        ax_left.set_xlabel('Time (ms)')
        ax_left.set_ylabel('Voltage (mV)')
        ax_left.grid(True, alpha=0.3)
        
        # ========== RIGHT PLOT: MUTATION SIMULATION ==========
        # Load the mutation simulation
        mutation_sim = bluepysnap.Simulation(str(mutation_map[percent_start]))
        mutation_voltage = mutation_sim.reports['Recording 0'].filter().report
        
        # Extract time and voltage data
        time = mutation_voltage.index.to_numpy()
        voltage = mutation_voltage.to_numpy()
        
        # Plot voltage traces (up to 5 cells)
        num_cells = min(5, voltage.shape[1] if voltage.ndim > 1 else 1)
        for i in range(num_cells):
            cell_voltage = voltage[:, i] if voltage.ndim > 1 else voltage
            ax_right.plot(time, cell_voltage, alpha=0.7, linewidth=1.2)
        
        # Count spikes
        mutation_spikes = mutation_sim.spikes[mutation_sim.spikes.population_names[0]].get()
        num_mutation_spikes = len(mutation_spikes)
        
        # Format the plot
        ax_right.set_title(f'MUTATION - Step_{int(percent_start)}\nTotal Spikes: {num_mutation_spikes}', 
                        fontweight='bold', fontsize=12)
        ax_right.set_xlabel('Time (ms)')
        ax_right.set_ylabel('Voltage (mV)')
        ax_right.grid(True, alpha=0.3)
        
        # Add overall title and display
        plt.suptitle(f'Control vs Mutation Comparison - Step_{int(percent_start)}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # # Print summary statistics
        # spike_diff = num_mutation_spikes - num_control_spikes
        # percent_change = (spike_diff / max(num_control_spikes, 1)) * 100
        
        print(f"\nSummary for Step_{int(percent_start)}:")
        print(f"   Control spikes:  {num_control_spikes}")
        print(f"   Mutation spikes: {num_mutation_spikes}")
        # print(f"   Difference:      {spike_diff:+d} ({percent_change:+.1f}%)")

    return plot_func_
