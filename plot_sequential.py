"""
Script to generate comprehensive visualizations for sequential clearing model results.
This script creates plots for generation mix, costs, pumped storage operations, and reserves.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import traceback
import sys

# Add debug print function
def debug_print(message):
    print(f"DEBUG: {message}")
    sys.stdout.flush()  # Force output to be displayed immediately

debug_print("Script started")

def gather_data(day_type, data_path, results_path):
    """
    Gathers all necessary data for a given day type from the sequential results directory.
    
    Returns a dictionary with all relevant data frames.
    """
    data = {}
    
    # Define paths to data files - specifically from sequential results folder
    sequential_path = os.path.join(results_path, "sequential")
    project_root = os.path.dirname(results_path)
    
    # Energy generation from the sequential model
    energy_generation_path = os.path.join(sequential_path, f"energy_generation_{day_type}.csv")
    
    # Reserve provision from the reserve step
    reserve_path = os.path.join(sequential_path, f"reserve_solution_{day_type}.csv")
    
    # Demand data
    demand_path = os.path.join(sequential_path, f"energy_demand_{day_type}.csv")
    
    # Pumped storage data
    ps_data_path = os.path.join(sequential_path, f"ps_timeseries_{day_type}.csv")
    
    coopt_path = os.path.join(results_path, "coopt")

    # Generator data for fuel types
    gen_data_path = os.path.join(data_path, "Generators.csv")
    re_data_path = os.path.join(data_path, "GeneratorsRE.csv")
    
    # Try to load all files with error handling
    try:
        data['energy_generation'] = pd.read_csv(energy_generation_path)
        print(f"Loaded energy generation from {energy_generation_path}")
    except Exception as e:
        print(f"Warning: Could not load energy generation: {e}")
        data['energy_generation'] = pd.DataFrame()
    
    try:
        data['reserve'] = pd.read_csv(reserve_path)
        print(f"Loaded reserve solution from {reserve_path}")
    except Exception as e:
        print(f"Warning: Could not load reserve solution: {e}")
        data['reserve'] = pd.DataFrame()
    
    try:
        data['demand'] = pd.read_csv(demand_path)
        print(f"Loaded demand data from {demand_path}")
    except Exception as e:
        print(f"Warning: Could not load demand data: {e}")
        data['demand'] = pd.DataFrame(columns=['Demand'])
    
    try:
        # Try to load pumped storage timeseries data
        data['ps_timeseries'] = pd.read_csv(ps_data_path)
        print(f"Loaded pumped storage timeseries from {ps_data_path}")
    except Exception as e:
        print(f"Warning: Could not load pumped storage timeseries: {e}")
        data['ps_timeseries'] = pd.DataFrame()
    
    try:
        # Look for generator costs in both locations
        cost_path = os.path.join(coopt_path, f"marginal_costs_{day_type}.csv")
        if not os.path.exists(cost_path):
            cost_path = os.path.join(results_path, f"marginal_costs_{day_type}.csv")
            if not os.path.exists(cost_path):
                cost_path = os.path.join(project_root, f"marginal_costs_{day_type}.csv")
        
        data['costs'] = pd.read_csv(cost_path)
        print(f"Loaded generator costs from {cost_path}")
    except Exception as e:
        print(f"Warning: Could not load generator costs: {e}")
        data['costs'] = pd.DataFrame()
    
    try:
        data['generators'] = pd.read_csv(gen_data_path)
        print(f"Loaded generator data from {gen_data_path}")
    except Exception as e:
        print(f"Warning: Could not load generator data: {e}")
        data['generators'] = pd.DataFrame()
    
    try:
        data['renewables'] = pd.read_csv(re_data_path, delimiter=';')
        print(f"Loaded renewable generator data from {re_data_path}")
    except Exception as e:
        print(f"Warning: Could not load renewable generator data: {e}")
        data['renewables'] = pd.DataFrame()
    
    # Load additional data specific for renewables if available
    try:
        re_profiles_path = os.path.join(data_path, "DeterministicProfileRates.csv")
        data['renewable_profiles'] = pd.read_csv(re_profiles_path)
        print(f"Loaded renewable profiles from {re_profiles_path}")
    except Exception as e:
        print(f"Warning: Could not load renewable profiles: {e}")
        data['renewable_profiles'] = pd.DataFrame()
    
    # Try to get detailed generator summary with fuel types
    try:
        gen_summary_path = os.path.join(sequential_path, f"generators_summary_{day_type}.csv")
        if os.path.exists(gen_summary_path):
            data['generator_summary'] = pd.read_csv(gen_summary_path)
            print(f"Loaded generator summary from {gen_summary_path}")
        else:
            # Try looking in data directory
            gen_summary_path = os.path.join(data_path, f"unit_characteristics_{day_type}.csv")
            if os.path.exists(gen_summary_path):
                data['generator_summary'] = pd.read_csv(gen_summary_path)
                print(f"Loaded generator summary from {gen_summary_path}")
    except Exception as e:
        print(f"Warning: Could not load generator summary: {e}")
        data['generator_summary'] = pd.DataFrame()
    
    # Create an empty production key to avoid KeyError
    data['production'] = pd.DataFrame()
    
    return data

def create_fuel_mappings(data):
    """
    Creates mappings between generator names and fuel types.
    
    Returns a dictionary mapping generator names to fuel categories.
    """
    # Prepare generator data with fuel types
    dfG = data['generators'].rename(columns={
        "Generator": "Name",
        "FuelGenerator": "FuelType"
    })
    
    dfRE = data['renewables'].rename(columns={
        "GeneratorRE": "Name",
        "FuelGeneratorRE": "FuelType"
    })
    
    # Try to use generator summary if available as it has the most comprehensive fuel type info
    if not data['generator_summary'].empty and 'FuelType' in data['generator_summary'].columns:
        if 'Name' in data['generator_summary'].columns:
            dfSum = data['generator_summary'][['Name', 'FuelType']]
        else:
            # If Name column doesn't exist, try to create it from the index
            dfSum = data['generator_summary'].copy()
            dfSum['Name'] = dfSum.index
            dfSum = dfSum[['Name', 'FuelType']]
            
        # Clean up any index or numbering issues
        dfSum = dfSum.reset_index(drop=True)
    else:
        dfSum = pd.DataFrame(columns=['Name', 'FuelType'])
    
    # Combine all sources of fuel type information
    dfG = dfG[["Name", "FuelType"]].drop_duplicates()
    dfRE = dfRE[["Name", "FuelType"]].drop_duplicates()
    dfAll = pd.concat([dfG, dfRE, dfSum], ignore_index=True).drop_duplicates()
    
    # Map fuel types to categories - expand with all known fuel types
    fuel_map = {
        # Nuclear
        "NUCLEAR": "Nuclear",
        
        # Fossil fuels
        "COAL": "Coal",
        "NAT_GAS": "Gas",
        "GAS": "Gas",
        "OTHER_GAS": "Gas",
        "BLAST_GAS": "Gas",
        "GASOIL": "Oil",
        "OIL": "Oil", 
        "FUELOIL_1": "Oil",
        "LIGNITE": "Coal",
        
        # Renewables
        "SOLAR": "Solar",
        "WIND": "Wind",
        "WINDOFF": "Wind",
        "WINDON": "Wind",
        "BIOMASS": "Biomass",
        
        # Hydro
        "HYDRAULIC_PS": "PumpedStorage",
        "HYDRAULIC": "Hydro",
        "HYDRAULIC_RoR": "Hydro",
        
        # Other
        "WASTE": "Waste",
        "FREE_FUEL": "Other"
    }
    
    # Use fuel mapping to categorize generators
    dfAll["FuelCategory"] = dfAll["FuelType"].map(fuel_map).fillna("Other")
    name_to_cat = dict(zip(dfAll["Name"], dfAll["FuelCategory"]))
    
    # Add manual mapping for pumped storage units based on naming patterns
    production_df = data['production']
    if not production_df.empty:
        # Look for patterns in column names
        ps_patterns = ["_Gen", "_Pump", "PS_", "PLATETAILLE", "COO_A", "Pumped", "Storage"]
        for col in production_df.columns:
            if any(ps_pattern in col for ps_pattern in ps_patterns):
                name_to_cat[col] = "PumpedStorage"
        
        # Look for any wind or solar patterns
        wind_patterns = ["WIND", "Wind"]
        solar_patterns = ["SOLAR", "Solar", "PV"]
        for col in production_df.columns:
            if any(pattern in col.upper() for pattern in wind_patterns):
                name_to_cat[col] = "Wind"
            elif any(pattern in col.upper() for pattern in solar_patterns):
                name_to_cat[col] = "Solar"
    
    return name_to_cat

def plot_generation_vs_load(day_type, data, name_to_cat, plots_path, results_path):
    """
    Plots generation vs load, both by unit and by fuel type.
    """
    production_df = data['production']
    demand_df = data['demand']
    
    if production_df.empty or demand_df.empty:
        print("Error: Missing production or demand data, skipping generation plots")
        return
    
    # Check and integrate renewable energy data if available 
    if 'energy_generation' in data and not data['energy_generation'].empty:
        print("Processing renewable energy data for inclusion in generation plots...")
        # Find renewable generators based on fuel mapping
        renewable_gens = [name for name, cat in name_to_cat.items() 
                         if cat in ['Wind', 'Solar', 'Biomass', 'Hydro'] 
                         and name not in production_df.columns]
        
        # If we found renewable generators that aren't in the production data, process them
        if renewable_gens:
            print(f"Found {len(renewable_gens)} renewable generators to add to the unit-level plot")
            
            # Check if energy_generation has renewable columns we can use
            energy_gen_df = data['energy_generation']
            for gen in renewable_gens:
                if gen in energy_gen_df.columns:
                    print(f"Adding renewable generator {gen} to production data")
                    production_df[gen] = energy_gen_df[gen]
            
            # Also check renewable_profiles if available
            if 'renewable_profiles' in data and not data['renewable_profiles'].empty:
                re_profiles = data['renewable_profiles']
                if 'GeneratorRE' in re_profiles.columns and 'Value' in re_profiles.columns and 'Hour' in re_profiles.columns:
                    for gen in renewable_gens:
                        gen_data = re_profiles[re_profiles['GeneratorRE'] == gen]
                        if not gen_data.empty and gen not in production_df.columns:
                            # Create time-series data from renewable profiles
                            for hour in gen_data['Hour'].unique():
                                value = gen_data[gen_data['Hour'] == hour]['Value'].values[0]
                                # Expand hourly values to 15-min intervals (same value for all 4 periods)
                                for i in range(4):
                                    idx = hour * 4 + i
                                    if idx < len(production_df):
                                        production_df.at[idx, gen] = value
                            print(f"Added renewable generator {gen} from profiles")
    
    # Ensure production data is numeric
    for col in production_df.columns:
        production_df[col] = pd.to_numeric(production_df[col], errors='coerce')
    
    # Ensure demand data is numeric
    if "Demand" in demand_df.columns:
        demand_df["Demand"] = pd.to_numeric(demand_df["Demand"], errors='coerce')
    
    # -----------------------------------------------------
    # PLOT 1: Each column individually (stacked bar)
    # -----------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Color generators by fuel type for more informative plot
    colors_by_gen = {}
    fuel_colors = {
        'Nuclear': '#7b6888', 
        'Gas': '#6b486b',
        'Coal': '#a05d56',
        'Oil': '#d0743c',
        'Wind': '#98abc5',
        'Solar': '#8a89a6',
        'PumpedStorage': '#ff8c00',
        'Waste': '#b15928',
        'Biomass': '#8c564b',
        'Hydro': '#17becf',
        'Other': '#cab2d6'
    }
    
    for gen in production_df.columns:
        fuel = name_to_cat.get(gen, 'Unknown')
        colors_by_gen[gen] = fuel_colors.get(fuel, '#cccccc')
    
    # Plot with color coding by fuel type
    production_df.plot(kind='bar', stacked=True, ax=ax1, alpha=0.8, width=0.8, 
                      color=[colors_by_gen.get(col, '#cccccc') for col in production_df.columns])
    
    if "Demand" in demand_df.columns:
        ax1.plot(demand_df.index, demand_df["Demand"].values,
                 marker='o', linestyle='-', color='black',
                 linewidth=2, label="Load")
    
    ax1.set_xlabel("Interval (15-min)")
    ax1.set_ylabel("MW")
    ax1.set_title(f"{day_type} - Generation (by Unit) vs. Load - Sequential")
    
    # Create a legend that groups generators by fuel type
    handles, labels = ax1.get_legend_handles_labels()
    by_fuel = {}
    for i, label in enumerate(labels):
        if label == "Load":  # Keep Load separate
            continue
        fuel = name_to_cat.get(label, "Unknown")
        if fuel not in by_fuel:
            by_fuel[fuel] = []
        by_fuel[fuel].append(i)
    
    # Create fuel type headers in legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for fuel in sorted(by_fuel.keys()):
        # Add a header for the fuel type
        legend_elements.append(Line2D([0], [0], color='w', marker='', linestyle='', label=f"--- {fuel} ---"))
        # Add generators of this fuel type
        for i in by_fuel[fuel]:
            legend_elements.append(handles[i])
    
    # Add Load at the end
    for i, label in enumerate(labels):
        if label == "Load":
            legend_elements.append(handles[i])
    
    ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure to plots directory
    plot_path = os.path.join(plots_path, f"sequential_generation_by_unit_{day_type}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig1)
    
    # -----------------------------------------------------
    # PLOT 2: Group by Fuel Category (stacked bar)
    # -----------------------------------------------------
    grouped_sums = {}
    for col in production_df.columns:
        gen_name = col
        cat = name_to_cat.get(gen_name, "Unknown")
        if cat not in grouped_sums:
            grouped_sums[cat] = production_df[col].copy()
        else:
            grouped_sums[cat] += production_df[col]
    
    grouped_df = pd.DataFrame(grouped_sums, index=production_df.index)
    
    # Ensure all columns in grouped_df are numeric
    for col in grouped_df.columns:
        grouped_df[col] = pd.to_numeric(grouped_df[col], errors='coerce')
    
    # Save grouped data to results directory
    sequential_path = os.path.join(results_path, "sequential")
    grouped_output_path = os.path.join(sequential_path, f"grouped_generation_{day_type}.csv")
    grouped_df.to_csv(grouped_output_path)
    print(f"Grouped generation data saved to {grouped_output_path}")
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Use a more appealing color palette for fuel types
    colors = {
        'Nuclear': '#7b6888', 
        'Gas': '#6b486b',
        'Coal': '#a05d56',
        'Oil': '#d0743c',
        'Wind': '#98abc5',
        'Solar': '#8a89a6',
        'PumpedStorage': '#ff8c00',
        'Waste': '#b15928',
        'Other': '#cab2d6'
    }
    
    # Reorder columns to have renewables, then conventional, then pumped storage
    ordered_cols = []
    if 'Solar' in grouped_df.columns: ordered_cols.append('Solar')
    if 'Wind' in grouped_df.columns: ordered_cols.append('Wind')
    if 'Nuclear' in grouped_df.columns: ordered_cols.append('Nuclear')
    if 'Coal' in grouped_df.columns: ordered_cols.append('Coal')
    if 'Gas' in grouped_df.columns: ordered_cols.append('Gas')
    if 'Oil' in grouped_df.columns: ordered_cols.append('Oil')
    if 'Waste' in grouped_df.columns: ordered_cols.append('Waste')
    if 'PumpedStorage' in grouped_df.columns: ordered_cols.append('PumpedStorage')
    
    # Add any remaining columns
    for col in grouped_df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    
    # Select only columns that exist in the dataframe
    ordered_cols = [col for col in ordered_cols if col in grouped_df.columns]
    
    # Instead of plotting column by column, plot the entire DataFrame at once
    if ordered_cols:  # Check if we have any columns to plot
        # Create a new DataFrame with ordered columns
        plot_df = grouped_df[ordered_cols].copy()
        
        # Use custom colors for the columns where available
        column_colors = [colors.get(col, None) for col in ordered_cols]
        
        # Plot the stacked bar chart
        plot_df.plot(
            kind='bar', 
            stacked=True, 
            ax=ax2, 
            color=column_colors,
            alpha=0.8, 
            width=0.8
        )
    
    # Add demand line
    if "Demand" in demand_df.columns:
        ax2.plot(demand_df.index, demand_df["Demand"].values,
                marker='o', linestyle='-', color='black',
                linewidth=2, label="Load")
    
    ax2.set_xlabel("Interval (15-min)")
    ax2.set_ylabel("MW")
    ax2.set_title(f"{day_type} - Generation (by Fuel Type) vs. Load - Sequential")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Fuel Types")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure to plots directory
    plot_path = os.path.join(plots_path, f"sequential_generation_by_fuel_{day_type}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig2)
    
    # -----------------------------------------------------
    # PLOT 3: Hourly generation with area plot (smoother visual)
    # -----------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    # Use same ordered columns for area plot
    if ordered_cols:  # Check if we have any columns to plot
        try:
            plot_df = grouped_df[ordered_cols].copy()
            plot_df.plot(
                kind='area', 
                stacked=True, 
                ax=ax3, 
                color=[colors.get(col, None) for col in ordered_cols],
                alpha=0.7, 
                linewidth=0
            )
        except Exception as e:
            print(f"Warning: Could not create area plot: {e}")
    
    # Add demand line
    if "Demand" in demand_df.columns:
        ax3.plot(demand_df.index, demand_df["Demand"].values,
                marker='o', linestyle='-', color='black',
                linewidth=2, label="Load")
    
    ax3.set_xlabel("Interval (15-min)")
    ax3.set_ylabel("MW")
    ax3.set_title(f"{day_type} - Generation Profile (by Fuel Type) - Sequential")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Fuel Types")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure to plots directory
    plot_path = os.path.join(plots_path, f"sequential_generation_area_{day_type}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig3)

def process_ps_timeseries(day_type, data, plots_path, results_path):
    """
    Processes the pumped storage timeseries data from the ps_timeseries file.
    This function handles the specific format of the ps_timeseries files in the sequential model results.
    """
    ps_timeseries = data.get('ps_timeseries', pd.DataFrame())
    
    if ps_timeseries.empty:
        print("Error: Missing pumped storage timeseries data, skipping pumped storage plots")
        return
    
    # Check if we have the TimeStep column
    if 'TimeStep' not in ps_timeseries.columns:
        print("Error: PS timeseries file does not have a TimeStep column")
        return
    
    # Extract PS unit names from column headers
    ps_columns = ps_timeseries.columns.tolist()
    ps_columns.remove('TimeStep')  # Remove the TimeStep column
    
    # Group columns by PS unit
    ps_units = {}
    for col in ps_columns:
        # Skip 'net' columns - we'll calculate these
        if col.endswith('_net'):
            continue
            
        # Extract the unit name by removing _gen or _pump suffix
        if col.endswith('_gen'):
            unit_name = col[:-4]  # Remove _gen
            if unit_name not in ps_units:
                ps_units[unit_name] = {'gen': col}
            else:
                ps_units[unit_name]['gen'] = col
        elif col.endswith('_pump'):
            unit_name = col[:-5]  # Remove _pump
            if unit_name not in ps_units:
                ps_units[unit_name] = {'pump': col}
            else:
                ps_units[unit_name]['pump'] = col
    
    if not ps_units:
        print("Error: Could not identify any pumped storage units in the timeseries data")
        return
    
    print(f"Found {len(ps_units)} pumped storage units: {', '.join(ps_units.keys())}")
    
    # Create a combined figure for all PS units
    fig_combined, axes_combined = plt.subplots(len(ps_units), 1, figsize=(14, 4 * len(ps_units)), sharex=True)
    if len(ps_units) == 1:
        axes_combined = [axes_combined]  # Make it iterable if there's only one unit
    
    # Track PS statistics for reporting
    ps_stats = []
    
    # Create individual plots for each PS unit
    for i, (unit_name, columns) in enumerate(sorted(ps_units.items())):
        gen_col = columns.get('gen')
        pump_col = columns.get('pump')
        
        # Skip if we don't have both generation and pumping data
        if not gen_col or not pump_col:
            print(f"Warning: Missing generation or pumping column for unit {unit_name}")
            continue
        
        # Extract generation and pumping data
        generation = ps_timeseries[gen_col]
        pumping = ps_timeseries[pump_col]
        
        # Calculate net power and state of charge
        # Note: pumping values are already negative in the data
        net_power = generation + pumping  # This works because pumping is already negative
        
        # Calculate state of charge by integrating net power over time (15-min intervals = 0.25 hours)
        # Assuming we start at 50% of the max observed SoC for visualization purposes
        soc = net_power.cumsum() * 0.25  # Integrate power (MW) to energy (MWh)
        max_discharge = soc.min()
        if max_discharge < 0:
            # Shift the SoC curve so it's always positive
            soc = soc - max_discharge + 10  # Add 10 MWh buffer
        
        # Calculate statistics
        max_gen = generation.max()
        max_pump = abs(pumping.min())  # Convert to positive for reporting
        total_gen = generation.sum() * 0.25  # Convert from MW to MWh (15-min intervals)
        total_pump = abs(pumping.sum()) * 0.25  # Convert from MW to MWh and make positive
        max_soc = soc.max()
        min_soc = soc.min()
        reservoir_capacity = max_soc - min_soc
        
        # Calculate utilization percentages
        if reservoir_capacity > 0:
            gen_utilization = (total_gen / reservoir_capacity) * 100
            pump_utilization = (total_pump / reservoir_capacity) * 100
            cycles = min(total_gen, total_pump) / reservoir_capacity
        else:
            gen_utilization = 0
            pump_utilization = 0
            cycles = 0
        
        # Store statistics
        ps_stats.append({
            'Unit': unit_name,
            'MaxGeneration_MW': max_gen,
            'MaxPumping_MW': max_pump,
            'TotalGeneration_MWh': total_gen,
            'TotalPumping_MWh': total_pump,
            'ReservoirCapacity_MWh': reservoir_capacity,
            'GenerationUtilization_pct': gen_utilization,
            'PumpingUtilization_pct': pump_utilization,
            'Cycles': cycles
        })
        
        # Create individual figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax2 = ax1.twinx()
        
        # Plot generation and pumping as bars
        x = range(len(ps_timeseries))
        ax1.bar(x, generation.values, color='green', alpha=0.7, label='Generation')
        ax1.bar(x, pumping.values, color='red', alpha=0.7, label='Pumping')
        
        # Plot state of charge as a line
        ax2.plot(x, soc.values, color='blue', marker=None, linestyle='-', linewidth=2, label='State of Charge')
        
        # Add zero line and grid
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(True, alpha=0.3)
        
        # Add labels and title
        ax1.set_xlabel("Time Period (15-min)")
        ax1.set_ylabel("Power (MW)")
        ax2.set_ylabel("State of Charge (MWh)", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.title(f"{day_type} - Pumped Storage Operations: {unit_name} - Sequential")
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Save individual figure
        plt.tight_layout()
        plot_path = os.path.join(plots_path, f"sequential_ps_timeseries_{unit_name}_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
        
        # Plot on combined figure
        ax_combined = axes_combined[i]
        
        # Bar chart for generation and pumping
        ax_combined.bar(x, generation.values, color='green', alpha=0.7, label='Generation')
        ax_combined.bar(x, pumping.values, color='red', alpha=0.7, label='Pumping')
        
        # Add second y-axis for state of charge
        ax_soc = ax_combined.twinx()
        ax_soc.plot(x, soc.values, color='blue', linewidth=2, label='SoC')
        ax_soc.set_ylabel('State of Charge (MWh)', color='blue')
        ax_soc.tick_params(axis='y', labelcolor='blue')
        
        # Add zero line and formatting
        ax_combined.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax_combined.grid(True, alpha=0.3)
        ax_combined.set_ylabel("Power (MW)")
        ax_combined.set_title(f"PS Unit: {unit_name}")
        
        # Add legends for both axes
        ax_combined.legend(loc='upper left')
        ax_soc.legend(loc='upper right')
    
    # Finalize combined figure
    fig_combined.suptitle(f"{day_type} - All Pumped Storage Operations - Sequential", fontsize=16)
    axes_combined[-1].set_xlabel("Time Period (15-min)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    
    # Save combined figure
    plot_path = os.path.join(plots_path, f"sequential_ps_operations_all_{day_type}.png")
    plt.savefig(plot_path)
    print(f"Combined plot saved to {plot_path}")
    plt.close(fig_combined)
    
    # Create summary statistics DataFrame
    stats_df = pd.DataFrame(ps_stats)
    
    # Save to CSV
    sequential_path = os.path.join(results_path, "sequential")
    os.makedirs(sequential_path, exist_ok=True)
    stats_output_path = os.path.join(sequential_path, f"ps_utilization_stats_{day_type}.csv")
    stats_df.to_csv(stats_output_path, index=False)
    print(f"PS utilization statistics saved to {stats_output_path}")
    
    # Print summary
    print("\nPumped Storage Utilization Summary:")
    print(stats_df.to_string(index=False))
    
    # Create a bar chart for utilization percentages
    if not stats_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(stats_df))
        width = 0.35
        
        ax.bar(x - width/2, stats_df['GenerationUtilization_pct'], width, label='Generation', color='green', alpha=0.7)
        ax.bar(x + width/2, stats_df['PumpingUtilization_pct'], width, label='Pumping', color='red', alpha=0.7)
        
        ax.set_xlabel('Pumped Storage Unit')
        ax.set_ylabel('Utilization (% of Reservoir Capacity)')
        ax.set_title(f"{day_type} - Pumped Storage Utilization - Sequential")
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['Unit'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_path, f"sequential_ps_utilization_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
        
        # Create a chart for cycles
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(stats_df['Unit'], stats_df['Cycles'], color='purple', alpha=0.7)
        ax.set_xlabel('Pumped Storage Unit')
        ax.set_ylabel('Number of Cycles')
        ax.set_title(f"{day_type} - Pumped Storage Cycles - Sequential")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_path, f"sequential_ps_cycles_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
    
    return stats_df

def process_pumped_storage_data(day_type, data, plots_path, results_path):
    """
    Processes and plots pumped storage data from the ps_timeseries file.
    Specifically handles potential negative values for pumped storage operations.
    """
    ps_timeseries = data.get('ps_timeseries', pd.DataFrame())
    
    if ps_timeseries.empty:
        print("Error: Missing pumped storage timeseries data, skipping pumped storage plots")
        return
    
    # Check if the timeseries data has the expected structure
    required_cols = ['Unit', 'TimeStep', 'Generation', 'Pumping', 'StateOfCharge']
    missing_cols = [col for col in required_cols if col not in ps_timeseries.columns]
    
    if missing_cols:
        print(f"Warning: PS timeseries is missing columns: {missing_cols}")
        # Try to work with what we have
        if 'Unit' not in ps_timeseries.columns or 'TimeStep' not in ps_timeseries.columns:
            print("Error: Cannot process pumped storage data without Unit and TimeStep columns")
            return
    
    # Get unique pumped storage units
    ps_units = ps_timeseries['Unit'].unique()
    
    # Create a combined figure for all PS units
    fig_combined, axes_combined = plt.subplots(len(ps_units), 1, figsize=(12, 4 * len(ps_units)), sharex=True)
    if len(ps_units) == 1:
        axes_combined = [axes_combined]  # Make it iterable if there's only one unit
    
    # Create individual plots for each PS unit
    for i, unit in enumerate(sorted(ps_units)):
        # Filter data for this unit
        unit_data = ps_timeseries[ps_timeseries['Unit'] == unit]
        
        # Pivot data to get a clean time series
        if 'StateOfCharge' in unit_data.columns:
            soc_df = unit_data.pivot_table(index='TimeStep', values='StateOfCharge')
            has_soc = True
        else:
            has_soc = False
        
        # Create generation and pumping series
        if 'Generation' in unit_data.columns:
            gen_df = unit_data.pivot_table(index='TimeStep', values='Generation')
            has_gen = True
        else:
            has_gen = False
            
        if 'Pumping' in unit_data.columns:
            pump_df = unit_data.pivot_table(index='TimeStep', values='Pumping')
            # Check if pumping is already represented as negative
            if pump_df.min().min() >= 0:
                # If pumping is positive, convert to negative for visualization
                pump_df = -pump_df
            has_pump = True
        else:
            has_pump = False
        
        # Create individual figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx() if has_soc else None
        
        # Plot generation and pumping on left y-axis
        if has_gen:
            ax1.bar(gen_df.index, gen_df['Generation'].values, color='green', alpha=0.7, label='Generation')
        
        if has_pump:
            ax1.bar(pump_df.index, pump_df['Pumping'].values, color='red', alpha=0.7, label='Pumping')
        
        # Plot state of charge on right y-axis if available
        if has_soc:
            ax2.plot(soc_df.index, soc_df['StateOfCharge'].values, color='blue', marker='o', linestyle='-', label='State of Charge')
            ax2.set_ylabel('State of Charge (MWh)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
        
        # Add zero line and grid for generation/pumping
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(True, alpha=0.3)
        
        # Add labels and title
        ax1.set_xlabel("Time Period (15-min)")
        ax1.set_ylabel("Power (MW)")
        ax2.set_ylabel("State of Charge (MWh)", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.title(f"{day_type} - Pumped Storage Operations: {unit} - Sequential")
        
        # Create combined legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        if has_soc:
            handles2, labels2 = ax2.get_legend_handles_labels()
            fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.0, 0.9))
        else:
            ax1.legend()
        
        # Save individual figure
        plt.tight_layout()
        plot_path = os.path.join(plots_path, f"sequential_ps_timeseries_{unit}_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
        
        # Plot on combined figure
        ax_combined = axes_combined[i]
        
        if has_gen:
            ax_combined.bar(gen_df.index, gen_df['Generation'].values, color='green', alpha=0.7, label='Generation')
        
        if has_pump:
            ax_combined.bar(pump_df.index, pump_df['Pumping'].values, color='red', alpha=0.7, label='Pumping')
        
        # Add a second y-axis for state of charge if available
        if has_soc:
            ax_soc = ax_combined.twinx()
            ax_soc.plot(soc_df.index, soc_df['StateOfCharge'].values, color='blue', marker='.', linestyle='-', label='SoC')
            ax_soc.set_ylabel('State of Charge (MWh)', color='blue')
            ax_soc.tick_params(axis='y', labelcolor='blue')
        
        ax_combined.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax_combined.grid(True, alpha=0.3)
        ax_combined.set_ylabel("Power (MW)")
        ax_combined.set_title(f"PS Unit: {unit}")
        ax_combined.legend(loc='upper left')
        
        # Add SoC legend entry if applicable
        if has_soc:
            ax_soc.legend(loc='upper right')
    
    # Finalize combined figure
    fig_combined.suptitle(f"{day_type} - All Pumped Storage Operations - Sequential", fontsize=16)
    axes_combined[-1].set_xlabel("Time Period (15-min)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    
    # Save combined figure
    plot_path = os.path.join(plots_path, f"sequential_ps_operations_all_{day_type}.png")
    plt.savefig(plot_path)
    print(f"Combined plot saved to {plot_path}")
    plt.close(fig)
    
    # Calculate and return pumped storage utilization statistics
    return calculate_ps_utilization(day_type, ps_timeseries, plots_path, results_path)

def calculate_ps_utilization(day_type, ps_timeseries, plots_path, results_path):
    """
    Calculates the pumped storage utilization statistics from the timeseries data.
    """
    # Check if we have the necessary columns
    if any(col not in ps_timeseries.columns for col in ['Unit', 'TimeStep', 'Generation', 'Pumping', 'StateOfCharge']):
        print("Warning: Missing required columns for pumped storage utilization calculation")
        return pd.DataFrame()
    
    # Group by unit to calculate summary statistics
    ps_stats = []
    for unit in ps_timeseries['Unit'].unique():
        unit_data = ps_timeseries[ps_timeseries['Unit'] == unit]
        
        # Get max SoC as reservoir capacity
        max_soc = unit_data['StateOfCharge'].max()
        
        # Calculate total energy generated and pumped (MWh)
        total_gen = unit_data['Generation'].sum() / 4  # 15-min intervals
        total_pump = unit_data['Pumping'].sum() / 4  # 15-min intervals
        
        # Find max power values
        max_gen = unit_data['Generation'].max()
        max_pump = unit_data['Pumping'].max() if 'Pumping' in unit_data.columns else 0
        
        # Calculate utilization percentages
        gen_utilization = (total_gen / max_soc) * 100 if max_soc > 0 else 0
        pump_utilization = (total_pump / max_soc) * 100 if max_soc > 0 else 0
        
        # Calculate number of cycles
        cycles = min(total_gen, total_pump) / max_soc if max_soc > 0 else 0
        
        # Store the results
        ps_stats.append({
            'Unit': unit,
            'ReservoirCapacity_MWh': max_soc,
            'MaxGeneration_MW': max_gen,
            'MaxPumping_MW': max_pump,
            'EnergyGenerated_MWh': total_gen,
            'EnergyPumped_MWh': total_pump,
            'GenerationUtilization_pct': gen_utilization,
            'PumpingUtilization_pct': pump_utilization,
            'Cycles': cycles
        })
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(ps_stats)
    
    # Save to CSV
    sequential_path = os.path.join(results_path, "sequential")
    os.makedirs(sequential_path, exist_ok=True)
    stats_output_path = os.path.join(sequential_path, f"ps_utilization_stats_{day_type}.csv")
    stats_df.to_csv(stats_output_path, index=False)
    print(f"PS utilization statistics saved to {stats_output_path}")
    
    # Print summary
    print("\nPumped Storage Utilization Summary:")
    print(stats_df.to_string(index=False))
    
    # Create a bar chart for utilization percentages
    if not stats_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(stats_df))
        width = 0.35
        
        ax.bar(x - width/2, stats_df['GenerationUtilization_pct'], width, label='Generation', color='green', alpha=0.7)
        ax.bar(x + width/2, stats_df['PumpingUtilization_pct'], width, label='Pumping', color='red', alpha=0.7)
        
        ax.set_xlabel('Pumped Storage Unit')
        ax.set_ylabel('Utilization (% of Reservoir Capacity)')
        ax.set_title(f"{day_type} - Pumped Storage Utilization - Sequential")
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['Unit'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_path, f"sequential_ps_utilization_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
        
        # Create a chart for cycles
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(stats_df['Unit'], stats_df['Cycles'], color='purple', alpha=0.7)
        ax.set_xlabel('Pumped Storage Unit')
        ax.set_ylabel('Number of Cycles')
        ax.set_title(f"{day_type} - Pumped Storage Cycles - Sequential")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_path, f"sequential_ps_cycles_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
    
    return stats_df

def plot_reserve_provision(day_type, data, name_to_cat, plots_path, results_path):
    """
    Plots the reserve provision by unit and reserve type.
    """
    reserve_df = data['reserve']
    
    if reserve_df.empty:
        print("Error: Missing reserve data, skipping reserve plots")
        return
    
    # Get unique reserve products
    reserve_products = reserve_df['ReserveProduct'].unique()
    
    # Create a separate plot for each reserve product
    for product in reserve_products:
        # Filter data for this product
        product_df = reserve_df[reserve_df['ReserveProduct'] == product]
        
        # Create a pivot table with generators as columns and timesteps as rows
        pivot_df = product_df.pivot_table(index='TimeStep', columns='Generator', values='s_res', fill_value=0)
        
        # Plot reserve provision by unit
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_df.plot(kind='bar', stacked=True, ax=ax)
        ax.set_xlabel("Time Period (15-min)")
        ax.set_ylabel("Reserve Provision (MW)")
        ax.set_title(f"{day_type} - Reserve Provision: Product {product} - Sequential")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(plots_path, f"sequential_reserve_product_{product}_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
    
    # Create a plot of total reserve provision by fuel type
    if not reserve_df.empty:
        # Group reserves by time step and fuel type
        reserve_by_fuel = {}
        for _, row in reserve_df.iterrows():
            gen_name = row['Generator']
            time_step = row['TimeStep']
            reserve_val = row['s_res']
            
            # Get fuel category for this generator
            fuel_cat = name_to_cat.get(gen_name, "Unknown")
            
            # Initialize nested dictionary if needed
            if time_step not in reserve_by_fuel:
                reserve_by_fuel[time_step] = {}
            
            if fuel_cat not in reserve_by_fuel[time_step]:
                reserve_by_fuel[time_step][fuel_cat] = 0
            
            # Add reserve value
            reserve_by_fuel[time_step][fuel_cat] += reserve_val
        
        # Convert to DataFrame
        fuel_rows = []
        for time_step, fuel_dict in reserve_by_fuel.items():
            for fuel_cat, value in fuel_dict.items():
                fuel_rows.append({
                    'TimeStep': time_step,
                    'FuelType': fuel_cat,
                    'ReserveValue': value
                })
        
        fuel_df = pd.DataFrame(fuel_rows)
        
        # Pivot to get fuel types as columns
        pivot_fuel_df = fuel_df.pivot_table(index='TimeStep', columns='FuelType', values='ReserveValue', fill_value=0)
        
        # Plot reserve provision by fuel type
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_fuel_df.plot(kind='bar', stacked=True, ax=ax)
        ax.set_xlabel("Time Period (15-min)")
        ax.set_ylabel("Reserve Provision (MW)")
        ax.set_title(f"{day_type} - Total Reserve Provision by Fuel Type - Sequential")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(plots_path, f"sequential_reserve_by_fuel_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)

def plot_cost_breakdown(day_type, data, name_to_cat, plots_path, results_path):
    """
    Plots the cost breakdown by generator and fuel type.
    """
    cost_df = data['costs']
    
    if cost_df.empty:
        print("Error: Missing cost data, skipping cost plots")
        return
    
    # Ensure the cost dataframe has the expected columns
    required_cols = ['Generator', 'MarginalCost', 'StartupCost', 'NoLoadCost', 'TotalCost']
    missing_cols = [col for col in required_cols if col not in cost_df.columns]
    
    if missing_cols:
        print(f"Warning: Cost dataframe is missing columns: {missing_cols}")
        # Try to work with what we have
        if 'Generator' not in cost_df.columns or 'TotalCost' not in cost_df.columns:
            print("Error: Cannot plot costs without Generator and TotalCost columns")
            return
    
    # Create a copy to avoid modifying the original
    plot_df = cost_df.copy()
    
    # Add fuel type information
    plot_df['FuelType'] = plot_df['Generator'].map(name_to_cat).fillna('Unknown')
    
    # Create pie chart of total costs by fuel type
    fuel_costs = plot_df.groupby('FuelType')['TotalCost'].sum()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        fuel_costs, 
        labels=fuel_costs.index, 
        autopct='%1.1f%%',
        textprops={'fontsize': 10}
    )
    ax.set_title(f"{day_type} - Total Cost by Fuel Type - Sequential")
    plt.tight_layout()
    
    plot_path = os.path.join(plots_path, f"sequential_cost_by_fuel_{day_type}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)
    
    # Create bar chart of top generators by cost
    top_n = 20  # Show top 20 generators
    top_generators = plot_df.nlargest(top_n, 'TotalCost')
    
    # If we have the cost component columns, create a stacked bar for cost components
    if all(col in top_generators.columns for col in ['MarginalCost', 'StartupCost', 'NoLoadCost']):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create stacked bars with cost components
        bottom = np.zeros(len(top_generators))
        
        for cost_type, color in [('MarginalCost', 'blue'), ('StartupCost', 'green'), ('NoLoadCost', 'red')]:
            if cost_type in top_generators.columns:
                ax.bar(
                    top_generators['Generator'], 
                    top_generators[cost_type], 
                    bottom=bottom, 
                    label=cost_type,
                    color=color,
                    alpha=0.7
                )
                bottom += top_generators[cost_type].values
        
        ax.set_xlabel("Generator")
        ax.set_ylabel("Cost (€)")
        ax.set_title(f"{day_type} - Top {top_n} Generators by Cost Components - Sequential")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = os.path.join(plots_path, f"sequential_top_generator_costs_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
    else:
        # Simple bar chart without cost components
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(top_generators['Generator'], top_generators['TotalCost'])
        ax.set_xlabel("Generator")
        ax.set_ylabel("Total Cost (€)")
        ax.set_title(f"{day_type} - Top {top_n} Generators by Total Cost - Sequential")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = os.path.join(plots_path, f"sequential_top_generator_costs_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)

def compare_sequential_vs_coopt(day_type, results_path, plots_path):
    """
    Compares results between sequential and co-optimization models.
    Creates visualizations that highlight the differences in costs, generation mix, and commitment decisions.
    
    Parameters:
    -----------
    day_type : str
        The day type to analyze (e.g., "AutumnWD")
    results_path : str
        Path to the results directory
    plots_path : str
        Path to save the plots
    """
    print(f"\nComparing Sequential vs. Co-optimization for {day_type}")
    
    # Define paths
    sequential_path = os.path.join(results_path, "sequential")
    coopt_path = os.path.join(results_path, "coopt")
    
    # Try to load objective values
    seq_obj = None
    coopt_obj = None
    
    # Sequential objective
    seq_obj_path = os.path.join(sequential_path, f"energy_objective_{day_type}.txt")
    if os.path.exists(seq_obj_path):
        try:
            with open(seq_obj_path, 'r') as f:
                seq_obj = float(f.read().strip())
            print(f"Sequential objective: {seq_obj:,.2f} EUR")
        except Exception as e:
            print(f"Error reading sequential objective: {e}")
    
    # Get reserve cost from reserve_cost file
    reserve_cost_path = os.path.join(sequential_path, f"reserve_cost_{day_type}.csv")
    reserve_obj = None
    if os.path.exists(reserve_cost_path):
        try:
            reserve_df = pd.read_csv(reserve_cost_path)
            if 'Total_Cost' in reserve_df.columns:
                reserve_obj = float(reserve_df['Total_Cost'].iloc[0])
                print(f"Reserve objective: {reserve_obj:,.2f} EUR")
        except Exception as e:
            print(f"Error reading reserve objective: {e}")
    
    # Co-optimization objective
    coopt_obj_path = os.path.join(coopt_path, f"objective_{day_type}.txt")
    if os.path.exists(coopt_obj_path):
        try:
            with open(coopt_obj_path, 'r') as f:
                coopt_obj = float(f.read().strip())
            print(f"Co-optimization objective: {coopt_obj:,.2f} EUR")
        except Exception as e:
            print(f"Error reading co-optimization objective: {e}")
    
    # Calculate total sequential cost
    if seq_obj is not None and reserve_obj is not None:
        total_seq = seq_obj + reserve_obj
        print(f"Total sequential cost: {total_seq:,.2f} EUR")
    else:
        total_seq = None
        print("Could not calculate total sequential cost due to missing data")
    
    # Calculate cost difference
    if total_seq is not None and coopt_obj is not None:
        diff = total_seq - coopt_obj
        diff_pct = (diff / coopt_obj) * 100
        print(f"Sequential - Co-optimization difference: {diff:,.2f} EUR ({diff_pct:.2f}%)")
    
    # Create bar chart comparing costs
    if total_seq is not None or coopt_obj is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Data for the bar chart
        costs = []
        if reserve_obj is not None:
            costs.append(('Sequential (Reserve)', reserve_obj))
        if seq_obj is not None:
            costs.append(('Sequential (Energy)', seq_obj))
        if total_seq is not None:
            costs.append(('Sequential (Total)', total_seq))
        if coopt_obj is not None:
            costs.append(('Co-optimization', coopt_obj))
        
        # Create bar chart
        labels = [item[0] for item in costs]
        values = [item[1] for item in costs]
        
        # Define colors
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0']
        
        bars = ax.bar(labels, values, color=colors)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax.set_ylabel('Cost (EUR)')
        ax.set_title(f'Cost Comparison: Sequential vs. Co-optimization - {day_type}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add text annotation for difference percentage if available
        if total_seq is not None and coopt_obj is not None:
            ax.text(0.02, 0.95, 
                   f"Sequential approach costs {diff_pct:.2f}% more than co-optimization",
                   transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(plots_path, f"cost_comparison_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Cost comparison plot saved to {plot_path}")
        plt.close(fig)
    
    # Compare commitment decisions if available
    try:
        # Load sequential commitment decisions
        seq_commit_path = os.path.join(sequential_path, f"commitment_{day_type}.csv")
        if not os.path.exists(seq_commit_path):
            print(f"Sequential commitment file not found: {seq_commit_path}")
            return
        
        seq_commit = pd.read_csv(seq_commit_path)
        
        # Load co-optimization commitment decisions
        coopt_commit_path = os.path.join(coopt_path, f"commitment_{day_type}.csv")
        if not os.path.exists(coopt_commit_path):
            print(f"Co-optimization commitment file not found: {coopt_commit_path}")
            return
        
        coopt_commit = pd.read_csv(coopt_commit_path)
        
        # Ensure we have the same columns in both dataframes
        if 'Generator' not in seq_commit.columns or 'Generator' not in coopt_commit.columns:
            print("Cannot compare commitments: Generator column missing")
            return
            
        if 'Hour' not in seq_commit.columns or 'Hour' not in coopt_commit.columns:
            print("Cannot compare commitments: Hour column missing")
            return
        
        # Check if we have a column with commitment values
        seq_commit_col = next((col for col in ['Committed', 'wValue'] if col in seq_commit.columns), None)
        coopt_commit_col = next((col for col in ['Committed', 'wValue'] if col in coopt_commit.columns), None)
        
        if seq_commit_col is None or coopt_commit_col is None:
            print("Cannot compare commitments: Commitment column missing")
            return
            
        # Create pivot tables for easy comparison
        seq_pivot = seq_commit.pivot(index='Generator', columns='Hour', values=seq_commit_col)
        coopt_pivot = coopt_commit.pivot(index='Generator', columns='Hour', values=coopt_commit_col)
        
        # Find generators that are in both datasets
        common_gens = list(set(seq_pivot.index).intersection(set(coopt_pivot.index)))
        
        if not common_gens:
            print("No common generators found between sequential and co-optimization results")
            return
            
        # Find differences in commitment decisions
        different_commitments = []
        
        for gen in common_gens:
            for hour in range(24):
                if hour in seq_pivot.columns and hour in coopt_pivot.columns:
                    seq_val = seq_pivot.loc[gen, hour]
                    coopt_val = coopt_pivot.loc[gen, hour]
                    
                    if seq_val != coopt_val:
                        different_commitments.append({
                            'Generator': gen,
                            'Hour': hour,
                            'Sequential': seq_val,
                            'CoOpt': coopt_val
                        })
        
        # Create a summary DataFrame
        diff_df = pd.DataFrame(different_commitments)
        
        if not diff_df.empty:
            print(f"Found {len(diff_df)} different commitment decisions")
            
            # Count differences by generator
            gen_diff_count = diff_df['Generator'].value_counts()
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot differences by hour for top N generators with differences
            top_n = 10
            top_gens = gen_diff_count.nlargest(top_n).index.tolist()
            
            # Filter diff_df to only include top generators
            plot_df = diff_df[diff_df['Generator'].isin(top_gens)]
            
            # Create a matrix for visualization
            matrix_data = np.zeros((len(top_gens), 24))
            
            for idx, gen in enumerate(top_gens):
                gen_data = plot_df[plot_df['Generator'] == gen]
                for _, row in gen_data.iterrows():
                    # Use 1 for seq only, 2 for coopt only
                    if row['Sequential'] == 1 and row['CoOpt'] == 0:
                        matrix_data[idx, int(row['Hour'])] = 1  # Sequential only
                    elif row['Sequential'] == 0 and row['CoOpt'] == 1:
                        matrix_data[idx, int(row['Hour'])] = 2  # Co-opt only
            
            # Create the heatmap
            im = ax.imshow(matrix_data, cmap='coolwarm', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
            cbar.set_ticklabels(['Same', 'Sequential Only', 'Co-opt Only'])
            
            # Set up axes
            ax.set_xticks(np.arange(24))
            ax.set_yticks(np.arange(len(top_gens)))
            ax.set_xticklabels([f"{h}" for h in range(24)])
            ax.set_yticklabels(top_gens)
            
            ax.set_xlabel('Hour')
            ax.set_ylabel('Generator')
            ax.set_title(f'Commitment Differences: Sequential vs. Co-optimization - {day_type}')
            
            plt.tight_layout()
            
            # Save figure
            plot_path = os.path.join(plots_path, f"commitment_diff_{day_type}.png")
            plt.savefig(plot_path)
            print(f"Commitment difference plot saved to {plot_path}")
            plt.close(fig)
            
            # Save difference data to CSV
            diff_path = os.path.join(results_path, f"commitment_diff_{day_type}.csv")
            diff_df.to_csv(diff_path, index=False)
            print(f"Commitment differences saved to {diff_path}")
        else:
            print("No differences found in commitment decisions")
    
    except Exception as e:
        print(f"Error comparing commitment decisions: {e}")

def plot_renewables_contribution(day_type, data, name_to_cat, plots_path, results_path):
    """
    Creates specialized plots highlighting renewable energy contributions to the generation mix.
    
    Parameters:
    -----------
    day_type : str
        Day type to analyze
    data : dict
        Dictionary containing data frames with all relevant data
    name_to_cat : dict
        Mapping between generator names and fuel categories
    plots_path : str
        Path to save the plots
    results_path : str
        Path to the results directory
    """
    print("\nGenerating renewable energy contribution plots...")
    
    production_df = data['production']
    demand_df = data['demand']
    
    if production_df.empty or demand_df.empty:
        print("Error: Missing production or demand data, skipping renewable plots")
        return
    
    # Group generation by fuel category
    grouped_sums = {}
    for col in production_df.columns:
        gen_name = col
        cat = name_to_cat.get(gen_name, "Unknown")
        if cat not in grouped_sums:
            grouped_sums[cat] = production_df[col].copy()
        else:
            grouped_sums[cat] += production_df[col]
    
    grouped_df = pd.DataFrame(grouped_sums, index=production_df.index)
    
    # 1. Pie chart showing average generation mix by fuel type
    avg_generation = grouped_df.mean()
    total_gen = avg_generation.sum()
    
    # Create labels with percentages
    labels = [f"{fuel} ({avg_generation[fuel]/total_gen*100:.1f}%)" for fuel in avg_generation.index]
    
    # Color mapping for consistent colors across plots
    colors = {
        'Nuclear': '#7b6888',
        'Gas': '#6b486b',
        'Coal': '#a05d56',
        'Oil': '#d0743c',
        'Wind': '#98abc5',
        'Solar': '#8a89a6',
        'PumpedStorage': '#ff8c00',
        'Biomass': '#8c564b',
        'Hydro': '#17becf',
        'Waste': '#b15928',
        'Other': '#cab2d6'
    }
    
    # Get colors for the pie chart
    pie_colors = [colors.get(fuel, '#cccccc') for fuel in avg_generation.index]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        avg_generation, 
        labels=labels, 
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10}
    )
    ax.set_title(f"{day_type} - Average Generation Mix by Fuel Type - Sequential")
    plt.tight_layout()
    
    plot_path = os.path.join(plots_path, f"sequential_avg_generation_mix_{day_type}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)
    
    # 2. Renewables vs. Non-renewables stacked area plot
    # Identify renewable columns
    renewable_cats = ['Wind', 'Solar', 'Biomass', 'Hydro']
    renewable_cols = [col for col in grouped_df.columns if col in renewable_cats]
    non_renewable_cols = [col for col in grouped_df.columns if col not in renewable_cats and col != 'PumpedStorage']
    storage_cols = [col for col in grouped_df.columns if col == 'PumpedStorage']
    
    # Create aggregated dataframe for renewables vs non-renewables
    agg_df = pd.DataFrame(index=grouped_df.index)
    
    # Sum all renewable generation
    if renewable_cols:
        agg_df['Renewable'] = grouped_df[renewable_cols].sum(axis=1)
    else:
        agg_df['Renewable'] = 0
    
    # Sum all non-renewable generation
    if non_renewable_cols:
        agg_df['Non-Renewable'] = grouped_df[non_renewable_cols].sum(axis=1)
    else:
        agg_df['Non-Renewable'] = 0
    
    # Include storage separately if available
    if storage_cols:
        agg_df['Storage'] = grouped_df[storage_cols].sum(axis=1)
    
    # Create an area plot for renewables vs non-renewables
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use lighter colors for the area plot
    area_colors = {
        'Renewable': '#98fb98',  # light green
        'Non-Renewable': '#ffb6c1',  # light red
        'Storage': '#ffd700'  # gold
    }
    
    agg_df.plot(
        kind='area', 
        stacked=True,
        ax=ax,
        color=[area_colors.get(col, '#cccccc') for col in agg_df.columns],
        alpha=0.7,
        linewidth=0
    )
    
    # Add demand line
    ax.plot(
        demand_df.index, 
        demand_df["Demand"].values,
        marker='o', 
        linestyle='-', 
        color='black',
        linewidth=2, label="Load"
    )
    
    ax.set_xlabel("Interval (15-min)")
    ax.set_ylabel("MW")
    ax.set_title(f"{day_type} - Renewable vs. Non-Renewable Generation - Sequential")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_path, f"sequential_renewables_vs_non_renewables_{day_type}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)
    
    # 3. Renewable energy detail breakdown (if we have renewable generation)
    if renewable_cols:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot each renewable source separately
        grouped_df[renewable_cols].plot(
            kind='bar', 
            stacked=True,
            ax=ax,
            color=[colors.get(col, '#cccccc') for col in renewable_cols],
            alpha=0.8,
            width=0.8
        )
        
        # Add total renewables line
        renewable_total = grouped_df[renewable_cols].sum(axis=1)
        ax.plot(
            range(len(renewable_total)), 
            renewable_total.values,
            marker='o', 
            linestyle='-', 
            color='green',
            linewidth=2, 
            label="Total Renewables"
        )
        
        ax.set_xlabel("Interval (15-min)")
        ax.set_ylabel("MW")
        ax.set_title(f"{day_type} - Detailed Renewable Generation - Sequential")
        ax.legend(title="Renewable Sources", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_path, f"sequential_renewable_detail_{day_type}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
        
    # 4. Renewable penetration as percentage of total generation
    total_generation = grouped_df.sum(axis=1)
    renewable_total = grouped_df[renewable_cols].sum(axis=1) if renewable_cols else pd.Series(0, index=grouped_df.index)
    renewable_percentage = (renewable_total / total_generation) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        renewable_percentage.index, 
        renewable_percentage.values,
        marker='o', 
        linestyle='-', 
        color='green',
        linewidth=2
    )
    
    ax.set_xlabel("Interval (15-min)")
    ax.set_ylabel("Percentage of Total Generation (%)")
    ax.set_title(f"{day_type} - Renewable Energy Penetration - Sequential")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)  # Set y-axis from 0-100%
    
    # Add average line
    avg_renewable_pct = renewable_percentage.mean()
    ax.axhline(
        y=avg_renewable_pct, 
        color='r', 
        linestyle='--', 
        label=f"Avg: {avg_renewable_pct:.1f}%"
    )
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(plots_path, f"sequential_renewable_penetration_{day_type}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)
    
    # 5. Calculate and save renewable statistics
    renewable_stats = {
        'DayType': day_type,
        'AvgRenewablePenetration_pct': avg_renewable_pct,
        'MaxRenewablePenetration_pct': renewable_percentage.max(),
        'MinRenewablePenetration_pct': renewable_percentage.min(),
        'TotalRenewableEnergy_MWh': renewable_total.sum() * 0.25,  # Convert from MW to MWh (15-min intervals)
        'TotalEnergy_MWh': total_generation.sum() * 0.25
    }
    
    # Add individual renewable source averages
    for source in renewable_cols:
        source_avg = grouped_df[source].mean()
        source_total = grouped_df[source].sum() * 0.25  # Convert to MWh
        renewable_stats[f'Avg_{source}_MW'] = source_avg
        renewable_stats[f'Total_{source}_MWh'] = source_total
    
    # Save statistics to CSV
    stats_df = pd.DataFrame([renewable_stats])
    sequential_path = os.path.join(results_path, "sequential")
    os.makedirs(sequential_path, exist_ok=True)
    stats_path = os.path.join(sequential_path, f"renewable_stats_{day_type}.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Renewable statistics saved to {stats_path}")
    
    return renewable_stats

def main():
    """
    Main function to generate all plots for the sequential clearing model.
    """
    debug_print("Starting main function")
    
    # Define paths
    DATA_PATH = "/Users/tommie/Documents/thesis/project/data/"
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
    BASE_PATH = os.path.dirname(PROJECT_PATH)
    RESULTS_PATH = os.path.join(BASE_PATH, "results")
    SEQUENTIAL_PATH = os.path.join(RESULTS_PATH, "sequential")
    PLOTS_PATH = os.path.join(BASE_PATH, "plots")
    SEQUENTIAL_PLOTS_PATH = os.path.join(PLOTS_PATH, "sequential")
    
    debug_print(f"Using paths:")
    debug_print(f"- DATA_PATH: {DATA_PATH}")
    debug_print(f"- RESULTS_PATH: {RESULTS_PATH}")
    debug_print(f"- SEQUENTIAL_PATH: {SEQUENTIAL_PATH}")
    debug_print(f"- PLOTS_PATH: {PLOTS_PATH}")
    debug_print(f"- SEQUENTIAL_PLOTS_PATH: {SEQUENTIAL_PLOTS_PATH}")
    
    # Create directories if they don't exist
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(SEQUENTIAL_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)
    os.makedirs(SEQUENTIAL_PLOTS_PATH, exist_ok=True)
    
    DAY_TYPES = ["AutumnWD", "WinterWD", "SpringWD", "SummerWD"]
    debug_print(f"Will process day types: {DAY_TYPES}")
    
    # Generate plots for each day type
    for day_type in DAY_TYPES:
        debug_print(f"\n==== Processing {day_type} ====")
        
        try:
            # Gather all necessary data
            debug_print(f"Gathering data for {day_type}")
            data = gather_data(day_type, DATA_PATH, RESULTS_PATH)
            debug_print(f"Data gathered, keys: {list(data.keys())}")
            
            try:
                # Create fuel mappings
                debug_print("Creating fuel mappings")
                name_to_cat = create_fuel_mappings(data)
                debug_print(f"Created fuel mappings with {len(name_to_cat)} entries")
            except Exception as e:
                print(f"ERROR in create_fuel_mappings: {e}")
                traceback.print_exc()
                continue  # Skip this day type if we can't create fuel mappings
            
            # Process energy generation data if production data is missing
            if data['production'].empty and not data['energy_generation'].empty:
                debug_print("Converting energy_generation data for plotting...")
                try:
                    # Try to create a production dataframe from energy_generation
                    if 'TimeStep' in data['energy_generation'].columns:
                        debug_print("Found TimeStep column in energy_generation")
                        # The energy_generation data is already in the right format with TimeStep as index
                        timesteps = data['energy_generation']['TimeStep'].unique()
                        # Remove the TimeStep column if it exists
                        cols_to_use = [col for col in data['energy_generation'].columns if col != 'TimeStep']
                        data['production'] = data['energy_generation'][cols_to_use]
                        data['production'].index = timesteps
                        debug_print(f"Created production dataframe with shape {data['production'].shape}")
                    elif data['energy_generation'].index.name == 'TimeStep':
                        debug_print("Energy generation already indexed by TimeStep")
                        # Already indexed by TimeStep

                        data['production'] = data['energy_generation']
                        debug_print(f"Using existing production dataframe with shape {data['production'].shape}")
                except Exception as e:
                    print(f"ERROR converting energy_generation data: {e}")
                    traceback.print_exc()
            
            # Generate main plots with try/except for each to continue even if one fails
            try:
                # Skip generation vs load plot if still missing production data
                if not data['production'].empty:
                    debug_print("\nGenerating generation vs load plots...")
                    plot_generation_vs_load(day_type, data, name_to_cat, SEQUENTIAL_PLOTS_PATH, RESULTS_PATH)
                    debug_print("Successfully generated generation vs load plots")
                else:
                    debug_print("Warning: Skipping generation vs load plot due to missing production data")
            except Exception as e:
                print(f"ERROR in plot_generation_vs_load: {e}")
                traceback.print_exc()
            
            try:
                # Process pumped storage data from timeseries
                if not data['ps_timeseries'].empty:
                    debug_print("\nProcessing pumped storage timeseries...")
                    print(f"PS timeseries columns: {data['ps_timeseries'].columns}")
                    process_ps_timeseries(day_type, data, SEQUENTIAL_PLOTS_PATH, RESULTS_PATH)
                    debug_print("Successfully processed pumped storage data")
                else:
                    debug_print("Warning: No pumped storage timeseries data available")
            except Exception as e:
                print(f"ERROR in process_ps_timeseries: {e}")
                traceback.print_exc()
            
            try:
                # Plot reserve data
                debug_print("\nGenerating reserve provision plots...")
                if not data['reserve'].empty:
                    print(f"Reserve data columns: {data['reserve'].columns}")
                    plot_reserve_provision(day_type, data, name_to_cat, SEQUENTIAL_PLOTS_PATH, RESULTS_PATH)
                    debug_print("Successfully generated reserve provision plots")
                else:
                    debug_print("Warning: No reserve data available for plotting")
            except Exception as e:
                print(f"ERROR in plot_reserve_provision: {e}")
                traceback.print_exc()
            
            try:
                # Cost breakdown plots - may be skipped if cost data is missing
                debug_print("\nGenerating cost breakdown plots...")
                plot_cost_breakdown(day_type, data, name_to_cat, SEQUENTIAL_PLOTS_PATH, RESULTS_PATH)
                debug_print("Successfully generated cost breakdown plots")
            except Exception as e:
                print(f"ERROR in plot_cost_breakdown: {e}")
                traceback.print_exc()
            
            try:
                # Compare sequential model with co-optimization results
                debug_print("\nComparing with co-optimization results...")
                compare_sequential_vs_coopt(day_type, RESULTS_PATH, PLOTS_PATH)
                debug_print("Successfully compared sequential vs co-optimization results")
            except Exception as e:
                print(f"ERROR in compare_sequential_vs_coopt: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"ERROR processing {day_type}: {e}")
            traceback.print_exc()
    
    debug_print("\nAll sequential clearing plots and comparisons generation attempts completed.")

# Add this to the bottom of the file to actually run the script
if __name__ == "__main__":
    debug_print("Script is being run directly")
    try:
        main()
    except Exception as e:
        print(f"ERROR in main function: {e}")
        traceback.print_exc()