#!/usr/bin/env python3
"""
compare_costs.py
---------------
Compares the costs of cooptimization vs. sequential clearing and produces
detailed output files for dispatch and reserve provision.

This script:
1. Reads the objective values from cooptimization and sequential clearing
2. Calculates the total cost of sequential clearing (reserve + energy)
3. Produces detailed output files with generator dispatch and reserve provision
4. Prepares data for plotting generator dispatch and load curves
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Path constants
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(PROJECT_PATH)
RESULTS_PATH = os.path.join(BASE_PATH, "results")
COOPT_PATH = os.path.join(RESULTS_PATH, "coopt")
SEQUENTIAL_PATH = os.path.join(RESULTS_PATH, "sequential")
PLOTS_PATH = os.path.join(BASE_PATH, "plots")
DAY_TYPES = ["AutumnWD", "WinterWD", "SpringWD", "SummerWD"]

# Create directories if they don't exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(COOPT_PATH, exist_ok=True)
os.makedirs(SEQUENTIAL_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

def get_coopt_objective(day_type):
    """Read the objective value from the cooptimization run."""
    file_path = os.path.join(COOPT_PATH, f"coopt_objective_{day_type}.txt")
    try:
        with open(file_path, "r") as f:
            return float(f.read().strip())
    except FileNotFoundError:
        # Try fallback to results directory
        fallback_path = os.path.join(RESULTS_PATH, f"coopt_objective_{day_type}.txt")
        try:
            with open(fallback_path, "r") as f:
                print(f"Using coopt objective from results directory: {fallback_path}")
                return float(f.read().strip())
        except FileNotFoundError:
            # Try project directory as final fallback
            fallback_path = os.path.join(BASE_PATH, f"coopt_objective_{day_type}.txt")
            try:
                with open(fallback_path, "r") as f:
                    print(f"Using coopt objective from project directory: {fallback_path}")
                    return float(f.read().strip())
            except FileNotFoundError:
                print(f"Warning: Could not find cooptimization objective for {day_type}")
                return None

def get_sequential_objectives(day_type):
    """Read the objective values from sequential clearing (reserve + energy)."""
    # Read reserve objective
    reserve_csv = os.path.join(SEQUENTIAL_PATH, f"reserve_solution_{day_type}.csv")
    try:
        reserve_df = pd.read_csv(reserve_csv)
        reserve_obj = reserve_df["Objective"].iloc[0]
    except (FileNotFoundError, KeyError):
        # Try fallback to results directory
        fallback_path = os.path.join(RESULTS_PATH, f"reserve_solution_{day_type}.csv")
        try:
            reserve_df = pd.read_csv(fallback_path)
            reserve_obj = reserve_df["Objective"].iloc[0]
            print(f"Using reserve objective from results directory: {fallback_path}")
        except (FileNotFoundError, KeyError):
            # Try project directory as final fallback
            fallback_path = os.path.join(BASE_PATH, f"reserve_solution_{day_type}.csv")
            try:
                reserve_df = pd.read_csv(fallback_path)
                reserve_obj = reserve_df["Objective"].iloc[0]
                print(f"Using reserve objective from project directory: {fallback_path}")
            except (FileNotFoundError, KeyError):
                print(f"Warning: Could not find or read reserve objective for {day_type}")
                reserve_obj = None
    
    # Read energy objective
    energy_file = os.path.join(SEQUENTIAL_PATH, f"energy_objective_{day_type}.txt")
    try:
        with open(energy_file, "r") as f:
            energy_obj = float(f.read().strip())
    except FileNotFoundError:
        # Try fallback to results directory
        fallback_path = os.path.join(RESULTS_PATH, f"energy_objective_{day_type}.txt")
        try:
            with open(fallback_path, "r") as f:
                energy_obj = float(f.read().strip())
                print(f"Using energy objective from results directory: {fallback_path}")
        except FileNotFoundError:
            # Try project directory as final fallback
            fallback_path = os.path.join(BASE_PATH, f"energy_objective_{day_type}.txt")
            try:
                with open(fallback_path, "r") as f:
                    energy_obj = float(f.read().strip())
                    print(f"Using energy objective from project directory: {fallback_path}")
            except FileNotFoundError:
                print(f"Warning: Could not find energy objective for {day_type}")
                energy_obj = None
    
    return reserve_obj, energy_obj

def combine_generator_dispatch(day_type):
    """Combine generator dispatch data from energy step with reserve provision from reserve step."""
    # Read energy generation data
    energy_gen_csv = os.path.join(SEQUENTIAL_PATH, f"energy_generation_{day_type}.csv")
    if not os.path.exists(energy_gen_csv):
        # Try fallback to results directory
        fallback_path = os.path.join(RESULTS_PATH, f"energy_generation_{day_type}.csv")
        if os.path.exists(fallback_path):
            energy_gen_csv = fallback_path
            print(f"Using energy generation data from results directory: {fallback_path}")
        else:
            # Try project directory as final fallback
            fallback_path = os.path.join(BASE_PATH, f"energy_generation_{day_type}.csv")
            if os.path.exists(fallback_path):
                energy_gen_csv = fallback_path
                print(f"Using energy generation data from project directory: {fallback_path}")
            else:
                print(f"Warning: Could not find energy generation data for {day_type}")
                return None
    
    try:
        energy_df = pd.read_csv(energy_gen_csv, index_col="TimeStep")
    except FileNotFoundError:
        print(f"Warning: Could not find energy generation data for {day_type}")
        return None
    
    # Read reserve provision data
    reserve_csv = os.path.join(SEQUENTIAL_PATH, f"reserve_solution_{day_type}.csv")
    if not os.path.exists(reserve_csv):
        # Try fallback to results directory
        fallback_path = os.path.join(RESULTS_PATH, f"reserve_solution_{day_type}.csv")
        if os.path.exists(fallback_path):
            reserve_csv = fallback_path
            print(f"Using reserve solution data from results directory: {fallback_path}")
        else:
            # Try project directory as final fallback
            fallback_path = os.path.join(BASE_PATH, f"reserve_solution_{day_type}.csv")
            if os.path.exists(fallback_path):
                reserve_csv = fallback_path
                print(f"Using reserve solution data from project directory: {fallback_path}")
            else:
                print(f"Warning: Could not find reserve solution data for {day_type}")
                return None
    
    try:
        reserve_df = pd.read_csv(reserve_csv)
    except FileNotFoundError:
        print(f"Warning: Could not find reserve solution data for {day_type}")
        return None
    
    # Create a new DataFrame for detailed generator output
    # First, get all thermal generators from energy_df
    thermal_gens = [col for col in energy_df.columns if col not in ['TimeStep']]
    
    # Initialize the detailed output DataFrame
    detailed_df = pd.DataFrame(index=range(96))
    detailed_df.index.name = "TimeStep"
    
    # Add energy dispatch for each generator
    for gen in thermal_gens:
        if gen in energy_df.columns:
            detailed_df[f"{gen}_energy"] = energy_df[gen]
    
    # Process reserve provision
    # Get unique reserve products
    reserve_products = reserve_df["ReserveProduct"].unique()
    
    # For each generator and reserve product, add the reserve provision
    for gen in thermal_gens:
        for prod in reserve_products:
            # Filter reserve data for this generator and product
            gen_reserve = reserve_df[(reserve_df["Generator"] == gen) & 
                                    (reserve_df["ReserveProduct"] == prod)]
            
            # Create a series with the reserve provision
            if not gen_reserve.empty:
                reserve_series = pd.Series(index=range(96), data=0.0)
                for _, row in gen_reserve.iterrows():
                    time_step = int(row["TimeStep"])
                    reserve_series[time_step] = row["s_res_Value"]
                
                detailed_df[f"{gen}_reserve_{prod}"] = reserve_series
    
    # Add commitment status (w)
    for gen in thermal_gens:
        # Filter for commitment status
        gen_commitment = reserve_df[(reserve_df["Generator"] == gen) & 
                                  (~reserve_df["Hour"].isna())]
        
        if not gen_commitment.empty:
            commitment_series = pd.Series(index=range(96), data=0.0)
            for _, row in gen_commitment.iterrows():
                hour = int(row["Hour"])
                # For each hour, set the commitment status for 4 time steps
                for t in range(hour*4, (hour+1)*4):
                    if t < 96:  # Make sure we don't go beyond the day
                        commitment_series[t] = row["wValue"]
            
            detailed_df[f"{gen}_commitment"] = commitment_series
    
    # Save the detailed output
    output_file = os.path.join(RESULTS_PATH, f"sequential_detailed_{day_type}.csv")
    detailed_df.to_csv(output_file)
    print(f"Detailed generator dispatch saved to {output_file}")
    
    return detailed_df

def calculate_cost_comparison():
    """Calculate and display the cost comparison between cooptimization and sequential clearing."""
    # Create a DataFrame to store the results
    results = pd.DataFrame(index=DAY_TYPES, 
                          columns=["Coopt_Cost", "Sequential_Reserve_Cost", "Sequential_Energy_Cost", 
                                  "Sequential_Total_Cost", "Cost_Difference", "Percentage_Difference",
                                  "Sequential_Reserve_Profit", "Sequential_Net_Cost"])
    
    total_coopt = 0
    total_sequential = 0
    total_net_sequential = 0
    
    for day_type in DAY_TYPES:
        # Get costs
        coopt_obj = get_coopt_objective(day_type)
        reserve_obj, energy_obj = get_sequential_objectives(day_type)
        
        if coopt_obj is not None and reserve_obj is not None and energy_obj is not None:
            # Properly interpret reserve costs
            # If reserve_obj is negative, it represents profit, not cost
            reserve_profit = abs(reserve_obj) if reserve_obj < 0 else 0
            reserve_cost = reserve_obj if reserve_obj > 0 else 0
            
            # Calculate net sequential cost (energy cost minus reserve profit)
            sequential_net = energy_obj - reserve_profit
            
            # Original total sequential cost (for reference)
            sequential_total = reserve_obj + energy_obj
            
            # Calculate difference using the net cost
            diff = sequential_net - coopt_obj
            perc_diff = (diff / coopt_obj) * 100
            
            # Update results
            results.loc[day_type, "Coopt_Cost"] = coopt_obj
            results.loc[day_type, "Sequential_Reserve_Cost"] = reserve_obj
            results.loc[day_type, "Sequential_Energy_Cost"] = energy_obj
            results.loc[day_type, "Sequential_Total_Cost"] = sequential_total
            results.loc[day_type, "Sequential_Reserve_Profit"] = reserve_profit
            results.loc[day_type, "Sequential_Net_Cost"] = sequential_net
            results.loc[day_type, "Cost_Difference"] = diff
            results.loc[day_type, "Percentage_Difference"] = perc_diff
            
            # Update totals
            total_coopt += coopt_obj
            total_sequential += sequential_total
            total_net_sequential += sequential_net
    
    # Add total row
    results.loc["Total", "Coopt_Cost"] = total_coopt
    results.loc["Total", "Sequential_Reserve_Cost"] = results["Sequential_Reserve_Cost"].sum()
    results.loc["Total", "Sequential_Energy_Cost"] = results["Sequential_Energy_Cost"].sum()
    results.loc["Total", "Sequential_Total_Cost"] = total_sequential
    results.loc["Total", "Sequential_Reserve_Profit"] = results["Sequential_Reserve_Profit"].sum()
    results.loc["Total", "Sequential_Net_Cost"] = total_net_sequential
    results.loc["Total", "Cost_Difference"] = total_net_sequential - total_coopt
    results.loc["Total", "Percentage_Difference"] = ((total_net_sequential - total_coopt) / total_coopt) * 100
    
    # Save results
    results.to_csv(os.path.join(RESULTS_PATH, "cost_comparison.csv"))
    print(f"Cost comparison saved to {os.path.join(RESULTS_PATH, 'cost_comparison.csv')}")
    
    # Create visualization
    # Bar chart comparing coopt vs sequential costs
    plt.figure(figsize=(12, 6))
    ind = np.arange(len(DAY_TYPES))
    width = 0.35
    
    plt.bar(ind, [results.loc[day, "Coopt_Cost"] for day in DAY_TYPES], width, 
            label='Co-optimization Cost')
    plt.bar(ind + width, [results.loc[day, "Sequential_Net_Cost"] for day in DAY_TYPES], width,
            label='Sequential Net Cost')
    
    plt.xlabel('Day Type')
    plt.ylabel('Cost (€)')
    plt.title('Cost Comparison: Co-optimization vs. Sequential Clearing')
    plt.xticks(ind + width/2, DAY_TYPES)
    plt.legend(loc='best')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_PATH, "overall_cost_comparison.png")
    plt.savefig(plot_path)
    print(f"Cost comparison plot saved to {plot_path}")
    plt.close()
    
    # Stacked bar chart for sequential costs breakdown
    plt.figure(figsize=(12, 6))
    
    # For each day, show energy cost and reserve cost/profit
    energy_costs = [results.loc[day, "Sequential_Energy_Cost"] for day in DAY_TYPES]
    reserve_costs = []
    reserve_profits = []
    
    for day in DAY_TYPES:
        if results.loc[day, "Sequential_Reserve_Cost"] > 0:
            reserve_costs.append(results.loc[day, "Sequential_Reserve_Cost"])
            reserve_profits.append(0)
        else:
            reserve_costs.append(0)
            reserve_profits.append(results.loc[day, "Sequential_Reserve_Profit"])
    
    plt.bar(ind, energy_costs, width, label='Energy Cost')
    plt.bar(ind, reserve_costs, width, bottom=energy_costs, label='Reserve Cost')
    plt.bar(ind, [-p for p in reserve_profits], width, bottom=[0]*len(DAY_TYPES), label='Reserve Profit')
    
    plt.xlabel('Day Type')
    plt.ylabel('Cost/Profit (€)')
    plt.title('Sequential Clearing: Cost Breakdown')
    plt.xticks(ind, DAY_TYPES)
    plt.legend(loc='best')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_PATH, "sequential_cost_breakdown.png")
    plt.savefig(plot_path)
    print(f"Sequential cost breakdown plot saved to {plot_path}")
    plt.close()
    
    # Print results
    print("\n=== Cost Comparison: Cooptimization vs. Sequential Clearing ===")
    print(results[["Coopt_Cost", "Sequential_Energy_Cost", "Sequential_Reserve_Cost", "Sequential_Net_Cost", "Percentage_Difference"]])
    
    return results

def create_sequential_dispatch_files():
    """Create detailed dispatch files for each day type."""
    for day_type in DAY_TYPES:
        print(f"\nProcessing detailed dispatch for {day_type}...")
        detailed_df = combine_generator_dispatch(day_type)
        
        if detailed_df is not None:
            print(f"  ✓ Created detailed dispatch file for {day_type}")

def main():
    """Main function to run all comparisons and create output files."""
    print("Creating detailed dispatch files for sequential clearing...")
    create_sequential_dispatch_files()
    
    print("\nCalculating cost comparison...")
    cost_comparison = calculate_cost_comparison()
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main() 