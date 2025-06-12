import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_generation_vs_load(day_type, data_path, results_path, plots_path, method="coopt"):
    """
    Reads generator production results + demand for a given day_type,
    groups them, and plots.
    
    Parameters:
    -----------
    method: str
        Either "coopt" or "sequential" to determine which subdirectory to use
    """
    # Determine the appropriate directory based on method
    method_path = os.path.join(results_path, method)
    
    # 1) Load final production (all generators) + demand
    production_file = os.path.join(method_path, f"generator_production_results_{day_type}.csv")
    demand_file = os.path.join(method_path, f"hourly_demand_{day_type}.csv")

    # Check if files exist, if not try looking in results directory
    if not os.path.exists(production_file):
        fallback_path = os.path.join(results_path, f"generator_production_results_{day_type}.csv")
        if os.path.exists(fallback_path):
            production_file = fallback_path
            print(f"Using production file from results directory: {fallback_path}")
        else:
            # Try project directory as final fallback
            fallback_path = os.path.join(os.path.dirname(results_path), f"generator_production_results_{day_type}.csv")
            if os.path.exists(fallback_path):
                production_file = fallback_path
                print(f"Using production file from project directory: {fallback_path}")
            else:
                print(f"Error: Production file not found at {production_file}")
                return

    if not os.path.exists(demand_file):
        fallback_path = os.path.join(results_path, f"hourly_demand_{day_type}.csv")
        if os.path.exists(fallback_path):
            demand_file = fallback_path
            print(f"Using demand file from results directory: {fallback_path}")
        else:
            # Try project directory as final fallback
            fallback_path = os.path.join(os.path.dirname(results_path), f"hourly_demand_{day_type}.csv")
            if os.path.exists(fallback_path):
                demand_file = fallback_path
                print(f"Using demand file from project directory: {fallback_path}")
            else:
                print(f"Error: Demand file not found at {demand_file}")
                return

    production_df = pd.read_csv(production_file, index_col=0)
    demand_df = pd.read_csv(demand_file, index_col=0)

    # 2) Read Generators + GeneratorsRE to unify the fuel info
    dfG  = pd.read_csv(data_path + "Generators.csv")
    dfRE = pd.read_csv(data_path + "GeneratorsRE.csv", delimiter=";")

    # Rename columns to unify
    dfG_ren = dfG.rename(columns={
        "Generator":     "Name",
        "FuelGenerator": "FuelType"
    })
    dfRE_ren = dfRE.rename(columns={
        "GeneratorRE":   "Name",
        "FuelGeneratorRE": "FuelType"
    })

    dfG_ren = dfG_ren[["Name","FuelType"]].drop_duplicates()
    dfRE_ren = dfRE_ren[["Name","FuelType"]].drop_duplicates()
    dfAll = pd.concat([dfG_ren, dfRE_ren], ignore_index=True).drop_duplicates()

    fuel_map = {
        "NUCLEAR":       "Nuclear",
        "WASTE":         "Waste",
        "GAS":           "Gas",
        "COAL":          "Coal",
        "OIL":           "Oil",
        "HYDRAULIC_PS":  "PumpedStorage",
        "SOLAR":         "Solar",
        "WIND":          "Wind",
        "BESS":          "Battery",
    }
    dfAll["FuelCategory"] = dfAll["FuelType"].map(fuel_map).fillna("Other")
    name_to_cat = dict(zip(dfAll["Name"], dfAll["FuelCategory"]))
    
    # Add manual mapping for pumped storage units and BESS units
    for col in production_df.columns:
        if "_Gen" in col or "_Pump" in col or "PLATETAILLE" in col or "COO_A" in col:
            name_to_cat[col] = "PumpedStorage"
        elif "BESS" in col:
            name_to_cat[col] = "Battery"
    # -----------------------------------------------------
    # PLOT 1: Each column individually (stacked bar)
    # -----------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12,6))
    production_df.plot(kind='bar', stacked=True, ax=ax1)
    ax1.plot(demand_df.index, demand_df["Demand"].values,
             marker='o', linestyle='-', color='black',
             linewidth=2, label="Load")
    ax1.set_xlabel("Interval (15-min)")
    ax1.set_ylabel("MW")
    ax1.set_title(f"{day_type} - Hourly Generation (by Unit) vs. Load - {method.capitalize()}")
    ax1.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax1.grid(True)
    plt.tight_layout()
    
    # Save figure to plots directory
    plot_path = os.path.join(plots_path, f"{method}_generation_by_unit_{day_type}.png")
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
    
    # Save grouped data to results directory
    grouped_output_path = os.path.join(method_path, f"grouped_generation_{day_type}.csv")
    grouped_df.to_csv(grouped_output_path)
    print(f"Grouped generation data saved to {grouped_output_path}")

    fig2, ax2 = plt.subplots(figsize=(12,6))
    grouped_df.plot(kind='bar', stacked=True, ax=ax2)
    ax2.plot(demand_df.index, demand_df["Demand"].values,
             marker='o', linestyle='-', color='black',
             linewidth=2, label="Load")
    ax2.set_xlabel("Interval (15-min)")
    ax2.set_ylabel("MW")
    ax2.set_title(f"{day_type} - Hourly Generation (by Fuel Type) vs. Load - {method.capitalize()}")
    ax2.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax2.grid(True)
    plt.tight_layout()
    
    # Save figure to plots directory
    plot_path = os.path.join(plots_path, f"{method}_generation_by_fuel_{day_type}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig2)


def main():
    DATA_PATH = "/Users/tommie/Documents/thesis/project/data/"
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
    BASE_PATH = os.path.dirname(PROJECT_PATH)
    RESULTS_PATH = os.path.join(BASE_PATH, "results")
    COOPT_PATH = os.path.join(RESULTS_PATH, "coopt")
    SEQUENTIAL_PATH = os.path.join(RESULTS_PATH, "sequential")
    PLOTS_PATH = os.path.join(BASE_PATH, "plots")
    
    # Create directories if they don't exist
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(COOPT_PATH, exist_ok=True)
    os.makedirs(SEQUENTIAL_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    DAY_TYPES = ["AutumnWD", "WinterWD", "SpringWD", "SummerWD"]

    # Generate plots for co-optimization results
    print("Generating plots for co-optimization results...")
    for dtype in DAY_TYPES:
        print(f"Processing {dtype} ...")
        plot_generation_vs_load(dtype, DATA_PATH, RESULTS_PATH, PLOTS_PATH, method="coopt")
    
    # Generate plots for sequential clearing results
    print("\nGenerating plots for sequential clearing results...")
    for dtype in DAY_TYPES:
        print(f"Processing {dtype} ...")
        plot_generation_vs_load(dtype, DATA_PATH, RESULTS_PATH, PLOTS_PATH, method="sequential")

    print("All plots generated.")

if __name__ == "__main__":
    main()
