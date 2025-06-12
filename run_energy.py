#!/usr/bin/env python3
"""
run_energy.py
-------------
Sequential energy-only optimization step.

Reads fixed commitment decisions from reserve step and optimizes energy dispatch only.
Pumped storage is treated as fixed time series (no optimization variables).
"""

import pandas as pd
import numpy as np
import gurobipy as gp
from data_processing import prepare_data
from energy import build_energy_only_model
import os
import argparse

DATA_PATH = "/Users/tommie/Documents/thesis/project/data/"
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(PROJECT_PATH)
RESULTS_PATH = os.path.join(BASE_PATH, "results")
SEQUENTIAL_PATH = os.path.join(RESULTS_PATH, "sequential")
DAY_TYPES = ["AutumnWD", "WinterWD", "SpringWD", "SummerWD"]




def solve_energy(day_type: str,
                 allow_decommitment: bool = True,
                 include_bess: bool = True,
                 use_arbitrage: bool = False,      # ← NEW
                 output_suffix: str = "") -> float:        # ← return obj
    """
    Solve energy-only optimisation for a given day type.

    Returns
    -------
    float   The objective value (€) of the solved model.
    """
    tag = "with BESS" if include_bess else "no BESS"
    suffix_text = (" (decommit)" if allow_decommitment else " (fixed)") + f", {tag}"
    print(f"\n=== Sequential energy step: {day_type}{suffix_text} ===")

    data = prepare_data(DATA_PATH, day_type=day_type)
    # ----------------------------------------------------------
# Inject λ-series for arbitrage runs
# ----------------------------------------------------------
    if use_arbitrage and include_bess:
        mode_tag = "decommit" if allow_decommitment else "fixed"
        lambda_csv = ("/Users/tommie/Documents/thesis/project/results/"
                      f"coopt/lambda_values_{day_type}.csv")
        if not os.path.exists(lambda_csv):
            raise FileNotFoundError(f"λ‑file {lambda_csv} not found")
        lam_df = pd.read_csv(lambda_csv)
        # Drop obvious index column if present (named 'Unnamed: 0' or empty)
        for col in ["Unnamed: 0", "index"]:
            if col in lam_df.columns:
                lam_df = lam_df.drop(columns=[col])
        # Pick column named 'lambda' or the first numeric column
        if "Lambda_EUR_per_MWh" in lam_df.columns:
            lam_series = lam_df["Lambda_EUR_per_MWh"].astype(float).tolist()[:96]
        else:
            # Try case-insensitive lambda column matching
            lambda_col = next((col for col in lam_df.columns if col.lower().startswith("lambda")), None)
            if lambda_col:
                lam_series = lam_df[lambda_col].astype(float).tolist()[:96]
            else:
                # Fall back to first numeric column
                num_cols = lam_df.select_dtypes(include=[float, int]).columns
                if len(num_cols) == 0:
                    raise ValueError("λ‑CSV has no numeric column")
                lam_series = lam_df[num_cols[0]].astype(float).tolist()[:96]
        # Simple sanity check: if values equal 0…95, warn the user
        if all(abs(lam_series[t] - t) < 1e-6 for t in range(min(96, len(lam_series)))):
            print("⚠️  λ‑vector looks like row numbers (0…95). "
                  "Check the CSV – did you save the index column?")
        data["lambda_series"] = lam_series
    else:
        # guarantee the key exists so build_energy_only_model can .get(...)
        data["lambda_series"] = [0.0] * 96

    # --- PATCH A ---------------------------------------------------
    # Remove batteries if scenario demands it
    if not include_bess:
        data["bess_units"] = pd.DataFrame(columns=[
            "PowerCapacity", "EnergyCapacity", "ChargeEff",
            "DischargeEff", "InitialSoC", "RampRateMax",
            "RampRateMin", "CycleCost"
        ])

    # Create results directory if it doesn't exist
    os.makedirs(SEQUENTIAL_PATH, exist_ok=True)
    
    # Look for reserve solution file
    reserve_csv = os.path.join(SEQUENTIAL_PATH, f"reserve_solution_{day_type}.csv")
    if not os.path.exists(reserve_csv):
        # Try results directory as fallback
        fallback_path = os.path.join(RESULTS_PATH, f"reserve_solution_{day_type}.csv")
        if os.path.exists(fallback_path):
            reserve_csv = fallback_path
        else:
            print(f"  Error: {reserve_csv} not found. Run reserve step first.")
            return

    # Build and solve energy-only model with specified decommitment option
    model = build_energy_only_model(data, reserve_csv,
                                allow_decommitment=allow_decommitment,
                                include_bess=include_bess,
                                use_arbitrage=use_arbitrage)
    model.setParam("OutputFlag", 1)
    model.optimize()

    if model.Status != gp.GRB.OPTIMAL:
        print(f"  → Model status = {model.Status}")
        if model.Status == gp.GRB.INFEASIBLE:
            model.computeIIS()
            output_file = os.path.join(SEQUENTIAL_PATH, f"energy_iis_{day_type}{output_suffix}.ilp")
            model.write(output_file)
            print(f"  IIS written to {output_file}")
        return

    # Extract results
    generators = data["generators"] # Changed from just using index to get full dataframe
    hourly_renewables = data["hourly_renewables"]
    hourly_ps_gen = data["hourly_ps_gen"].set_index("Name")
    hourly_ps_pump = data["hourly_ps_pump"].set_index("Name")
    demand_vec = data["hourly_demand"]
    bess_units = data["bess_units"]
    segment_data = data["segment_data"]
    
    T = range(96)

    # Build generation results DataFrame
    generation_data = {}

    # Initialize empty BESS dataframes that will be used later
    bess_dis_data = {}
    bess_chg_data = {}
    bess_net_data = {}

    # ──────────────────────────────────────────────────────────────
    # Battery Energy Storage – discharge / charge / net
    # ──────────────────────────────────────────────────────────────
    extract_bess = (include_bess and not bess_units.empty)
    if extract_bess:
        for i in bess_units.index:
            dis = [model.getVarByName(f"p_discharge_bess[{i},{t}]").X for t in T]
            chg = [model.getVarByName(f"p_charge_bess[{i},{t}]").X    for t in T]
            bess_dis_data[f"{i}_Dis"] = dis                        # positive MW
            bess_chg_data[f"{i}_Chg"] = chg                        # positive MW (charging)
            bess_net_data[f"{i}_Net"] = [d - c for d, c in zip(dis, chg)]
        soc_data = {}
        for i in bess_units.index:
            soc_data[i] = [model.getVarByName(f"soc_bess[{i},{t}]").X for t in T]
    

        
        pd.DataFrame(soc_data, index=T, columns=bess_units.index).to_csv(os.path.join(SEQUENTIAL_PATH, f"bess_soc_{day_type}{output_suffix}.csv"), index_label="TimeStep")
            # Extract state of charge for each BESS unit
    # Thermal generators - use proper names instead of indices
    for g in generators.index:
        generation_data[generators.loc[g, "Name"] if "Name" in generators.columns else g] = [model.getVarByName(f"p[{g},{t}]").X for t in T]
    
    # Renewables (fixed time series) - truncate to 96 time steps
    for r in hourly_renewables.index:
        renewable_values = hourly_renewables.loc[r].values.tolist()
        renewable_name = hourly_renewables.loc[r, "Name"] if "Name" in hourly_renewables.columns else r
        # Ensure we only take the first 96 values to match T = range(96)
        generation_data[renewable_name] = renewable_values[:96]
    
    # Pumped storage - separate generation and pumping for clarity
    # Generation (positive values)
    ps_gen_data = {}
    for ps_name in hourly_ps_gen.index:
        gen_series = hourly_ps_gen.loc[ps_name].values.tolist()[:96]
        ps_gen_data[f"{ps_name}_Gen"] = gen_series
        
    # Pumping (negative values)
    ps_pump_data = {}
    for ps_name in hourly_ps_pump.index:
        pump_series = hourly_ps_pump.loc[ps_name].values.tolist()[:96]
        ps_pump_data[f"{ps_name}_Pump"] = pump_series
        
    # Also include net values for backward compatibility
    ps_net_data = {}
    for ps_name in hourly_ps_gen.index:
        if ps_name in hourly_ps_pump.index:
            gen_series = hourly_ps_gen.loc[ps_name].values.tolist()[:96]
            pump_series = hourly_ps_pump.loc[ps_name].values.tolist()[:96]
            net_series = [g + p for g, p in zip(gen_series, pump_series)]
            ps_net_data[ps_name] = net_series
    
    # Create DataFrames with proper index (96 rows)
    thermal_df = pd.DataFrame(generation_data, index=T)
    ps_gen_df = pd.DataFrame(ps_gen_data, index=T)
    ps_pump_df = pd.DataFrame(ps_pump_data, index=T)
    ps_net_df = pd.DataFrame(ps_net_data, index=T)
    bess_dis_df = pd.DataFrame(bess_dis_data, index=T)
    bess_chg_df = pd.DataFrame(bess_chg_data, index=T)
    bess_net_df = pd.DataFrame(bess_net_data, index=T)


    
    # Combine all generator data, making sure we have exactly 96 rows
    all_gen_df = pd.concat([thermal_df, ps_gen_df, ps_pump_df, ps_net_df, bess_dis_df, bess_chg_df, bess_net_df], axis=1)
    
    # Ensure the index is properly set
    all_gen_df.index = list(T)
    all_gen_df.index.name = "TimeStep"
    
    # Save generation results
    generation_file = os.path.join(SEQUENTIAL_PATH, f"energy_generation_{day_type}{output_suffix}.csv")
    all_gen_df.to_csv(generation_file, index=True)
    print(f"  ✓ Generation data saved to {generation_file} with {len(all_gen_df)} rows and {len(all_gen_df.columns)} generator columns")

    # ----- diagnostics -------------------------------------------------
    # Lambda vector (already padded to length‑96 in solve_energy)
    lam_vec = data["lambda_series"] if "lambda_series" in data else [0.0]*96

    bess_dis_total = [0.0]*96
    bess_chg_total = [0.0]*96
    if extract_bess:
        for t in T:
            bess_dis_total[t] = sum(bess_dis_data[f"{i}_Dis"][t] for i in bess_units.index)
            bess_chg_total[t] = sum(bess_chg_data[f"{i}_Chg"][t] for i in bess_units.index)

    # Marginal‑cost statistics
    avg_mc  = [0.0]*96
    max_mc  = [0.0]*96
    for t in T:
        total_p  = 0.0
        tot_cost = 0.0
        max_cost = 0.0
        for g in generators.index:
            pg = model.getVarByName(f"p[{g},{t}]").X
            if pg < 0.01:
                continue
            slope = segment_data[generators.loc[g, "HeatRateCurve"]][0][0]
            mc_eur = slope * generators.loc[g, "Price"] 
            total_p  += pg
            tot_cost += pg * mc_eur
            max_cost = max(max_cost, mc_eur)
        if total_p > 0:
            avg_mc[t] = tot_cost / total_p
            max_mc[t] = max_cost

    df_diag = pd.DataFrame({
        "Lambda":        lam_vec,
        "BESS_Discharge": bess_dis_total,
        "BESS_Charge":    bess_chg_total,
        "BESS_Net":       [d - c for d,c in zip(bess_dis_total, bess_chg_total)],
        "Avg_MC":         avg_mc,
        "Max_MC":         max_mc
    }, index=T)
    df_diag.index.name = "TimeStep"
    diag_file = os.path.join(SEQUENTIAL_PATH,
                             f"energy_diagnostics_{day_type}{output_suffix}.csv")
    df_diag.to_csv(diag_file, index=True)
    print(f"  ✓ Diagnostics saved to {diag_file}")

    # Save demand and load shedding
    load_shed = [model.getVarByName(f"l[{t}]").X for t in T]
    load_shed_neg = [model.getVarByName(f"l_neg[{t}]").X for t in T]
    
    # Flatten demand data to ensure 1D array
    demand_values = demand_vec.values.flatten() if len(demand_vec.values.shape) > 1 else demand_vec.values
    
    df_demand = pd.DataFrame({
        "Demand": demand_values[:96],  # Ensure only 96 values
        "LoadShedding": load_shed,
        "LoadSheddingNeg": load_shed_neg
    }, index=T)
    df_demand.index.name = "TimeStep"
    df_demand.to_csv(os.path.join(SEQUENTIAL_PATH, f"energy_demand_{day_type}{output_suffix}.csv"), index=True)

    # Save commitment decisions
    commitment_data = {}
    for g in generators.index:
        commitment_data[generators.loc[g, "Name"] if "Name" in generators.columns else g] = [model.getVarByName(f"w[{g},{h}]").X for h in range(24)]
    
    df_commit = pd.DataFrame(commitment_data, index=range(24))
    df_commit.index.name = "Hour"
    df_commit.to_csv(os.path.join(SEQUENTIAL_PATH, f"energy_commitment_{day_type}{output_suffix}.csv"), index=True)

    # Save objective value
    with open(os.path.join(SEQUENTIAL_PATH, f"energy_objective_{day_type}{output_suffix}.txt"), "w") as f:
        f.write(str(model.ObjVal))

    print(f"  ✓ Solved – objective = {model.ObjVal:,.2f} EUR")
    
    # Return the objective value for comparison 
    return model.ObjVal


if __name__ == "__main__":
    # ------------------------------------------------------------
    # 1) Command-line options
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=("Run sequential energy model with optional "
                     "de-commitment and BESS / no-BESS comparison"))
    parser.add_argument('--no-decommit', action='store_true',
                        help='Disable decommitment (fixed commitment)')
    parser.add_argument('--run-both', action='store_true',
                        help='Run both fixed-commitment and de-commitment cases')
    parser.add_argument('--compare-bess', action='store_true',
                        help='For every case, also run a scenario with the '
                             'battery removed and compare total cost')
    parser.add_argument('--day-types', nargs='+', choices=DAY_TYPES,
                        default=DAY_TYPES, help='Day types to process')
    parser.add_argument('--compare-arb', action='store_true',
    help='Also run a scenario with battery and arbitrage revenue term')
    args = parser.parse_args()

    os.makedirs(SEQUENTIAL_PATH, exist_ok=True)

    # ------------------------------------------------------------
    # 2) Which unit-commitment variants to solve?
    # ------------------------------------------------------------
    uc_variants = []      # list of tuples: (allow_decommitment, suffix)
    if args.run_both:
        uc_variants = [(False, "_fixed"), (True, "_decommit")]
    else:
        uc_variants = [(not args.no_decommit, "")]

    # ------------------------------------------------------------
    # 3) Master loop over days and variants
    # ------------------------------------------------------------
    for day_type in args.day_types:
        for allow_dec, suf in uc_variants:

            # ---------- baseline: battery, no arbitrage -------------
            obj_bess = solve_energy(day_type, allow_dec,
                                    include_bess=True,
                                    use_arbitrage=False,
                                    output_suffix=suf + "_BESS")

            results = {"BESS": obj_bess}

            # ---------- no-battery comparison ------------------------
            if args.compare_bess:
                obj_no = solve_energy(day_type, allow_dec,
                                    include_bess=False,
                                    use_arbitrage=False,
                                    output_suffix=suf + "_noBESS")
                results["noBESS"] = obj_no

            # ---------- battery with λ-revenue -----------------------
            if args.compare_arb:
                obj_arb = solve_energy(day_type, allow_dec,
                                    include_bess=True,
                                    use_arbitrage=True,
                                    output_suffix=suf + "_BESS_ARB")
                results["BESS_ARB"] = obj_arb

            # ---------- print & save pairwise deltas ----------------
            ref_tag = "noBESS" if "noBESS" in results else "BESS"
            ref_val = results[ref_tag]

            for tag, val in results.items():
                if tag == ref_tag:      # skip self
                    continue
                delta = ref_val - val
                pct   = delta / ref_val * 100 if ref_val else 0.0
                print(f"\n=== {tag} vs {ref_tag} "
                    f"({day_type}, {'decommit' if allow_dec else 'fixed'}) ===")
                print(f"   {ref_tag:9s}: {ref_val:,.2f} €")
                print(f"   {tag:9s}: {val:,.2f} €")
                print(f"   Δ = {delta:,.2f} € ({pct:.2f} %)")

                pd.DataFrame([{
                    "Day":          day_type,
                    "UC_mode":      "decommit" if allow_dec else "fixed",
                    "Scenario":     tag,
                    "Reference":    ref_tag,
                    "Cost_ref":     ref_val,
                    "Cost_scn":     val,
                    "Delta":        delta,
                    "Delta_%":      pct
                }]).to_csv(os.path.join(
                    SEQUENTIAL_PATH,
                    f"energy_comparison_{ref_tag}_vs_{tag}_{day_type}{suf}.csv"),
                    index=False)

    print("\nAll sequential energy runs complete.")