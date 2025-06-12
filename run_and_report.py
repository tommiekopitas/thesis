import pandas as pd
import matplotlib.pyplot as plt
import os

VERBOSE = False   # Set True if you want deep debug prints

from data_processing import prepare_data
from build_model import build_cooptimization_model
from build_model import TIME_STEP_HOURS   # 0.25 h per 15‑min interval

# Helper constants for time intervals
dt = TIME_STEP_HOURS          # 0.25 h per 15‑min interval
T96 = range(96)
H24 = range(24)

def main():
    DATA_PATH = "/Users/tommie/Documents/thesis/project/data/"
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
    BASE_PATH = os.path.dirname(PROJECT_PATH)
    RESULTS_PATH = os.path.join(BASE_PATH, "results")
    COOPT_PATH = os.path.join(RESULTS_PATH, "coopt")
    PLOTS_PATH = os.path.join(BASE_PATH, "plots")
    
    # Create directories if they don't exist
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(COOPT_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    DAY_TYPES = ["AutumnWD", "WinterWD", "SpringWD", "SummerWD"]

    for dtype in DAY_TYPES:
        print(f"=== Running co-optimization for {dtype} ===")

        # 1) Load & prepare data for this day type
        data = prepare_data(DATA_PATH, day_type=dtype)

        # Extract needed data variables 
        generators = data["generators"]
        ps_units = data["ps_units"]
        hourly_renewables = data["hourly_renewables"]
        hourly_demand = data["hourly_demand"]
        segment_data = data["segment_data"]
        bess_units = data["bess_units"]
        T_range = range(96)

        # 2) First, build and solve the model with binary variables for accurate dispatch
        print("Running model with binary variables for accurate dispatch...")
        model_bin, p_bin, w_bin, z_bin, c_g_bin, l_bin, l_neg_bin, p_gen_ps_bin, p_pump_ps_bin, soc_ps_bin, p_charge_bess_bin, p_discharge_bess_bin, soc_bess_bin = build_cooptimization_model(data, use_lp_relaxation=False, day_type=dtype)
        model_bin.optimize()
        
        if model_bin.status == 3:  # GRB.INFEASIBLE
            model_bin.computeIIS()
            output_file = os.path.join(COOPT_PATH, f"model_{dtype}_bin.ilp")
            model_bin.write(output_file)
            print(f"Model for {dtype} is infeasible; see {output_file} for details.")
            continue
            
        # Save the binary solution results
        binary_solution = {
            'p': {(g,t): p_bin[g,t].X for g in generators.index for t in range(96)},
            'w': {(g,h): w_bin[g,h].X for g in generators.index for h in range(24)},
            'z': {(g,h): z_bin[g,h].X for g in generators.index for h in range(24)},
            'p_gen_ps': {(i,t): p_gen_ps_bin[i,t].X for i in ps_units.index for t in range(96)},
            'p_pump_ps': {(i,t): p_pump_ps_bin[i,t].X for i in ps_units.index for t in range(96)},
            'soc_ps': {(i,t): soc_ps_bin[i,t].X for i in ps_units.index for t in range(96)},
            'p_charge_bess': {(i,t): p_charge_bess_bin[i,t].X for i in bess_units.index for t in range(96)},
            'p_discharge_bess': {(i,t): p_discharge_bess_bin[i,t].X for i in bess_units.index for t in range(96)},
            'soc_bess': {(i,t): soc_bess_bin[i,t].X for i in bess_units.index for t in range(96)}
        }
        
        # Record the objective value from the binary solution
        binary_obj_val = model_bin.ObjVal
        
        # 3) Now build and solve the LP relaxation model for duals
        print("Running model with LP relaxation for dual values...")
        model, p, w, z, c_g, l, l_neg, p_gen_ps, p_pump_ps, soc_ps, p_charge_bess_bin, p_discharge_bess_bin, soc_bess_bin = build_cooptimization_model(data, use_lp_relaxation=True,day_type=dtype)

        # Set parameter to compute dual values (shadow prices)
        model.setParam('InfUnbdInfo', 1)
        model.setParam('DualReductions', 0)
        model.setParam('OutputFlag', 1)  # Show detailed output
        model.setParam('Method', 1)      # Use dual simplex method
        
        # 4) Solve LP model
        model.optimize()
        if model.status == 3:  # GRB.INFEASIBLE
            model.computeIIS()
            output_file = os.path.join(COOPT_PATH, f"model_{dtype}.ilp")
            model.write(output_file)
            print(f"Model for {dtype} is infeasible; see {output_file} for details.")
            continue
        
        # 4) Collect results & export to CSV
        hourly_renewables_df = hourly_renewables.set_index("Name").T
        hourly_renewables_df.index = T_range
        
        # Thermal production - use binary solution
        production_results = {
            t: {g: binary_solution['p'][(g,t)] for g in generators.index} for t in T_range
        }
        production_df = pd.DataFrame(production_results).T
        production_df.columns = generators["Name"].values

        # Pumped Storage - show generation and pumping separately - use binary solution
        ps_gen_results = {
            t: {h: binary_solution['p_gen_ps'][(h,t)] for h in ps_units.index} for t in T_range
        }
        ps_pump_results = {
            t: {h: -binary_solution['p_pump_ps'][(h,t)] for h in ps_units.index} for t in T_range
        }  # Negative for pumping
        
        ps_gen_df = pd.DataFrame(ps_gen_results).T
        ps_gen_df.columns = [name + "_Gen" for name in ps_units["Name"].values]
        
        ps_pump_df = pd.DataFrame(ps_pump_results).T
        ps_pump_df.columns = [name + "_Pump" for name in ps_units["Name"].values]
        
        # Also calculate net for backward compatibility
        ps_net_results = {
            t: {h: binary_solution['p_gen_ps'][(h,t)] - binary_solution['p_pump_ps'][(h,t)] for h in ps_units.index} for t in T_range
        }
        ps_net_df = pd.DataFrame(ps_net_results).T
        ps_net_df.columns = ps_units["Name"].values

        # Merge all results - prevent duplicating columns
        # First combine thermal and pumped storage results
        all_results_df = pd.concat([production_df, ps_gen_df, ps_pump_df, ps_net_df], axis=1)
        
        bess_gen_results = {
            t: {i: binary_solution['p_discharge_bess'][(i,t)] for i in bess_units.index} for t in T_range
        }

        bess_pump_results = {
            t: {i: -binary_solution['p_charge_bess'][(i,t)] for i in bess_units.index} for t in T_range
        }  # Negative for pumping

        bess_gen_df = pd.DataFrame(bess_gen_results).T
        bess_gen_df.columns = [name + "_Discharge" for name in bess_units["Name"].values]
        bess_pump_df = pd.DataFrame(bess_pump_results).T
        bess_pump_df.columns = [name + "_Charge" for name in bess_units["Name"].values]

        #Also calculate net
        bess_net_results = {
            t: {i: binary_solution['p_discharge_bess'][(i,t)] - binary_solution['p_charge_bess'][(i,t)] for i in bess_units.index} for t in T_range
        }
        bess_net_df = pd.DataFrame(bess_net_results).T
        bess_net_df.columns = bess_units["Name"].values

        # Add BESS results to all_results_df
        all_results_df = pd.concat([all_results_df, bess_gen_df, bess_pump_df, bess_net_df], axis=1)

        # Now carefully add renewables (avoid duplicating columns)
        for col in hourly_renewables_df.columns:
            if col not in all_results_df.columns:
                all_results_df[col] = hourly_renewables_df[col]

        # Save final production results
        output_file = os.path.join(COOPT_PATH, f"generator_production_results_{dtype}.csv")
        all_results_df.to_csv(output_file)
        print(f"Generator production results saved to {output_file}")
        
        # Save state of charge timeseries for pumped storage units - use binary solution
        ps_soc_results = {
            t: {h: binary_solution['soc_ps'][(h,t)] for h in ps_units.index} for t in T_range
        }
        ps_soc_df = pd.DataFrame(ps_soc_results).T
        ps_soc_df.columns = [name + "_SoC" for name in ps_units["Name"].values]
        output_file = os.path.join(COOPT_PATH, f"ps_timeseries_{dtype}.csv")
        ps_soc_df.to_csv(output_file)
        print(f"Pumped storage timeseries saved to {output_file}")

        # Also, save total demand for plotting
        total_hourly_demand = hourly_demand.sum(axis=0)
        demand_df = pd.DataFrame(total_hourly_demand, columns=["Demand"])
        output_file = os.path.join(COOPT_PATH, f"hourly_demand_{dtype}.csv")
        demand_df.to_csv(output_file)
        print(f"Hourly demand saved to {output_file}")

        # Save marginal cost information (optional) - use binary solution
        rows = []
        for g in generators.index:
            hrc_key = generators.loc[g,"HeatRateCurve"]
            for t in T_range:
                if hrc_key in segment_data:
                    # Get the segments for this generator
                    segments = segment_data[hrc_key]
                    dispatch = binary_solution['p'][(g,t)]
                    
                    if dispatch > 1e-6:  # Generator is dispatched
                        # For dispatched generators, find the appropriate segment
                        for (slope, intercept) in segments:
                            total_cost = slope*generators.loc[g,"Price"]*dispatch + intercept
                            MC = slope * generators.loc[g,"Price"] + intercept
                            rows.append({
                                "Generator": g,
                                "Name": generators.loc[g,"Name"] if "Name" in generators.columns else g,
                                "Interval": t,
                                "DispatchMW": dispatch,
                                "TotalCost": total_cost,
                                "MC": MC,
                            })
                    else:  # Generator is not dispatched
                        # For non-dispatched generators, use the first segment for MC
                        # but set total cost to 0
                        first_segment = segments[0]  # Get the first segment (lowest output level)
                        slope, intercept = first_segment
                        MC = slope * generators.loc[g,"Price"] + intercept
                        rows.append({
                            "Generator": g,
                            "Name": generators.loc[g,"Name"] if "Name" in generators.columns else g,
                            "Interval": t,
                            "DispatchMW": 0.0,
                            "TotalCost": 0.0,  # Total cost is 0 since not dispatched
                            "MC": MC,  # MC is based on the first segment
                        })
        marginal_cost_df = pd.DataFrame(rows)
        output_file = os.path.join(COOPT_PATH, f"marginal_costs_{dtype}.csv")
        marginal_cost_df.to_csv(output_file, index=False)
        print(f"Marginal costs saved to {output_file}")
        
        # Save objective value of co-optimization for this day type
        # Save objective value (single number) so compare_costs_full.py can pick it up
        # Use the binary objective value for accurate cost reporting
        output_file = os.path.join(COOPT_PATH, f"coopt_objective_{dtype}.txt")
        with open(output_file, "w") as objfile:
            objfile.write(str(binary_obj_val))
        print(f"Co-optimization objective saved to {output_file}")

        # ---------------------------------------------------------------
        # EXTRA OUTPUT ①: system‑level cost buckets (EUR) – binary model
        # ---------------------------------------------------------------
        fuel   = dt * sum(c_g_bin[g, t].X for g in generators.index for t in T96)
        no_ld  = dt * sum(
                    generators.at[g, "K_eur"] * w_bin[g, h].X
                    for g in generators.index if "K_eur" in generators.columns
                    for h in H24)
        start  =       sum(
                    generators.at[g, "S_eur"] * z_bin[g, h].X
                    for g in generators.index if "S_eur" in generators.columns
                    for h in H24)
        pos_ls = dt * 3000 * sum(l_bin[t].X     for t in T96)
        neg_ls = dt *  100 * sum(l_neg_bin[t].X for t in T96)

        cost_df = pd.DataFrame([{
            "Fuel": fuel, "NoLoad": no_ld, "Startup": start,
            "PosShed": pos_ls, "NegShed": neg_ls, "Total": model_bin.ObjVal
        }])
        cost_file = os.path.join(COOPT_PATH, f"cost_components_{dtype}.csv")
        cost_df.to_csv(cost_file, index=False)
        print(f"Cost components saved to {cost_file}")

        # ---------------------------------------------------------------
        # EXTRA OUTPUT ②: unit‑level KPIs – binary model
        # ---------------------------------------------------------------
        kpi_rows = []
        for g in generators.index:
            # 15‑min power trajectory for this generator
            power_qh = [p_bin[g, t].X for t in T96]

            # Energy [MWh] = Σ (MW * dt)
            energy_mwh = dt * sum(power_qh)

            # Run‑hours counted at quarter‑hour resolution
            run_hours = dt * sum(1 for mw in power_qh if mw > 1e-3)

            # Start‑ups detected from 15‑min trajectory
            starts = sum(
                1
                for idx, mw in enumerate(power_qh)
                if mw > 1e-3 and (idx == 0 or power_qh[idx - 1] <= 1e-3)
            )

            # Optional capacity factor (if Pmax is available in generator data)
            p_max = generators.at[g, "Pmax"] if "Pmax" in generators.columns else None
            cap_factor = energy_mwh / (p_max * 24) if p_max else None

            kpi_rows.append([g, starts, run_hours, energy_mwh, cap_factor])

        kpi_df = pd.DataFrame(
            kpi_rows,
            columns=["Generator", "Starts", "RunHours", "EnergyMWh", "CapacityFactor"],
        )
        kpi_file = os.path.join(COOPT_PATH, f"unit_kpi_{dtype}.csv")
        kpi_df.to_csv(kpi_file, index=False)
        print(f"Unit‑level KPIs saved to {kpi_file}")
        
        lambda_values = {}
        reserve_duals = {}
        if model.status == 2:  # GRB.OPTIMAL
            # Energy‑balance duals (only defined for the LP model)
            if model.IsMIP == 0:                       # confirm it is an LP
                for t in range(96):
                    constr = model.getConstrByName(f"E_2_EnergyBalance_{t}")
                    if constr and hasattr(constr, "Pi"):
                        lambda_values[t] = constr.Pi / TIME_STEP_HOURS   # €/MWh
                    else:
                        lambda_values[t] = 0.0

            # Reserve requirement duals
            for t in range(96):
                reserve_duals[t] = {}
                # aFRR requirements
                try:
                    constr_up = model.getConstrByName(f"E9_UpResReq_aFRR_{t}")
                    if constr_up and hasattr(constr_up, 'Pi'):
                        reserve_duals[t]['aFRR_up'] = constr_up.Pi
                    else:
                        reserve_duals[t]['aFRR_up'] = 0.0
                except:
                    reserve_duals[t]['aFRR_up'] = 0.0

                try:
                    constr_down = model.getConstrByName(f"E9_DownResReq_aFRR_{t}")
                    if constr_down and hasattr(constr_down, 'Pi'):
                        reserve_duals[t]['aFRR_down'] = constr_down.Pi
                    else:
                        reserve_duals[t]['aFRR_down'] = 0.0
                except:
                    reserve_duals[t]['aFRR_down'] = 0.0

                # mFRR requirements
                try:
                    constr_up = model.getConstrByName(f"E9_UpResReq_total_{t}")
                    if constr_up and hasattr(constr_up, 'Pi'):
                        reserve_duals[t]['mFRR_up'] = constr_up.Pi
                    else:
                        reserve_duals[t]['mFRR_up'] = 0.0
                except:
                    reserve_duals[t]['mFRR_up'] = 0.0

                try:
                    constr_down = model.getConstrByName(f"E9_DownResReq_total_{t}")
                    if constr_down and hasattr(constr_down, 'Pi'):
                        reserve_duals[t]['mFRR_down'] = constr_down.Pi
                    else:
                        reserve_duals[t]['mFRR_down'] = 0.0
                except:
                    reserve_duals[t]['mFRR_down'] = 0.0

        # Save energy balance duals
        if lambda_values:
            lambda_df = pd.DataFrame(list(lambda_values.items()),
                                     columns=["Interval", "Lambda_EUR_per_MWh"])
            output_file = os.path.join(COOPT_PATH, f"lambda_values_{dtype}.csv")
            lambda_df.to_csv(output_file, index=False)
            print(f"Lambda values saved to {output_file}")
        
        # Save reserve requirement duals
        reserve_duals_df = pd.DataFrame.from_dict(reserve_duals, orient='index')
        reserve_duals_df.index.name = 'Interval'
        output_file = os.path.join(COOPT_PATH, f"reserve_duals_{dtype}.csv")
        reserve_duals_df.to_csv(output_file)
        print(f"Reserve duals saved to {output_file}")

        # ---------------------------------------------------------------
        # EXTRA STEP: LP with binaries fixed to integer solution – duals ②
        # ---------------------------------------------------------------
        print("Running LP with binaries fixed to integer solution for consistent duals...")

        # Build an LP model again (all integrality relaxed)
        model_fix, p_fix, w_fix, z_fix, c_g_fix, l_fix, l_neg_fix, p_gen_ps_fix, p_pump_ps_fix, soc_ps_fix, p_charge_bess_fix, p_discharge_bess_fix, soc_bess_fix = build_cooptimization_model(
            data, use_lp_relaxation=True, day_type=dtype
        )

        # Fix commitment variables to the integer (binary) solution
        for g in generators.index:
            for h in H24:
                w_fix[g, h].LB = w_bin[g, h].X
                w_fix[g, h].UB = w_bin[g, h].X
                z_fix[g, h].LB = z_bin[g, h].X
                z_fix[g, h].UB = z_bin[g, h].X

        # Solve the fixed‑commitment LP
        model_fix.setParam("InfUnbdInfo", 1)
        model_fix.setParam("DualReductions", 0)
        model_fix.setParam("OutputFlag", 0)
        model_fix.optimize()

        lambda_values_fixed = {}
        reserve_duals_fixed = {}
        if model_fix.status == 2:  # GRB.OPTIMAL
            # Energy‑balance duals
            for t in T96:
                constr = model_fix.getConstrByName(f"E_2_EnergyBalance_{t}")
                if constr and hasattr(constr, "Pi"):
                    lambda_values_fixed[t] = constr.Pi / TIME_STEP_HOURS
                else:
                    lambda_values_fixed[t] = 0.0

            # Reserve requirement duals
            for t in T96:
                reserve_duals_fixed[t] = {}
                try:
                    constr_up = model_fix.getConstrByName(f"E9_UpResReq_aFRR_{t}")
                    reserve_duals_fixed[t]["aFRR_up"] = constr_up.Pi if constr_up and hasattr(constr_up, "Pi") else 0.0
                except:
                    reserve_duals_fixed[t]["aFRR_up"] = 0.0

                try:
                    constr_down = model_fix.getConstrByName(f"E9_DownResReq_aFRR_{t}")
                    reserve_duals_fixed[t]["aFRR_down"] = constr_down.Pi if constr_down and hasattr(constr_down, "Pi") else 0.0
                except:
                    reserve_duals_fixed[t]["aFRR_down"] = 0.0

                try:
                    constr_up = model_fix.getConstrByName(f"E9_UpResReq_total_{t}")
                    reserve_duals_fixed[t]["mFRR_up"] = constr_up.Pi if constr_up and hasattr(constr_up, "Pi") else 0.0
                except:
                    reserve_duals_fixed[t]["mFRR_up"] = 0.0

                try:
                    constr_down = model_fix.getConstrByName(f"E9_DownResReq_total_{t}")
                    reserve_duals_fixed[t]["mFRR_down"] = constr_down.Pi if constr_down and hasattr(constr_down, "Pi") else 0.0
                except:
                    reserve_duals_fixed[t]["mFRR_down"] = 0.0

        # Save fixed‑commitment lambda values
        if lambda_values_fixed:
            lambda_df_fixed = pd.DataFrame(
                list(lambda_values_fixed.items()),
                columns=["Interval", "Lambda_EUR_per_MWh"],
            )
            output_file = os.path.join(COOPT_PATH, f"lambda_values_fixed_{dtype}.csv")
            lambda_df_fixed.to_csv(output_file, index=False)
            print(f"Fixed‑commitment lambda values saved to {output_file}")

        # Save fixed‑commitment reserve duals
        reserve_duals_fixed_df = pd.DataFrame.from_dict(reserve_duals_fixed, orient="index")
        reserve_duals_fixed_df.index.name = "Interval"
        output_file = os.path.join(COOPT_PATH, f"reserve_duals_fixed_{dtype}.csv")
        reserve_duals_fixed_df.to_csv(output_file)
        print(f"Fixed‑commitment reserve duals saved to {output_file}")
        
        if VERBOSE:
            print(f"\nDEBUG: Sample of reserve requirement duals:")
            print(reserve_duals_df.head())

        # Save load shedding values (both positive and negative)
        load_shedding = {
            t: {'positive': l[t].X, 'negative': l_neg[t].X} for t in T_range
        }
        load_shedding_df = pd.DataFrame.from_dict(load_shedding, orient='index')
        load_shedding_df.index.name = 'Interval'
        output_file = os.path.join(COOPT_PATH, f"load_shedding_{dtype}.csv")
        load_shedding_df.to_csv(output_file)
        print(f"Load shedding values saved to {output_file}")

    print("All runs complete.")

if __name__ == "__main__":
    main()
