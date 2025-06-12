import gurobipy as gp
from gurobipy import GRB
import pandas as pd
# Toggle extra console output
VERBOSE = False
# Duration of one modelling interval (15 minutes) expressed in hours.
TIME_STEP_HOURS = 0.25

def build_cooptimization_model(data, use_lp_relaxation, day_type):
    """
    Builds the day-ahead co-optimization model (energy + reserves + pumped storage).
    Returns the constructed Gurobi model and any additional results 
    (like T range, references to data frames, etc.).
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all input data
    use_lp_relaxation : bool, optional
        If True, uses continuous variables instead of binary for commitment status,
        which allows for retrieving dual values. Default is False.
    """
    # Extract the needed data frames from data dict
    generators = data["generators"]
    ps_units   = data["ps_units"]
    segment_data = data["segment_data"]
    be_reserves         = data["be_reserves"]
    bess_units = data["bess_units"]



    # mFRR and aFRR tables from prepare_data()
    up_products_mFRR    = data["up_products_mFRR"]
    down_products_mFRR  = data["down_products_mFRR"]
    up_products_aFRR    = data["up_products_aFRR"]
    down_products_aFRR  = data["down_products_aFRR"]

    # unified tables / index lists
    up_products   = pd.concat([up_products_mFRR,   up_products_aFRR])
    down_products = pd.concat([down_products_mFRR, down_products_aFRR])

    up_list_total   = up_products.index.tolist()     # IDs of ALL up products
    down_list_total = down_products.index.tolist()   # IDs of ALL down products

    if VERBOSE:
        print(f"BE reserves: {be_reserves}")    
    hourly_reserve_req = data["hourly_reserve_req"]
    hourly_demand = data["hourly_demand"]
    hourly_renewables = data["hourly_renewables"]
    
    # Safely extract reserve requirement data and initialize if missing
    hourly_up_reserve_mFRR = data.get("hourly_up_reserve_mFRR", {})
    hourly_down_reserve_mFRR = data.get("hourly_down_reserve_mFRR", {})
    hourly_up_reserve_aFRR = data.get("hourly_up_reserve_aFRR", {})
    hourly_down_reserve_aFRR = data.get("hourly_down_reserve_aFRR", {})
    
    # Create model
    model = gp.Model("Day-Ahead Co-Optimization")

    # Time set = 24 hours
    T = range(96)
    
    def hour_of_t(t):
        return t // 4
        
    # Ensure all time periods have entries in reserve dictionaries (even if empty)
    for t in T:
        if t not in hourly_up_reserve_mFRR:
            hourly_up_reserve_mFRR[t] = pd.DataFrame()
        if t not in hourly_down_reserve_mFRR:
            hourly_down_reserve_mFRR[t] = pd.DataFrame()
        if t not in hourly_up_reserve_aFRR:
            hourly_up_reserve_aFRR[t] = pd.DataFrame()
        if t not in hourly_down_reserve_aFRR:
            hourly_down_reserve_aFRR[t] = pd.DataFrame()
            
    # Print data structure info for debugging
    if VERBOSE:
        print(f"Up mFRR data contains {len(hourly_up_reserve_mFRR)} time periods")
        print(f"Down mFRR data contains {len(hourly_down_reserve_mFRR)} time periods")
        print(f"Up aFRR data contains {len(hourly_up_reserve_aFRR)} time periods")
        print(f"Down aFRR data contains {len(hourly_down_reserve_aFRR)} time periods")

    # Decision variables
    p = model.addVars(generators.index, T, lb=0, name="p")
    
    # Use continuous variables instead of binary if LP relaxation is enabled
    if use_lp_relaxation:
        w = model.addVars(generators.index, range(24), lb=0, ub=1, name="w")
        z = model.addVars(generators.index, range(24), lb=0, ub=1, name="z")
    else:
        w = model.addVars(generators.index, range(24), vtype=GRB.BINARY, name="w")
        z = model.addVars(generators.index, range(24), vtype=GRB.BINARY, name="z")
        
    c_g = model.addVars(generators.index, T, lb=0, name="c_g")
    l   = model.addVars(T, lb=0, name="l")  # Positive load shedding (demand > generation)
    l_neg = model.addVars(T, lb=0, name="l_neg")  # Negative load shedding (generation > demand)

    soc_ps = model.addVars(ps_units.index, T, lb=0,
                           ub={i: ps_units.loc[i, "ReservoirCap"] for i in ps_units.index},
                           name="soc_ps")
    p_gen_ps = model.addVars(ps_units.index, T, lb=0, 
                             ub={i: ps_units.loc[i, "GenMax"] for i in ps_units.index},
                             name="p_gen_ps")
    p_pump_ps = model.addVars(ps_units.index, T, lb=0,
                              ub={i: ps_units.loc[i, "PumpMax"] for i in ps_units.index},
                              name="p_pump_ps")
    
    p_discharge_bess = model.addVars(bess_units.index, T, lb = 0,
                                     ub={i: bess_units.loc[i, "PowerCapacity"] for i in bess_units.index},
                                     name="p_discharge_bess")
    
    p_charge_bess = model.addVars(bess_units.index, T, lb=0,
                                  ub={i:bess_units.loc[i, "PowerCapacity"] for i in bess_units.index},
                                    name="p_charge_bess")
    soc_bess         = model.addVars(bess_units.index, T, lb=0, 
                                 ub={i: bess_units.loc[i, "EnergyCapacity"] for i in bess_units.index},
                                 name="soc_bess")
    
    reserve_idxs = be_reserves.index.tolist()
    s_res = model.addVars(generators.index, T, reserve_idxs, lb=0, name="s_res")
    s_res_ps = model.addVars(ps_units.index, T, reserve_idxs, lb=0, name="s_res_ps")
    s_res_bess = model.addVars(bess_units.index, T, reserve_idxs, lb=0, name="s_res_bess")
    # Check total demand vs capacity
    total_max_capacity = generators["P_max"].sum()
    total_demand = hourly_demand.sum().sum()
    if VERBOSE:
        print(f"Total max capacity: {total_max_capacity}, total demand: {total_demand}")

    # 4) COST CONSTRAINTS (Piecewise)
    for g in generators.index:
        hrc_key = generators.loc[g,"HeatRateCurve"]
        for t in T:
            seg_costs = []
            if hrc_key in segment_data:
                for (slope, intercept) in segment_data[hrc_key]:
                    cost_var = model.addVar()
                    model.addConstr(
                        cost_var == slope*generators.loc[g,"Price"]*(p[g,t] + gp.quicksum(s_res[g,t,i] for i in reserve_idxs)) + intercept
                    )
                    seg_costs.append(cost_var)
                for cvar in seg_costs:
                    model.addConstr(c_g[g,t] >= cvar)
            else:
                # If no piecewise data, just do linear cost or set c_g=0
                model.addConstr(c_g[g,t] == generators.loc[g,"Price"]*p[g,t])

    model.update()

    # 5) OBJECTIVE
    # --- objective with consistent quarter‑hour scaling -----------------------
    dt = TIME_STEP_HOURS  # 0.25 h per time‑step

    model.setObjective(
          # Fuel + piece‑wise intercepts
          dt * gp.quicksum(c_g[g, t] for g in generators.index for t in T)

        # No‑load cost (€/h) × hours
        + dt * gp.quicksum(
              generators.loc[g, 'K_eur'] * w[g, hour_of_t(t)]
              for g in generators.index if 'K_eur' in generators.columns
              for t in T
          )

        # Startup cost (lump‑sum, *not* time‑scaled)
        + gp.quicksum(
              generators.loc[g, 'S_eur'] * z[g, hour_of_t(t)]
              for g in generators.index if 'S_eur' in generators.columns
              for t in T
          )

        # Positive / negative load‑shedding penalties
        + dt * 3000 * gp.quicksum(l[t]      for t in T)   # VOLL
        + dt * 100  * gp.quicksum(l_neg[t] for t in T)  # over‑generation spill

        + TIME_STEP_HOURS * gp.quicksum(bess_units.loc[i, "CycleCost"] * (p_charge_bess[i,t] + p_discharge_bess[i,t])
                                          for i in bess_units.index for t in T),
        GRB.MINIMIZE
    )

    

    

    # 7.1 Energy Balance
    for t in T:
        model.addConstr(
            gp.quicksum(p[g,t] for g in generators.index)
          + gp.quicksum(hourly_renewables.loc[r, t] for r in hourly_renewables.index)
          + gp.quicksum(p_gen_ps[i,t] for i in ps_units.index)
          - gp.quicksum(p_pump_ps[i,t] for i in ps_units.index)
          + gp.quicksum(p_discharge_bess[i,t] for i in bess_units.index)
            - gp.quicksum(p_charge_bess[i,t] for i in bess_units.index)
          + l[t] - l_neg[t]  # Added negative load shedding
          == hourly_demand[t],
            name=f"E_2_EnergyBalance_{t}"
        )

    # Create lists for specific reserve types
    up_list_mFRR = up_products_mFRR.index.tolist()
    down_list_mFRR = down_products_mFRR.index.tolist()
    up_list_aFRR = up_products_aFRR.index.tolist()
    down_list_aFRR = down_products_aFRR.index.tolist()
    
    # Now implementing all constraints from E.8 to E.27 in order
    
    # E.8: Reserve Requirements (maximum between products)
    # Note: This is implemented implicitly through the individual reserve product constraints
    
    # E.9: System-level aFRR requirements
    for t in T:
        try:
            up_aFRR_req = 0
            for j in up_list_aFRR:
                if j in hourly_reserve_req and not hourly_reserve_req[j].empty and t in hourly_reserve_req[j].columns:
                    up_aFRR_req += hourly_reserve_req[j].loc[:, t].sum()
                    
            model.addConstr(
                gp.quicksum(s_res[g, t, j] for g in generators.index if generators.loc[g, "FuelType"] != "NUCLEAR" for j in up_list_aFRR)
              + gp.quicksum(s_res_ps[u, t, j] for u in ps_units.index for j in up_list_aFRR)
              + gp.quicksum(s_res_bess[i,t,j] for i in bess_units.index for j in up_list_aFRR)
              >= up_aFRR_req,
              name=f"E9_UpResReq_aFRR_{t}"
            )
        except (KeyError, AttributeError) as e:
            print(f"Warning: Error with up aFRR system requirements for time {t}: {e}")
            model.addConstr(
                gp.quicksum(s_res[g, t, j] for g in generators.index if generators.loc[g, "FuelType"] != "NUCLEAR" for j in up_list_aFRR)
              + gp.quicksum(s_res_ps[u, t, j] for u in ps_units.index for j in up_list_aFRR)
              + gp.quicksum(s_res_bess[i,t,j] for i in bess_units.index for j in up_list_aFRR)
              >= 0,
              name=f"E9_UpResReq_aFRR_{t}"
            )
            
        try:
            down_aFRR_req = 0
            for j in down_list_aFRR:
                if j in hourly_reserve_req and not hourly_reserve_req[j].empty and t in hourly_reserve_req[j].columns:
                    down_aFRR_req += hourly_reserve_req[j].loc[:, t].sum()
                    
            model.addConstr(
                gp.quicksum(s_res[g, t, j] for g in generators.index if generators.loc[g, "FuelType"] != "NUCLEAR" for j in down_list_aFRR)
              + gp.quicksum(s_res_ps[u, t, j] for u in ps_units.index for j in down_list_aFRR)
              + gp.quicksum(s_res_bess[i,t,j] for i in bess_units.index for j in down_list_aFRR)
              >= down_aFRR_req,
              name=f"E9_DownResReq_aFRR_{t}"
            )
        except (KeyError, AttributeError) as e:
            print(f"Warning: Error with down aFRR system requirements for time {t}: {e}")
            model.addConstr(
                gp.quicksum(s_res[g, t, j] for g in generators.index if generators.loc[g, "FuelType"] != "NUCLEAR" for j in down_list_aFRR)
              + gp.quicksum(s_res_ps[u, t, j] for u in ps_units.index for j in down_list_aFRR)
              + gp.quicksum(s_res_bess[i,t,j] for i in bess_units.index for j in down_list_aFRR)
              >= 0,
              name=f"E9_DownResReq_aFRR_{t}"
            )

    # ------------------------------------------------------------------
    # mFRR requirements (those can be met by aFRR, which is faster, too)
    # ------------------------------------------------------------------
    for t in T:
        # ---- Upward mFRR requirement ----------------------------------
        up_mFRR_req = 0
        for j in up_list_mFRR:
            if j in hourly_reserve_req \
            and not hourly_reserve_req[j].empty \
            and t in hourly_reserve_req[j].columns:
                up_mFRR_req += hourly_reserve_req[j].loc[:, t].sum()

        model.addConstr(
            gp.quicksum(s_res[g, t, j]    for g in generators.index
                                            if generators.loc[g,"FuelType"]!="NUCLEAR"
                                            for j in up_list_mFRR)
        + gp.quicksum(s_res_ps[u,t,j]   for u in ps_units.index
                                            for j in up_list_mFRR)
        + gp.quicksum(s_res_bess[i,t,j] for i in bess_units.index
                                            for j in up_list_mFRR)

        + gp.quicksum(s_res[g,t,r] for g in generators.index
                                    if generators.loc[g,"FuelType"]!="NUCLEAR"
                                    for r in up_list_aFRR)
        + gp.quicksum(s_res_ps[u,t,r] for u in ps_units.index
                                    for r in up_list_aFRR)
        + gp.quicksum(s_res_bess[i,t,r] for i in bess_units.index
                                    for r in up_list_aFRR)

        >= up_mFRR_req,
        name=f"E9_UpResReq_mFRR_{t}"
        )

        # ---- Downward mFRR requirement --------------------------------
        down_mFRR_req = 0
        for j in down_list_mFRR:
            if j in hourly_reserve_req \
            and not hourly_reserve_req[j].empty \
            and t in hourly_reserve_req[j].columns:
                down_mFRR_req += hourly_reserve_req[j].loc[:, t].sum()

        model.addConstr(
            gp.quicksum(s_res[g, t, j]    for g in generators.index
                                            if generators.loc[g,"FuelType"]!="NUCLEAR"
                                            for j in down_list_mFRR)
        + gp.quicksum(s_res_ps[u,t,j]   for u in ps_units.index
                                            for j in down_list_mFRR)
        + gp.quicksum(s_res_bess[i,t,j] for i in bess_units.index
                                            for j in down_list_mFRR)
                            
        + gp.quicksum(s_res[g,t,r] for g in generators.index
                                    if generators.loc[g,"FuelType"]!="NUCLEAR"
                                    for r in down_list_aFRR)
        + gp.quicksum(s_res_ps[u,t,r] for u in ps_units.index
                                    for r in down_list_aFRR)
        + gp.quicksum(s_res_bess[i,t,r] for i in bess_units.index
                                    for r in down_list_aFRR)
        
        >= down_mFRR_req,
        name=f"E9_DownResReq_mFRR_{t}"
        )

    # ------------------------------------------------------------------
    # >>>>>  UPWARD mFRR UNIT-LEVEL CAPACITY (mirror of E.10)  <<<<<
    # ------------------------------------------------------------------
    for g in generators.index:
        for t in T:
            dt_mfrr = max([up_products_mFRR.loc[i,"DeliveryTime"]
                        for i in up_list_mFRR]) if up_list_mFRR else 0
            model.addConstr(
                gp.quicksum(s_res[g, t, i] for i in up_list_mFRR)
                <= min(generators.loc[g,"P_max"],
                    dt_mfrr * generators.loc[g,"R_max"]),
                name=f"E10_mFRR_UpCapacity_{g}_{t}"
            )
    
    
    # E.10: mFRR Capacity Constraints
    for g in generators.index:
        for t in T:
            dt_mfrr = max([down_products_mFRR.loc[i, "DeliveryTime"] for i in down_list_mFRR]) if down_list_mFRR else 0
            model.addConstr(
                gp.quicksum(s_res[g, t, i] for i in down_list_mFRR) <= 
                min(generators.loc[g, "P_max"], dt_mfrr * abs(generators.loc[g, "R_min"])),
                name=f"E10_mFRR_DownCapacity_{g}_{t}"
            )
    
    # E.11: aFRR Capacity Constraints
    for g in generators.index:
        for t in T:
            dt_afrr = max([down_products_aFRR.loc[i, "DeliveryTime"] for i in down_list_aFRR]) if down_list_aFRR else 0
            model.addConstr(
                gp.quicksum(s_res[g, t, i] for i in down_list_aFRR) <= 
                min(generators.loc[g, "P_max"], dt_afrr * abs(generators.loc[g, "R_min"])),
                name=f"E11_aFRR_DownCapacity_{g}_{t}"
            )
    
    # E.12: FAST generators aFRR capacity
    for g in generators.index:
        if generators.loc[g, "GeneratorType"] == "FAST":
            for t in T:
                model.addConstr(
                    p[g, t] + gp.quicksum(s_res[g, t, i] for i in up_list_aFRR) <= 
                    generators.loc[g, "P_max"] * w[g, hour_of_t(t)],
                    name=f"E12_FastGen_aFRR_Capacity_{g}_{t}"
                )
    
    # E.13: FAST generators total capacity (mFRR + aFRR)
    for g in generators.index:
        if generators.loc[g, "GeneratorType"] == "FAST":
            for t in T:
                model.addConstr(
                    p[g, t] + gp.quicksum(s_res[g, t, i] for i in up_list_mFRR) + 
                    gp.quicksum(s_res[g, t, i] for i in up_list_aFRR) <= 
                    generators.loc[g, "P_max"],
                    name=f"E13_FastGen_Total_Capacity_{g}_{t}"
                )
    
    # E.14: SLOW generators total capacity
    for g in generators.index:
        if generators.loc[g, "GeneratorType"] == "SLOW":
            for t in T:
                model.addConstr(
                    p[g, t] + gp.quicksum(s_res[g, t, i] for i in up_list_mFRR) + 
                    gp.quicksum(s_res[g, t, i] for i in up_list_aFRR) <= 
                    generators.loc[g, "P_max"] * w[g, hour_of_t(t)],
                    name=f"E14_SlowGen_Total_Capacity_{g}_{t}"
                )
    
    # E.15: Lower bound constraint for all generators
    for g in generators.index:
        for t in T:
            model.addConstr(
                p[g, t] - gp.quicksum(s_res[g, t, i] for i in down_list_mFRR) - 
                gp.quicksum(s_res[g, t, i] for i in down_list_aFRR) >= 
                generators.loc[g, "P_min"] * w[g, hour_of_t(t)],
                name=f"E15_Gen_Lower_Bound_{g}_{t}"
            )
    
    # E.16-E.17: Ramping constraints with reserves
    ramp_window = 15  # 15-minute intervals
    ramp_up_slack = model.addVars(generators.index, T, lb=0, name="ramp_up_slack")
    ramp_down_slack = model.addVars(generators.index, T, lb=0, name="ramp_down_slack")
    
    for g in generators.index:
        for t in range(1, 96):
            # E.16: Upward ramping with reserves
            model.addConstr(
                (p[g, t] - p[g, t-1]) + 
                gp.quicksum(s_res[g, t, i] for i in up_list_mFRR) + 
                gp.quicksum(s_res[g, t, i] for i in up_list_aFRR) <= 
                ramp_window * generators.loc[g, "R_max"] + ramp_up_slack[g,t],
                name=f"E16_RampUp_WithReserves_{g}_{t}"
            )
            
            # E.17: Downward ramping with reserves
            model.addConstr(
                (p[g, t-1] - p[g, t]) + 
                gp.quicksum(s_res[g, t, i] for i in down_list_mFRR) + 
                gp.quicksum(s_res[g, t, i] for i in down_list_aFRR) >= 
                ramp_window * abs(generators.loc[g, "R_min"]) - ramp_down_slack[g,t],
                name=f"E17_RampDown_WithReserves_{g}_{t}"
            )
    
    # E.18: Minimum up time constraint
    for g in generators.index:
        UT = int(generators.loc[g,'UT_g'])
        for t in T:
            if t >= UT-1:
                model.addConstr(
                    gp.quicksum(z[g, hour_of_t(q)] for q in range(t-UT+1, t+1)) <= w[g, hour_of_t(t)],
                    name=f"E18_MinUpTime_{g}_{t}"
                )
    
    # E.19: Minimum down time constraint
    for g in generators.index:
        DT = int(generators.loc[g,'DT_g'])
        for t in T:
            if t <= (95 - DT):
                model.addConstr(
                    gp.quicksum(z[g, hour_of_t(q)] for q in range(t+1, t+DT+1)) <= 1 - w[g, hour_of_t(t)],
                    name=f"E19_MinDownTime_{g}_{t}"
                )
    
    # E.20: Startup binary variable constraint
    for g in generators.index:
        for t in range(24):  # Using T_60 as the hourly periods
            model.addConstr(
                z[g, t] <= 1,
                name=f"E20_Startup_Binary_{g}_{t}"
            )
    
    # E.21: Startup definition constraint
    for g in generators.index:
        for t in range(24):  # Using T_60 as the hourly periods
            if t > 0:
                model.addConstr(
                    z[g, t] >= w[g, t] - w[g, t-1],
                    name=f"E21_StartupDef_{g}_{t}"
                )
            else:
                model.addConstr(
                    z[g, 0] >= w[g, 0] - w[g, 23],
                    name=f"E21_StartupDef_{g}_0"
                )
    
    # E.22-E.23: Pumped Storage state equations
    for i in ps_units.index:
        # Initial state of charge constraint
        model.addConstr(
            soc_ps[i, 0] == ps_units.loc[i, "InitialSoC"],
            name=f"E22_PS_InitialSoC_{i}"
        )
        
        # Ensure final SoC matches initial SoC (daily cycle)
        model.addConstr(
            soc_ps[i, 95] >= ps_units.loc[i, "InitialSoC"],
            name=f"E22_PS_FinalSoC_{i}"
        )
        
        for t in range(1, 96):
            prev_t = t - 1
            
            # E.22 and E.23 combined into one reservoir balance constraint
            model.addConstr(
                soc_ps[i, t] == soc_ps[i, prev_t] + 
                ps_units.loc[i, "EtaPump"] * p_pump_ps[i, prev_t] - 
                (1/ps_units.loc[i, "EtaGen"]) * p_gen_ps[i, t],
                name=f"E22_E23_PS_StateEquation_{i}_{t}"
            )
            
            # Additional constraints to ensure proper behavior
            # Ensure SoC doesn't exceed reservoir capacity
            model.addConstr(
                soc_ps[i, t] <= ps_units.loc[i, "ReservoirCap"],
                name=f"E22_PS_MaxSoC_{i}_{t}"
            )
            
            # Cannot generate more than available energy
            model.addConstr(
                (1/ps_units.loc[i, "EtaGen"]) * p_gen_ps[i, t] <= soc_ps[i, prev_t],
                name=f"E22_PS_MaxGeneration_{i}_{t}"
            )
            
            # Cannot pump if reservoir is full
            model.addConstr(
                ps_units.loc[i, "EtaPump"] * p_pump_ps[i, t] <= ps_units.loc[i, "ReservoirCap"] - soc_ps[i, t],
                name=f"E22_PS_MaxPumping_{i}_{t}"
            )
        
        # Add binary variables to ensure pumping and generation don't happen simultaneously
        for t in T:
            # Use continuous variables instead of binary if LP relaxation is enabled
            if use_lp_relaxation:
                is_gen = model.addVar(lb=0, ub=1, name=f"is_gen_{i}_{t}")
            else:
                is_gen = model.addVar(vtype=GRB.BINARY, name=f"is_gen_{i}_{t}")
            
            # Cannot generate and pump at the same time
            model.addConstr(
                p_gen_ps[i, t] <= ps_units.loc[i, "GenMax"] * is_gen,
                name=f"E22_PS_GenMode_{i}_{t}"
            )
            model.addConstr(
                p_pump_ps[i, t] <= ps_units.loc[i, "PumpMax"] * (1 - is_gen),
                name=f"E22_PS_PumpMode_{i}_{t}"
            )
    
    # E.24-E.27: Pumped Storage ramping constraints
    for i in ps_units.index:
        for t in range(1, 96):
            prev_t = t - 1
            
            # E.24: Upward generation ramping with reserves
            model.addConstr(
                p_gen_ps[i, t] - p_gen_ps[i, prev_t] + 
                gp.quicksum(s_res_ps[i, t, j] for j in up_list_mFRR) +
                gp.quicksum(s_res_ps[i, t, j] for j in up_list_aFRR) <= 
                ramp_window * ps_units.loc[i, "R_max"],
                name=f"E24_PS_GenRampUp_WithReserves_{i}_{t}"
            )
            
            # E.25: Downward generation ramping with reserves
            model.addConstr(
                p_gen_ps[i, prev_t] - p_gen_ps[i, t] + 
                gp.quicksum(s_res_ps[i, t, j] for j in down_list_mFRR) +
                gp.quicksum(s_res_ps[i, t, j] for j in down_list_aFRR) >= 
                ramp_window * ps_units.loc[i, "R_min"],
                name=f"E25_PS_GenRampDown_WithReserves_{i}_{t}"
            )
            
            # E.26: Upward pumping ramping with reserves (d_z represents pumping)
            model.addConstr(
                p_pump_ps[i, t] - p_pump_ps[i, prev_t] + 
                gp.quicksum(s_res_ps[i, t, j] for j in up_list_mFRR) +
                gp.quicksum(s_res_ps[i, t, j] for j in up_list_aFRR) <= 
                ramp_window * ps_units.loc[i, "R_max"],
                name=f"E26_PS_PumpRampUp_WithReserves_{i}_{t}"
            )
            
            # E.27: Downward pumping ramping with reserves
            model.addConstr(
                p_pump_ps[i, prev_t] - p_pump_ps[i, t] + 
                gp.quicksum(s_res_ps[i, t, j] for j in down_list_mFRR) +
                gp.quicksum(s_res_ps[i, t, j] for j in down_list_aFRR) >= 
                ramp_window * ps_units.loc[i, "R_min"],
                name=f"E27_PS_PumpRampDown_WithReserves_{i}_{t}"
            )

    # Force nuclear units to have zero reserve provision
    for g in generators.index:
        if generators.loc[g, "FuelType"] == "NUCLEAR":
            for t in T:
                for i in reserve_idxs:
                    model.addConstr(
                        s_res[g, t, i] == 0,
                        name=f"E_NuclearNoReserve_{g}_{t}_{i}"
                    )
    ### Battery Energy Storage System (BESS) constraints

    # Initial state of charge and final state of charge constraints\
    #Ensuring final SoC >= initial SoC enforces that the battery ends the day at least as charged as it started (here to prevent end of horizon energy extraction)

    for i in bess_units.index:
        model.addConstr(
            soc_bess[i, 0] == bess_units.loc[i, "InitialSoC"],
            name=f"BESS_InitialSoC_{i}"
        )
        model.addConstr(
            soc_bess[i, 95] >= bess_units.loc[i, "InitialSoC"],
            name=f"BESS_FinalSoC_{i}"
        )

    #SoC Balance Equation

    for i in bess_units.index:
        for t in range(1, 96):
            prev = t - 1
            model.addConstr(
                soc_bess[i, t] == soc_bess[i, prev] 
                                + bess_units.loc[i, "ChargeEff"] * p_charge_bess[i, prev] * TIME_STEP_HOURS 
                                - (1 / bess_units.loc[i, "DischargeEff"]) * p_discharge_bess[i, t] * TIME_STEP_HOURS,
                name=f"BESS_SoC_Update_{i}_{t}"
            )

    #Non simultaneous charging and discharging
    # To enforce this in MILP, introduce a binary indicator (unless using LP relaxation) for mode in each interval, similar to the pumped storage is_gen.
    # --------------------------------------------------------------
#  BESS operating-mode, power-capacity and energy-availability
# --------------------------------------------------------------
    for i in bess_units.index:
        for t in range(1,96):
            
            # ---- 0/1 indicator: 1 = discharge mode, 0 = charge mode ----
            if use_lp_relaxation:
                mode_bess = model.addVar(lb=0, ub=1,
                                        name=f"mode_bess_{i}_{t}")
            else:
                mode_bess = model.addVar(vtype=GRB.BINARY,
                                        name=f"mode_bess_{i}_{t}")

            # Non-simultaneous charge / discharge
            model.addConstr(p_discharge_bess[i, t] <=
                            bess_units.loc[i, "PowerCapacity"] * mode_bess,
                            name=f"BESS_DischargeMode_{i}_{t}")
            model.addConstr(p_charge_bess[i, t]    <=
                            bess_units.loc[i, "PowerCapacity"] * (1 - mode_bess),
                            name=f"BESS_ChargeMode_{i}_{t}")

            # Power-capacity headroom for *reserve* offers
            model.addConstr(p_discharge_bess[i, t] +
                            gp.quicksum(s_res_bess[i, t, r]
                                        for r in up_list_total)
                            <= bess_units.loc[i, "PowerCapacity"],
                            name=f"BESS_PowerCap_Up_{i}_{t}")

            model.addConstr(p_charge_bess[i, t] +
                            gp.quicksum(s_res_bess[i, t, r]
                                        for r in down_list_total)
                            <= bess_units.loc[i, "PowerCapacity"],
                            name=f"BESS_PowerCap_Down_{i}_{t}")

            # Energy-availability for upward reserves & discharge
            model.addConstr(
                (1 / bess_units.loc[i, "DischargeEff"]) *
                (p_discharge_bess[i, t] +
                gp.quicksum(s_res_bess[i, t, r] for r in up_list_total)
                ) * TIME_STEP_HOURS
                <= soc_bess[i, t-1],
                name=f"BESS_EnergyAvail_Up_{i}_{t}"
            )

            # Empty-space availability for downward reserves & charge
            model.addConstr(
                bess_units.loc[i, "ChargeEff"] *
                (p_charge_bess[i, t] +
                gp.quicksum(s_res_bess[i, t, r] for r in down_list_total)
                ) * TIME_STEP_HOURS
                <= bess_units.loc[i, "EnergyCapacity"] - soc_bess[i, t-1],
                name=f"BESS_EnergyAvail_Down_{i}_{t}"
            )


            # Energy-availability for upward reserves & discharge
            model.addConstr(
                (1 / bess_units.loc[i, "DischargeEff"]) *
                (p_discharge_bess[i, 0] +
                gp.quicksum(s_res_bess[i, 0, r] for r in up_list_total)
                ) * TIME_STEP_HOURS
                <= soc_bess[i, 95],
                name=f"BESS_EnergyAvail_Up_{i}_{0}"
            )
            model.addConstr(
                bess_units.loc[i, "ChargeEff"] *
                (p_charge_bess[i, 0] +
                gp.quicksum(s_res_bess[i, 0, r] for r in down_list_total)
                ) * TIME_STEP_HOURS
                <= bess_units.loc[i, "EnergyCapacity"] - soc_bess[i, 95],
                name=f"BESS_EnergyAvail_Down_{i}_{0}"
            )
        

    model.update()
    model.optimize()

    return model, p, w, z, c_g, l, l_neg, p_gen_ps, p_pump_ps, soc_ps, p_charge_bess, p_discharge_bess, soc_bess
