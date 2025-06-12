import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from pathlib import Path
import argparse
import sys
from typing import Dict, Tuple, Any



# Duration of one modelling interval (15 minutes) expressed in hours
TIME_STEP_HOURS = 0.25
VOLL = 3000  # Value of Lost Load (€/MWh)


def build_energy_only_model(data, 
                            reserve_csv, 
                            allow_decommitment=True,
                            include_bess=True,
                            use_arbitrage=False):
    """
    Build energy-only optimization model with fixed commitment from reserve step.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all input data
    reserve_csv : str
        Path to CSV file containing fixed commitment and reserve decisions
    allow_decommitment : bool, default=True
        If True, allows the energy model to decommit generators that were committed in reserve step
        If False, generators committed in reserve step must remain committed
        
    Returns:
    --------
    model : gurobi model
        Optimized energy-only model
    """
    
    # Load fixed decisions from reserve step
    reserve_df = pd.read_csv(reserve_csv)
    
    # Extract fixed commitment decisions (w variables)
    commitment_rows = reserve_df.dropna(subset=["Hour"])
    w_fixed = {}
    for _, row in commitment_rows.iterrows():
        g = row["Generator"]
        h = int(row["Hour"])
        w_val = float(row["wValue"])
        w_fixed[(g, h)] = w_val
    
    # Extract data
    generators = data["generators"]
    segment_data = data["segment_data"] 
    hourly_demand = data["hourly_demand"]
    hourly_renewables = data["hourly_renewables"]
    hourly_ps_gen = data["hourly_ps_gen"].set_index("Name")
    hourly_ps_pump = data["hourly_ps_pump"].set_index("Name")
    bess_units = data["bess_units"]
    # --- read lambda -------------------------------------------------
    lambda_t = data.get("lambda_series", [0.0]*96)[:96]

    
    
    # Create model
    model = gp.Model("Energy-Only")
    
    # Time periods (96 x 15-minute intervals)
    T = range(96)
    
    def hour_of_t(t):
        return t // 4
    
    # Decision variables
    p = model.addVars(generators.index, T, lb=0, name="p")  # Energy generation
    c_g = model.addVars(generators.index, T, lb=0, name="c_g")  # Generation costs
    l = model.addVars(T, lb=0, name="l")  # Positive load shedding
    l_neg = model.addVars(T, lb=0, name="l_neg")  # Negative load shedding

    ###BESS variables 
    if include_bess:
        p_discharge_bess = model.addVars(
            bess_units.index, T, lb=0,
            ub = {i: bess_units.loc[i, "PowerCapacity"] for i in bess_units.index},
            name="p_discharge_bess")
        
        p_charge_bess = model.addVars(
            bess_units.index, T, lb=0,
            ub = {i:bess_units.loc[i, "PowerCapacity"] for i in bess_units.index},
            name = "p_charge_bess" )
        
        soc_bess = model.addVars(
            bess_units.index, T, lb=0,
            ub={i: bess_units.loc[i, "EnergyCapacity"] for i in bess_units.index},
            name = "soc_bess"
        )
    else:
        p_discharge_bess = {}
        p_charge_bess = {}
        soc_bess = {}

    

    # Commitment variables - need to be defined BEFORE referencing them
    w = model.addVars(generators.index, range(24), vtype=GRB.BINARY, name="w")
    z = model.addVars(generators.index, range(24), vtype=GRB.BINARY, name="z")
    
    # Fix commitment variables based on reserve solution
    if not allow_decommitment:
        # Original approach: Only enforce that generators committed in reserve step must stay committed
        # Allow energy model to commit additional generators if needed
        for g in generators.index:
            for h in range(24):
                if (g, h) in w_fixed and w_fixed[(g, h)] > 0.5:  # If committed in reserve step
                    model.addConstr(w[g, h] == 1, name=f"w_fixed_{g}_{h}")
                # Don't constrain generators that were not committed - the energy model can choose
    
    # Piecewise cost constraints (same as in build_model.py)
    for g in generators.index:
        hrc_key = generators.loc[g, "HeatRateCurve"]
        for t in T:
            seg_costs = []
            if hrc_key in segment_data:
                for (slope, intercept) in segment_data[hrc_key]:
                    cost_var = model.addVar()
                    model.addConstr(
                        cost_var == slope * generators.loc[g, "Price"] * p[g, t] + intercept
                    )
                    seg_costs.append(cost_var)
                for cvar in seg_costs:
                    model.addConstr(c_g[g, t] >= cvar)
            else:
                # Linear cost if no piecewise data
                model.addConstr(c_g[g, t] == generators.loc[g, "Price"] * p[g, t])
    
    # Objective function (energy costs only, without fixed costs)
    dt = TIME_STEP_HOURS
    
    obj  = dt * gp.quicksum(c_g[g,t] for g in generators.index for t in T)
    obj += dt * VOLL * gp.quicksum(l[t] for t in T)
    obj += dt * VOLL * gp.quicksum(l_neg[t] for t in T)




    


    if not bess_units.empty:
        # degradation cost (always)
        obj += dt * gp.quicksum(
                bess_units.loc[i,"CycleCost"] *
                (p_charge_bess[i,t] + p_discharge_bess[i,t])
                for i in bess_units.index for t in T)
        # arbitrage revenue (only if flag true)
        if use_arbitrage:
            obj -= dt * gp.quicksum(
                    lambda_t[t] *
                (p_discharge_bess[i,t] - p_charge_bess[i,t])
                for i in bess_units.index for t in T)

    model.setObjective(obj, GRB.MINIMIZE)
    
    # Energy balance constraint
    for t in T:
        # Pumped storage net generation (fixed time series)
        ps_net = 0
        for ps_name in hourly_ps_gen.index:
            if ps_name in hourly_ps_pump.index:
                ps_gen = hourly_ps_gen.loc[ps_name, t]
                ps_pump = hourly_ps_pump.loc[ps_name, t]  # Already negative
                ps_net += ps_gen + ps_pump

        bess_term = 0
        if include_bess:
            bess_term = ( gp.quicksum(p_discharge_bess[i,t] for i in bess_units.index)
                        - gp.quicksum(p_charge_bess[i,t]    for i in bess_units.index) )

        model.addConstr(
        gp.quicksum(p[g,t] for g in generators.index)
        + gp.quicksum(hourly_renewables.loc[r,t] for r in hourly_renewables.index)
        + ps_net
        + bess_term
        + l[t] - l_neg[t]
        == hourly_demand[t],
        name=f"EnergyBalance_{t}")
    
    # Generator capacity constraints
    for g in generators.index:
        for t in T:
            h = hour_of_t(t)
            # Upper bound
            model.addConstr(
                p[g, t] <= generators.loc[g, "P_max"] * w[g, h],
                name=f"MaxGen_{g}_{t}"
            )
            # Lower bound  
            model.addConstr(
                p[g, t] >= generators.loc[g, "P_min"] * w[g, h],
                name=f"MinGen_{g}_{t}"
            )
    
    # Ramping constraints
    for g in generators.index:
        for t in range(1, 96):
            ramp_up_limit = 15 * generators.loc[g, "R_max"]  # 15 minutes
            ramp_down_limit = 15 * abs(generators.loc[g, "R_min"])
            
            model.addConstr(
                p[g, t] - p[g, t-1] <= ramp_up_limit,
                name=f"RampUp_{g}_{t}"
            )
            model.addConstr(
                p[g, t-1] - p[g, t] <= ramp_down_limit,
                name=f"RampDown_{g}_{t}"
            )
    
    # Startup definition constraints
    for g in generators.index:
        for h in range(24):
            if h > 0:
                model.addConstr(
                    z[g, h] >= w[g, h] - w[g, h-1],
                    name=f"StartupDef_{g}_{h}"
                )
            else:
                model.addConstr(
                    z[g, 0] >= w[g, 0] - w[g, 23],
                    name=f"StartupDef_{g}_0"
                )
    
    # Minimum up time constraints
    for g in generators.index:
        UT = int(generators.loc[g, 'UT_g'])
        for t in T:
            if t >= UT - 1:
                model.addConstr(
                    gp.quicksum(z[g, hour_of_t(q)] for q in range(t - UT + 1, t + 1)) <= w[g, hour_of_t(t)],
                    name=f"MinUpTime_{g}_{t}"
                )
    
    # Minimum down time constraints  
    for g in generators.index:
        DT = int(generators.loc[g, 'DT_g'])
        for t in T:
            if t <= (95 - DT):
                model.addConstr(
                    gp.quicksum(z[g, hour_of_t(q)] for q in range(t + 1, t + DT + 1)) <= 1 - w[g, hour_of_t(t)],
                    name=f"MinDownTime_{g}_{t}"
                )


    # ──────────────────────────────────────────────────────────────
    # BESS constraints
    # ──────────────────────────────────────────────────────────────
    if include_bess:
        for i in bess_units.index:
            # Initial and final SoC
            model.addConstr(soc_bess[i, 0]  == bess_units.loc[i, "InitialSoC"],
                            name=f"BESS_InitSoC_{i}")
            model.addConstr(soc_bess[i, 95] >= bess_units.loc[i, "InitialSoC"],
                            name=f"BESS_FinalSoC_{i}")

            for t in range(1, 96):
                # SoC dynamics
                model.addConstr(
                    soc_bess[i, t] ==
                    soc_bess[i, t-1]
                + bess_units.loc[i, "ChargeEff"]    * p_charge_bess[i, t] * TIME_STEP_HOURS
                - (1 / bess_units.loc[i, "DischargeEff"]) * p_discharge_bess[i, t]   * TIME_STEP_HOURS,
                    name=f"BESS_SoC_Update_{i}_{t}"
                )

                # Energy-availability for discharge (can’t take out what isn’t there)
                model.addConstr(
                    (1 / bess_units.loc[i, "DischargeEff"])
                    * p_discharge_bess[i, t] * TIME_STEP_HOURS
                    <= soc_bess[i, t-1],
                    name=f"BESS_EnergyAvail_Up_{i}_{t}"
                )

                # Empty-space availability for charge
                model.addConstr(
                    bess_units.loc[i, "ChargeEff"]
                    * p_charge_bess[i, t] * TIME_STEP_HOURS
                    <= bess_units.loc[i, "EnergyCapacity"] - soc_bess[i, t-1],
                    name=f"BESS_EnergyAvail_Down_{i}_{t}"
                )

            # Non-simultaneous charge / discharge (mode binary)
            for t in T:
                if allow_decommitment:   # just reuse flag; has nothing to do with UC here
                    mode = model.addVar(lb=0, ub=1, name=f"BESS_mode_{i}_{t}")
                else:
                    mode = model.addVar(vtype=GRB.BINARY, name=f"BESS_mode_{i}_{t}")

                model.addConstr(p_discharge_bess[i, t] <=
                                bess_units.loc[i, "PowerCapacity"] * mode,
                                name=f"BESS_DisMode_{i}_{t}")
                model.addConstr(p_charge_bess[i, t]    <=
                                bess_units.loc[i, "PowerCapacity"] * (1 - mode),
                                name=f"BESS_ChargeMode_{i}_{t}")
        
    return model