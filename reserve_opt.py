"""Reserve sub‑problem *without* pumped‑storage optimisation.

The model follows the mathematical formulation shown in equations (E.29)–(E.37)
of your appendix.  All pumped‑storage (PS) units are now treated as *exogenous* –
whatever time–series you pass in the `data` dictionary (e.g. in `hourly_ps_gen`
and `hourly_ps_pump`) will **not** be decision variables and **do not** appear in
any reserve constraints.  Consequently, *no* reserve provision or optimisation
is possible from PS units.

Main structural changes
-----------------------
1. **Removed** every decision variable, constraint and expression that involved
   `ps_units` – including the auxiliary binaries `is_up` and the ramping
   constraints (E.42–E.45 in the original extended model).
2. System‑level reserve requirements (E.32 & E.33) no longer sum over
   `s_res_ps`, because that variable has been deleted.
3. Helper dataframes `hourly_ps_gen` / `hourly_ps_pump` are still accepted in
   the `data` dict, but are *not* referenced inside the optimisation model –
   they are left untouched so you can use them later in a combined energy +
   reserve pass if you want.
4. Reduced the argument list: the `s_res_ps_sol` parameter is gone because we
   do not solve anything for PS any more.
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


def build_reserve_subproblem_no_ps(
        data,
        lambda_dict,
        marginal_costs_dict,
        day_label: str = "AutumnWD",
):
    """Create reserve‐only sub‑problem **without** pumped storage optimisation.

    Parameters
    ----------
    data : dict
        Output of ``prepare_data``; *must* contain at least the following keys:
        ``generators``, ``be_reserves``, ``up_products_mFRR``,
        ``down_products_mFRR``, ``up_products_aFRR``, ``down_products_aFRR``,
        and the four hourly reserve requirement series.
    lambda_dict : dict[int, float]
        Clearing price λₜ for each 15‑min period *t* ∈ 0…95.
    marginal_costs_dict : dict[tuple[str,int], float]
        Mapping (generator, t) → MC_{g,t}.
    day_label : str, optional
        Used only for the Gurobi model name.

    Returns
    -------
    model : gp.Model
        The fully populated but *unsolved* Gurobi model.  Call ``model.optimize``
        outside if you want to solve it.
    tuple
        (p, w, z, c_g, s_res, s_res_bess, ocf) – the Gurobi ``Var`` objects.
    """

    # ------------------------------------------------------------------
    # 1) UNPACK INPUT DATA
    # ------------------------------------------------------------------
    generators        = data["generators"]
    segment_data      = data["segment_data"]
    be_reserves       = data["be_reserves"]
    up_products_mFRR  = data["up_products_mFRR"]
    down_products_mFRR = data["down_products_mFRR"]
    up_products_aFRR  = data["up_products_aFRR"]
    down_products_aFRR = data["down_products_aFRR"]

    # ------------- NEW: Battery storage --------------------------
    bess_units = data.get("bess_units", pd.DataFrame())  # may be empty

    hourly_up_reserve_mFRR   = data["hourly_up_reserve_mFRR"]
    hourly_down_reserve_mFRR = data["hourly_down_reserve_mFRR"]
    hourly_up_reserve_aFRR   = data["hourly_up_reserve_aFRR"]
    hourly_down_reserve_aFRR = data["hourly_down_reserve_aFRR"]

    # ---- Time indices -------------------------------------------------
    T15 = range(96)    # 15‑min intervals 0…95
    T60 = range(24)    # hours

    dt = 0.25   # hours per 15‑min step

    def hour_of_t(t: int) -> int:
        """Return the hour index h for quarter‑hour t."""
        return t // 4

    up_unique   = sorted(set(up_products_mFRR.index)   | set(up_products_aFRR.index))
    down_unique = sorted(set(down_products_mFRR.index) | set(down_products_aFRR.index))

    # ---------------------------------------------------------------
    # Requirement vectors (product‑specific, 96 timesteps)
    # ---------------------------------------------------------------
    # The runner attaches one 96‑element pandas.Series per product id
    # under data["req_vec"]  →  {product id: Series}.
    req_vec = data["req_vec"]

    # ------------------------------------------------------------------
    # 2) CREATE GUROBI MODEL
    # ------------------------------------------------------------------
    model = gp.Model(f"ReserveSubproblem_noPS_{day_label}")
    model.ModelSense = GRB.MINIMIZE

    # ------------------------------------------------------------------
    # 3) DECISION VARIABLES (only conventional units now)
    # ------------------------------------------------------------------
    p = model.addVars(generators.index, T15, lb=0, name="p")              # (E.29) generation level (not energy‑balanced)
    w = model.addVars(generators.index, T60, vtype=GRB.BINARY, name="w")    # commitment
    z = model.addVars(generators.index, T60, vtype=GRB.BINARY, name="z")    # startup
    c_g = model.addVars(generators.index, T15, lb=0, name="c_g")            # generation cost proxy

    reserve_idxs = be_reserves.index.tolist()  # e.g. "mFRR_up", "aFRR_down", …
    s_res = model.addVars(generators.index, T15, reserve_idxs, lb=0, name="s_res")

    # BESS reserve variables  ‑ only if batteries exist
    if not bess_units.empty:
        s_res_bess = model.addVars(bess_units.index, T15, reserve_idxs, lb=0,
                                   name="s_res_bess")
        soc_bess   = model.addVars(bess_units.index, T15, lb=0,
                                   ub={i: bess_units.loc[i,"EnergyCapacity"]
                                       for i in bess_units.index},
                                   name="soc_bess")
    else:
        s_res_bess = {}
        soc_bess   = {}

    # Nuclear units cannot offer reserves (business rule)
    for g in generators.index:
        if generators.loc[g, "FuelType"] == "NUCLEAR":
            for t in T15:
                for r in reserve_idxs:
                    s_res[g, t, r].ub = 0

    # ------------------------------------------------------------------
    # 4) CAPACITY & RESERVE CONSTRAINTS
    # ------------------------------------------------------------------

    # Up to max generation when committed (E.30 analogue, but without PS)
    for g in generators.index:
        for t in T15:
            model.addConstr(
                p[g, t] <= generators.loc[g, "P_max"] * w[g, hour_of_t(t)],
                name=f"MaxGen_{g}_{t}"
            )

    # ------------------------------------------------------------------
    # 4‑b) PRODUCT‑SPECIFIC RESERVE REQUIREMENTS  (replaces E.32 & E.33)
    # ------------------------------------------------------------------
    # Upward reserves (E.32)
    for t in T15:
        for r in up_products_mFRR.index:
            req_value = 0.0
            if isinstance(hourly_up_reserve_mFRR, pd.DataFrame):
                if r in hourly_up_reserve_mFRR.index and t in hourly_up_reserve_mFRR.columns:
                    req_value = float(hourly_up_reserve_mFRR.loc[r, t])
            elif r in hourly_up_reserve_mFRR:
                req_value = float(hourly_up_reserve_mFRR[r][t])
            
            model.addConstr(
                gp.quicksum(s_res[g, t, r] for g in generators.index) +
                ( gp.quicksum(s_res_bess[i, t, r] for i in bess_units.index)
                  if s_res_bess else 0 )
                >= req_value,
                name=f"UpResReq_mFRR_{r}_{t}"
            )
    
    for t in T15:
        for r in up_products_aFRR.index:
            req_value = 0.0
            if isinstance(hourly_up_reserve_aFRR, pd.DataFrame):
                if r in hourly_up_reserve_aFRR.index and t in hourly_up_reserve_aFRR.columns:
                    req_value = float(hourly_up_reserve_aFRR.loc[r, t])
            elif r in hourly_up_reserve_aFRR:
                req_value = float(hourly_up_reserve_aFRR[r][t])
                
            model.addConstr(
                gp.quicksum(s_res[g, t, r] for g in generators.index) +
                ( gp.quicksum(s_res_bess[i, t, r] for i in bess_units.index)
                  if s_res_bess else 0 )
                >= req_value,
                name=f"UpResReq_aFRR_{r}_{t}"
            )

    for t in T15:
        for r in down_products_mFRR.index:
            req_value = 0.0
            if isinstance(hourly_down_reserve_mFRR, pd.DataFrame):
                if r in hourly_down_reserve_mFRR.index and t in hourly_down_reserve_mFRR.columns:
                    req_value = float(hourly_down_reserve_mFRR.loc[r, t])
            elif r in hourly_down_reserve_mFRR:
                req_value = float(hourly_down_reserve_mFRR[r][t])
                
            model.addConstr(
                gp.quicksum(s_res[g, t, r] for g in generators.index) +
                ( gp.quicksum(s_res_bess[i, t, r] for i in bess_units.index)
                  if s_res_bess else 0 )
                >= req_value,
                name=f"DownResReq_mFRR_{r}_{t}"
            )

    for t in T15:
        for r in down_products_aFRR.index:
            req_value = 0.0
            if isinstance(hourly_down_reserve_aFRR, pd.DataFrame):
                if r in hourly_down_reserve_aFRR.index and t in hourly_down_reserve_aFRR.columns:
                    req_value = float(hourly_down_reserve_aFRR.loc[r, t])
            elif r in hourly_down_reserve_aFRR:
                req_value = float(hourly_down_reserve_aFRR[r][t])
                
            model.addConstr(
                gp.quicksum(s_res[g, t, r] for g in generators.index) +
                ( gp.quicksum(s_res_bess[i, t, r] for i in bess_units.index)
                  if s_res_bess else 0 )
                >= req_value,
                name=f"DownResReq_aFRR_{r}_{t}"
            )



    # Generator‑specific headroom constraints (E.34–E.37)
    ramp_window = 15  # min
    for g in generators.index:
        is_fast = generators.loc[g, "GeneratorType"] == "FAST"
        is_slow = generators.loc[g, "GeneratorType"] == "SLOW"
        R_max   = generators.loc[g, "R_max"]
        R_min   = generators.loc[g, "R_min"]

        for t in T15:
            # E.34 / E.35 FAST units capacity for upward reserves
            if is_fast:
                model.addConstr(
                    generators.loc[g, "P_min"] * w[g, hour_of_t(t)]
                    + gp.quicksum(s_res[g, t, r] for r in up_unique)
                    <= generators.loc[g, "P_max"] * w[g, hour_of_t(t)],
                    name=f"E34_FAST_upcap_{g}_{t}"
                )
                model.addConstr(
                    generators.loc[g, "P_min"]
                    + gp.quicksum(s_res[g, t, r] for r in up_unique)
                    <= generators.loc[g, "P_max"],
                    name=f"E35_FAST_headroom_{g}_{t}"
                )

            # E.36 SLOW units capacity for upward reserves
            if is_slow:
                model.addConstr(
                    generators.loc[g, "P_min"] * w[g, hour_of_t(t)]
                    + gp.quicksum(s_res[g, t, r] for r in up_unique)
                    <= generators.loc[g, "P_max"] * w[g, hour_of_t(t)],
                    name=f"E36_SLOW_headroom_{g}_{t}"
                )

            # E.37 ramp limits for reserves (up & down)
            model.addConstr(
                gp.quicksum(s_res[g, t, r] for r in down_unique) +
                (gp.quicksum(s_res_bess[i, t, r] for i in bess_units.index for r in down_unique)
                 if s_res_bess else 0)
                <= ramp_window * R_min,
                name=f"E37_ramp_down_{g}_{t}"
            )
            model.addConstr(
                gp.quicksum(s_res[g, t, r] for r in up_unique) +
                (gp.quicksum(s_res_bess[i, t, r] for i in bess_units.index for r in up_unique)
                 if s_res_bess else 0)
                <= ramp_window * R_max,
                name=f"E37_ramp_up_{g}_{t}"
            )

    # ------------------------------------------------------------------
    # BESS technical limits for reserve provision
    # ------------------------------------------------------------------
    if not bess_units.empty:
        eta_c = bess_units["ChargeEff"]
        eta_d = bess_units["DischargeEff"]

        # Initial SoC (assume given) and daily closure ≥ initial
        for i in bess_units.index:
            model.addConstr(soc_bess[i,0] == bess_units.loc[i,"InitialSoC"],
                            name=f"BESS_InitSoC_{i}")
            model.addConstr(soc_bess[i,95] >= bess_units.loc[i,"InitialSoC"],
                            name=f"BESS_FinalSoC_{i}")

        for i in bess_units.index:
            for t in T15:
                up_sum   = gp.quicksum(s_res_bess[i,t,r] for r in up_unique)
                down_sum = gp.quicksum(s_res_bess[i,t,r] for r in down_unique)

                # Power capacity
                model.addConstr(
                    up_sum + down_sum <= bess_units.loc[i,"PowerCapacity"],
                    name=f"BESS_PowerCap_{i}_{t}")

                # Energy head‑room (upward reserve consumes energy)
                if t > 0:
                    model.addConstr(
                        (1/ bess_units.loc[i,"DischargeEff"]) * up_sum * dt
                        <= soc_bess[i,t-1],
                        name=f"BESS_EnergyUp_{i}_{t}")
                    model.addConstr(
                        bess_units.loc[i,"ChargeEff"] * down_sum * dt
                        <= bess_units.loc[i,"EnergyCapacity"] - soc_bess[i,t-1],
                        name=f"BESS_EnergyDown_{i}_{t}")

                # Simple SoC progression ignoring energy market
                if t > 0:
                    model.addConstr(
                        soc_bess[i,t] == soc_bess[i,t-1],   # flat SoC in reserve stage
                        name=f"BESS_SoC_Flat_{i}_{t}")

    # ------------------------------------------------------------------
    # 5) OPPORTUNITY COST FUNCTION (E.29) – linearised per your image
    # ------------------------------------------------------------------
    ocf = {}
    for g in generators.index:
        for t in T15:
            MC = marginal_costs_dict.get((g, t), None)
            if MC is None:
                continue
            lambda_t = lambda_dict[t]
            key_term = MC/4 - lambda_t

            s_up = gp.quicksum(s_res[g, t, r] for r in up_unique)

            # Aux binary for the sign of (MC/4 - λ) → "is_pos" (1 if ≥ 0)
            is_pos = model.addVar(vtype=GRB.BINARY, name=f"signPos_{g}_{t}")
            M_big  = 10_000  # big‑M large enough compared to possible |MC - λ|
            model.addConstr(key_term - M_big * is_pos <= 0, name=f"signPosDef1_{g}_{t}")
            model.addConstr(-key_term - M_big * (1 - is_pos) <= 0, name=f"signPosDef2_{g}_{t}")

            # OCF linearisation
            ocf_pos = (key_term) * (generators.loc[g, "P_min"] + s_up)  # when key_term ≥ 0
            ocf_neg = (key_term) * (-s_up)                               # when key_term < 0
            ocf[(g, t)] = is_pos * ocf_pos + (1 - is_pos) * ocf_neg

    # ------------------------------------------------------------------
    # 6) OBJECTIVE (1/4 Σ_t Σ_g OCF + no‑load + start‑up) – Eq. (E.29)
    # ------------------------------------------------------------------
    obj = (0.25 * gp.quicksum(ocf[(g, t)] for g in generators.index for t in T15)
           # include fixed costs once – in reserve stage
           + dt * gp.quicksum(generators.loc[g, "K_eur"] * w[g, h]
                              for g in generators.index if "K_eur" in generators.columns
                              for h in T60)
           + gp.quicksum(generators.loc[g, "S_eur"] * z[g, h]
                         for g in generators.index if "S_eur" in generators.columns
                         for h in T60))
    model.setObjective(obj)

    # ------------------------------------------------------------------
    # 7) MIN UP / DOWN TIMES & START‑UP LOGIC (E.38–E.41)
    # ------------------------------------------------------------------
    for g in generators.index:
        UT = int(generators.loc[g, "UT_g"])
        DT = int(generators.loc[g, "DT_g"])
        for h in T60:
            # Startup definition w_h - w_{h-1}
            w_prev = w[g, h - 1] if h > 0 else w[g, 23]
            model.addConstr(z[g, h] >= w[g, h] - w_prev, name=f"StartupDef_{g}_{h}")

            # Min‑up time
            if h >= UT - 1:
                model.addConstr(
                    gp.quicksum(z[g, hh] for hh in range(h - UT + 1, h + 1)) <= w[g, h],
                    name=f"MinUp_{g}_{h}"
                )
            # Min‑down time
            if h <= 23 - DT:
                model.addConstr(
                    gp.quicksum(z[g, hh] for hh in range(h + 1, h + DT + 1)) <= 1 - w[g, h],
                    name=f"MinDown_{g}_{h}"
                )

    # Wrap‑around commitment (w_0 == w_23)
    for g in generators.index:
        model.addConstr(w[g, 0] == w[g, 23], name=f"Wrap_{g}")

    # ------------------------------------------------------------------
    # 8) LINK COMMITMENT TO RESERVE PROVISION (big‑M)
    # ------------------------------------------------------------------
    M_big = 10_000
    for g in generators.index:
        for t in T15:
            model.addConstr(
                gp.quicksum(s_res[g, t, r] for r in reserve_idxs) <= M_big * w[g, hour_of_t(t)],
                name=f"CommitIfReserves_{g}_{t}"
            )

    # ------------------------------------------------------------------
    # 9) Finalise model
    # ------------------------------------------------------------------
    model.update()
    return model, p, w, z, c_g, s_res, s_res_bess, ocf
