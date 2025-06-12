"""Runner script for the **pumped‑storage‑free** reserve sub‑problem.

Outputs **system‑level** *and* **per‑product** reserve requirements so you can
trace exactly what the model is asked to cover.

The optimisation itself is unchanged: still builds the model via
``build_reserve_subproblem_no_ps`` and dumps `.lp`, `.ilp`, variable values,
etc.  Only the input‑reporting block is richer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from reserve_opt import build_reserve_subproblem_no_ps
from data_processing import prepare_data

# ----------------------------------------------------------------------
# Helper I/O ------------------------------------------------------------
# ----------------------------------------------------------------------

def read_lambda_values(csv_path: str | Path) -> Dict[int, float]:
    df = pd.read_csv(csv_path)
    lam_col = next((c for c in df.columns if c.lower().startswith("lambda")), None)
    if lam_col is None:
        raise ValueError("No 'Lambda' column in " + str(csv_path))
    lam = {int(r["Interval"]): float(r[lam_col]) for _, r in df.iterrows()}
    for t in range(96):
        lam.setdefault(t, 0.0)
    return lam


def read_marginal_costs(csv_path: str | Path) -> Dict[Tuple[str, int], float]:
    df = pd.read_csv(csv_path)
    return {(r["Generator"], int(r["Interval"])): float(r["MC"])
            for _, r in df.iterrows()}

# ----------------------------------------------------------------------
# Reserve requirement helper (same as co‑opt flow) ----------------------
# ----------------------------------------------------------------------

def req_at(req_obj: Any, t: int) -> float:
    """Return MW requirement of a reserve *object* at quarter‑hour *t* (0‑95).

    The reserve object can be:
    • DataFrame (wide or long) with 96 values
    • Series / list of length 96
    • dict {product_id → DataFrame/Series/number}
    • scalar (int/float)
    Any 24‑step input is considered an upstream error and raises ValueError.
    """
    # -------------------- DataFrame -----------------------------
    if isinstance(req_obj, pd.DataFrame):
        # Wide (columns = 0‑95)
        if t in req_obj.columns:
            return float(req_obj[t].sum())
        # Long (index = 0‑95)
        if t in req_obj.index:
            return float(req_obj.loc[t].sum())
        raise ValueError("DataFrame reserve input must have 96 columns or index rows.")

    # -------------------- Series / list -------------------------
    if isinstance(req_obj, (pd.Series, list)):
        if len(req_obj) != 96:
            raise ValueError("Series/list reserve input must be length 96.")
        return float(req_obj[t])

    # -------------------- dict (bucket of products) -------------
    if isinstance(req_obj, dict):
        total = 0.0
        for v in req_obj.values():
            total += req_at(v, t)      # recurse – each *v* must pass one of the above branches
        return total

    # -------------------- scalar -------------------------------
    return float(req_obj)

# ----------------------------------------------------------------------
# Core runner -----------------------------------------------------------
# ----------------------------------------------------------------------

def run_reserve_subproblem_no_ps(data: Dict[str, Any],
                                 lambda_csv: str | Path,
                                 mc_csv: str | Path,
                                 day_label: str,
                                 out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1)  WRITE RESERVE REQUIREMENT CSVs
    # ---------------------------------------------------------------
    sys_df = pd.DataFrame({
        "aFRR_up":  [req_at(data["hourly_up_reserve_aFRR"],   t) for t in range(96)],
        "aFRR_down":[req_at(data["hourly_down_reserve_aFRR"], t) for t in range(96)],
        "mFRR_up":  [req_at(data["hourly_up_reserve_mFRR"],   t) for t in range(96)],
        "mFRR_down":[req_at(data["hourly_down_reserve_mFRR"], t) for t in range(96)],
    })
    sys_df.index.name = "Interval"
    sys_df.to_csv(out_dir / f"reserve_requirements_{day_label}.csv")

    # Per‑product detail
    prod_vals: Dict[str, list] = {}
    def _vec(src, prod):
        series = src.loc[prod] if isinstance(src, pd.DataFrame) else src[prod]
        return [req_at(series, t) for t in range(96)]

    for prod in data["up_products_aFRR"].index:
        prod_vals[prod] = _vec(data["hourly_up_reserve_aFRR"], prod)
    for prod in data["down_products_aFRR"].index:
        prod_vals[prod] = _vec(data["hourly_down_reserve_aFRR"], prod)
    for prod in data["up_products_mFRR"].index:
        prod_vals[prod] = _vec(data["hourly_up_reserve_mFRR"], prod)
    for prod in data["down_products_mFRR"].index:
        prod_vals[prod] = _vec(data["hourly_down_reserve_mFRR"], prod)

    prod_df = pd.DataFrame(prod_vals); prod_df.index.name = "Interval"
    prod_df.to_csv(out_dir / f"reserve_requirements_products_{day_label}.csv")

    # ---------------------------------------------------------------
    # Build a per‑product 96‑step requirement vector that the core
    # optimiser can consume directly.  Each entry is a pandas.Series.
    # ---------------------------------------------------------------
    req_vec: Dict[str, pd.Series] = {c: prod_df[c] for c in prod_df.columns}

    # Attach to the data dict so `build_reserve_subproblem_no_ps` can
    # read it.  (Shallow‑copy first to avoid side‑effects upstream.)
    data = dict(data)
    data["req_vec"] = req_vec


    # Basic generator techs for sanity check
    cols = [c for c in ["P_min", "P_max", "R_max", "R_min", "FuelType", "GeneratorType"]
            if c in data["generators"].columns]
    data["generators"][cols].to_csv(out_dir / f"generators_summary_{day_label}.csv")

    # ---------------------------------------------------------------
    # 2)  BUILD MODEL
    # ---------------------------------------------------------------
    model, p, w, z, c_g, s_res, s_res_bess, ocf = build_reserve_subproblem_no_ps(
        data,
        read_lambda_values(lambda_csv),
        read_marginal_costs(mc_csv),
        day_label=day_label)

    model.write((out_dir / f"reserve_model_{day_label}.lp").as_posix())
    print(f"LP model written → reserve_model_{day_label}.lp")

    # ---------------------------------------------------------------
    # 3)  SOLVE
    # ---------------------------------------------------------------
    print("Optimising reserve sub-problem...")
    model.optimize()
    
    # Get status name using model.Status directly
    status_code = model.Status
    if status_code == GRB.OPTIMAL:
        status_name = "OPTIMAL"
    elif status_code == GRB.INFEASIBLE:
        status_name = "INFEASIBLE"
    elif status_code == GRB.INF_OR_UNBD:
        status_name = "INFEASIBLE_OR_UNBOUNDED"
    else:
        status_name = f"Status code: {status_code}"
    
    print(f"Model status: {status_name}")

    if model.Status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD}:
        try:
            print("Computing IIS to identify infeasibility cause...")
            model.computeIIS()
            ilp_path = out_dir / f"reserve_{day_label}.ilp"
            model.write(ilp_path.as_posix())
            print(f"IIS written → {ilp_path}")
        except gp.GurobiError as e:
            print(f"IIS computation failed: {e}")

    # ---------------------------------------------------------------
    # 4)  RETURN BASIC SOLUTION DICT AND EXPORT RESULTS TO CSV
    # ---------------------------------------------------------------
    sol = {"Objective": None, "w": {}, "s_res": {}}
    if model.Status == GRB.OPTIMAL:
        sol["Objective"] = model.ObjVal
        gens = data["generators"].index
        bess_ids = data.get("bess_units", pd.DataFrame()).index
        sol["w"] = {(g, h): w[g, h].X for g in gens for h in range(24)}
        prod_set = data["be_reserves"].index
        sol["s_res"] = {(g, t, r): s_res[g, t, r].X for g in gens for t in range(96) for r in prod_set}
        if s_res_bess:
            sol["s_res_bess"] = {(i, t, r): s_res_bess[i, t, r].X
                                 for i in bess_ids for t in range(96) for r in prod_set}

        # Export results to CSV files
        print(f"Exporting results to CSV files...")

        # 1. Generator commitment decisions (hourly)
        commit_data = []
        for g in gens:
            for h in range(24):
                commit_data.append({
                    'Generator': g,
                    'Hour': h,
                    'Committed': w[g, h].X
                })
        commit_df = pd.DataFrame(commit_data)
        commit_df.to_csv(out_dir / f"commitment_{day_label}.csv", index=False)
        print(f"Commitment decisions written → commitment_{day_label}.csv")

        # 2. Reserve provision by provider (generators and BESS), time step and product (quarter-hourly)
        reserve_data = []
        # conventional generators
        for g in gens:
            for t in range(96):
                for r in prod_set:
                    val = s_res[g, t, r].X
                    if val > 0.001:
                        reserve_data.append({
                            'Provider': g,
                            'Type': 'GEN',
                            'Interval': t,
                            'Product': r,
                            'Reserve_MW': val
                        })
        # batteries
        if s_res_bess:
            for i in bess_ids:
                for t in range(96):
                    for r in prod_set:
                        val = s_res_bess[i, t, r].X
                        if val > 0.001:
                            reserve_data.append({
                                'Provider': i,
                                'Type': 'BESS',
                                'Interval': t,
                                'Product': r,
                                'Reserve_MW': val
                            })
        reserve_df = pd.DataFrame(reserve_data)
        reserve_df.to_csv(out_dir / f"reserve_provision_{day_label}.csv", index=False)
        print(f"Reserve provision written → reserve_provision_{day_label}.csv "
              "(includes generators and BESS)")

        # 3. Summary of total reserve by product and time interval (include BESS)
        summary_data = {}
        for t in range(96):
            summary_data[t] = {}
            for r in prod_set:
                total_reserve = sum(s_res[g, t, r].X for g in gens)
                if s_res_bess:
                    total_reserve += sum(s_res_bess[i, t, r].X for i in bess_ids)
                summary_data[t][r] = total_reserve

        summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
        summary_df.index.name = 'Interval'
        summary_df.to_csv(out_dir / f"reserve_summary_{day_label}.csv")
        print(f"Reserve summary written → reserve_summary_{day_label}.csv")

        # 4. Cost breakdown
        if hasattr(model, 'objVal'):
            cost_data = {
                'Total_Cost': model.objVal,
                'Day': day_label
            }
            cost_df = pd.DataFrame([cost_data])
            cost_df.to_csv(out_dir / f"reserve_cost_{day_label}.csv", index=False)
            print(f"Cost information written → reserve_cost_{day_label}.csv")

        # 5. Create the reserve solution file needed by run_energy.py
        # Include 96 rows per generator with complete data in all columns
        reserve_solution_data = []
        objective_value = model.ObjVal

        # Create 96 rows for each generator
        for g in gens:
            for t in range(96):
                h = t // 4  # The hour this timestep belongs to

                # Initialize dictionary with zeros for all s_res values
                s_res_values = {r: 0 for r in prod_set}

                # Fill in actual s_res values where they exist
                for r in prod_set:
                    if s_res[g, t, r].X > 0.001:
                        s_res_values[r] = s_res[g, t, r].X

                # For each reserve product, create a row
                for r in prod_set:
                    reserve_solution_data.append({
                        'Generator': g,
                        'Hour': h,  # The corresponding hour
                        'wValue': w[g, h].X,  # The commitment decision
                        'TimeStep': t,  # The timestep (0-95)
                        'ReserveProduct': r,  # The reserve product
                        's_res': s_res_values[r],  # The reserve amount (0 if none)
                        'Objective': objective_value  # Total objective value
                    })

        # Create the dataframe and save it
        solution_df = pd.DataFrame(reserve_solution_data)
        solution_df.to_csv(out_dir / f"reserve_solution_{day_label}.csv", index=False)
        print(f"Reserve solution written → reserve_solution_{day_label}.csv")

    return sol

# ----------------------------------------------------------------------
# CLI ------------------------------------------------------------------
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reserve optimisation without PS")
    parser.add_argument("--season", choices=["SummerWD", "AutumnWD", "WinterWD", "SpringWD"],
                        help="Season to process (default: all)")
    parser.add_argument("--data-dir", default="./data/", help="Input data root dir")
    parser.add_argument("--results-dir", default="./results", help="Directory for outputs & artefacts")
    args = parser.parse_args()

    seasons = [args.season] if args.season else ["SummerWD", "AutumnWD", "WinterWD", "SpringWD"]

    ok = 0
    all_costs = []
    all_commits = []
    all_reserves = []
    
    for s in seasons:
        print(f"\n===== {s} =====")
        try:
            data = prepare_data(args.data_dir, s)
            lam_csv = Path(args.results_dir) / f"coopt/lambda_values_fixed_{s}.csv"
            mc_csv  = Path(args.results_dir) / f"coopt/marginal_costs_{s}.csv"
            if not lam_csv.exists() or not mc_csv.exists():
                print("Missing λ or MC CSV – skipping.")
                continue
            res = run_reserve_subproblem_no_ps(data, lam_csv, mc_csv,
                                               day_label=s,
                                               out_dir=Path(args.results_dir, "sequential"))
            if res["Objective"] is not None:
                ok += 1
                
                # Collect data for consolidated results
                out_dir = Path(args.results_dir) / "sequential"
                
                # Collect cost data
                cost_file = out_dir / f"reserve_cost_{s}.csv"
                if cost_file.exists():
                    cost_df = pd.read_csv(cost_file)
                    all_costs.append(cost_df)
                
                # Collect commitment data
                commit_file = out_dir / f"commitment_{s}.csv"
                if commit_file.exists():
                    commit_df = pd.read_csv(commit_file)
                    commit_df['Season'] = s
                    all_commits.append(commit_df)
                
                # Collect reserve provision data
                reserve_file = out_dir / f"reserve_provision_{s}.csv"
                if reserve_file.exists():
                    reserve_df = pd.read_csv(reserve_file)
                    reserve_df['Season'] = s
                    all_reserves.append(reserve_df)
                    
        except Exception as e:
            print(f"ERROR processing {s}: {e}")
            import traceback; traceback.print_exc()
    
    # Create consolidated results if we have any successful runs
    if ok > 0:
        out_dir = Path(args.results_dir) / "sequential"
        
        # Combined costs across seasons
        if all_costs:
            combined_costs = pd.concat(all_costs, ignore_index=True)
            combined_costs.to_csv(out_dir / "all_reserve_costs.csv", index=False)
            print("Combined cost data written → all_reserve_costs.csv")
        
        # Combined commitments across seasons
        if all_commits:
            combined_commits = pd.concat(all_commits, ignore_index=True)
            combined_commits.to_csv(out_dir / "all_commitments.csv", index=False)
            print("Combined commitment data written → all_commitments.csv")
        
        # Combined reserve provisions across seasons
        if all_reserves:
            combined_reserves = pd.concat(all_reserves, ignore_index=True)
            combined_reserves.to_csv(out_dir / "all_reserve_provisions.csv", index=False)
            print("Combined reserve provision data written → all_reserve_provisions.csv")
    
    print("\n===== SUMMARY =====")
    if ok > 0:
        print(f"Successfully solved {ok}/{len(seasons)} seasons.")
    else:
        print("No season solved – investigate logs.")
