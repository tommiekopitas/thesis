import pandas as pd

def prepare_data(DATA_PATH, day_type):
    """
    Loads and prepares all data (generators, renewables, loads, etc.)
    for a specified day_type (e.g., 'AutumnWD', 'WinterWD', etc.).
    Returns a dictionary of all necessary DataFrames and series.
    """

    # ---------------------------
    # 1) DATA LOADING
    # ---------------------------
    generators      = pd.read_csv(DATA_PATH + "Generators.csv")
    renewables      = pd.read_csv(DATA_PATH + "GeneratorsRE.csv", delimiter=";")
    loads           = pd.read_csv(DATA_PATH + "Loads.csv")
    fuel_prices     = pd.read_csv(DATA_PATH + "FuelPrice.csv")
    reserves        = pd.read_csv(DATA_PATH + "Reserves.csv", delimiter=";")
    heat_rates      = pd.read_csv(DATA_PATH + "HeatRateCurves.csv")
    det_profile     = pd.read_csv(DATA_PATH + "DeterministicProfileRates.csv")
    dynamic_profile = pd.read_csv(DATA_PATH + "DeterministicProfileRates.csv")
    stohastic_profiles = pd.read_csv(DATA_PATH + "StochasticProfileRates.csv")
    bess_units = pd.read_csv(DATA_PATH + "BESS.csv")

    # Load BESS data
    try:
        bess_units = pd.read_csv(DATA_PATH + "BESS.csv")
        print(f"Loaded {len(bess_units)} BESS units")
    except Exception as e:
        print(f"Warning: Could not load BESS data: {e}")
        bess_units = pd.DataFrame()

    # ---------------------------
    # 2) DATA PROCESSING & PREP
    # ---------------------------

    # 2.1 Filter Generators for Belgium only
    generators = generators[generators['BusGenerator'].str.contains("B-|B_", regex=True, na=False)]

    # 2.2 Rename columns for clarity
    generators = generators.rename(columns={
        "Generator":        "Name",
        "MaxRunCapacity":   "P_max",
        "MinRunCapacity":   "P_min",
        "RampUp":           "R_max",
        "RampDown":         "R_min",
        "StartupCost":      "S",
        "NoLoadConsumption":"K",
        "FuelGenerator":    "FuelType",
        "HRSGenerator":     "HeatRateCurve",
        "UT":               "UT_g",
        "DT":               "DT_g"
    })
    renewables = renewables.rename(columns={
        "GeneratorRE":      "Name",
        "ReferenceValueRE": "P_max",
        "FuelGeneratorRE":  "FuelType"
    })

    # 2.3 Filter Renewables for Belgium
    belgium_renewables = renewables[renewables["BusGeneratorRE"]
                                    .str.contains("B-|B_", regex=True, na=False)]
    # Remove WINDOFF_BELWIND,WINDOFF_CPOWER
   #belgium_renewables = belgium_renewables[
    ##]
    # Separate pumped storage from other renewables
    ps_units = belgium_renewables[belgium_renewables["FuelType"] == "HYDRAULIC_PS"].copy()
    other_units = belgium_renewables[belgium_renewables["FuelType"] != "HYDRAULIC_PS"].copy()

    # -----------------------------------------
    # How many pumped‑storage units do we have?
    # (Needed elsewhere so we avoid hard‑coding)
    # -----------------------------------------
    num_ps_units = ps_units.shape[0]

    # Adjust pumped storage parameters as needed
    ps_units['PumpMax'] = 200
    ps_units['ReservoirCap'] = 400
    ps_units['EtaGen'] = 0.77
    ps_units['EtaPump'] = 0.77
    ps_units['InitialSoC'] = 0
    ps_units.rename(columns={"P_max":"GenMax"}, inplace=True)
    ps_units['R_max'] = 0.77*ps_units['PumpMax']
    ps_units['R_min'] = 0.77*ps_units['GenMax']

    # Process BESS units if available
    num_bess_units = 0
    if not bess_units.empty:
        # Set up BESS parameters - add ramp rates if not specified
        if 'RampRateMax' not in bess_units.columns:
            # Default to full power capacity in 15 minutes
            bess_units['RampRateMax'] = bess_units['PowerCapacity']
        if 'RampRateMin' not in bess_units.columns:
            bess_units['RampRateMin'] = bess_units['PowerCapacity']
            
        # Standardize column names to match model expectations
        bess_units = bess_units.rename(columns={
            'PowerCapacity': 'PowerCapacity',
            'EnergyCapacity': 'EnergyCapacity',
            'ChargeEff': 'ChargeEff',
            'DischargeEff': 'DischargeEff',
            'RampRateMax': 'RampRateMax',
            'RampRateMin': 'RampRateMin'
        })
        
        num_bess_units = len(bess_units)
        print(f"Processed {num_bess_units} BESS units")

    # 2.4 Merge Fuel Prices into Generators
    generators = generators.merge(
        fuel_prices[['Fuel', 'FuelHub', 'Price']],
        left_on=["FuelType", "FHGenerator"],
        right_on=["Fuel", "FuelHub"],
        how="left"
    )
    generators.drop(columns=["Fuel","FHGenerator"], inplace=True, errors='ignore')
    generators = generators.drop_duplicates()
    # ------------------------------------------------------------------
    # Convert energy‑based cost inputs (MWh, MWh/h) to monetary values.
    #  - K : no‑load consumption given in MWh/h
    #  - S : startup energy given in MWh/start
    # Multiply by the unit‑specific fuel price (€/MWh) so downstream
    # optimisation models can use euro costs directly.
    # ------------------------------------------------------------------
    generators["Price"] = pd.to_numeric(generators["Price"], errors="coerce").fillna(0)

    for col in ["K", "S"]:
        generators[col] = pd.to_numeric(generators[col], errors="coerce").fillna(0)

    generators["K_eur"] = generators["K"] * generators["Price"]   # €/h
    generators["S_eur"] = generators["S"] * generators["Price"]   # €/start
    print(f"Diagnostic about startup costs: {generators[['Name', 'S', 'Price', 'S_eur']]}")
    # 2.5 Build a separate dictionary for piecewise heat-rate segments
    print(generators.shape)
    segment_data = {}
    for _, row in heat_rates.iterrows():
        key = row["SeriesHRC"]
        if key not in segment_data:
            segment_data[key] = []
        segment_data[key].append((row["SlopeHRC"], row["InterceptHRC"]))
    # Drop rows that have any NaN
    generators = generators.dropna().drop_duplicates()
    
    # 2.6 Build Demand timeseries at 15-min resolution (96 intervals)
    #    - Filter for the chosen day_type
    det_profile = det_profile[det_profile["DayType"] == day_type]
    det_profile = det_profile[det_profile["DynamicProfile"].str.contains("BE_LOAD")]
    
    # Assume columns 2..end are the 96 intervals
    demand_profile = det_profile.iloc[:,2:].copy()
    demand_profile.columns = range(96)

    # Filter loads (and keep only Belgium loads)
    be_load = loads[loads["DynamicProfileLoad"].str.contains("BE_LOAD")]
    be_load = be_load[loads["BusLoad"].str.contains("B-")]
    
    # Multiply normalized profile by total load
    total_be_load = be_load["ReferenceValueLoad"].sum()
    hourly_demand = demand_profile.mul(total_be_load, axis=1)

    # 2.6b Build Reserves timeseries at 15-min resolution
    be_reserves = reserves[reserves["ZoneReserve"].str.strip() == "BE"]
    be_reserves_mFRR = be_reserves[be_reserves["Reserve"].str.contains("mFRR")]
    be_reserves_aFRR = be_reserves[be_reserves["Reserve"].str.contains("aFRR")]

    up_products_mFRR = be_reserves_mFRR[be_reserves_mFRR["Category"].str.contains("UP")]
    down_products_mFRR = be_reserves_mFRR[be_reserves_mFRR["Category"].str.contains("DOWN")]
    up_products_aFRR = be_reserves_aFRR[be_reserves_aFRR["Category"].str.contains("UP")]
    down_products_aFRR = be_reserves_aFRR[be_reserves_aFRR["Category"].str.contains("DOWN")]

    # Restore the original logic that uses demand profiles to scale reserve requirements
    hourly_reserve_req = {}
    for i in be_reserves.index:
        ref_val = be_reserves.loc[i, "ReferenceValueR"]
        df_req = demand_profile.mul(ref_val, axis=1)
        hourly_reserve_req[i] = df_req

    hourly_up_reserve_mFRR = {}
    for i in up_products_mFRR.index:
        ref_val_up = up_products_mFRR.loc[i, "ReferenceValueR"]
        df_req_up = demand_profile.mul(ref_val_up, axis=1)
        hourly_up_reserve_mFRR[i] = df_req_up

    hourly_down_reserve_mFRR = {}
    for i in down_products_mFRR.index:
        ref_val_down = down_products_mFRR.loc[i, "ReferenceValueR"]
        df_req_down = demand_profile.mul(ref_val_down, axis=1)
        hourly_down_reserve_mFRR[i] = df_req_down

    hourly_up_reserve_aFRR = {}
    for i in up_products_aFRR.index:
        ref_val_up = up_products_aFRR.loc[i, "ReferenceValueR"]
        df_req_up = demand_profile.mul(ref_val_up, axis=1)
        hourly_up_reserve_aFRR[i] = df_req_up
    
    hourly_down_reserve_aFRR = {}
    for i in down_products_aFRR.index:
        ref_val_down = down_products_aFRR.loc[i, "ReferenceValueR"]
        df_req_down = demand_profile.mul(ref_val_down, axis=1)
        hourly_down_reserve_aFRR[i] = df_req_down


    SEED_MAP = {
    "AutumnWD": "SEED_2013.09.23",
    "WinterWD": "SEED_2013.01.10",
    "SpringWD": "SEED_2013.04.15",
    "SummerWD": "SEED_2013.07.05",
    "AutumnWE": "SEED_2020.09.23",
    "WinterWE": "SEED_2020.01.10",
    "SpringWE": "SEED_2020.04.15",
    "SummerWE": "SEED_2020.07.05",
    # etc.
    }

    seed = SEED_MAP.get(day_type, "SEED_2013.09.23")
    # 2.7 Build Renewables timeseries at 15-min resolution
    dynamic_profile = dynamic_profile[dynamic_profile["DayType"] == day_type]
    stohastic_profiles = stohastic_profiles[
        (stohastic_profiles["DayType"] == day_type) &
        (stohastic_profiles["Sample"] == seed)
    ]
    matched_profiles = dynamic_profile[
        dynamic_profile["DynamicProfile"].isin(other_units["DynamicProfileRE"])
    ]
    matched_stohastic_profiles = stohastic_profiles[
        stohastic_profiles["DynamicProfile"].isin(other_units["DynamicProfileRE"])
    ]
    # Assume columns 2..end of matched_profiles are the 96 intervals
    renew_profile = matched_profiles.iloc[:,2:].copy()
    renew_profile.columns = range(96)
    stoch_renew_profile = matched_stohastic_profiles.iloc[:,3:].copy()
    stoch_renew_profile.columns = range(96)
    renew_profile.insert(0, "DynamicProfile", matched_profiles["DynamicProfile"].values)
    stoch_renew_profile.insert(0, "DynamicProfile", matched_stohastic_profiles["DynamicProfile"].values)
    renew_profile_full = pd.concat([renew_profile, stoch_renew_profile], ignore_index=True)

    hourly_renewables = other_units.merge(
        renew_profile_full,
        left_on="DynamicProfileRE", right_on="DynamicProfile", how="left"
    )
    for t in range(96):
        hourly_renewables[t] *= hourly_renewables["P_max"]
    hourly_renewables = hourly_renewables[["Name"] + list(range(96))]
    
     
    # 2.8 Build Pumped Storage generation timeseries from DynamicProfileRE
    matched_ps_profiles = dynamic_profile[
        dynamic_profile["DynamicProfile"].isin(ps_units["DynamicProfileRE"])
    ]
    matched_stoch_ps_profiles = stohastic_profiles[
        stohastic_profiles["DynamicProfile"].isin(ps_units["DynamicProfileRE"])
    ]

    ps_profile = matched_ps_profiles.iloc[:,2:].copy()
    ps_profile.columns = range(96)
    stoch_ps_profile = matched_stoch_ps_profiles.iloc[:,3:].copy()
    stoch_ps_profile.columns = range(96)

    ps_profile.insert(0, "DynamicProfile", matched_ps_profiles["DynamicProfile"].values)
    stoch_ps_profile.insert(0, "DynamicProfile", matched_stoch_ps_profiles["DynamicProfile"].values)
    ps_profile_full = pd.concat([ps_profile, stoch_ps_profile], ignore_index=True)

    hourly_ps_gen = ps_units.merge(
        ps_profile_full,
        left_on="DynamicProfileRE", right_on="DynamicProfile", how="left"
    )

    hourly_ps_pump = ps_units.merge(
        ps_profile_full,
        left_on="DynamicProfileRE", right_on="DynamicProfile", how="left"
    )

    for t in range(96):
        # Identify rows generating (positive values) and pumping (non‑positive values)
        positive_mask = hourly_ps_gen[t] > 0.0

        # Generation: scale by GenMax and zero out corresponding pump column
        hourly_ps_gen.loc[positive_mask, t] = (
            hourly_ps_gen.loc[positive_mask, t] * hourly_ps_gen.loc[positive_mask, "GenMax"]
        )
        hourly_ps_pump.loc[positive_mask, t] = 0.0

        # Pumping: scale pump profile by GenMax and zero out corresponding gen column
        negative_mask = ~positive_mask
        hourly_ps_pump.loc[negative_mask, t] = (
            hourly_ps_pump.loc[negative_mask, t] * hourly_ps_gen.loc[negative_mask, "GenMax"]
        )
        hourly_ps_gen.loc[negative_mask, t] = 0.0
        
    hourly_ps_gen = hourly_ps_gen[["Name"] + list(range(96))]
    hourly_ps_pump = hourly_ps_pump[["Name"] + list(range(96))]
    
    print(f"  ✓ data loaded and prepared for {day_type}")
    print(f"These are the hourly ps profiles: {hourly_ps_gen}")
    print(f"These are the hourly ps pump profiles: {hourly_ps_pump}")
    print(f"Number of pumped‑storage units detected: {num_ps_units}")

    # 2.9 Export a CSV listing every unit and its key characteristics
    # ---------------------------------------------------------------
    unit_summary = pd.concat([generators, ps_units, other_units], ignore_index=True, sort=False)
    if not bess_units.empty:
        unit_summary = pd.concat([unit_summary, bess_units], ignore_index=True, sort=False)
    summary_path = DATA_PATH + f"unit_characteristics_{day_type}.csv"
    unit_summary.to_csv(summary_path, index=False)
    print(f"Unit‑characteristics summary written to: {summary_path} (total {len(unit_summary)} units)")


    # Return everything
    data_dict = {
        "generators":         generators,
        "ps_units":           ps_units,
        "bess_units":         bess_units,
        "other_units":        other_units,
        "segment_data":       segment_data,
        "hourly_demand":      hourly_demand,
        "be_reserves":        be_reserves,
        "up_products_mFRR":        up_products_mFRR,
        "down_products_mFRR":      down_products_mFRR,
        "up_products_aFRR":        up_products_aFRR,
        "down_products_aFRR":      down_products_aFRR,
        "hourly_reserve_req": hourly_reserve_req,
        "hourly_renewables":  hourly_renewables,
        "hourly_up_reserve_mFRR":  hourly_up_reserve_mFRR,
        "hourly_down_reserve_mFRR":hourly_down_reserve_mFRR,
        "hourly_up_reserve_aFRR":  hourly_up_reserve_aFRR,
        "hourly_down_reserve_aFRR":hourly_down_reserve_aFRR,
        "hourly_ps_gen":      hourly_ps_gen,
        "hourly_ps_pump":      hourly_ps_pump,
        "num_ps_units":      num_ps_units,
        "num_bess_units":    num_bess_units,
    }
    return data_dict
