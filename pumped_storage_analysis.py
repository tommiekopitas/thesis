# pumped_storage_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

###############################################################################
# 0. Configuration – edit as needed
###############################################################################
CSV_PATH   = "/Users/tommie/Documents/thesis/project/data/generation_per_fuel_BE_2023.csv"  # path to the CSV file
TZ         = "Europe/Brussels"          # MTU is in CET/CEST
PLOT_DAY   = "2023-01-01"               # any date in the file
TOP_N_BINS = 4                          # 4 × 15 min = top hour of each day

###############################################################################
# 1. Read & tidy
###############################################################################
df = (
    pd.read_csv(CSV_PATH)
      .replace({"n/e": None, "": None})
      .assign(**{
          "Hydro Pumped Storage  - Actual Aggregated [MW]":
              lambda d: pd.to_numeric(d["Hydro Pumped Storage  - Actual Aggregated [MW]"], errors="coerce"),
          "Hydro Pumped Storage  - Actual Consumption [MW]":
              lambda d: pd.to_numeric(d["Hydro Pumped Storage  - Actual Consumption [MW]"], errors="coerce")
      })
)

###############################################################################
# 2. Make a timestamp for the **start** of each MTU block
###############################################################################
def mtu_to_start(ts: str) -> pd.Timestamp:
    """
    Convert the *left* boundary of an MTU string to a timezone‑aware Timestamp.

    The raw MTU cell looks like
        "01.01.2023 00:00 - 01.01.2023 01:00 (CET/CEST)"
    We keep the left boundary ("01.01.2023 00:00"), parse it as a naive
    datetime and then localise it to Europe/Brussels.

    Older/newer pandas versions sometimes reject `ambiguous="infer"`.  Using
    `ambiguous="NaT"` is widely supported and safely marks the repeated hour
    during the DST fall‑back as NaT; those rows are later dropped.  For the
    spring‑forward gap we try `nonexistent="shift_forward"` (pandas ≥1.1).  If
    that keyword is unavailable we fall back to a simpler call.
    """
    ts_left = ts.split(" - ")[0]
    naive = pd.to_datetime(ts_left, format="%d.%m.%Y %H:%M")
    try:
        # Preferred path (works on pandas ≥1.1)
        return naive.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    except TypeError:
        # Fallback for very old pandas that lack the `nonexistent` keyword
        return naive.tz_localize(TZ, ambiguous="NaT")

df["timestamp"] = df["MTU"].map(mtu_to_start)
# Drop any MTU rows where the timestamp could not be localised (NaT)
df = df.dropna(subset=["timestamp"])
df = df.set_index("timestamp").sort_index()

###############################################################################
# 3. Keep only the two pumped-storage columns
###############################################################################
gen   = df["Hydro Pumped Storage  - Actual Aggregated [MW]"].rename("generation_MW")
pump  = df["Hydro Pumped Storage  - Actual Consumption [MW]"].rename("pumping_MW")

data_hr = pd.concat([gen, pump], axis=1).dropna(how="all")

###############################################################################
# 4. Upsample to 15-min resolution (forward-fill each hourly average)
###############################################################################
data_qh = (
    data_hr
      .resample("15T")
      .ffill()
)

###############################################################################
# 5. Daily aggregates & capacity proxy
###############################################################################
# Energy = MW × 0.25 h per 15-min bin
daily = pd.DataFrame({
    "MWh_generation": data_qh["generation_MW"].clip(lower=0).groupby(pd.Grouper(freq="1D")).sum() * 0.25,
    "MWh_pumping"   : data_qh["pumping_MW"].clip(lower=0).groupby(pd.Grouper(freq="1D")).sum() * 0.25,
})

# “Rated power” proxy: mean of the day’s TOP_N_BINS highest MW values
def top_mean(s: pd.Series) -> float:
    return s.nlargest(TOP_N_BINS).mean()

daily["MW_max_gen"]  = data_qh["generation_MW"].groupby(pd.Grouper(freq="1D")).apply(top_mean)
daily["MW_max_pump"] = data_qh["pumping_MW"   ].groupby(pd.Grouper(freq="1D")).apply(top_mean)
daily["MW_capacity"] = daily[["MW_max_gen", "MW_max_pump"]].max(axis=1)   # bigger of the two
daily["MWh_net"]     = daily["MWh_generation"] - daily["MWh_pumping"]

###############################################################################
# 5b. Infer number of pumped‑storage units and build ps_units table
###############################################################################
# Whole‑plant maximum power (over the full data set) in both directions
plant_max_gen  = daily["MW_max_gen"].max()      # max generation MW
plant_max_pump = daily["MW_max_pump"].max()     # max pumping   MW
# A robust 95‑percentile figure is still kept for the ramp‑step heuristic
pump_max_p95   = daily["MW_max_pump"].quantile(0.95)

# Usable storage swing from cumulative net energy
data_qh["net_MW"]  = data_qh["generation_MW"].fillna(0) - data_qh["pumping_MW"].fillna(0)
data_qh["net_MWh"] = data_qh["net_MW"] * 0.25  # 15‑min bin → hours
cum = data_qh["net_MWh"].cumsum()
reservoir_MWh = cum.max() - cum.min()

# ---------------------------------------------------------------------------
# Detect how many reversible units exist
# 1) Try metadata (generators.csv next to the raw file)               
# 2) Fallback: infer from most common MW step size in generation ramps
# ---------------------------------------------------------------------------
N_UNITS = None
meta_path = Path(CSV_PATH).parent / "generators.csv"
try:
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        mask = meta["GeneratorType"].astype(str).str.contains(r"pumped|ps", case=False, na=False)
        if mask.any():
            N_UNITS = int(mask.sum())
except Exception:
    N_UNITS = None  # fall through to heuristic

if not N_UNITS:
    steps = data_qh["generation_MW"].diff().abs()
    # ignore AGC jitter (<10 % of plant max) and coarse‑bin the rest to nearest 100 MW
    candidate = steps[(steps > plant_max_pump * 0.10) & (steps < plant_max_pump)].round(-2)
    if not candidate.empty:
        mode_step = candidate.value_counts().idxmax()
        # ignore small regulation moves (<10 % of plant max) or impossible jumps (>plant max)
        N_UNITS = max(1, int(round(plant_max_pump / mode_step)))
    else:
        N_UNITS = 1  # conservative default

print(f"Detected {N_UNITS} pumped‑storage unit(s)")
print(f"Plant max generation power: {plant_max_gen:.0f} MW")
print(f"Plant max pumping power   : {plant_max_pump:.0f} MW")

#
# ---------------------------------------------------------------------------
# Per‑unit parameters and table
# ---------------------------------------------------------------------------
per_unit = {
    "MaxGen"      : plant_max_gen  / N_UNITS,
    "MaxPump"     : plant_max_pump / N_UNITS,
    "ReservoirCap": reservoir_MWh  / N_UNITS,
    "EtaGen"      : 0.90,
    "EtaPump"     : 0.90,
    "InitialSoC"  : 0.50 * (reservoir_MWh / N_UNITS),
}

unit_names = [f"PS_Unit{u+1}" for u in range(N_UNITS)]
ps_units = pd.DataFrame([per_unit] * N_UNITS, index=unit_names)

ps_units.to_csv("ps_units_summary.csv", float_format="%.2f")
print("Wrote: ps_units_summary.csv")

###############################################################################
# 6. Save the daily summary
###############################################################################
daily.to_csv("coo_daily_summary_2023.csv", float_format="%.2f")
print("Wrote: coo_daily_summary_2023.csv")

###############################################################################
# 7. PLOTS
###############################################################################
# --- 7a. 24-h profile for PLOT_DAY ------------------------------------------
sel = data_qh.loc[PLOT_DAY]
ax = sel.plot(title=f"Coo-Trois-Ponts – 15-min profile on {PLOT_DAY}", figsize=(10,4))
ax.set_ylabel("MW")
ax.legend(["Generation (turbines)", "Pumping (motors)"])
plt.tight_layout()
plt.savefig(f"profile_{PLOT_DAY}.png", dpi=120)

# --- 7b. Daily capacity estimate -------------------------------------------
fig, ax = plt.subplots(figsize=(10,4))
daily["MW_capacity"].plot(ax=ax)
ax.set_title("Estimated rated power (daily rolling proxy)")
ax.set_ylabel("MW")
plt.tight_layout()
plt.savefig("daily_capacity_proxy.png", dpi=120)

# --- 7c. Daily net energy ---------------------------------------------------
fig, ax = plt.subplots(figsize=(10,4))
daily["MWh_net"].plot(ax=ax, color="grey")
ax.set_title("Daily net energy (generation – pumping)")
ax.set_ylabel("MWh")
plt.tight_layout()
plt.savefig("daily_net_energy.png", dpi=120)

print("Plots saved:")
print(" • profile_<date>.png")
print(" • daily_capacity_proxy.png")
print(" • daily_net_energy.png")