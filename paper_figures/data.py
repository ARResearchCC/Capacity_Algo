"""
data.py — tidy long-form loaders for the FOB capacity-planning results.

Reads the three result workbooks and returns ONE clean long-form DataFrame so
every figure pulls from the same source and computes its own means/dispersion.

Architecture is encoded by *file* (there is no architecture column in the raw
workbooks):
    PCM     -> FOB_Sensitivity_Results.xlsx            (PV + battery + hot/cold PCM)
    PVB     -> FOB_PVB_Sensitivity_Results.xlsx        (PV + battery; PCM fixed at 0)
    Diesel  -> FOB_Diesel_Results/FOB_Diesel_Sensitivity_Results.xlsx (genset only)

The wide sheets carry paired 'Training <x>' / 'Testing <x>' columns; these are
melted into a `split` column in {train, test}. Each fold's Testing/Training value
is already a per-year mean over that fold's 5 test / 20 train years; the Summary
sheet is the mean over the 5 folds.

UNITS (all costs are ANNUALIZED $/yr; capacities are installed size):
    pv_kw [kW], battery_kwh [kWh], pcm_hot_kwh / pcm_cold_kwh [kWh_thermal],
    generator_kw [kW], *_cost / *_penalty [$/yr], *_lol_hours [h/yr],
    *_lol_events [events/yr], *_worst_event_h [h], fuel_gallons [gal/yr],
    diesel_price_usd_per_gal [$/gal].

Key functions:
    load_folds(...)   -> per-fold rows (fold in 1..5) for dispersion across folds
    load_summary(...) -> fold-mean rows (matches each file's Summary sheet)
    agg_folds(...)    -> mean + SD across folds for chosen value columns
"""

from __future__ import annotations

import os
import pandas as pd

import style as S

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "PCM":    "FOB_Sensitivity_Results.xlsx",
    "PVB":    "FOB_PVB_Sensitivity_Results.xlsx",
    "Diesel": os.path.join("FOB_Diesel_Results", "FOB_Diesel_Sensitivity_Results.xlsx"),
}

# ---- wide -> tidy column maps -------------------------------------------- #
ID_MAP = {                       # identifiers
    "location": "Location",
    "voll": "VoLL Level",
    "method": "Method",
    "fold": "Fold",
}
INVARIANT_MAP = {                # split-invariant attributes (attach to both splits)
    "pv_kw": "PV_Size",
    "battery_kwh": "Battery_Size",
    "pcm_hot_kwh": "PCM_Heating_Size",
    "pcm_cold_kwh": "PCM_Cooling_Size",
    "generator_kw": "Generator_Size",
    "peak_electric_demand_kw": "Peak_Electric_Demand",
    "reserve_margin": "Reserve_Margin",
    "diesel_price_usd_per_gal": "Diesel Price ($/gal)",
    "fuel_price_level": "Fuel Price Level",
    "worst_training_year": "Worst Training Year",
}
# split-varying metrics: tidy name -> suffix after 'Training '/'Testing '
SUFFIX_MAP = {
    "total_cost": "Total Cost",
    "capital_cost": "Capital Cost",
    "operating_cost": "Operation Cost",
    "hvac_penalty": "HVAC Cost",
    "critical_penalty": "Critical Load Cost",
    "hvac_lol_hours": "HVAC LoL Hours",
    "critical_lol_hours": "Critical LoL Hours",
    "hvac_lol_events": "HVAC LoL Events",
    "critical_lol_events": "Critical LoL Events",
    "hvac_worst_event_h": "HVAC Max LoL Event Hours",
    "critical_worst_event_h": "Critical Max LoL Event Hours",
    # SO_CVaR training-only risk quantities
    "expected_outage_cost": "Expected Outage Cost",
    "cvar_outage_cost": "CVaR Outage Cost",
    "cvar_eta": "CVaR_eta",
    # diesel-only operating breakdown
    "fixed_om_cost": "Fixed OM Cost",
    "variable_om_cost": "Variable OM Cost",
    "fuel_cost": "Fuel Cost",
    "fuel_gallons": "Fuel Gallons",
}

FOLD_SHEETS = [f"Fold_{k}" for k in range(1, 6)]

# tidy columns that are numeric (coerced early so all-NA columns get float dtype,
# which keeps pd.concat from raising the all-NA dtype FutureWarning)
NUMERIC_TIDY = set(SUFFIX_MAP) | {
    "pv_kw", "battery_kwh", "pcm_hot_kwh", "pcm_cold_kwh", "generator_kw",
    "peak_electric_demand_kw", "reserve_margin", "diesel_price_usd_per_gal",
    "worst_training_year", "fold",
}


def _abspath(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def _read_wide(path, kind):
    """Read a workbook into a single wide frame. kind in {folds, summary}."""
    if kind == "summary":
        df = pd.read_excel(path, sheet_name="Summary")
        df["Fold"] = pd.NA
        return df
    frames = []
    xl = pd.ExcelFile(path)
    for sh in FOLD_SHEETS:
        if sh in xl.sheet_names:
            d = pd.read_excel(path, sheet_name=sh)
            if "Fold" not in d.columns:
                d["Fold"] = int(sh.split("_")[1])
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _split_frame(wide, architecture, split):
    prefix = "Training" if split == "train" else "Testing"
    out = pd.DataFrame(index=wide.index)
    out["architecture"] = architecture
    for tidy, col in ID_MAP.items():
        out[tidy] = wide[col] if col in wide.columns else pd.NA
    for tidy, col in INVARIANT_MAP.items():
        out[tidy] = wide[col] if col in wide.columns else pd.NA
    for tidy, suf in SUFFIX_MAP.items():
        col = f"{prefix} {suf}"
        out[tidy] = wide[col] if col in wide.columns else pd.NA
    out["split"] = split
    for c in NUMERIC_TIDY:                      # float dtype even when all-NA
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _tidy(wide, architecture):
    return pd.concat(
        [_split_frame(wide, architecture, "train"),
         _split_frame(wide, architecture, "test")],
        ignore_index=True,
    )


def _finalize(df):
    """Add climate label, ordered categoricals, and pretty method labels."""
    df = df.copy()
    df["climate"] = df["location"].map(S.CLIMATE_LABEL).fillna(df["location"])
    df["method_label"] = df["method"].map(S.METHOD_LABEL).fillna(df["method"])
    df["location"] = pd.Categorical(df["location"], S.LOCATION_ORDER, ordered=True)
    df["climate"] = pd.Categorical(
        df["climate"], [S.CLIMATE_LABEL[l] for l in S.LOCATION_ORDER], ordered=True)
    df["voll"] = pd.Categorical(df["voll"], S.VOLL_ORDER, ordered=True)
    df["method"] = pd.Categorical(
        df["method"], S.METHOD_ORDER + ["Diesel"], ordered=True)
    df["architecture"] = pd.Categorical(
        df["architecture"], S.ARCH_ORDER + ["Diesel"], ordered=True)
    df["split"] = pd.Categorical(df["split"], ["train", "test"], ordered=True)
    # numeric coercion for all value columns
    num_cols = list(INVARIANT_MAP) + list(SUFFIX_MAP) + ["fold"]
    for c in num_cols:
        if c in df.columns and c != "fuel_price_level":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load(kind="folds", architectures=("PCM", "PVB", "Diesel")):
    """Return the tidy long-form table. kind in {folds, summary}."""
    parts = []
    for arch in architectures:
        path = _abspath(FILES[arch])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{arch} results not found: {path}")
        parts.append(_tidy(_read_wide(path, kind), arch))
    return _finalize(pd.concat(parts, ignore_index=True))


def load_folds(architectures=("PCM", "PVB", "Diesel")):
    return load("folds", architectures)


def load_summary(architectures=("PCM", "PVB", "Diesel")):
    return load("summary", architectures)


# VoLL levels ($/kWh): thermal (HVAC) and electrical (critical). From every
# workbook's VoLL_Scenarios sheet; also used to convert penalty $ -> unmet kWh.
VOLL_TABLE = {
    "Low":  {"hvac": 1.0,  "critical": 30.0},
    "Med":  {"hvac": 3.0,  "critical": 100.0},
    "High": {"hvac": 10.0, "critical": 300.0},
}


def add_unmet_energy(df):
    """Add VoLL columns + expected annual UNMET ENERGY (kWh/yr) recovered from the
    penalty terms (penalty_$ = VoLL_$/kWh x unmet_kWh), and cost excluding the VoLL
    penalty (= capital + non-penalty operating), for reliability/frontier figures.

        thermal_unmet_kwh   <- hvac_penalty     / hvac_voll   (HVAC = thermal service)
        electric_unmet_kwh  <- critical_penalty / critical_voll (Critical = electrical)
        total_unmet_kwh     = thermal + electric
        cost_ex_penalty     = total_cost - hvac_penalty - critical_penalty
    """
    df = df.copy()
    v = df["voll"].astype(str)
    hv = v.map(lambda x: VOLL_TABLE.get(x, {}).get("hvac"))
    cv = v.map(lambda x: VOLL_TABLE.get(x, {}).get("critical"))
    df["hvac_voll"] = hv
    df["critical_voll"] = cv
    df["thermal_unmet_kwh"] = df["hvac_penalty"] / hv
    df["electric_unmet_kwh"] = df["critical_penalty"] / cv
    df["total_unmet_kwh"] = df["thermal_unmet_kwh"] + df["electric_unmet_kwh"]
    df["cost_ex_penalty"] = (df["total_cost"]
                             - df["hvac_penalty"].fillna(0)
                             - df["critical_penalty"].fillna(0))
    return df


def agg_folds(df, value_cols, by):
    """Mean + SD across folds. `by` are grouping columns (e.g.
    ['architecture','location','voll','method','split'])."""
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    g = df.groupby(by, observed=True)[value_cols]
    out = g.agg(["mean", "std", "count"])
    out.columns = [f"{v}_{stat}" for v, stat in out.columns]
    return out.reset_index()


if __name__ == "__main__":
    # smoke test / schema print
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    folds = load_folds()
    summ = load_summary()
    print("folds  :", folds.shape, "| summary:", summ.shape)
    print("columns:", list(folds.columns))
    print("\narchitectures:", list(folds["architecture"].cat.categories))
    print("methods      :", sorted(folds["method"].dropna().unique().tolist()))
    print("voll         :", list(folds["voll"].cat.categories))
    print("climates     :", list(folds["climate"].cat.categories))
    print("\nrows per (architecture, split):")
    print(folds.groupby(["architecture", "split"], observed=True).size())
    print("\nexample — test-split, PCM, Med VoLL, California:")
    ex = summ[(summ.architecture == "PCM") & (summ.split == "test")
              & (summ.voll == "Med") & (summ.location == "California")]
    print(ex[["method", "total_cost", "capital_cost", "hvac_penalty",
              "critical_penalty", "hvac_lol_hours", "critical_lol_hours"]].to_string(index=False))
