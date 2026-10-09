"""
FOB scenario, diesel-only baseline.

Diesel generator (or generator fleet, modeled as one continuously-sizable capacity)
supplies all electricity: the same heat pump serves heating and cooling, but there is
no PV, battery, or PCM. Generator capacity is a single deterministic design per
location, sized once from the peak electric demand (critical load + heat-pump
electrical demand) across all 25 weather years, plus a reserve margin -- see
Diesel_Model.size_generator. There is no LP_Avg/LP_Worst/SO_CVaR method comparison:
diesel sizing is a peak-plus-margin rule, not a cost/reliability trade-off
optimization.

Delivered diesel price is swept over FUEL_PRICE_SCENARIOS (Low/Med/High), reflecting
the wide, literature-documented range between a secure-base commodity price and a
contested forward-operating-base "fully burdened cost of fuel" (FBCF: procurement +
transport + force protection for the logistics tail). See the citations below. Because
fuel cost is linear in price at fixed dispatch, this sweep is exactly what a
breakeven-delivered-fuel-price comparison against a renewable design's total cost needs
(Diesel_Model.breakeven_fuel_price).

Sources for FUEL_PRICE_SCENARIOS:
  JASON/MITRE Corp., "Reducing DoD Fossil-Fuel Dependence," JSR-06-135 (Sep 2006):
    fully burdened cost of fuel (FBCF) = $100-$600/gal in theater, depending on
    "front line" to "back line" separation in distance, terrain, and defense.
  Deloitte, "Energy Security: America's Best Defense" (2009): ~$400/gal for
    division-scale remote FOB resupply, the figure most widely cited by DoD/press.
  Army Environmental Policy Institute, "Sustain the Mission Project: Casualty Factors
    for Fuel and Water Resupply Convoys" (2009): quantifies the convoy casualty risk
    (~1 per 24 resupply convoys in Afghanistan) underlying FBCF's force-protection
    component.

The output workbook (FOB_Diesel_Results/FOB_Diesel_Sensitivity_Results.xlsx) matches
the sheet/column layout of FOB_Sensitivity_Results.xlsx / FOB_PVB_Sensitivity_Results.xlsx
(Config, VoLL_Scenarios, Summary, Fold_1..Fold_5, plus a Fuel_Price_Scenarios sheet) so
a downstream plotting script can read all three files the same way. Since the
generator design does not depend on the fold split, "Training"/"Testing" columns here
are the same fixed design evaluated on that fold's training-year subset vs. held-out
test-year subset (not a trained result) -- this reuses the exact fold splits from
FOB_PVB.py so test years line up across files. CAPACITY_KEYS (PV/battery/PCM sizes)
are reported as 0; Generator_Size, Peak_Electric_Demand, Reserve_Margin, and
Fuel Price Level / Diesel Price ($/gal) are appended as diesel-specific columns.

Load data (weather, electrical load, heating/cooling load) is built with the exact
same function used for the PV-battery architecture (FOB_PVB.build_input_data), so the
diesel baseline is evaluated against an identical load to both FOB_PVB.py and FOB.py.
"""

import os

import pandas as pd

import Diesel_Model
import FOB_PVB
import Input_Parameters

scenario = "FOB"
locations = FOB_PVB.locations
weather_year_list = FOB_PVB.weather_year_list
VOLL_SCENARIOS = FOB_PVB.VOLL_SCENARIOS
RESERVE_MARGIN = Input_Parameters.Gen_Reserve_Margin
fold = FOB_PVB.fold

# Delivered diesel fuel price ($/gal) sensitivity -- see module docstring for sources.
FUEL_PRICE_SCENARIOS = {
    "Low": 13.0,                           # peacetime forward ground delivery (Nat'l Defense Mag 2010)
    "Med": Input_Parameters.Diesel_Price,  # protected convoy, ~$45/gal (defensible burdened tier)
    "High": 400.0,                         # contested/helicopter extreme (Afghanistan; documented upper bound)
}

algorithms = ["Diesel"]
folder_name = "FOB_Diesel_Results"
output_file = os.path.join(folder_name, "FOB_Diesel_Sensitivity_Results.xlsx")

CAPACITY_KEYS = FOB_PVB.CAPACITY_KEYS
TRAINING_KEYS = FOB_PVB.TRAINING_KEYS
TESTING_COST_KEYS = FOB_PVB.TESTING_COST_KEYS
TESTING_LOL_KEYS = FOB_PVB.TESTING_LOL_KEYS
TESTING_KEYS = FOB_PVB.TESTING_KEYS

GROUP_COLS = ["Location", "VoLL Level", "Fuel Price Level", "Method"]

BASE_COL_ORDER = (
    ["Location", "VoLL Level", "Fuel Price Level", "Method", "Fold"]
    + CAPACITY_KEYS
    + TRAINING_KEYS
    + TESTING_COST_KEYS
    + TESTING_LOL_KEYS
)


def prefix_keys(result, prefix):
    """{'Total Cost': x, ...} -> {'Training Total Cost': x, ...} (or 'Testing ...')."""
    return {f"{prefix} {k}": v for k, v in result.items()}


def zero_capacities():
    return {key: 0.0 for key in CAPACITY_KEYS}


def run_location(nested_dict, location, fold_splits):
    all_years_data = [nested_dict[location][year] for year in weather_year_list]
    gen_size, peak_demand = Diesel_Model.size_generator(
        all_years_data, reserve_margin=RESERVE_MARGIN, scenario=scenario
    )

    rows = []
    for voll_name, voll in VOLL_SCENARIOS.items():
        for price_name, diesel_price in FUEL_PRICE_SCENARIOS.items():
            with FOB_PVB.hvac_voll_context(voll["hvac_voll"]):
                per_year = {
                    year: Diesel_Model.simulate_diesel(
                        nested_dict[location][year],
                        voll["critical_voll"],
                        gen_size,
                        diesel_price=diesel_price,
                        scenario=scenario,
                    )
                    for year in weather_year_list
                }

            for k, (training_years, testing_years) in enumerate(fold_splits):
                train_avg = Diesel_Model.average_results(
                    {y: per_year[y] for y in training_years}
                )
                test_avg = Diesel_Model.average_results(
                    {y: per_year[y] for y in testing_years}
                )

                row = {
                    "Location": location,
                    "VoLL Level": voll_name,
                    "Fuel Price Level": price_name,
                    "Diesel Price ($/gal)": diesel_price,
                    "Method": "Diesel",
                    "Fold": k + 1,
                    "Generator_Size": round(gen_size, 3),
                    "Peak_Electric_Demand": round(peak_demand, 3),
                    "Reserve_Margin": RESERVE_MARGIN,
                }
                row.update(zero_capacities())
                row.update(prefix_keys(train_avg, "Training"))
                row.update(prefix_keys(test_avg, "Testing"))
                rows.append(row)
    return rows


def build_summary(detail_df):
    numeric_cols = [
        c
        for c in detail_df.columns
        if c not in GROUP_COLS + ["Fold"] and pd.api.types.is_numeric_dtype(detail_df[c])
    ]
    summary_rows = []
    for group_key, grp in detail_df.groupby(GROUP_COLS):
        row = dict(zip(GROUP_COLS, group_key))
        for col in numeric_cols:
            row[col] = grp[col].mean()
        row["Folds Averaged"] = fold
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).round(3)


def make_config_dataframe():
    voll_rows = [
        {
            "VoLL Level": name,
            "HVAC VoLL ($/kWh)": vals["hvac_voll"],
            "Critical VoLL ($/kWh)": vals["critical_voll"],
        }
        for name, vals in VOLL_SCENARIOS.items()
    ]
    fuel_price_rows = [
        {"Fuel Price Level": name, "Diesel Price ($/gal)": price}
        for name, price in FUEL_PRICE_SCENARIOS.items()
    ]
    n_runs = len(locations) * len(VOLL_SCENARIOS) * len(FUEL_PRICE_SCENARIOS)
    config = pd.DataFrame(
        [
            {"Parameter": "Scenario", "Value": scenario},
            {
                "Parameter": "Sizing",
                "Value": (
                    "Generator sized once per location to (1 + reserve margin) x peak "
                    "electric demand (critical + heat pump) across all weather years; "
                    "no PV, battery, or PCM; fixed design evaluated on each fold's "
                    "training/testing year split"
                ),
            },
            {"Parameter": "Reserve margin", "Value": RESERVE_MARGIN},
            {"Parameter": "Locations", "Value": ", ".join(locations)},
            {"Parameter": "Folds", "Value": fold},
            {
                "Parameter": "Weather years",
                "Value": f"{weather_year_list[0]}-{weather_year_list[-1]}",
            },
            {"Parameter": "Total sensitivity runs", "Value": n_runs},
            {"Parameter": "Methods", "Value": ", ".join(algorithms)},
            {"Parameter": "Rows per fold sheet", "Value": n_runs * len(algorithms)},
            {"Parameter": "Generator capital cost ($/kW)", "Value": Input_Parameters.C_Gen},
            {"Parameter": "Generator fixed O&M ($/kW/yr)", "Value": Input_Parameters.C_Gen_OP},
            {"Parameter": "Generator variable O&M ($/kWh)", "Value": Input_Parameters.C_Gen_VOM},
            {"Parameter": "Generator lifetime (yr)", "Value": Input_Parameters.Gen_Lifetime},
            {"Parameter": "Fuel curve F0 (gal/hr per kW rated)", "Value": Input_Parameters.Fuel_Curve_F0},
            {"Parameter": "Fuel curve F1 (gal/hr per kW output)", "Value": Input_Parameters.Fuel_Curve_F1},
            {
                "Parameter": "Fuel price source",
                "Value": (
                    "FBCF ladder (Nat'l Defense Mag 2010; JASON JSR-06-135 2006; GAO-09-300): "
                    "$13/gal peacetime forward, $45/gal protected convoy, $400/gal contested "
                    "extreme; hostile-area FBCF range $100-600/gal"
                ),
            },
        ]
    )
    return config, pd.DataFrame(voll_rows), pd.DataFrame(fuel_price_rows)


def order_columns(df, base_cols):
    ordered = [c for c in base_cols if c in df.columns]
    extra = [c for c in df.columns if c not in base_cols]
    return ordered + extra


def write_excel(all_fold_rows, summary_df, config_df, voll_df, fuel_price_df):
    os.makedirs(folder_name, exist_ok=True)
    fold_frames = {k: [] for k in range(fold)}
    for row in all_fold_rows:
        fold_frames[row["Fold"] - 1].append(row)

    expected_rows = len(locations) * len(VOLL_SCENARIOS) * len(FUEL_PRICE_SCENARIOS) * len(algorithms)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        config_df.to_excel(writer, sheet_name="Config", index=False)
        voll_df.to_excel(writer, sheet_name="VoLL_Scenarios", index=False)
        fuel_price_df.to_excel(writer, sheet_name="Fuel_Price_Scenarios", index=False)

        summary_df = summary_df.sort_values(GROUP_COLS).reset_index(drop=True)
        summary_cols = order_columns(summary_df, BASE_COL_ORDER)
        summary_df[summary_cols].to_excel(writer, sheet_name="Summary", index=False)

        for k in range(fold):
            df = pd.DataFrame(fold_frames[k])
            if len(df) != expected_rows:
                raise ValueError(
                    f"Fold_{k + 1}: expected {expected_rows} rows, got {len(df)}"
                )
            df = df.sort_values(GROUP_COLS).reset_index(drop=True)
            use_cols = order_columns(df, BASE_COL_ORDER)
            df[use_cols].to_excel(writer, sheet_name=f"Fold_{k + 1}", index=False)


def main():
    fold_splits = FOB_PVB.get_fold_splits()
    FOB_PVB.validate_fold_splits(fold_splits)

    print("Building input data for all locations and years...")
    nested_dict = FOB_PVB.build_input_data()

    all_rows = []
    for loc in locations:
        print(f"\n=== Diesel baseline: {loc} ===")
        all_rows.extend(run_location(nested_dict, loc, fold_splits))

    detail_df = pd.DataFrame(all_rows)
    expected_total_rows = (
        len(locations) * len(VOLL_SCENARIOS) * len(FUEL_PRICE_SCENARIOS) * len(algorithms) * fold
    )
    if len(detail_df) != expected_total_rows:
        raise ValueError(
            f"Expected {expected_total_rows} result rows, got {len(detail_df)}"
        )

    summary_df = build_summary(detail_df)
    config_df, voll_df, fuel_price_df = make_config_dataframe()
    write_excel(all_rows, summary_df, config_df, voll_df, fuel_price_df)

    print(f"\nDone. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
