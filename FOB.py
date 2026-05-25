"""
FOB scenario: 5-fold cross-validation with 2D sensitivity (5 locations x 3 VoLL levels).

Methods:
  - LP_Avg:    run LP once per training year, average capacities (as in New_SA_Scenario.py)
  - LP_Worst:  run LP once per training year, keep capacities from highest-cost training year
  - SO_CVaR:   CVaR-based stochastic optimization (SO_CVaR.py)
"""

import os
from contextlib import contextmanager

import numpy as np
import pandas as pd

import Baseline_CO
import Data_Conversion
import Electrical_Load
import Input_Parameters
import Passive_Model
import SO_CVaR
import Solar_Generation
import Simulate

# =============================================================================
# User inputs
# =============================================================================

data_dir = "Data"
scenario = "FOB"
locations = ["California", "Arizona", "Alaska", "Minnesota", "Florida"]

# VoLL sensitivity: HVAC ($/kWh) and critical electrical ($/kWh) for FOB
VOLL_SCENARIOS = {
    "Low": {"hvac_voll": 1.0, "critical_voll": 100.0},
    "Med": {
        "hvac_voll": Input_Parameters.HVAC_lol_cost,
        "critical_voll": Input_Parameters.lossofloadcost,
    },
    "High": {"hvac_voll": 10.0, "critical_voll": 600.0},
}

# CVaR stochastic optimization settings
CVAR_LAMBDA = 1.0
CVAR_ALPHA = Input_Parameters.CVaR_alpha

fold = 5
weather_year_list = list(range(1998, 2023))
capacity_costs = [
    Input_Parameters.C_PV,
    Input_Parameters.C_PV_OP,
    Input_Parameters.C_B,
    Input_Parameters.C_B_OP,
]

algorithms = ["LP_Avg", "LP_Worst", "SO_CVaR"]
folder_name = "FOB_Results"
output_file = os.path.join(folder_name, "FOB_Sensitivity_Results.xlsx")

CAPACITY_KEYS = ["PV_Size", "Battery_Size", "PCM_Heating_Size", "PCM_Cooling_Size"]
TRAINING_KEYS = [
    "Training Total Cost",
    "Training Capital Cost",
    "Training Operation Cost",
    "Training HVAC Cost",
    "Training Critical Load Cost",
]
TESTING_COST_KEYS = [
    "Testing Total Cost",
    "Testing Capital Cost",
    "Testing Operation Cost",
    "Testing HVAC Cost",
    "Testing Critical Load Cost",
]
TESTING_LOL_KEYS = [
    "Testing HVAC LoL Hours",
    "Testing Critical LoL Hours",
    "Testing HVAC LoL Events",
    "Testing Critical LoL Events",
    "Testing HVAC Max LoL Event Hours",
    "Testing Critical Max LoL Event Hours",
]
TESTING_KEYS = TESTING_COST_KEYS + TESTING_LOL_KEYS
CVAR_TRAINING_KEYS = [
    "Training Expected Outage Cost",
    "Training CVaR Outage Cost",
    "Training CVaR_eta",
]


@contextmanager
def hvac_voll_context(hvac_voll):
    """Temporarily set HVAC VoLL (read from Input_Parameters in LP/SO/Simulate)."""
    original = Input_Parameters.HVAC_lol_cost
    Input_Parameters.HVAC_lol_cost = hvac_voll
    try:
        yield
    finally:
        Input_Parameters.HVAC_lol_cost = original


def pack_lp_result(
    pv, battery, pcm_h, pcm_c, obj, first, second, hvac, critical
):
    return {
        "PV_Size": pv,
        "Battery_Size": battery,
        "PCM_Heating_Size": pcm_h,
        "PCM_Cooling_Size": pcm_c,
        "Training Total Cost": obj,
        "Training Capital Cost": first,
        "Training Operation Cost": second,
        "Training HVAC Cost": hvac,
        "Training Critical Load Cost": critical,
    }


def run_lp_training(nested_dict, location, training_years, critical_voll):
    per_year = {}
    for year in training_years:
        input_df = nested_dict[location][year]
        result = Baseline_CO.Cap_Baseline_V1(
            input_df, critical_voll, capacity_costs, scenario
        )
        per_year[year] = pack_lp_result(*result)
    return per_year


def average_results(per_year, num_years, keys):
    averaged = {key: 0.0 for key in keys}
    for row in per_year.values():
        for key in keys:
            averaged[key] += row[key]
    for key in keys:
        averaged[key] /= num_years
    return averaged


def worst_year_results(per_year):
    worst_year = max(
        per_year,
        key=lambda y: per_year[y]["Training Total Cost"],
    )
    result = {k: per_year[worst_year][k] for k in per_year[worst_year]}
    result["Worst Training Year"] = worst_year
    return result


def pack_simulate_result(sim_result):
    """Map Simulate.simulate return tuple to labeled testing metrics."""
    if len(sim_result) != 11:
        raise ValueError(
            f"Simulate.simulate must return 11 values; got {len(sim_result)}. "
            "Update pack_simulate_result if Simulate.py changed."
        )
    return {
        "Testing Total Cost": sim_result[0],
        "Testing Capital Cost": sim_result[1],
        "Testing Operation Cost": sim_result[2],
        "Testing HVAC Cost": sim_result[3],
        "Testing Critical Load Cost": sim_result[4],
        "Testing HVAC LoL Hours": sim_result[5],
        "Testing Critical LoL Hours": sim_result[6],
        "Testing HVAC LoL Events": sim_result[7],
        "Testing Critical LoL Events": sim_result[8],
        "Testing HVAC Max LoL Event Hours": sim_result[9],
        "Testing Critical Max LoL Event Hours": sim_result[10],
    }


def run_testing(nested_dict, location, testing_years, critical_voll, capacities):
    """Simulate fixed capacities on each test year; average metrics across test years."""
    per_year = {}
    for year in testing_years:
        input_df = nested_dict[location][year]
        sim_result = Simulate.simulate(
            input_df, critical_voll, capacities, capacity_costs, scenario
        )
        per_year[year] = pack_simulate_result(sim_result)
    return per_year


def average_testing(per_year, num_years):
    return average_results(per_year, num_years, TESTING_KEYS)


def run_so_cvar_training(nested_dict, location, training_years, critical_voll):
    input_df_list = [nested_dict[location][year] for year in training_years]
    result = SO_CVaR.SO_CVaR_training(
        input_df_list,
        critical_voll,
        capacity_costs,
        scenario,
        cvar_alpha=CVAR_ALPHA,
        cvar_lambda=CVAR_LAMBDA,
    )
    return {
        "PV_Size": result[0],
        "Battery_Size": result[1],
        "PCM_Heating_Size": result[2],
        "PCM_Cooling_Size": result[3],
        "Training Total Cost": result[4],
        "Training Capital Cost": result[5],
        "Training Operation Cost": result[6],
        "Training HVAC Cost": result[7],
        "Training Critical Load Cost": result[8],
        "Training Expected Outage Cost": result[9],
        "Training CVaR Outage Cost": result[10],
        "Training CVaR_eta": result[11],
    }


def capacities_from_training(training_row):
    return [
        training_row["PV_Size"],
        training_row["Battery_Size"],
        training_row["PCM_Heating_Size"],
        training_row["PCM_Cooling_Size"],
    ]


def combine_training_testing(training_row, testing_row):
    combined = {}
    combined.update(training_row)
    combined.update(testing_row)
    return combined


def build_input_data():
    j = len(weather_year_list)
    random_seeds = np.arange(1, j + 1)
    lats, lons, timezones = Data_Conversion.get_timezones(data_dir, locations)
    nested_dict = {
        loc: {year: {} for year in weather_year_list} for loc in locations
    }

    for i, loc in enumerate(locations):
        for j_idx, year in enumerate(weather_year_list):
            random_seed = random_seeds[j_idx]
            nsrdb = Data_Conversion.read_NSRDB(data_dir, loc, year)
            weather_data = Data_Conversion.prepare_NSRDB(
                nsrdb, lats[i], lons[i], timezones[i]
            )
            net_heat = Passive_Model.passive_model(
                Input_Parameters.calibration_file_path,
                weather_data,
                Input_Parameters.T_indoor_constant,
                lats[i],
            )
            pv_cf = Solar_Generation.generate_pv(weather_data, lats[i])
            load_sched = Electrical_Load.generate_schedules(
                scenario, weather_data, random_seed
            )
            nested_dict[loc][year] = Data_Conversion.combine_input_NSRDB(
                weather_data, load_sched, pv_cf, net_heat
            )
    return nested_dict


def get_fold_splits():
    """Same 5-fold year blocks as New_SA_Scenario.py (5 test years per fold)."""
    splits = []
    for k in range(fold):
        start_idx = k * 5
        testing_years = weather_year_list[start_idx : start_idx + 5]
        training_years = [
            y for y in weather_year_list if y not in testing_years
        ]
        splits.append((training_years, testing_years))
    return splits


def validate_fold_splits(splits):
    expected_test = 5
    expected_train = len(weather_year_list) - expected_test
    for k, (train_years, test_years) in enumerate(splits):
        if len(test_years) != expected_test:
            raise ValueError(
                f"Fold {k + 1}: expected {expected_test} test years, got {len(test_years)}"
            )
        if len(train_years) != expected_train:
            raise ValueError(
                f"Fold {k + 1}: expected {expected_train} training years, "
                f"got {len(train_years)}"
            )
        overlap = set(train_years) & set(test_years)
        if overlap:
            raise ValueError(f"Fold {k + 1}: train/test overlap {overlap}")


def build_summary(detail_df):
    """Mean over folds; omit metrics that do not apply to a given method."""
    group_cols = ["Location", "VoLL Level", "Method"]
    summary_rows = []
    for group_key, grp in detail_df.groupby(group_cols):
        if isinstance(group_key, tuple):
            row = dict(zip(group_cols, group_key))
        else:
            row = {group_cols[0]: group_key}
        method = row["Method"]
        keys = CAPACITY_KEYS + TRAINING_KEYS + TESTING_KEYS
        if method == "SO_CVaR":
            keys = keys + CVAR_TRAINING_KEYS
        # Worst Training Year is per-fold metadata; do not average across folds
        for col in keys:
            if col in grp.columns and grp[col].notna().any():
                row[col] = grp[col].mean()
        row["Folds Averaged"] = fold
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).round(3)


def make_config_dataframe():
    voll_rows = []
    for name, vals in VOLL_SCENARIOS.items():
        voll_rows.append(
            {
                "VoLL Level": name,
                "HVAC VoLL ($/kWh)": vals["hvac_voll"],
                "Critical VoLL ($/kWh)": vals["critical_voll"],
            }
        )
    config = pd.DataFrame(
        [
            {"Parameter": "Scenario", "Value": scenario},
            {"Parameter": "Locations", "Value": ", ".join(locations)},
            {"Parameter": "CVaR lambda", "Value": CVAR_LAMBDA},
            {"Parameter": "CVaR alpha", "Value": CVAR_ALPHA},
            {"Parameter": "Folds", "Value": fold},
            {
                "Parameter": "Weather years",
                "Value": f"{weather_year_list[0]}-{weather_year_list[-1]}",
            },
            {"Parameter": "Total sensitivity runs", "Value": len(locations) * len(VOLL_SCENARIOS)},
            {"Parameter": "Methods", "Value": ", ".join(algorithms)},
            {"Parameter": "Rows per fold sheet", "Value": len(locations) * len(VOLL_SCENARIOS) * len(algorithms)},
        ]
    )
    return config, pd.DataFrame(voll_rows)


def results_to_row(location, voll_level, method, fold_idx, combined):
    row = {
        "Location": location,
        "VoLL Level": voll_level,
        "Method": method,
        "Fold": fold_idx + 1,
    }
    row.update(combined)
    return row


def write_excel(all_fold_rows, summary_df, config_df, voll_df):
    os.makedirs(folder_name, exist_ok=True)
    fold_frames = {k: [] for k in range(fold)}
    for row in all_fold_rows:
        fold_frames[row["Fold"] - 1].append(row)

    expected_rows = len(locations) * len(VOLL_SCENARIOS) * len(algorithms)
    col_order = (
        ["Location", "VoLL Level", "Method", "Fold"]
        + CAPACITY_KEYS
        + TRAINING_KEYS
        + CVAR_TRAINING_KEYS
        + ["Worst Training Year"]
        + TESTING_COST_KEYS
        + TESTING_LOL_KEYS
    )

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        config_df.to_excel(writer, sheet_name="Config", index=False)
        voll_df.to_excel(writer, sheet_name="VoLL_Scenarios", index=False)
        summary_df = summary_df.sort_values(
            ["Location", "VoLL Level", "Method"]
        ).reset_index(drop=True)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        for k in range(fold):
            df = pd.DataFrame(fold_frames[k])
            if len(df) != expected_rows:
                raise ValueError(
                    f"Fold_{k + 1}: expected {expected_rows} rows, got {len(df)}"
                )
            df["Method"] = pd.Categorical(
                df["Method"], categories=algorithms, ordered=True
            )
            df = df.sort_values(
                ["Location", "VoLL Level", "Method"]
            ).reset_index(drop=True)
            use_cols = [c for c in col_order if c in df.columns]
            df[use_cols].to_excel(writer, sheet_name=f"Fold_{k + 1}", index=False)


def main():
    fold_splits = get_fold_splits()
    validate_fold_splits(fold_splits)

    print("Building input data for all locations and years...")
    nested_dict = build_input_data()

    all_rows = []
    run_count = 0
    total_runs = len(locations) * len(VOLL_SCENARIOS)

    for loc in locations:
        for voll_name, voll in VOLL_SCENARIOS.items():
            run_count += 1
            critical_voll = voll["critical_voll"]
            hvac_voll = voll["hvac_voll"]
            print(
                f"\n=== Run {run_count}/{total_runs}: "
                f"{loc} | VoLL={voll_name} "
                f"(HVAC={hvac_voll}, Critical={critical_voll}) ==="
            )

            with hvac_voll_context(hvac_voll):
                for k, (training_years, testing_years) in enumerate(fold_splits):
                    n_train = len(training_years)
                    n_test = len(testing_years)

                    # ----- LP per year -----
                    lp_per_year = run_lp_training(
                        nested_dict, loc, training_years, critical_voll
                    )
                    lp_avg = average_results(
                        lp_per_year, n_train, CAPACITY_KEYS + TRAINING_KEYS
                    )
                    lp_worst = worst_year_results(lp_per_year)

                    # ----- SO_CVaR -----
                    so_cvar_train = run_so_cvar_training(
                        nested_dict, loc, training_years, critical_voll
                    )

                    method_training = {
                        "LP_Avg": lp_avg,
                        "LP_Worst": lp_worst,
                        "SO_CVaR": so_cvar_train,
                    }

                    for method, train_row in method_training.items():
                        test_caps = capacities_from_training(train_row)
                        test_per_year = run_testing(
                            nested_dict,
                            loc,
                            testing_years,
                            critical_voll,
                            test_caps,
                        )
                        test_avg = average_testing(test_per_year, n_test)
                        combined = combine_training_testing(train_row, test_avg)
                        all_rows.append(
                            results_to_row(loc, voll_name, method, k, combined)
                        )
                    print(f"  Fold {k + 1}/{fold} complete.")

    detail_df = pd.DataFrame(all_rows)
    expected_total_rows = (
        len(locations) * len(VOLL_SCENARIOS) * len(algorithms) * fold
    )
    if len(detail_df) != expected_total_rows:
        raise ValueError(
            f"Expected {expected_total_rows} result rows, got {len(detail_df)}"
        )

    summary_df = build_summary(detail_df)

    config_df, voll_df = make_config_dataframe()
    write_excel(all_rows, summary_df, config_df, voll_df)
    print(f"\nDone. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
