"""
Diesel-only baseline: generator supplies all electricity (critical electrical load
plus heat-pump electrical demand for heating/cooling). No PV, battery, or PCM.

Unlike the renewable architectures (Baseline_CO_PVB.py / SO_CVaR_PVB.py / Simulate.py),
diesel dispatch has no storage and therefore no intertemporal coupling: at every
timestep the generator simply serves as much load as its fixed capacity allows, so
sizing and dispatch are computed directly (no LP/solver needed).

Sizing: generator capacity = peak total electric demand (critical + heat pump) across
the supplied years, times (1 + reserve margin). This mirrors how diesel gensets are
sized in practice (nameplate peak + margin), not via a stochastic capacity trade-off.

Dispatch under a capacity shortfall (only possible if a simulated year's peak demand
exceeds the sized capacity): critical electrical load is served first; heat-pump
electrical demand is shed first, matching the VoLL hierarchy used elsewhere in this
study (critical VoLL >> HVAC VoLL).
"""

import numpy as np

import Input_Parameters
import Simulate

RESULT_KEYS = [
    "Total Cost",
    "Capital Cost",
    "Operation Cost",
    "Generator Capital Cost",
    "Heat Pump Capital Cost",
    "Fixed OM Cost",
    "Generator Fixed OM Cost",
    "Heat Pump Fixed OM Cost",
    "Variable OM Cost",
    "Fuel Cost",
    "Fuel Gallons",
    "HVAC Cost",
    "Critical Load Cost",
    "HVAC LoL Hours",
    "Critical LoL Hours",
    "HVAC LoL Events",
    "Critical LoL Events",
    "HVAC Max LoL Event Hours",
    "Critical Max LoL Event Hours",
]


def _hp_size(scenario):
    return Input_Parameters.HPSize_DC if scenario == "DC" else Input_Parameters.HPSize


def heat_pump_electric_demand(input_df, scenario="FOB"):
    """
    Electric power drawn by the heat pump to fully serve Heating_Load/Cooling_Load
    at fixed COP, capped at the heat pump's electrical capacity (HPSize). No PCM
    buffering is available in the diesel architecture.
    """
    HPSize = _hp_size(scenario)
    Heating_Load = input_df["Heating_Load"].to_numpy(dtype=float)
    Cooling_Load = input_df["Cooling_Load"].to_numpy(dtype=float)
    hp_elec = Heating_Load / Input_Parameters.COP_H + Cooling_Load / Input_Parameters.COP_C
    return np.minimum(hp_elec, HPSize)


def size_generator(input_df_list, reserve_margin=None, scenario="FOB"):
    """
    Generator capacity = (1 + reserve_margin) * peak of (E_Load + heat-pump electric
    demand) across all timesteps in all supplied years (a single deterministic design,
    not fold-dependent).

    Returns (gen_size, peak_demand).
    """
    if reserve_margin is None:
        reserve_margin = Input_Parameters.Gen_Reserve_Margin

    peak_demand = 0.0
    for input_df in input_df_list:
        E_Load = input_df["E_Load"].to_numpy(dtype=float)
        hp_elec = heat_pump_electric_demand(input_df, scenario)
        peak_demand = max(peak_demand, float(np.max(E_Load + hp_elec)))

    gen_size = peak_demand * (1 + reserve_margin)
    return gen_size, peak_demand


def simulate_diesel(input_df, critical_voll, gen_size, diesel_price=None, scenario="FOB"):
    """
    Simulate a fixed-capacity diesel generator serving one year of load.

    critical_voll and diesel_price are passed explicitly (as in Baseline_CO_PVB/
    Simulate); HVAC VoLL is read from Input_Parameters.HVAC_lol_cost, which callers
    should set via the same hvac_voll_context pattern used for the renewable
    architectures. diesel_price defaults to Input_Parameters.Diesel_Price (the FOB-
    representative "Med" tier; see FOB_Diesel.FUEL_PRICE_SCENARIOS for the literature-
    backed Low/Med/High delivered fuel price range).

    Returns a dict keyed by RESULT_KEYS.
    """
    if diesel_price is None:
        diesel_price = Input_Parameters.Diesel_Price

    datetime_col = input_df["DateTime"]
    dt = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600
    num_time = len(datetime_col)

    HPSize = _hp_size(scenario)
    E_Load = input_df["E_Load"].to_numpy(dtype=float)
    hp_elec = heat_pump_electric_demand(input_df, scenario)

    # Priority dispatch: critical load first, heat-pump demand second.
    served_crit = np.minimum(E_Load, gen_size)
    remaining_capacity = gen_size - served_crit
    served_hp = np.minimum(hp_elec, remaining_capacity)

    unserved_crit = E_Load - served_crit
    unserved_hp = hp_elec - served_hp
    gen_output = served_crit + served_hp

    # Annualized capital + fixed O&M, scaled to the simulated horizon (matches the
    # (NumTime/8760) convention used in Baseline_CO_PVB.py / Simulate.py). The heat
    # pump is common equipment across every architecture in this study, so it is
    # costed the same way here as in Baseline_CO_PVB.py (Input_Parameters.CRF, the
    # general 20-year capital recovery factor, not the generator's shorter Gen_CRF).
    year_fraction = num_time / 8760
    generator_capital_cost = Input_Parameters.C_Gen * gen_size * Input_Parameters.Gen_CRF * year_fraction
    heat_pump_capital_cost = Input_Parameters.C_HP * HPSize * Input_Parameters.CRF * year_fraction
    capital_cost = generator_capital_cost + heat_pump_capital_cost

    generator_fixed_om_cost = Input_Parameters.C_Gen_OP * gen_size * year_fraction
    heat_pump_fixed_om_cost = Input_Parameters.C_HP_OP * HPSize * year_fraction
    fixed_om_cost = generator_fixed_om_cost + heat_pump_fixed_om_cost

    variable_om_cost = Input_Parameters.C_Gen_VOM * dt * float(np.sum(gen_output))

    fuel_rate_gal_per_hr = np.where(
        gen_output > 0,
        Input_Parameters.Fuel_Curve_F0 * gen_size + Input_Parameters.Fuel_Curve_F1 * gen_output,
        0.0,
    )
    fuel_gallons = dt * float(np.sum(fuel_rate_gal_per_hr))
    fuel_cost = fuel_gallons * diesel_price

    hvac_cost = dt * Input_Parameters.HVAC_lol_cost * float(np.sum(unserved_hp))
    critical_load_cost = dt * critical_voll * float(np.sum(unserved_crit))

    operation_cost = fixed_om_cost + variable_om_cost + fuel_cost
    total_cost = capital_cost + operation_cost + hvac_cost + critical_load_cost

    hvac_lol_hours, hvac_lol_events, hvac_max_lol_event_h = Simulate.loss_of_load_metrics(
        unserved_hp, dt
    )
    crit_lol_hours, crit_lol_events, crit_max_lol_event_h = Simulate.loss_of_load_metrics(
        unserved_crit, dt
    )

    return {
        "Total Cost": round(total_cost, 3),
        "Capital Cost": round(capital_cost, 3),
        "Operation Cost": round(operation_cost, 3),
        "Generator Capital Cost": round(generator_capital_cost, 3),
        "Heat Pump Capital Cost": round(heat_pump_capital_cost, 3),
        "Fixed OM Cost": round(fixed_om_cost, 3),
        "Generator Fixed OM Cost": round(generator_fixed_om_cost, 3),
        "Heat Pump Fixed OM Cost": round(heat_pump_fixed_om_cost, 3),
        "Variable OM Cost": round(variable_om_cost, 3),
        "Fuel Cost": round(fuel_cost, 3),
        "Fuel Gallons": round(fuel_gallons, 3),
        "HVAC Cost": round(hvac_cost, 3),
        "Critical Load Cost": round(critical_load_cost, 3),
        "HVAC LoL Hours": round(hvac_lol_hours, 3),
        "Critical LoL Hours": round(crit_lol_hours, 3),
        "HVAC LoL Events": hvac_lol_events,
        "Critical LoL Events": crit_lol_events,
        "HVAC Max LoL Event Hours": round(hvac_max_lol_event_h, 3),
        "Critical Max LoL Event Hours": round(crit_max_lol_event_h, 3),
    }


def average_results(per_year):
    """Average a {year: result_dict} mapping (from simulate_diesel) across years."""
    num_years = len(per_year)
    averaged = {key: 0.0 for key in RESULT_KEYS}
    for row in per_year.values():
        for key in RESULT_KEYS:
            averaged[key] += row[key]
    for key in RESULT_KEYS:
        averaged[key] = round(averaged[key] / num_years, 3)
    return averaged


def breakeven_fuel_price(diesel_row, renewable_total_cost):
    """
    Delivered diesel price ($/gal) at which the diesel system's Total Cost equals
    renewable_total_cost, holding capacity, O&M, and VoLL outage costs fixed.

    diesel_row must contain "Total Cost", "Fuel Cost", and "Fuel Gallons" (e.g. one
    row/averaged result from simulate_diesel / average_results).
    """
    fuel_gallons = diesel_row["Fuel Gallons"]
    if fuel_gallons <= 0:
        return float("nan")
    fixed_cost = diesel_row["Total Cost"] - diesel_row["Fuel Cost"]
    return (renewable_total_cost - fixed_cost) / fuel_gallons
