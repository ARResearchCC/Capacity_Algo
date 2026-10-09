import numpy as np
import pandas as pd
import random

import Input_Parameters

def generate_schedules(schedule, weather, random_seed):
    # Stochastic occupancy/electrical-load schedule for the forward operating base (FOB),
    # the only application modelled in this study.
    if schedule == "FOB":
        return bayes_schedule_FOB(weather, random_seed)
    raise ValueError(f"Invalid schedule type {schedule!r}; only 'FOB' is supported")

# Helper function for weighted random sampling
def weighted_sample(choices, weights):
    return random.choices(choices, weights=weights, k=1)[0]

def draw_sched_FOB(weather):
    datetime_col = weather['DateTime']
    weekday = datetime_col.dt.weekday + 1  # Python weekday is 0-6 (Monday=0), so +1 to match Julia's 1-7
    occ_vec = np.zeros(len(datetime_col), dtype=int)

    # Define Probability Distributions
    weekday_probs = [0.42, 0.35, 0.35, 0.35, 0.42, 0, 0]  # Probability of showing up for each day of the week
    arrive_probs = [0.3, 0.4, 0.3]  # Probability of arriving at 8, 9, or 10 AM
    arrival_hours = [8, 9, 10]  # Corresponding to probabilities in arrive_probs
    depart_probs = [0.2, 0.4, 0.2, 0.2]  # Probability of leaving at 4, 5, 6, or 7 PM
    departure_hours = [16, 17, 18, 19]  # Corresponding to probabilities in depart_probs (4, 5, 6, 7 PM)

    for beginning_of_day in range(0, len(datetime_col), 24):
        day_of_week = weekday[beginning_of_day]
        if random.random() <= weekday_probs[day_of_week - 1]:  # Sample whether they arrive that day
            arrival_hour = weighted_sample(arrival_hours, arrive_probs)  # Sample arrival hour based on probabilities
            departure_hour = weighted_sample(departure_hours, depart_probs)  # Sample departure hour based on probabilities
            amount_spent = departure_hour - arrival_hour
            for hour in range(amount_spent):
                occ_vec[beginning_of_day + arrival_hour + hour] = 1
    return occ_vec

def bayes_schedule_FOB(weather, input_random_seed):
    random_seed = int(input_random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    hour_step = np.arange(weather.shape[0])
    datetime_col = weather['DateTime']
    month = datetime_col.dt.month
    day = datetime_col.dt.day
    weekday = datetime_col.dt.weekday + 1  # Python weekday is 0-6 (Monday=0), so +1 to match Julia's 1-7
    hour = datetime_col.dt.hour

    # Initialize DataFrame for readability
    load_sched = pd.DataFrame({
        "Hour Step": hour_step,
        "DateTime": datetime_col,
        "Month": month,
        "Day": day,
        "Weekday": weekday,
        "Hour": hour
    })

    # ------ Draw Occupancy Schedule ---
    occupancy_df = load_sched.copy()

    for i in range(1, Input_Parameters.MaxOccupancy + 1):
        occupancy_vector = draw_sched_FOB(weather)
        occupancy_df[f"Person {i}"] = occupancy_vector

    tot_occ = occupancy_df.iloc[:, 6:].sum(axis=1)  # Sum occupancy rows at each timestep
    occupancy_df["Total Occupancy"] = tot_occ
    load_sched["Total Occupancy"] = tot_occ

    # ------ Draw Loads Based on Occupancy ---

    # Load Probabilities and Specifications
    lighting_max_load = Input_Parameters.PeakLighting  # [kW] on if occupied
    comp_charge_low_weights = [0.50, 0.50, 0.0]
    comp_charge_high_weights = [0.25, 0.50, 0.25]
    comp_charge_loads = [0, 0.5, 1]  # [kW]
    kettle_low_weights = [0.70, 0.20, 0.10]
    kettle_high_weights = [0.50, 0.30, 0.20]
    kettle_loads = [0, 0.1333, 0.266]  # [kW]

    # Initializing new columns for loads
    load_sched["Lighting Load"] = 0.0
    load_sched["Computer Charging Load"] = 0.0
    load_sched["Kettle Load"] = 0.0

    # Assign loads based on occupancy
    for i, occ in enumerate(load_sched["Total Occupancy"]):
        if occ == 0:
            load_sched.at[i, "Computer Charging Load"] = 0
            load_sched.at[i, "Kettle Load"] = 0
            load_sched.at[i, "Lighting Load"] = 0
        elif occ <= 2:  # Light Occupancy
            load_sched.at[i, "Computer Charging Load"] = weighted_sample(comp_charge_loads, comp_charge_low_weights)
            load_sched.at[i, "Kettle Load"] = weighted_sample(kettle_loads, kettle_low_weights)
            load_sched.at[i, "Lighting Load"] = lighting_max_load
        else:  # High Occupancy
            load_sched.at[i, "Computer Charging Load"] = weighted_sample(comp_charge_loads, comp_charge_high_weights)
            load_sched.at[i, "Kettle Load"] = weighted_sample(kettle_loads, kettle_high_weights)
            load_sched.at[i, "Lighting Load"] = lighting_max_load

    load_sched["Total Load"] = np.round(load_sched["Lighting Load"] + load_sched["Computer Charging Load"] + load_sched["Kettle Load"], 3)

    # option to return occupancy_df
    return load_sched
