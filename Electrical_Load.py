import numpy as np
import pandas as pd
import random

import Input_Parameters

def generate_schedules(schedule, weather, random_seed):
    datetime_col = weather['DateTime']
    time_resolution = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600  # Time resolution in hours
    total_intervals = len(datetime_col)
    intervals_per_hour = int(1 / time_resolution)
    intervals_per_day = 24 * intervals_per_hour
    
    # Calculate the number of days in the weather DataFrame
    num_days = total_intervals // intervals_per_day  # Full days only

    if schedule == "simple":
        return simple_schedule(weather, time_resolution, total_intervals, num_days)
    elif schedule == "complex":
        return complex_schedule(weather, time_resolution, total_intervals, num_days)
    elif schedule == "FOB":
        return bayes_schedule_FOB(weather, random_seed)
    elif schedule == "DC":
        return bayes_schedule_DC(weather, random_seed)
    elif schedule == "RC":
        return bayes_schedule_RC(weather, random_seed)
    else:
        raise ValueError("Invalid schedule type")

def simple_schedule(weather, time_resolution, total_intervals, num_days):
    intervals_per_hour = int(1 / time_resolution)
    SimpleSchedule = np.zeros((total_intervals, 3))
    
    # Loop over the number of days in the weather DataFrame (not hardcoded 365)
    for i in range(num_days):
        for j in range(10 * intervals_per_hour, 17 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:  # Prevent index overflow
                SimpleSchedule[idx, :] = 1
    
    schedule = pd.DataFrame()
    schedule['Lighting'] = SimpleSchedule[:, 0] * Input_Parameters.PeakLighting
    schedule['Plugs'] = SimpleSchedule[:, 1] * Input_Parameters.PeakPlugLoad
    schedule['Total Occupancy'] = SimpleSchedule[:, 2] * (Input_Parameters.MaxOccupancy)
    schedule['Total Load'] = schedule['Lighting']+ schedule['Plugs']
    schedule['DateTime'] = weather['DateTime']
    return schedule


def complex_schedule(weather, time_resolution, total_intervals, num_days):
    intervals_per_hour = int(1 / time_resolution)
    ComplexSchedule = np.zeros((total_intervals, 3))
    
    # Loop over the number of days in the weather DataFrame (not hardcoded 365)
    for i in range(num_days):
        # Time block 1: 1 AM - 6 AM
        for j in range(1 * intervals_per_hour, 6 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, 0:2] = 0  # Lighting and Plugs off
                ComplexSchedule[idx, 2] = 1    # Occupancy on
        
        # Time block 2: 7 AM - 8 AM
        for j in range(7 * intervals_per_hour, 8 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, 0] = 1  # Lighting on
                ComplexSchedule[idx, 1] = 0  # Plugs off
                ComplexSchedule[idx, 2] = 1  # Occupancy on
        
        # Time block 3: 9 AM - 7 PM
        for j in range(9 * intervals_per_hour, 19 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, 0] = 1  # Lighting on
                ComplexSchedule[idx, 1] = 0  # Plugs off
                ComplexSchedule[idx, 2] = 0.5  # Occupancy at 50%
        
        # Time block 4: 8 PM - 9 PM
        for j in range(20 * intervals_per_hour, 21 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, 0:2] = 1  # Lighting and Plugs on
                ComplexSchedule[idx, 2] = 0.5  # Occupancy at 50%
        
        # Time block 5: 9 PM - 10 PM
        for j in range(21 * intervals_per_hour, 22 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, :] = 1  # Everything on
        
        # Time block 6: 11 PM - 12 AM
        for j in range(23 * intervals_per_hour, 24 * intervals_per_hour):
            idx = i * 24 * intervals_per_hour + j
            if idx < total_intervals:
                ComplexSchedule[idx, 0] = 0  # Lighting off
                ComplexSchedule[idx, 1] = 1  # Plugs on
                ComplexSchedule[idx, 2] = 1  # Occupancy on
    
    # Create the schedule DataFrame
    schedule = pd.DataFrame()
    schedule['Lighting'] = ComplexSchedule[:, 0] * Input_Parameters.PeakLighting
    schedule['Plugs'] = ComplexSchedule[:, 1] * Input_Parameters.PeakPlugLoad
    schedule['Total Occupancy'] = ComplexSchedule[:, 2] * (Input_Parameters.MaxOccupancy)
    schedule['Total Load'] = schedule['Lighting']+ schedule['Plugs']
    schedule['DateTime'] = weather['DateTime']
    
    return schedule

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


# Function to simulate workload schedule for the data center
def draw_sched_DC(weather):
    datetime_col = weather['DateTime']
    weekday = datetime_col.dt.weekday  # 0 = Monday
    load_vec = np.zeros(len(datetime_col), dtype=float)

    # Define probabilities of high workload by weekday and time of day
    weekday_probs = [0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.2]  # More jobs Mon–Fri
    time_probs = {
        "low": [0, 1, 2, 3, 4, 5, 6, 21, 22, 23],
        "med": [7, 8, 9, 17, 18, 19, 20],
        "high": [10, 11, 12, 13, 14, 15, 16]
    }

    for i in range(0, len(datetime_col), 24):
        day_of_week = weekday[i]
        if random.random() <= weekday_probs[day_of_week]:
            for h in range(24):
                idx = i + h
                if idx >= len(load_vec): break

                # Assign load factor based on hour bucket
                if h in time_probs["low"]:
                    load_factor = weighted_sample([0.2, 0.3, 0.4], [0.5, 0.3, 0.2])
                elif h in time_probs["med"]:
                    load_factor = weighted_sample([0.4, 0.5, 0.6], [0.3, 0.5, 0.2])
                else:  # high workload hours
                    load_factor = weighted_sample([0.6, 0.8, 1.0], [0.2, 0.5, 0.3])
                load_vec[idx] = load_factor
    return load_vec

def bayes_schedule_DC(weather, input_random_seed):

    random_seed = int(input_random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    hour_step = np.arange(weather.shape[0])
    datetime_col = weather['DateTime']
    month = datetime_col.dt.month
    day = datetime_col.dt.day
    weekday = datetime_col.dt.weekday + 1
    hour = datetime_col.dt.hour

    # Load schedule DataFrame
    load_sched = pd.DataFrame({
        "Hour Step": hour_step,
        "DateTime": datetime_col,
        "Month": month,
        "Day": day,
        "Weekday": weekday,
        "Hour": hour
    })

    # Draw workload load vector (in 0–1)
    load_factor = draw_sched_DC(weather)

    # Scale by max possible load
    load_sched["Normalized Load Factor"] = load_factor
    load_sched["Total Load"] = np.round(load_factor * Input_Parameters.DataCenterPeakPower, 2)
    load_sched["Total Occupancy"] = 0
    return load_sched


def draw_sched_RC_full(weather):
    datetime_col = weather['DateTime']
    weekday = datetime_col.dt.weekday + 1  # 1=Mon
    day_index = np.arange(len(datetime_col))

    # Group time indices by day (assuming hourly steps, 24 per day)
    daily_blocks = [day_index[i:i+24] for i in range(0, len(datetime_col), 24)]
    weekdays = [weekday[i] for i in range(0, len(datetime_col), 24)]

    return daily_blocks, weekdays

def bayes_schedule_RC(weather, input_random_seed):
    random_seed = int(input_random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    hour_step = np.arange(weather.shape[0])
    datetime_col = weather['DateTime']
    month = datetime_col.dt.month
    day = datetime_col.dt.day
    weekday = datetime_col.dt.weekday + 1
    hour = datetime_col.dt.hour

    load_sched = pd.DataFrame({
        "Hour Step": hour_step,
        "DateTime": datetime_col,
        "Month": month,
        "Day": day,
        "Weekday": weekday,
        "Hour": hour
    })

    # Initialize occupancy columns
    occ_staff = np.zeros(len(weather), dtype=int)
    occ_patient = np.zeros(len(weather), dtype=int)

    # Pull daily blocks
    daily_blocks, weekdays = draw_sched_RC_full(weather)

    # Define scheduling probabilities
    staff_probs = [0.95, 0.95, 0.95, 0.95, 0.95, 0.85, 0.75]
    staff_arrival_hours = [7, 8, 9]
    staff_arrival_probs = [0.2, 0.6, 0.2]
    staff_departure_hours = [17, 18, 19]
    staff_depart_probs = [0.3, 0.5, 0.2]

    patient_arrival_hours = [8, 9, 10]
    patient_arrival_probs = [0.2, 0.4, 0.4]
    patient_durations = [1, 2, 3]
    patient_dwell_probs = [0.3, 0.5, 0.2]

    for day_indices, day_of_week in zip(daily_blocks, weekdays):
        staff_present = False  # default

        if random.random() <= staff_probs[day_of_week - 1]:
            staff_present = True
            start_hour = weighted_sample(staff_arrival_hours, staff_arrival_probs)
            end_hour = weighted_sample(staff_departure_hours, staff_depart_probs)
            for h in range(start_hour, end_hour):
                if h < 24:
                    occ_staff[day_indices[h]] = 1

        # Only generate patients if staff are present that day
        if staff_present:
            num_patients = np.random.poisson(Input_Parameters.Ave_Patient)
            for _ in range(num_patients):
                a_hour = weighted_sample(patient_arrival_hours, patient_arrival_probs)
                duration = weighted_sample(patient_durations, patient_dwell_probs)
                for h in range(duration):
                    idx = a_hour + h
                    if idx < 24:
                        occ_patient[day_indices[idx]] += 1

    # Assign back
    load_sched["Staff Present"] = occ_staff
    load_sched["Patients Present"] = occ_patient

    # Always-on fridge
    fridge_power = Input_Parameters.FridgePower
    load_sched["Fridge Load"] = fridge_power

    # Conditional loads
    load_sched["Ventilator Load"] = load_sched["Patients Present"] * Input_Parameters.VentilatorPower
    load_sched["Lighting Load"] = load_sched["Staff Present"] * Input_Parameters.PeakLighting
    load_sched["IT Load"] = load_sched["Staff Present"] * Input_Parameters.ITLoad

    # Diagnostic load (only if patients present)
    diag_loads = [0, 0.3, 0.6]
    diag_weights = [0.5, 0.3, 0.2]
    diagnostic = [
        weighted_sample(diag_loads, diag_weights) if p > 0 else 0
        for p in load_sched["Patients Present"]
    ]
    load_sched["Diagnostic Load"] = diagnostic

    load_sched["Total Load"] = np.round(
        load_sched["Fridge Load"] +
        load_sched["Ventilator Load"] +
        load_sched["Lighting Load"] +
        load_sched["IT Load"] +
        load_sched["Diagnostic Load"], 3
    )
    load_sched["Total Occupancy"] = load_sched["Staff Present"] + load_sched["Patients Present"]
    return load_sched
