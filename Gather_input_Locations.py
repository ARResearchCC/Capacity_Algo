import Data_Conversion
import Input_Parameters 
import Data_Conversion
import Passive_Model
import Solar_Generation
import Electrical_Load

import numpy as np
import pandas as pd
import os
from pathlib import Path

# Define the base data directory, list of locations, and the weather year.
data_dir = "Data"
locations = ["HalfMoonBay", "Arizona", "Alaska", "Minnesota", "Florida"]
# locations = ["HalfMoonBay"]
scenarios = ["FOB", "DC", "RC"]

# Define which columns to sum and which to average
cols = ['pv', 'E_Load', 'Cooling_Load', 'Heating_Load']  # Columns to sum daily

# 20 years of training data
weather_year_list = list(range(1998, 2023))  # The upper bound in range() is exclusive

# Set a random seed for consistency
# Define the number of rows (i) and columns (j)

j = len(weather_year_list)  # Number of columns

# Generate a sequential array of numbers starting from 1
total_cells = j  # Total number of cells in the DataFrame
sequential_numbers = np.arange(1, total_cells + 1)  # Generates 1, 2, 3, ..., i*j
random_seeds = sequential_numbers

# The base case should always be the first row of the random seed matrix, for sensitivity analysis on locations, scenarios, and capacity costs.

# Get the list of latitude, longitude, and timezones for all locations
lats, lons, timezones = Data_Conversion.get_timezones(data_dir, locations)

# Create the nested dictionary to store all input data
nested_dict = {location: {year: {} for year in weather_year_list} for location in locations}

output_dict = {location: {year: {} for year in weather_year_list} for location in locations}

# Define folder name
folder_name = "Yearly_Results"
# Create folder if it doesn't already exist
os.makedirs(folder_name, exist_ok=True)

# Get all input data for each of the 25 years
for i in range(len(locations)):
    for j in range(len(weather_year_list)):
        
        location = locations[i]
        year = weather_year_list[j]
        
        # Get a unique random seed number
        random_seed = random_seeds[j] # updated to keep the electrical load profile consistent across locations for the same year

        # Read NSRDB weather data of the given location of the given year
        # NSRDB_raw_weather = Data_Conversion.read_NSRDB(data_dir, location, year).head(24)
        NSRDB_raw_weather = Data_Conversion.read_NSRDB(data_dir, location, year)
        
        # Prepare weather data file using NSRDB data
        weather_data = Data_Conversion.prepare_NSRDB(NSRDB_raw_weather, lats[i], lons[i], timezones[i])

        # Prepare heating and cooling load using weather data and passive model
        NetHeatTransfers = Passive_Model.passive_model(Input_Parameters.calibration_file_path, weather_data, Input_Parameters.T_indoor_constant, lats[i])

        # Prepare solar PV capacity factor using weather data
        pv_cf = Solar_Generation.generate_pv(weather_data, lats[i])

        # Prepare occupancy and electrical load schedule using for a specific random seed number for a specific year at a specific location
        load_sched = Electrical_Load.generate_schedules("FOB", weather_data, random_seed)
        
        # Combine all relative input data as input_df, which will be the input of the capacity optimization algorithm
        input_df = Data_Conversion.combine_input_NSRDB(weather_data, load_sched, pv_cf, NetHeatTransfers)
        
        # Add input_df to list
        nested_dict[location][year] = input_df

        # Group by day and sum the selected columns
        daily_sums = input_df.groupby(input_df['DateTime'].dt.date)[cols].sum()

        # Compute coefficient of variation (CV = std / mean) for each column
        daily_cv = daily_sums.std() / daily_sums.mean()
        
        # Compute daily variances (variances of daily sums)
        daily_variances = daily_sums.var()
        
        # Compute yearly totals (sum of all daily sums)
        yearly_totals = daily_sums.sum()

        # Combine into a single list
        result_list = daily_variances.tolist() +  daily_cv.tolist() + yearly_totals.tolist()
        output_dict[location][year] = result_list

# Make a excel workbook file
output_file = os.path.join(folder_name, "locations_result.xlsx")

# Reporting (export to .xlsx file)
with pd.ExcelWriter(output_file) as writer:
    for location, year_data in output_dict.items():

        rows = [[year] + result_data for year, result_data in year_data.items()]
        
        # Create DataFrame with specified column names
        df = pd.DataFrame(rows, columns=["Year", "var PV", "var E_Load", "var Cooling_Load", "var Heating_Load",
        "CV PV", "CV E_Load", "CV Cooling_Load", "CV Heating_Load", 'Total pv (kWh/kW Capacity)', 'Total E_Load (kWh)', 'Total Cooling_Load (kWh T)', 'Total Heating_Load (kWh T)'])
        
        # Write to sheet named after the location
        df.to_excel(writer, sheet_name=str(location), index=False)



