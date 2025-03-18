import Data_Conversion
import Input_Parameters 
import Data_Conversion
import Passive_Model
import Solar_Generation
import Electrical_Load

import Baseline_CO


# Define the base data directory, list of locations, and the weather year.
data_dir = "Data"
locations = ["Arizona", "Alaska", "Minnesota", "Florida", "HalfMoonBay"]

# 20 years of training data
weather_year_list = list(range(1998, 2018))  # The upper bound in range() is exclusive

# Construct the folder path for the location.
weather_year = weather_year_list[3]
loc = locations[4]

df = Data_Conversion.read_NSRDB(data_dir, loc, weather_year)

print(df)

lats, lons, timezones = Data_Conversion.get_timezones(data_dir, locations)

df_new = Data_Conversion.prepare_NSRDB(df, lats[4], lons[4], timezones[4])

# Create the nested dictionary
nested_dict = {location: {year: {} for year in weather_year_list} for location in locations}

# now to passive model and PV model
# Predict heating and cooling load using weather data and passive model
NetHeatTransfers = Passive_Model.passive_model(Input_Parameters.calibration_file_path, df_new, Input_Parameters.T_indoor_constant)

# Prepare solar PV generation forecast using weather data
pv_cf = Solar_Generation.generate_pv(df_new)

# Set a random seed for consistency
random_seed = 30  # You can change this value to get a different sequence

load_sched = Electrical_Load.generate_schedules("bayes", df_new, random_seed)

input_df = Data_Conversion.combine_input_NSRDB(df_new, load_sched, pv_cf, NetHeatTransfers)
print(input_df.head(24))

