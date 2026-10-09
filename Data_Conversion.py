import pandas as pd
import os
import glob
import Input_Parameters
import Utility_functions

def prepare_NSRDB(df_raw, lat, lon, timezone):
    # Rename columns
    df_raw = df_raw.rename(columns={
        "Temperature": "Ta",
        "Wind Speed": "WS",
        "Relative Humidity": "RH"
    })

    # Create the DateTime column
    df_raw["DateTime"] = pd.to_datetime(df_raw[["Year", "Month", "Day", "Hour", "Minute"]])

    # Apply additional transformations (assuming these are defined functions)
    df_raw = Utility_functions.add_clearness_index(df_raw, lat, lon, timezone)
    df_raw = Utility_functions.add_solar_time_NSRDB(df_raw, timezone, lon)
    df_raw = Utility_functions.add_day_status(df_raw, lat)
    df_raw = Utility_functions.add_cloudcover(df_raw)

    return df_raw


def read_NSRDB(data_dir, loc, weather_year):
    
    folder_path = os.path.join(data_dir, f"NREL_NSRDB_{loc}")
    # Build the search pattern to match the CSV file for the given weather year.
    pattern = os.path.join(folder_path, f"*_{weather_year}.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No file found for location '{loc}' and weather year '{weather_year}' in folder '{folder_path}'. "
            f"Tried pattern: {pattern}"
        )
    # If multiple files match, take the first one.
    file_path = files[0]
    print(f"Reading file for {loc}: {file_path}")
    # Read the CSV file; adjust header= if necessary.
    df = pd.read_csv(file_path, header=2)
    return df

def get_timezones(data_dir, locations):

    timezones = []
    lats = []
    lons = []
    weather_year = 1998
    for loc in locations:
        folder_path = os.path.join(data_dir, f"NREL_NSRDB_{loc}")
        # Build the search pattern to match the CSV file for the given weather year.
        pattern = os.path.join(folder_path, f"*_{weather_year}.csv")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(
                f"No file found for location '{loc}' and weather year '{weather_year}' in folder '{folder_path}'. "
                f"Tried pattern: {pattern}"
            )
        # If multiple files match, take the first one.
        file_path = files[0]
        print(f"Reading file for {loc}: {file_path}")
        # Read the CSV file; adjust header= if necessary.
        # df = pd.read_csv(file_path, header=2)
        df = pd.read_csv(file_path)
        lats.append(df.iloc[0]["Latitude"])
        lons.append(df.iloc[0]["Longitude"])
        # tz = Utility_functions.get_timezone_name(int(df.iloc[0]["Time Zone"]))
        # timezones.append(tz)
        timezones.append(int(df.iloc[0]["Time Zone"]))
    
    return lats, lons, timezones

def combine_input_NSRDB(weather, schedule, pv, Q):

    # combine dataframes
    # calculate additional cooling load from occupancy and plug loads

    # Step 1: Create an empty DataFrame
    new_df = pd.DataFrame()

    # Step 2: Add columns from existing DataFrames to the new DataFrame
    new_df['DateTime'] = weather['DateTime']  
    new_df['Ta'] = weather['Ta'] # [°C]
    new_df['pv'] = pv # [kW/kW]
    new_df['E_Load'] = schedule['Total Load'] # [kW]
    
    Q_total = Q + schedule['Total Occupancy'] * Input_Parameters.TotalPersonHeat / 3412.14 + schedule['Total Load'] # [kW] internal heat gain
    
    # Convert Q_total to a pandas Series
    Q_total_series = pd.Series(Q_total)
    
    # Cooling and Heating Load
    new_df['Cooling_Load'] = Q_total_series.where(Q_total_series > 0, 0)  # [kW]
    new_df['Heating_Load'] = (-Q_total_series).where(Q_total_series < 0, 0)  # [kW]

    return new_df