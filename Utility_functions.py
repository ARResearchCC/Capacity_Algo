import pandas as pd
import numpy as np
from datetime import timedelta
import pvlib
from pvlib.location import Location
from pvlib.clearsky import ineichen
from pysolar.solar import get_altitude
from datetime import datetime, timedelta
import pytz
from math import sin, cos, acos, asin, pi, exp, log, radians, degrees
import Input_Parameters 

# Utility Functions
def get_timezone_name(offset):
    for timezone in pytz.all_timezones:
        tz = pytz.timezone(timezone)
        now = datetime.utcnow()  # Use current UTC time
        offset_hours = tz.utcoffset(now)  # Get the timezone offset

        if offset_hours is not None and offset_hours.total_seconds() / 3600 == offset:
            return timezone  # Return the first matching timezone

    return "Unknown Timezone"  # If no match is found

def add_cloudcover(df):
    df['Cloud'] = df['Cloud Type'].apply(cloud_type_to_percent)
    return df

def cloud_type_to_percent(cloud_type):
    if cloud_type < 0 or cloud_type > 10:
        raise ValueError("Invalid cloud type. Must be an integer between 0 and 9.")

    cloud_cover = {
        0: 0,     # Clear sky
        1: 0.1,   # Probably clear
        2: 0.95,  # Fog
        3: 0.8,   # Liquid water
        4: 0.9,   # Super cooled water
        5: 0.85,  # Mixed
        6: 0.9,   # Opaque ice
        7: 0.3,   # Cirrus
        8: 0.95,  # Overlapping
        9: 1,     # Overshooting
        10: 0.5   # Unknown
    }
    return cloud_cover[cloud_type]

def add_cloudcover_epw(df):
    df['Cloud'] = df.apply(lambda row: estimate_cloudcover(row['Ta'], row['Td'], row['RH']), axis=1)
    return df

def calculate_k_t(df, latitude=34.0522, longitude=-118.2437, timezone="America/Los_Angeles"):
    # Convert 'timestamp_utc' to datetime and local timezone
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    df['timestamp_local'] = df['timestamp_utc'].dt.tz_convert(timezone)
    
    # Set the index to the local timezone-aware datetime
    df = df.set_index('timestamp_local')
    
    # Define the location (Los Angeles)
    location = Location(latitude=latitude, longitude=longitude, tz=timezone)
    
    # Calculate solar zenith angle for each time step
    solar_zenith = []
    for date in df.index:
        solar_altitude = get_altitude(location.latitude, location.longitude, date)
        zenith = 90 - solar_altitude
        solar_zenith.append(zenith)
    df['solar_zenith'] = solar_zenith
    
    # Calculate clear-sky GHI using the index (DatetimeIndex)
    clearsky = location.get_clearsky(df.index, model='ineichen')
    df['clearsky_ghi'] = clearsky['ghi']
    
    # Calculate k_t using vectorized operations
    df['k_t'] = np.where(
        df['clearsky_ghi'] > 10,
        np.minimum(df['ghi'] / df['clearsky_ghi'], 1),
        0
    )
    
    return df.reset_index()

def add_clearness_index(df, latitude, longitude, timezone_offset):
    # Ensure latitude and longitude are floats
    latitude = float(latitude)
    longitude = float(longitude)

    # Create a Location object with UTC timezone (will be adjusted by the offset)
    site = Location(latitude, longitude, tz="UTC")

    # Create a datetime index for the DataFrame
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.set_index('datetime', inplace=True)

    # Create the correct Etc/GMT timezone string
    if timezone_offset <= 0:
        tz_str = f"Etc/GMT+{int(-timezone_offset)}"  # For negative offsets (e.g., UTC-8 -> Etc/GMT+8)
    else:
        tz_str = f"Etc/GMT-{int(timezone_offset)}"  # For positive offsets (e.g., UTC+7 -> Etc/GMT-7)

    # Localize the datetime index to the local timezone
    df.index = df.index.tz_localize(tz_str)

    # Convert the datetime index to UTC for solar calculations
    df_utc = df.tz_convert("UTC")

    # Calculate the theoretical clearsky GHI
    clearsky_ghi = site.get_clearsky(df_utc.index, model='ineichen')

    # Merge the clearsky GHI with the original DataFrame
    df = df.join(clearsky_ghi)

    # Calculate k_t using vectorized operations
    df['k_t'] = np.where(
        df['ghi'] > 10,  # Only calculate k_t if ghi > 10 to avoid division by small numbers
        np.minimum(df['GHI'] / df['ghi'], 1),  # Cap k_t at 1
        0  # Set k_t to 0 if ghi <= 10
    )

    # Drop the 'ghi' column as it's no longer needed
    df.drop(columns=['ghi'], inplace=True)

    # Reset the index to return the DataFrame to its original form
    df.reset_index(drop=True, inplace=True)
    
    return df

def correct_daylightsaving(df):
    for i in range(len(df)):
        if Input_Parameters.dls_start <= df.iloc[i]['DateTime'] <= Input_Parameters.dls_end:
            df.at[i, 'DateTime'] = df.iloc[i]['DateTime'] - timedelta(hours=1)
    return df

def add_day_status(df, latitude):
    df['DayStatus'] = df.apply(lambda row: IsDay(row['SolarTime'], row['DateTime'], latitude), axis=1)
    return df

def calculate_dew_point(T, RH):
    es = 6.112 * exp((17.67 * T) / (T + 243.5))
    e = (RH / 100) * es
    Td = (243.5 * log(e / 6.112)) / (17.67 - log(e / 6.112))
    return Td

def Q_convection(WS, Ta, Ti):
    T_indoor = Ti + 273.15
    delta_T = abs(T_indoor - Ta)
    Infiltration = Input_Parameters.Al * np.sqrt(Input_Parameters.Cs * delta_T + Input_Parameters.Cw * WS**2)
    CFM = Infiltration * 0.001
    Q_inf_sen = 1005 * 1.25 * CFM * (Ta - Ti)
    return Q_inf_sen

def Q_conduction(Ta, Ti):
    Q_cond = Input_Parameters.UA * (Ta - Ti)
    return Q_cond

def add_solar_time(df, timezone="America/Los_Angeles"):
    # Ensure the DateTime column is timezone-aware
    if df['DateTime'].dt.tz is None:
        df['DateTime'] = df['DateTime'].dt.tz_localize(timezone)
    
    solartimes = []
    for i in range(len(df)):
        local_time = df.iloc[i]['DateTime']
        
        # Check if the local time is in DST
        is_dst = local_time.dst() != pd.Timedelta(0)
        
        # Adjust for DST (subtract 1 hour if DST is active)
        if is_dst:
            local_time = local_time - pd.Timedelta(hours=1)
        
        # Calculate solar time
        LT = local_time.hour + local_time.minute / 60 + local_time.second / 3600
        N = local_time.timetuple().tm_yday
        Eqt = calculate_equation_of_time(N)
        T_solar = LT + Eqt / 60 + (4 * (Input_Parameters.standard_meridian_longitude - Input_Parameters.longitude)) / 60
        solartimes.append(T_solar)
    
    df['SolarTime'] = solartimes
    return df

def calculate_equation_of_time(N):
    """
    Calculate the equation of time for a given day of the year (N).
    
    Parameters:
        N (int): Day of the year (1-365).
    
    Returns:
        float: Equation of time in minutes.
    """
    B = radians((N - 1) * 360 / 365)
    Eqt = 229.2 * (0.000075 + 0.001868 * cos(B) - 0.032077 * sin(B) - 0.014615 * cos(2 * B) - 0.04089 * sin(2 * B))
    return Eqt

def add_solar_time_NSRDB(df, timezone_offset, longitude):
    """
    Add a solar time column to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'DateTime' column.
        timezone_offset (float): Timezone offset from UTC in hours (e.g., -7 for MST in Arizona).
        longitude (float): Longitude of the location.

    Returns:
        pd.DataFrame: DataFrame with an additional 'SolarTime' column.
    """
    # Ensure the DateTime column is timezone-naive
    if df['DateTime'].dt.tz is not None:
        df['DateTime'] = df['DateTime'].dt.tz_localize(None)
    
    # Calculate standard meridian longitude for the timezone
    standard_meridian_longitude = timezone_offset * 15
    
    # Calculate solar time for each row
    solartimes = []
    for i in range(len(df)):
        local_time = df.iloc[i]['DateTime']
        
        # Calculate solar time
        LT = local_time.hour + local_time.minute / 60 + local_time.second / 3600
        N = local_time.timetuple().tm_yday
        Eqt = calculate_equation_of_time(N)
        longitudinal_correction = (4 * (standard_meridian_longitude - float(longitude))) / 60  # Fixed sign
        T_solar = LT + longitudinal_correction + Eqt / 60
        
        # Ensure solar time is within 0-24 hours
        if T_solar < 0:
            T_solar += 24
        elif T_solar >= 24:
            T_solar -= 24
        
        solartimes.append(T_solar)
    
    # Add the SolarTime column to the DataFrame
    df['SolarTime'] = solartimes
    return df


def calculate_irradiance_new(solar_time, pyranometer, surface_azimuth, tilt_angle, k_t, local_time, lat):
    N = local_time.timetuple().tm_yday
    T_solar = solar_time
    delta = 23.45 * sin(radians((360 / 365) * (284 + N)))
    omega = 15 * (T_solar - 12)
    phi = radians(float(lat))
    alpha = asin(sin(phi) * sin(radians(delta)) + cos(phi) * cos(radians(delta)) * cos(radians(omega)))
    if omega < 0:
        Psi = acos((sin(radians(delta)) - sin(alpha) * sin(phi)) / (cos(alpha) * cos(phi)))
    else:
        Psi = 2 * pi - acos((sin(radians(delta)) - sin(alpha) * sin(phi)) / (cos(alpha) * cos(phi)))
    beta = radians(tilt_angle)
    gamma = radians(surface_azimuth)
    theta = acos(sin(alpha) * cos(beta) + cos(alpha) * sin(beta) * cos(gamma - Psi))
    if k_t <= 0.22:
        k_d = 1 - 0.09 * k_t
    elif 0.22 < k_t <= 0.8:
        k_d = 0.9511 - 0.1604 * k_t + 4.388 * k_t**2 - 16.638 * k_t**3 + 12.336 * k_t**4
    else:
        k_d = 0.165
    DHI = pyranometer * k_d
    theta_z = radians(0)
    DNI = min((pyranometer - DHI) / cos(pi / 2 - alpha - theta_z), pyranometer)
    if alpha < 0:
        DNI = 0
        DHI = pyranometer
    if cos(theta) < 0:
        I_b = 0
    else:
        I_b = DNI * cos(theta)
    GHI = pyranometer
    I_d = max(DHI * (1 + cos(beta)) / 2 + 0.5 * GHI * (0.012 * (pi / 2 - alpha) - 0.04) * (1 - cos(beta)), 0)
    albedo = 0.18
    I_r = max(GHI * albedo * (1 - cos(beta)) / 2, 0)
    I_total = I_d + I_r + I_b
    return I_total

def calculate_solarheatgain(solar_time, GHI, k_t, local_time, lat):
    Berg_tilt = 15
    GHI = max(GHI, 0)
    I_t = calculate_irradiance_new(solar_time, GHI, 0, 0, k_t, local_time, lat)
    I_e = calculate_irradiance_new(solar_time, GHI, 90 - Berg_tilt, 90, k_t, local_time, lat)
    I_s = calculate_irradiance_new(solar_time, GHI, 180 - Berg_tilt, 90, k_t, local_time, lat)
    I_w = calculate_irradiance_new(solar_time, GHI, 270 - Berg_tilt, 90, k_t, local_time, lat)
    I_n = calculate_irradiance_new(solar_time, GHI, 0 - Berg_tilt, 90, k_t, local_time, lat)
    TotalSolarHeatGain = I_t * Input_Parameters.L_wall * Input_Parameters.L_wall + (I_e + I_s + I_w + I_n) * Input_Parameters.L_wall * Input_Parameters.H_wall
    return TotalSolarHeatGain

def estimate_cloudcover(Ta, Td, rh):
    temp_dew_diff = Ta - Td
    if temp_dew_diff < 2 and rh > 90:
        return 0.95
    elif temp_dew_diff < 4 and rh > 80:
        return 0.8
    elif temp_dew_diff < 6 and rh > 70:
        return 0.5
    else:
        return 0.15

def estimate_Ts(Ti, Ta, solartime, day_status, GHI):
    if day_status == 0:
        Ts = (Ta + Ti) / 2
        return [Ts, Ts, Ts, Ts, Ts]
    Ts_E = max(Ti, Ti * ((GHI + 1000) / 2000) * (-13 / 7) * cos(2 * pi * (solartime + 2) / 24))
    Ts_S = max(Ti, Ti * ((GHI + 1000) / 2000) * (-13 / 7) * cos(2 * pi * (solartime - 2) / 24))
    Ts_W = max(Ti, Ti * ((GHI + 1000) / 2000) * (-13 / 7) * cos(2 * pi * (solartime - 4.5) / 24))
    Ts_N = max(Ti, Ti * (-8 / 7) * ((GHI + 1000) / 2000) * cos(2 * pi * (solartime - 48) / 72))
    Ts_R = max(Ti, Ti * ((GHI + 3000) / 4000) * (-12 / 7) * cos(2 * pi * (solartime - 52) / 72))
    return [Ts_E, Ts_S, Ts_W, Ts_N, Ts_R]

def Q_radiativeCooling(Ti, Ta, RH, GHI, solartime, day_status):
    temp_dewpoint = calculate_dew_point(Ta, RH)
    e_sky = 0.741 + 0.0062 * temp_dewpoint
    sigma = 5.67e-8
    exposed_area = [Input_Parameters.L_wall * Input_Parameters.H_wall, Input_Parameters.L_wall * Input_Parameters.H_wall, Input_Parameters.L_wall * Input_Parameters.H_wall, 
                    Input_Parameters.L_wall * Input_Parameters.H_wall, Input_Parameters.L_wall * Input_Parameters.L_wall]
    total_exposed_area = sum(exposed_area)
    Ts_list = estimate_Ts(Ti, Ta, solartime, day_status, GHI)
    Ts = [T + 273.15 for T in Ts_list]
    T_sky = Ta * e_sky**0.25
    Q_radcool = sigma * (Input_Parameters.e_Berg * sum((T**4 * area for T, area in zip(Ts, exposed_area))) - e_sky * total_exposed_area * (T_sky + 273.15)**4)
    return Q_radcool

def IsDay(solar_time, local_time, latitude):
    N = local_time.timetuple().tm_yday
    delta = 23.45 * sin(radians((360 / 365) * (284 + N)))
    omega = 15 * (solar_time - 12)
    phi = radians(float(latitude))
    alpha = asin(sin(phi) * sin(radians(delta)) + cos(phi) * cos(radians(delta)) * cos(radians(omega)))
    return 1 if alpha > 0 else 0

def normalize_row(row, cols, bounds):
    normalized_row = row.copy()
    for col in cols:
        if col in bounds:
            lower_bound, upper_bound = bounds[col]
            normalized_row[col] = (row[col] - lower_bound) / (upper_bound - lower_bound)
    return normalized_row

def find_nearest_centroid(normalized_row, centroids, cols):
    row_vector = np.array([normalized_row[col] for col in cols])
    distances = [np.linalg.norm(row_vector - centroid) for centroid in centroids.T]
    return np.argmin(distances)

