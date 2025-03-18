import pandas as pd
import numpy as np
import Utility_functions

# Define the columns and their bounds for normalization
cols_for_clustering_SHGC = ['WS', 'RH', 'Ta', 'Cloud', 'GHI', 'k_t']
cols_for_clustering_RCC = ['WS', 'RH', 'Ta', 'Cloud']
bounds = {
    'WS': (0.0, 10.0),  # Example bounds for Wind Speed [m/s]
    'RH': (0.0, 100.0),  # Example bounds for Relative Humidity [%]
    'Ta': (0.0, 40.0),  # Example bounds for Temperature in Celsius [°C]
    'GHI': (0.0, 1000.0),  # Example bounds for Global Horizontal Irradiance [W/m^2]
    'Cloud': (0.0, 1.0),  # Example bounds for Cloud cover fraction
    'k_t': (0.0, 1.0)  # Example bounds for clearness index
}

def passive_model(file_path, df, Ti_constant, lat, conversion_setting=1):
    # Load the Excel file
    dfs = pd.read_excel(file_path, sheet_name=None)
    
    SHGCs = dfs["Tuned Parameters"]["_SHGC_"]
    RCCs_day = dfs["Tuned Parameters"]["_RCC_day_"]
    RCCs_night = dfs["Tuned Parameters"]["_RCC_night_"]
    TC = dfs["Other Parameters"].iloc[0]["_Thermal_Capacitance_"]
    
    day_centroids_SHGC_df = dfs["SHGC Day Centroids"]
    day_centroids_RCC_df = dfs["RCC Day Centroids"]
    night_centroids_RCC_df = dfs["RCC Night Centroids"]
    
    day_centroids_SHGC = day_centroids_SHGC_df.to_numpy().T
    day_centroids_RCC = day_centroids_RCC_df.to_numpy().T
    night_centroids_RCC = night_centroids_RCC_df.to_numpy().T

    NumTime = df.shape[0]

    # Create space to store heat transfers
    NetHeatTransfers = []  # [W]
    
    # Create space to store various sources of heat transfers.
    Q_cond = []  # [W]
    Q_conv = []  # [W]
    Q_radcool = []  # [W]
    Q_radheat = []  # [W]

    TemperatureAmbient = df['Ta']  # [°C]
    WindSpeed = df['WS']  # [m/s]
    RelativeHumidity = df['RH']  # [%]
    GHI = df['GHI']  # [W/m^2]

    local_time = df['DateTime']
    k_t = df['k_t']
    solar_time = df['SolarTime']
    CloudCover = df['Cloud']
    day_status = df['DayStatus']

    # Timeseries simulation
    for t in range(NumTime):
        # Conduction
        q_cd = Utility_functions.Q_conduction(TemperatureAmbient.iloc[t], Ti_constant) / 1000  # [kW]
        Q_cond.append(q_cd)
        
        # Convection
        q_cv = Utility_functions.Q_convection(WindSpeed.iloc[t], TemperatureAmbient.iloc[t], Ti_constant) / 1000  # [kW]
        Q_conv.append(q_cv)
        
        # Radiative cooling
        q_rc = Utility_functions.Q_radiativeCooling(Ti_constant, TemperatureAmbient.iloc[t], RelativeHumidity.iloc[t], GHI.iloc[t], solar_time.iloc[t], day_status.iloc[t]) / 1000  # [kW]

        # Radiative heat gain to the structure
        q_rh = Utility_functions.calculate_solarheatgain(solar_time.iloc[t], GHI.iloc[t], k_t.iloc[t], local_time.iloc[t], lat) / 1000  # [kW]
        
        new_row = pd.DataFrame({
            'WS': [WindSpeed.iloc[t]],
            'RH': [RelativeHumidity.iloc[t]],
            'Ta': [TemperatureAmbient.iloc[t]],
            'GHI': [GHI.iloc[t]],
            'Cloud': [CloudCover.iloc[t]],
            'k_t': [k_t.iloc[t]]
        })
        
        if day_status.iloc[t] == 0.0:
            normalized_new_row_rcc = Utility_functions.normalize_row(new_row, cols_for_clustering_RCC, bounds)
            # Find the nearest centroid for night time data
            nearest_rcc_cluster = Utility_functions.find_nearest_centroid(normalized_new_row_rcc, night_centroids_RCC, cols_for_clustering_RCC)
            Q_radcool.append(q_rc * RCCs_night[nearest_rcc_cluster])
            Q_radheat.append(0)
            Q_net = q_cd + q_cv - q_rc * RCCs_night[nearest_rcc_cluster]  # [kW]
            NetHeatTransfers.append(Q_net)
        else:
            normalized_new_row_shgc = Utility_functions.normalize_row(new_row, cols_for_clustering_SHGC, bounds)
            normalized_new_row_rcc = Utility_functions.normalize_row(new_row, cols_for_clustering_RCC, bounds)
            # Find the nearest centroid for day time data
            nearest_shgc_cluster = Utility_functions.find_nearest_centroid(normalized_new_row_shgc, day_centroids_SHGC, cols_for_clustering_SHGC)
            nearest_rcc_cluster = Utility_functions.find_nearest_centroid(normalized_new_row_rcc, day_centroids_RCC, cols_for_clustering_RCC)
            Q_radcool.append(q_rc * RCCs_day[nearest_rcc_cluster])
            Q_radheat.append(q_rh * SHGCs[nearest_shgc_cluster])

            Q_net = q_cd + q_cv + q_rh * SHGCs[nearest_shgc_cluster] - q_rc * RCCs_day[nearest_rcc_cluster]  # [kW]
            NetHeatTransfers.append(Q_net)
    
    return NetHeatTransfers
