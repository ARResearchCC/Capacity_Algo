import pandas as pd
import numpy as np
import pvlib
from pvlib.temperature import fuentes
from pvlib.irradiance import get_total_irradiance
from datetime import datetime
import math
import Input_Parameters

def generate_pv(weather, lat):
    # Need to update the function (27 kW) facing 3 ways
    
    # Ensure the datetime column is in the correct format and set as the index
    weather["DateTime"] = pd.to_datetime(weather["DateTime"])
    weather = weather.set_index("DateTime")
    
    # Calculate PV module cell temperature with PVLib [°C]
    ambient_temperature = weather["Ta"]  # [°C]
    wind_speed = weather["WS"]  # [m/s]
    total_irradiance = weather["GHI"]  # [W/m^2]

    # Calculate PV cell temperature using the Fuentes Model
    temp_fuentes = fuentes(
        total_irradiance,
        ambient_temperature,
        wind_speed,
        Input_Parameters.noct_installed,
        module_height=Input_Parameters.module_height,
        wind_height=Input_Parameters.wind_height,
        emissivity=Input_Parameters.module_emissivity,
        absorption=Input_Parameters.module_absorption,
        surface_tilt=Input_Parameters.module_surface_tilt,
        module_width=Input_Parameters.module_width,
        module_length=Input_Parameters.module_length
    )  # [°C]

    # Store PV cell temperature into weather DataFrame
    weather["celltemp"] = temp_fuentes

    # Calculate PV capacity factor [kW DC Output/kW Capacity]
    PV_CF = np.zeros(len(weather))
    for i in range(len(weather)):
        pv_o = calculate_pv(
            weather.index[i],
            weather.iloc[i]["DNI"],
            weather.iloc[i]["DHI"],
            weather.iloc[i]["GHI"],
            180 - Input_Parameters.module_surface_tilt,  # Surface azimuth (180° - tilt)
            Input_Parameters.module_surface_tilt,
            weather.iloc[i]["celltemp"],
            weather.iloc[i]["SolarTime"],
            lat
        )
        PV_CF[i] = pv_o

    return PV_CF  # [kW/kW capacity]

def calculate_pv(local_time, DNI, DHI, GHI, surface_azimuth, tilt_angle, T_cell, solar_time, latitude):
    # Convert local time to Local Solar Time
    LT = local_time.hour + local_time.minute / 60 + local_time.second / 3600

    # Day of the year (N)
    N = local_time.timetuple().tm_yday

    # Solar Declination (δ)
    δ = 23.45 * math.sin(math.radians((360 / 365) * (284 + N)))  # [DEGREES]

    # Solar Time Angle (ω)
    ω = 15 * (solar_time - 12)  # [DEGREES]

    # Latitude (φ)
    φ = math.radians(float(latitude))  # [RADIANS]

    # Solar Elevation Angle (α)
    α = math.asin(math.sin(φ) * math.sin(math.radians(δ)) + math.cos(φ) * math.cos(math.radians(δ)) * math.cos(math.radians(ω)))  # [RADIANS]

    # Solar Azimuth Angle (Ψ)
    if ω < 0:
        Ψ = math.acos((math.sin(math.radians(δ)) - math.sin(α) * math.sin(φ)) / (math.cos(α) * math.cos(φ)))  # [RADIANS]
    else:
        Ψ = 2 * math.pi - math.acos((math.sin(math.radians(δ)) - math.sin(α) * math.sin(φ)) / (math.cos(α) * math.cos(φ)))  # [RADIANS]

    # Surface tilt (β)
    β = math.radians(tilt_angle)  # [RADIANS]
    # Surface azimuth (γ)
    γ = math.radians(surface_azimuth)  # [RADIANS]
    # Angle of Incidence (θ)
    θ = math.acos(math.sin(α) * math.cos(β) + math.cos(α) * math.sin(β) * math.cos(γ - Ψ))  # [RADIANS]

    # Beam Irradiance (I_b)
    if math.cos(θ) < 0:
        I_b = 0
    else:
        I_b = DNI * math.cos(θ)  # [W/m^2]

    # Diffuse irradiance (I_d)
    I_d = max(DHI * (1 + math.cos(β)) / 2 + 0.5 * GHI * (0.012 * (math.pi / 2 - α) - 0.04) * (1 - math.cos(β)), 0)  # [W/m^2]

    # Reflected irradiance (I_r)
    I_r = max(GHI * Input_Parameters.albedo * (1 - math.cos(β)) / 2, 0)  # [W/m^2]

    # Total irradiance (I_poa)
    I_poa = I_d + I_r + I_b  # [W/m^2]

    # Calculate transmittance
    θ_2 = math.asin((Input_Parameters.n_air / Input_Parameters.n_AR) * math.sin(θ))  # [RADIANS]
    τ_AR = 1 - 0.5 * ((math.sin(θ_2 - θ) ** 2) / (math.sin(θ_2 + θ) ** 2)) + ((math.tan(θ_2 - θ) ** 2) / (math.tan(θ_2 + θ) ** 2))  # [1]
    θ_3 = math.asin((Input_Parameters.n_AR / Input_Parameters.n_glass) * math.sin(θ_2))  # [RADIANS]
    τ_glass = 1 - 0.5 * ((math.sin(θ_3 - θ_2) ** 2) / (math.sin(θ_3 + θ_2) ** 2)) + ((math.tan(θ_3 - θ_2) ** 2) / (math.tan(θ_3 + θ_2) ** 2))  # [1]
    τ_cover = τ_AR * τ_glass  # [1]

    # Calculate transmitted POA irradiance [I_tr]
    I_tr = I_poa * τ_cover  # [W/m^2]

    # Calculate PV DC power output [P_dc]
    P_dc = (I_tr / 1000) * Input_Parameters.P_dc0 * (1 + Input_Parameters.Γ_t * (T_cell - Input_Parameters.T_ref))  # [kW/kW]

    # Calculate PV DC power output after system loss [P_PV]
    P_PV = P_dc * Input_Parameters.η_PV  # [kW/kW]

    return P_PV