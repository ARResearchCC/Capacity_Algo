import math
import numpy as np
import pandas as pd
import Input_Parameters
from pvlib.irradiance import get_total_irradiance
from pvlib.location import Location

def fuentes(poa_global, temp_air, wind_speed, noct, 
            module_height=1.0, wind_height=10.0, 
            emissivity=0.84, absorption=0.83, 
            surface_tilt=30, module_width=1.0, module_length=1.6):
    """
    Fuentes cell temperature model implementation
    """
    # Constants
    σ = 5.670373e-8  # Stefan-Boltzmann constant [W/m²·K⁴]
    h_conv = 5.7 + 3.8 * wind_speed  # Convective heat transfer coefficient [W/m²·K]
    
    # Wind speed adjustment to module height
    wind_speed_module = wind_speed * (module_height/wind_height)**0.2
    
    # Effective sky temperature
    temp_sky = (0.0552 * (temp_air + 273.15)**1.5) - 273.15  # [°C]
    
    # Module back temperature
    temp_back = temp_air + (noct - 20)/800 * poa_global
    
    # Cell temperature calculation
    numerator = poa_global * absorption
    numerator += h_conv * temp_air
    numerator += σ * emissivity * (temp_sky + 273.15)**4
    numerator -= σ * emissivity * (temp_back + 273.15)**4
    
    denominator = h_conv + 4 * σ * emissivity * (temp_back + 273.15)**3
    
    temp_cell = numerator / denominator
    
    return temp_cell

def generate_pv(weather, lat):

    lat = float(lat)
    # Ensure datetime index
    weather["DateTime"] = pd.to_datetime(weather["DateTime"])
    weather = weather.set_index("DateTime")
    
    # Create location object
    site = Location(latitude=lat, longitude=0, tz=weather.index.tz or 'UTC')
    solar_position = site.get_solarposition(weather.index)
    
    # Define three orientations (East, South, West)
    orientations = [
        {'azimuth': 165, 'weight': 1/3},  # East
        {'azimuth': 165, 'weight': 1/3},  # South
        {'azimuth': 165, 'weight': 1/3}   # West
    ]
    
    total_output = np.zeros(len(weather))
    
    for orient in orientations:
        # Calculate POA irradiance
        poa = get_total_irradiance(
            surface_tilt=Input_Parameters.module_surface_tilt,
            surface_azimuth=orient['azimuth'],
            dni=weather["DNI"],
            ghi=weather["GHI"],
            dhi=weather["DHI"],
            solar_zenith=solar_position["zenith"],
            solar_azimuth=solar_position["azimuth"],
            model='isotropic'
        )
        
        # Calculate cell temperature using Fuentes model
        temp_cell = fuentes(
            poa["poa_global"],
            weather["Ta"],
            weather["WS"],
            Input_Parameters.noct_installed,
            module_height=Input_Parameters.module_height,
            wind_height=Input_Parameters.wind_height,
            emissivity=Input_Parameters.module_emissivity,
            absorption=Input_Parameters.module_absorption,
            surface_tilt=Input_Parameters.module_surface_tilt,
            module_width=Input_Parameters.module_width,
            module_length=Input_Parameters.module_length
        )
        
        # Calculate PV output for this orientation
        for i in range(len(weather)):
            # Simple incidence angle modifier
            zenith = solar_position["zenith"].iloc[i]
            iam = 1 - 0.05 * ((90 - zenith)/90)**2 if zenith < 90 else 0
            
            # Calculate effective irradiance
            effective_poa = poa['poa_global'].iloc[i] * iam * Input_Parameters.optical_loss
            
            # PV output calculation
            P_dc = (effective_poa/1000) * Input_Parameters.P_dc0 * (
                1 + Input_Parameters.Γ_t * (temp_cell.iloc[i] - Input_Parameters.T_ref)
            )
            
            total_output[i] += orient['weight'] * P_dc * Input_Parameters.η_PV
    
    # Ensure no negative values
    total_output = np.maximum(total_output, 0)
    
    return total_output