# Input Parameters 

version = 1.0
code_name = "Capa-Algo"

calibration_file_path = "Calibration_Model_Input.xlsx"

############ Declare Parameters ############

# Berg Envelope Parameters
L_wall = 6.1 # [m] Length of the Wall
H_wall = 2.4 # [m] Height of the Wall
R_wall = 2.6 # [m^2·K/W] Thermal Resistance of the Wall  It's based on time units of seconds
R_floor = 3.0 # [m^2·K/W] Thermal Resistance of the Floor
H_Ceiling = 2.0 # [m] Interior Height of the Ceiling (78.5 inch) 
Area = L_wall * L_wall # [m^2] Floor/Ceiling Area 
UA = L_wall*H_wall*4*(1/R_wall) + Area*2*(1/R_floor) # [W/K]
Volume = L_wall * L_wall * H_Ceiling # [m^3]
e_Berg = 0.75 # fibreglass https://www.thermoworks.com/emissivity-table/
Berg_tilt = 15

# Infiltration Parameters

# Stack coefficient for building(house height = 1 story)
Cs = 0.000145 # [(L/s)^2/(cm^4*°K)]
# Wind coefficient for building(house height =1 story, shelter class=1(no obstructions or local shielding))
Cw = 0.000319 # [(L/s)^2/(cm^4*(m/s)^2)] (between 0.000319 and 0.000246 (house height =1 story, shelter class=2(typical shelter for an insolated rural house)))
# Effective leakage area measured during Blower Door Test 
# ELA =  47.1 # [in^2] (between 38.3(Dan's interpolation) and 47.1(Jessie's interpolation))
Al =  303.9 # [cm^2] using Jessie's interpolation
T_d = 273.15 - 1.67 # [°K] 99% Design Temperature for Half Moon Bay
T_indoor_constant = 22 # [°C] Constant Indoor Temperature (for the simplicity of a linear model)

# Environmental Parameters
wind_height = 9.144 # [m] The height above ground at which wind speed is measured. The PVWatts default is 9.144 m.
albedo = 0.18 # [1] albedo of grass
n_air = 1 # [1] Index of reflaction of air
n_AR = 1.3 # [1] Index of reflaction of AR coating
n_glass = 1.526 # [1] Index of reflaction of glass as in PVWatt model for standard modules

# Device initial SOC
Intial_B_SOC = 0.5
Intial_PCM_C_SOC = 0.5
Intial_PCM_H_SOC = 0.5

# Interior Standard

# Ventilation
Ventilation = 15 # [CFM/PPL] Building Standard
Ra = 0.06 # [CFM/ft^2] Area Outdoor Air Rate; Source:CEE226E Week 5 - Energy Modeling Questions - Slide 9
Rp = 5 # [CFM/PPL] People Outdoor Air Rate; Source:CEE226E Week 5 - Energy Modeling Questions - Slide 9   

# Data Center
TotalRack = 10
RackPower = 10.2 * 3 # [kW]
DataCenterPeakPower = TotalRack * RackPower # [kW]

# Forward Operating Base
PeakLighting = 0.1 # [KW] Dan's suggestion
PeakPlugLoad = 0.1 # [KW] Computer = 40W, Phone = 10W, 2*(40 + 10) = 100 [W] = 0.1 [kW]
MaxOccupancy = 4 # [PPL] 4-MAN Office Room Plan
PersonLatentHeat = 200; # [BTU/hr/PPL]  CEE226E Slide
PersonSensibleHeat = 300; # [BTU/hr/PPL]  CEE226E Slide
TotalPersonHeat = PersonSensibleHeat + PersonLatentHeat # [BTU/hr/PPL]

# Remote Clinic
Ave_Patient = 5
FridgePower = 0.15        # [kW]
VentilatorPower = 0.25    # [kW] per patient
ITLoad = 0.1              # [kW] Router, laptop, wifi

# Solar PV Parameters
# Temperature model parameters
noct_installed = 45  # Nominal Operating Cell Temperature [°C]
module_height = 1.0   # Module height above ground [m]
wind_height = 10.0    # Wind speed measurement height [m]
module_emissivity = 0.84
module_absorption = 0.83
module_surface_tilt = 30  # [degrees]
module_width = 1.0    # [m]
module_length = 1.6   # [m]

# PV performance parameters
optical_loss = 0.96   # Account for glass reflectance
P_dc0 = 1.0           # DC power at STC [kW/kWp]
Γ_t = -0.004          # Temperature coefficient [1/°C]
T_ref = 25            # Reference temperature [°C]
η_PV = 0.96           # System efficiency factor

# Equipments Parameters
     
# Economic parameters
HPSize = 10          # [kW] default constant electrical power consumption for heat pump
HPSize_DC = 100      # [kW] default constant electrical power consumption for heat pump for data center

BatteryLoss = 0.01/24# [/hr]
MaxDischarge = 0.8   # [1]
η = 0.98             # Battery inverter efficiency

C_PV = 1500          # [$/kW]
C_PV_OP = 15         # [$/kW/yr]
C_B = 500            # [$/kWh]
C_B_OP = 5           # [$/kWh/yr]
C_HP = 1000          # [$/kW]
C_HP_OP = 0.02 * C_HP # [$/yr]
C_PCM_H = 70         # [$/kWh]
C_PCM_H_OP = 0.04 * C_PCM_H # [$/kWh/yr]
C_PCM_C = 70         # [$/kWh]
C_PCM_C_OP = 0.04 * C_PCM_C # [$/kWh/yr]

C_PV_low = 500           # [$/kW]
C_PV_OP_low = 10         # [$/kW/yr]
C_B_low = 250            # [$/kWh]
C_B_OP_low = 0           # [$/kWh/yr]


Lifetime = 20        # [years]
d = 0.03             # Discount rate
CRF = (d * (1 + d)**Lifetime) / ((1 + d)**Lifetime - 1)  # Capital recovery factor (0.0672)
M = 10000            # Big M value

COP_H = 3.5           # COP heating
COP_C = 3.5           # COP cooling

HVAC_lol_cost = 3  # [$/kWh] loss of load cost due to thermal comfort (residential loss of load value)
HVAC_lol_cost_DC = 10  # [$/kWh] loss of load cost due to thermal comfort and computing for DC (residential loss of load value)
lossofloadcost_DC = 30 # [$/kWh] loss of load penalty for critical electrical load for DC (large C&I)
lossofloadcost = 300 # [$/kWh] loss of load penalty for critical electrical load for FOB & RC (small C&I)