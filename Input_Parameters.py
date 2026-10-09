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
Area = L_wall * L_wall # [m^2] Floor/Ceiling Area 
UA = L_wall*H_wall*4*(1/R_wall) + Area*2*(1/R_floor) # [W/K]
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
T_indoor_constant = 22 # [°C] Constant Indoor Temperature (for the simplicity of a linear model)

# Environmental Parameters
wind_height = 9.144 # [m] The height above ground at which wind speed is measured. The PVWatts default is 9.144 m.
albedo = 0.18 # [1] albedo of grass

# Device initial SOC
Intial_B_SOC = 0.5
Intial_PCM_C_SOC = 0.5
Intial_PCM_H_SOC = 0.5

# Forward Operating Base
PeakLighting = 0.1 # [KW] Dan's suggestion
MaxOccupancy = 4 # [PPL] 4-MAN Office Room Plan
PersonLatentHeat = 200; # [BTU/hr/PPL]  CEE226E Slide
PersonSensibleHeat = 300; # [BTU/hr/PPL]  CEE226E Slide
TotalPersonHeat = PersonSensibleHeat + PersonLatentHeat # [BTU/hr/PPL]

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
HPSize = 10          # [kW] default maximum electrical power consumption for heat pump

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

Lifetime = 20        # [years]
d = 0.03             # Discount rate
CRF = (d * (1 + d)**Lifetime) / ((1 + d)**Lifetime - 1)  # Capital recovery factor (0.0672)
M = 10000            # Big M value

COP_H = 3.5           # COP heating
COP_C = 3.5           # COP cooling

HVAC_lol_cost = 3  # [$/kWh] loss of load cost due to thermal comfort (residential loss of load value)
lossofloadcost = 100 # [$/kWh] Med critical-electrical VoLL for FOB. Within the peer-reviewed
# "typical" VoLL range $1-300/kWh (Anderson et al., IEEE Systems J. 2021) and near LBNL
# medium/large-C&I short-duration cost-per-unserved-kWh (Sullivan et al. 2015, LBNL-6941E).
# FOB VoLL sweep uses Low/Med/High = 30/100/300 (see FOB.py VOLL_SCENARIOS).

# CVaR stochastic optimization (SO_CVaR.py)
CVaR_alpha = 0.9   # confidence level; CVaR is the mean outage cost in the worst (1-alpha) tail of training years
CVaR_lambda = 0.9  # weight on CVaR vs expected outage cost in second stage (0 = SO, 1 = pure CVaR)

# Diesel generator baseline (Diesel_Model.py / FOB_Diesel.py)
# Capital, O&M and fuel-curve values (HOMER/NREL-style genset defaults) and their
# sources are documented in the paper's SI (Diesel benchmark section).
C_Gen = 800              # [$/kW] generator capital cost
C_Gen_OP = 0.03 * C_Gen  # [$/kW/yr] fixed O&M (3% of capital, consistent with C_HP_OP convention)
C_Gen_VOM = 0.02         # [$/kWh] variable O&M (non-fuel), per kWh generated

Gen_Lifetime = 15        # [years] diesel generator lifetime (shorter than PV/battery Lifetime)
Gen_CRF = (d * (1 + d)**Gen_Lifetime) / ((1 + d)**Gen_Lifetime - 1)  # Capital recovery factor for generator

# Delivered ("fully burdened") diesel fuel price for a forward operating base. A CONUS
# retail price is not representative: fuel delivered into theater carries large transport
# and force-protection costs on top of the commodity price, forming a documented ladder:
#   ~$2.8/gal commodity; ~$13/gal peacetime forward ground delivery; ~$42/gal aerial
#   refuelling; $100-600/gal hostile-area (combat-zone) delivery; ~$400/gal Afghanistan
#   helicopter resupply; up to ~$1000/gal in extreme cases.
# Diesel_Price below is the "Med" tier of FOB_Diesel.py's FUEL_PRICE_SCENARIOS
# (Low=$13 peacetime forward, Med=$45 protected convoy, High=$400 contested extreme).
# Sources (full citations in the paper's SI, "Diesel benchmark" section):
#   National Defense Magazine (2010): the $2.8/$13/$42/$100-600 ladder.
#   JASON/MITRE, "Reducing DoD Fossil-Fuel Dependence," JSR-06-135 (2006): FBCF
#     $100-600/gal in theater (hostile-area delivery).
#   GAO-09-300 (2009); Defense Science Board (2001/2016); PolitiFact (2011) notes the
#     $400/gal Afghanistan figure is a remote worst case, not the average.
#   Army Environmental Policy Institute, "Sustain the Mission Project" (2009): ~1 casualty
#     per 24 fuel-resupply convoys in Afghanistan (force-protection component of FBCF).
Diesel_Price = 45.00      # [$/gal] Med delivered diesel price = protected-convoy FBCF tier

# HOMER/NREL-style linear genset fuel curve: fuel [gal/hr] = Fuel_Curve_F0 * GenSize + Fuel_Curve_F1 * P_output
# (source coefficients ~0.0815 and 0.246 L/hr per kW, converted from L to gal by /3.785)
Fuel_Curve_F0 = 0.0215   # [gal/hr per kW rated capacity] no-load fuel intercept coefficient
Fuel_Curve_F1 = 0.0650   # [gal/hr per kW output] fuel curve slope coefficient

Gen_Reserve_Margin = 0.20  # default reserve margin added on top of peak electric demand when sizing