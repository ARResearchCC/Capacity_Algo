import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Expression, Set
import numpy as np
import pandas as pd

import Input_Parameters

def Cap_Baseline_V1(input_df, lossofloadcost):
    
    datetime_col = input_df['DateTime']
    δt = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600  # Time resolution in hours

    NumTime = len(datetime_col)

    PV = input_df['pv']
    E_Load = input_df['E_Load']
    Cooling_Load = input_df['Cooling_Load']
    Heating_Load = input_df['Heating_Load']

    # Create Pyomo model
    model = pyo.ConcreteModel()

    # Time index set
    model.T = pyo.RangeSet(0, NumTime - 1)
    model.NumTime = NumTime

    # Parameters
    # Time-dependent data
    model.PV = pyo.Param(model.T, initialize=lambda m, t: PV[t])
    model.E_Load = pyo.Param(model.T, initialize=lambda m, t: E_Load[t])
    model.Cooling_Load = pyo.Param(model.T, initialize=lambda m, t: Cooling_Load[t])
    model.Heating_Load = pyo.Param(model.T, initialize=lambda m, t: Heating_Load[t])

    # Constants as parameters
    model.C_IV = pyo.Param(initialize=Input_Parameters.C_IV)
    model.InverterSize = pyo.Param(initialize=Input_Parameters.InverterSize)
    model.HPSize = pyo.Param(initialize=Input_Parameters.HPSize)
    model.C_PV = pyo.Param(initialize=Input_Parameters.C_PV)
    model.C_PV_OP = pyo.Param(initialize=Input_Parameters.C_PV_OP)
    model.C_B = pyo.Param(initialize=Input_Parameters.C_B)
    model.C_B_OP = pyo.Param(initialize=Input_Parameters.C_B_OP)
    model.BatteryLoss = pyo.Param(initialize=Input_Parameters.BatteryLoss)
    model.MaxDischarge = pyo.Param(initialize=Input_Parameters.MaxDischarge)
    model.η = pyo.Param(initialize=Input_Parameters.η)
    model.C_HP = pyo.Param(initialize=Input_Parameters.C_HP)
    model.C_HP_OP = pyo.Param(initialize=Input_Parameters.C_HP_OP)
    model.COP_H = pyo.Param(initialize=Input_Parameters.COP_H)
    model.COP_C = pyo.Param(initialize=Input_Parameters.COP_C)
    model.C_PCM_H = pyo.Param(initialize=Input_Parameters.C_PCM_H)
    model.C_PCM_H_OP = pyo.Param(initialize=Input_Parameters.C_PCM_H_OP)
    model.C_PCM_C = pyo.Param(initialize=Input_Parameters.C_PCM_C)
    model.C_PCM_C_OP = pyo.Param(initialize=Input_Parameters.C_PCM_C_OP)
    model.d = pyo.Param(initialize=Input_Parameters.d)
    model.CRF = pyo.Param(initialize=Input_Parameters.CRF)
    model.δt = pyo.Param(initialize=δt)
    model.lossofloadcost = pyo.Param(initialize=lossofloadcost)
    model.η_PVIV = pyo.Param(initialize=0.94)

    model.Intial_B_SOC = pyo.Param(initialize=Input_Parameters.Intial_B_SOC)
    model.Intial_PCM_C_SOC = pyo.Param(initialize=Input_Parameters.Intial_PCM_C_SOC)
    model.Intial_PCM_H_SOC = pyo.Param(initialize=Input_Parameters.Intial_PCM_H_SOC)

    # Variables
    # Power flows
    model.PV2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.PV2G = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.PV2B = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.B2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.H2HP = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.HP2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.H2C = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.C2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.G2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.HP2PCM_H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.C2PCM_C = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.PCM_H2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.PCM_C2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    
    # Capacities
    model.PVSize = pyo.Var(within=pyo.NonNegativeReals)
    model.BatterySize = pyo.Var(within=pyo.NonNegativeReals)
    model.PCM_H_Size = pyo.Var(within=pyo.NonNegativeReals)
    model.PCM_C_Size = pyo.Var(within=pyo.NonNegativeReals)
    
    # Storage states
    model.InStorageBattery = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.InStoragePCM_H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.InStoragePCM_C = pyo.Var(model.T, within=pyo.NonNegativeReals)

    '''
    # Define expressions
    model.H_score = Expression(model.T)  # Heating score expression
    model.C_score = Expression(model.T)  # Cooling score expression
    model.E_score = Expression(model.T)  # Electric load score expression

    # These should be linear expressions since Heating_Load, Cooling_Load, and E_Load are inputs, although InStorage variables are all second stage decision variables.
    # The worst case scenario for each load (heating, cooling, and electrical) happens when the respective score is the lowest (always positive)
    for t in model.T:
        model.H_score[t] = model.InStoragePCM_H[t] / model.Heating_Load[t]
        model.C_score[t] = model.InStoragePCM_C[t] / model.Cooling_Load[t]
        model.E_score[t] = model.InStorageBattery[t] / model.E_Load[t]
    
    # Worst case: Highest H_score, C_score, and E_score, and lowest PV production
    '''

    # Objective Function
    # Total levelized capacity + fixed yearly operation & maintainence cost of the system (PV, electrochemical battery, 2 HPs (heating and cooling), Inverter, PCM H and C storages)
    capital_cost = (
        model.C_PV * model.PVSize +
        2 * model.C_B * model.BatterySize +
        2 * model.C_HP * model.HPSize +
        model.C_PCM_H * model.PCM_H_Size +
        model.C_PCM_C * model.PCM_C_Size +
        model.C_IV
    )

    fixed_OM_cost = (
        model.C_PV_OP * model.PVSize +
        model.C_B_OP * model.BatterySize +
        2 * model.C_HP_OP * model.HPSize +
        model.C_PCM_H_OP * model.PCM_H_Size +
        model.C_PCM_C_OP * model.PCM_C_Size
    )

    first_stage_cost = capital_cost * model.CRF + fixed_OM_cost
    
    outage_cost = model.δt * model.lossofloadcost * sum(model.G2H[t] for t in model.T)

    # First stage objective: capital_cost + fixed_OM_cost
    # Second stage objective: outage_cost
    model.objective = pyo.Objective(
        expr = first_stage_cost + outage_cost,
        sense = pyo.minimize
    )

    # Constraints

    def HVAC_load_balance_rule(m, t):
        net_thermal_load = m.HP2H[t] - m.C2H[t] + m.PCM_H2H[t] - m.PCM_C2H[t] # [kW] total active heat gain
        return (m.Cooling_Load[t] - m.Heating_Load[t] + net_thermal_load == 0)
    model.HVAC_load_balance = pyo.Constraint(model.T, rule=HVAC_load_balance_rule)

    # Battery storage initialization and termination
    def battery_init_rule(m):
        return m.InStorageBattery[0] == m.Intial_B_SOC * m.BatterySize
    model.battery_init = pyo.Constraint(rule=battery_init_rule)

    def battery_term_rule(m):
        return m.InStorageBattery[m.NumTime - 1] == m.Intial_B_SOC * m.BatterySize
    model.battery_term = pyo.Constraint(rule=battery_term_rule)

    # PV energy balance
    def pv_energy_balance_rule(m, t):
        return m.PV[t] * m.PVSize == m.PV2B[t] + m.PV2H[t] + m.PV2G[t]
    model.pv_energy_balance = pyo.Constraint(model.T, rule=pv_energy_balance_rule)

    # House electricity load balance
    def house_electricity_rule(m, t):
        return m.E_Load[t] + m.H2HP[t] + m.H2C[t] == m.PV2H[t] * m.η_PVIV + m.B2H[t] * m.η + m.G2H[t]
    model.house_electricity = pyo.Constraint(model.T, rule=house_electricity_rule)

    # Battery storage dynamics
    def battery_storage_balance_rule(m, t):
        if t < m.NumTime - 1:
            return m.InStorageBattery[t+1] == m.InStorageBattery[t] * (1 - m.BatteryLoss * m.δt) + m.δt * (m.PV2B[t] * m.η - m.B2H[t])
        else:
            return pyo.Constraint.Skip
    model.battery_storage_balance = pyo.Constraint(model.T, rule=battery_storage_balance_rule)

    # Battery discharge constraint
    def battery_discharge_rule(m, t):
        return m.δt * m.B2H[t] <= m.InStorageBattery[t]
    model.battery_discharge = pyo.Constraint(model.T, rule=battery_discharge_rule)

    # Inverter capacity constraint
    def battery_inverter_rule(m, t):
        return m.B2H[t] + m.PV2B[t] <= m.InverterSize
    model.battery_inverter = pyo.Constraint(model.T, rule=battery_inverter_rule)

    # Battery size constraint
    def battery_size_rule(m, t):
        return m.InStorageBattery[t] <= m.BatterySize
    model.battery_size = pyo.Constraint(model.T, rule=battery_size_rule)

    # Battery max discharge constraint
    def battery_max_discharge_rule(m, t):
        return m.InStorageBattery[t] >= m.BatterySize * (1 - m.MaxDischarge)
    model.battery_max_discharge = pyo.Constraint(model.T, rule=battery_max_discharge_rule)

    # Heating power balance
    def heating_power_balance_rule(m, t):
        return m.H2HP[t] == (m.HP2PCM_H[t] + m.HP2H[t]) / m.COP_H
    model.heating_power_balance = pyo.Constraint(model.T, rule=heating_power_balance_rule)

    # Cooling power balance
    def cooling_power_balance_rule(m, t):
        return m.H2C[t] == (m.C2PCM_C[t] + m.C2H[t]) / m.COP_C
    model.cooling_power_balance = pyo.Constraint(model.T, rule=cooling_power_balance_rule)

    # Heat pump heating capacity constraint
    def heat_pump_heating_capacity_rule(m, t):
        return m.H2HP[t] <= m.HPSize
    model.heat_pump_heating_capacity = pyo.Constraint(model.T, rule=heat_pump_heating_capacity_rule)

    # Heat pump cooling capacity constraint
    def heat_pump_cooling_capacity_rule(m, t):
        return m.H2C[t] <= m.HPSize
    model.heat_pump_cooling_capacity = pyo.Constraint(model.T, rule=heat_pump_cooling_capacity_rule)

    # PCM heating storage dynamics
    def pcm_h_storage_balance_rule(m, t):
        if t < m.NumTime - 1:
            return m.InStoragePCM_H[t+1] == m.InStoragePCM_H[t] + m.δt * (m.HP2PCM_H[t] - m.PCM_H2H[t])
        else:
            return pyo.Constraint.Skip
    model.pcm_h_storage_balance = pyo.Constraint(model.T, rule=pcm_h_storage_balance_rule)

    # PCM cooling storage dynamics
    def pcm_c_storage_balance_rule(m, t):
        if t < m.NumTime - 1:
            return m.InStoragePCM_C[t+1] == m.InStoragePCM_C[t] + m.δt * (m.C2PCM_C[t] - m.PCM_C2H[t])
        else:
            return pyo.Constraint.Skip
    model.pcm_c_storage_balance = pyo.Constraint(model.T, rule=pcm_c_storage_balance_rule)

    # PCM heating discharge constraint
    def pcm_h_discharge_rule(m, t):
        return m.δt * m.PCM_H2H[t] <= m.InStoragePCM_H[t]
    model.pcm_h_discharge = pyo.Constraint(model.T, rule=pcm_h_discharge_rule)

    # PCM cooling discharge constraint
    def pcm_c_discharge_rule(m, t):
        return m.δt * m.PCM_C2H[t] <= m.InStoragePCM_C[t]
    model.pcm_c_discharge = pyo.Constraint(model.T, rule=pcm_c_discharge_rule)

    # PCM heating size constraint
    def pcm_h_size_rule(m, t):
        return m.InStoragePCM_H[t] <= m.PCM_H_Size
    model.pcm_h_size = pyo.Constraint(model.T, rule=pcm_h_size_rule)

    # PCM cooling size constraint
    def pcm_c_size_rule(m, t):
        return m.InStoragePCM_C[t] <= m.PCM_C_Size
    model.pcm_c_size = pyo.Constraint(model.T, rule=pcm_c_size_rule)

    # PCM heating initialization and termination
    def pcm_h_init_rule(m):
        return m.InStoragePCM_H[0] == m.Intial_PCM_H_SOC * m.PCM_H_Size
    model.pcm_h_init = pyo.Constraint(rule=pcm_h_init_rule)

    def pcm_c_init_rule(m):
        return m.InStoragePCM_C[0] == m.Intial_PCM_C_SOC * m.PCM_C_Size
    model.pcm_c_init = pyo.Constraint(rule=pcm_c_init_rule)

    def pcm_h_term_rule(m):
        return m.InStoragePCM_H[m.NumTime - 1] == m.Intial_PCM_H_SOC * m.PCM_H_Size
    model.pcm_h_term = pyo.Constraint(rule=pcm_h_term_rule)

    def pcm_c_term_rule(m):
        return m.InStoragePCM_C[m.NumTime - 1] == m.Intial_PCM_C_SOC * m.PCM_C_Size
    model.pcm_c_term = pyo.Constraint(rule=pcm_c_term_rule)

    # Solve the model
    solver = pyo.SolverFactory('gurobi')  # Ensure Gurobi is installed or use another LP solver
    results = solver.solve(model, tee=True)  # tee=True for verbose output

    # Check if the solution is optimal
    if results.solver.status != pyo.SolverStatus.ok:
        raise RuntimeError("Solver did not exit normally")
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError("Solution is not optimal")

    # Retrieve results
    PV_Size = round(pyo.value(model.PVSize), 3)
    Battery_Size = round(pyo.value(model.BatterySize), 3)
    PCM_Heating_Size = round(pyo.value(model.PCM_H_Size), 3)
    PCM_Cooling_Size = round(pyo.value(model.PCM_C_Size), 3)
    
    ObjValue = round(pyo.value(model.objective), 3)
    First_stage_cost = round(pyo.value(first_stage_cost), 3)
    Second_stage_cost = round(pyo.value(outage_cost), 3)

    return PV_Size, Battery_Size, PCM_Heating_Size, PCM_Cooling_Size, ObjValue, First_stage_cost, Second_stage_cost


def simulate(input_df, lossofloadcost, capacities):
    
    datetime_col = input_df['DateTime']
    δt = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600  # Time resolution in hours

    NumTime = len(datetime_col)

    PV = input_df['pv']
    E_Load = input_df['E_Load']
    Cooling_Load = input_df['Cooling_Load']
    Heating_Load = input_df['Heating_Load']

    # Create Pyomo model
    model = pyo.ConcreteModel()

    # Time index set
    model.T = pyo.RangeSet(0, NumTime - 1)
    model.NumTime = NumTime

    # Parameters
    # Time-dependent data
    model.PV = pyo.Param(model.T, initialize=lambda m, t: PV[t])
    model.E_Load = pyo.Param(model.T, initialize=lambda m, t: E_Load[t])
    model.Cooling_Load = pyo.Param(model.T, initialize=lambda m, t: Cooling_Load[t])
    model.Heating_Load = pyo.Param(model.T, initialize=lambda m, t: Heating_Load[t])

    # Constants as parameters
    model.C_IV = pyo.Param(initialize=Input_Parameters.C_IV)
    model.InverterSize = pyo.Param(initialize=Input_Parameters.InverterSize)
    model.HPSize = pyo.Param(initialize=Input_Parameters.HPSize)
    model.C_PV = pyo.Param(initialize=Input_Parameters.C_PV)
    model.C_PV_OP = pyo.Param(initialize=Input_Parameters.C_PV_OP)
    model.C_B = pyo.Param(initialize=Input_Parameters.C_B)
    model.C_B_OP = pyo.Param(initialize=Input_Parameters.C_B_OP)
    model.BatteryLoss = pyo.Param(initialize=Input_Parameters.BatteryLoss)
    model.MaxDischarge = pyo.Param(initialize=Input_Parameters.MaxDischarge)
    model.η = pyo.Param(initialize=Input_Parameters.η)
    model.C_HP = pyo.Param(initialize=Input_Parameters.C_HP)
    model.C_HP_OP = pyo.Param(initialize=Input_Parameters.C_HP_OP)
    model.COP_H = pyo.Param(initialize=Input_Parameters.COP_H)
    model.COP_C = pyo.Param(initialize=Input_Parameters.COP_C)
    model.C_PCM_H = pyo.Param(initialize=Input_Parameters.C_PCM_H)
    model.C_PCM_H_OP = pyo.Param(initialize=Input_Parameters.C_PCM_H_OP)
    model.C_PCM_C = pyo.Param(initialize=Input_Parameters.C_PCM_C)
    model.C_PCM_C_OP = pyo.Param(initialize=Input_Parameters.C_PCM_C_OP)
    model.d = pyo.Param(initialize=Input_Parameters.d)
    model.CRF = pyo.Param(initialize=Input_Parameters.CRF)
    model.δt = pyo.Param(initialize=δt)
    model.lossofloadcost = pyo.Param(initialize=lossofloadcost)
    model.η_PVIV = pyo.Param(initialize=0.94)

    model.Intial_B_SOC = pyo.Param(initialize=Input_Parameters.Intial_B_SOC)
    model.Intial_PCM_C_SOC = pyo.Param(initialize=Input_Parameters.Intial_PCM_C_SOC)
    model.Intial_PCM_H_SOC = pyo.Param(initialize=Input_Parameters.Intial_PCM_H_SOC)

    # Variables
    # Power flows
    model.PV2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.PV2G = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.PV2B = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.B2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.H2HP = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.HP2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.H2C = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.C2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.G2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.HP2PCM_H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.C2PCM_C = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.PCM_H2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.PCM_C2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    
    # Capacities (Fixed)
    model.PVSize = pyo.Param(initialize=capacities[0])
    model.BatterySize = pyo.Param(initialize=capacities[1])
    model.PCM_H_Size = pyo.Param(initialize=capacities[2])
    model.PCM_C_Size = pyo.Param(initialize=capacities[3])
    
    # Storage states
    model.InStorageBattery = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.InStoragePCM_H = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.InStoragePCM_C = pyo.Var(model.T, within=pyo.NonNegativeReals)

    '''
    # Define expressions
    model.H_score = Expression(model.T)  # Heating score expression
    model.C_score = Expression(model.T)  # Cooling score expression
    model.E_score = Expression(model.T)  # Electric load score expression

    # These should be linear expressions since Heating_Load, Cooling_Load, and E_Load are inputs, although InStorage variables are all second stage decision variables.
    # The worst case scenario for each load (heating, cooling, and electrical) happens when the respective score is the lowest (always positive)
    for t in model.T:
        model.H_score[t] = model.InStoragePCM_H[t] / model.Heating_Load[t]
        model.C_score[t] = model.InStoragePCM_C[t] / model.Cooling_Load[t]
        model.E_score[t] = model.InStorageBattery[t] / model.E_Load[t]
    
    # Worst case: Highest H_score, C_score, and E_score, and lowest PV production
    '''

    # Objective Function
    # Total levelized capacity + fixed yearly operation & maintainence cost of the system (PV, electrochemical battery, 2 HPs (heating and cooling), Inverter, PCM H and C storages)
    capital_cost = (
        model.C_PV * model.PVSize +
        2 * model.C_B * model.BatterySize +
        2 * model.C_HP * model.HPSize +
        model.C_PCM_H * model.PCM_H_Size +
        model.C_PCM_C * model.PCM_C_Size +
        model.C_IV
    )

    fixed_OM_cost = (
        model.C_PV_OP * model.PVSize +
        model.C_B_OP * model.BatterySize +
        2 * model.C_HP_OP * model.HPSize +
        model.C_PCM_H_OP * model.PCM_H_Size +
        model.C_PCM_C_OP * model.PCM_C_Size
    )

    first_stage_cost = capital_cost * model.CRF + fixed_OM_cost
    
    outage_cost = model.δt * model.lossofloadcost * sum(model.G2H[t] for t in model.T)

    # First stage objective: capital_cost + fixed_OM_cost
    # Second stage objective: outage_cost
    model.objective = pyo.Objective(
        expr = outage_cost,
        sense = pyo.minimize
    )

    # Constraints

    def HVAC_load_balance_rule(m, t):
        net_thermal_load = m.HP2H[t] - m.C2H[t] + m.PCM_H2H[t] - m.PCM_C2H[t] # [kW] total active heat gain
        return (m.Cooling_Load[t] - m.Heating_Load[t] + net_thermal_load == 0)
    model.HVAC_load_balance = pyo.Constraint(model.T, rule=HVAC_load_balance_rule)

    # Battery storage initialization and termination
    def battery_init_rule(m):
        return m.InStorageBattery[0] == m.Intial_B_SOC * m.BatterySize
    model.battery_init = pyo.Constraint(rule=battery_init_rule)

    def battery_term_rule(m):
        return m.InStorageBattery[m.NumTime - 1] == m.Intial_B_SOC * m.BatterySize
    model.battery_term = pyo.Constraint(rule=battery_term_rule)

    # PV energy balance
    def pv_energy_balance_rule(m, t):
        return m.PV[t] * m.PVSize == m.PV2B[t] + m.PV2H[t] + m.PV2G[t]
    model.pv_energy_balance = pyo.Constraint(model.T, rule=pv_energy_balance_rule)

    # House electricity load balance
    def house_electricity_rule(m, t):
        return m.E_Load[t] + m.H2HP[t] + m.H2C[t] == m.PV2H[t] * m.η_PVIV + m.B2H[t] * m.η + m.G2H[t]
    model.house_electricity = pyo.Constraint(model.T, rule=house_electricity_rule)

    # Battery storage dynamics
    def battery_storage_balance_rule(m, t):
        if t < m.NumTime - 1:
            return m.InStorageBattery[t+1] == m.InStorageBattery[t] * (1 - m.BatteryLoss * m.δt) + m.δt * (m.PV2B[t] * m.η - m.B2H[t])
        else:
            return pyo.Constraint.Skip
    model.battery_storage_balance = pyo.Constraint(model.T, rule=battery_storage_balance_rule)

    # Battery discharge constraint
    def battery_discharge_rule(m, t):
        return m.δt * m.B2H[t] <= m.InStorageBattery[t]
    model.battery_discharge = pyo.Constraint(model.T, rule=battery_discharge_rule)

    # Inverter capacity constraint
    def battery_inverter_rule(m, t):
        return m.B2H[t] + m.PV2B[t] <= m.InverterSize
    model.battery_inverter = pyo.Constraint(model.T, rule=battery_inverter_rule)

    # Battery size constraint
    def battery_size_rule(m, t):
        return m.InStorageBattery[t] <= m.BatterySize
    model.battery_size = pyo.Constraint(model.T, rule=battery_size_rule)

    # Battery max discharge constraint
    def battery_max_discharge_rule(m, t):
        return m.InStorageBattery[t] >= m.BatterySize * (1 - m.MaxDischarge)
    model.battery_max_discharge = pyo.Constraint(model.T, rule=battery_max_discharge_rule)

    # Heating power balance
    def heating_power_balance_rule(m, t):
        return m.H2HP[t] == (m.HP2PCM_H[t] + m.HP2H[t]) / m.COP_H
    model.heating_power_balance = pyo.Constraint(model.T, rule=heating_power_balance_rule)

    # Cooling power balance
    def cooling_power_balance_rule(m, t):
        return m.H2C[t] == (m.C2PCM_C[t] + m.C2H[t]) / m.COP_C
    model.cooling_power_balance = pyo.Constraint(model.T, rule=cooling_power_balance_rule)

    # Heat pump heating capacity constraint
    def heat_pump_heating_capacity_rule(m, t):
        return m.H2HP[t] <= m.HPSize
    model.heat_pump_heating_capacity = pyo.Constraint(model.T, rule=heat_pump_heating_capacity_rule)

    # Heat pump cooling capacity constraint
    def heat_pump_cooling_capacity_rule(m, t):
        return m.H2C[t] <= m.HPSize
    model.heat_pump_cooling_capacity = pyo.Constraint(model.T, rule=heat_pump_cooling_capacity_rule)

    # PCM heating storage dynamics
    def pcm_h_storage_balance_rule(m, t):
        if t < m.NumTime - 1:
            return m.InStoragePCM_H[t+1] == m.InStoragePCM_H[t] + m.δt * (m.HP2PCM_H[t] - m.PCM_H2H[t])
        else:
            return pyo.Constraint.Skip
    model.pcm_h_storage_balance = pyo.Constraint(model.T, rule=pcm_h_storage_balance_rule)

    # PCM cooling storage dynamics
    def pcm_c_storage_balance_rule(m, t):
        if t < m.NumTime - 1:
            return m.InStoragePCM_C[t+1] == m.InStoragePCM_C[t] + m.δt * (m.C2PCM_C[t] - m.PCM_C2H[t])
        else:
            return pyo.Constraint.Skip
    model.pcm_c_storage_balance = pyo.Constraint(model.T, rule=pcm_c_storage_balance_rule)

    # PCM heating discharge constraint
    def pcm_h_discharge_rule(m, t):
        return m.δt * m.PCM_H2H[t] <= m.InStoragePCM_H[t]
    model.pcm_h_discharge = pyo.Constraint(model.T, rule=pcm_h_discharge_rule)

    # PCM cooling discharge constraint
    def pcm_c_discharge_rule(m, t):
        return m.δt * m.PCM_C2H[t] <= m.InStoragePCM_C[t]
    model.pcm_c_discharge = pyo.Constraint(model.T, rule=pcm_c_discharge_rule)

    # PCM heating size constraint
    def pcm_h_size_rule(m, t):
        return m.InStoragePCM_H[t] <= m.PCM_H_Size
    model.pcm_h_size = pyo.Constraint(model.T, rule=pcm_h_size_rule)

    # PCM cooling size constraint
    def pcm_c_size_rule(m, t):
        return m.InStoragePCM_C[t] <= m.PCM_C_Size
    model.pcm_c_size = pyo.Constraint(model.T, rule=pcm_c_size_rule)

    # PCM heating initialization and termination
    def pcm_h_init_rule(m):
        return m.InStoragePCM_H[0] == m.Intial_PCM_H_SOC * m.PCM_H_Size
    model.pcm_h_init = pyo.Constraint(rule=pcm_h_init_rule)

    def pcm_c_init_rule(m):
        return m.InStoragePCM_C[0] == m.Intial_PCM_C_SOC * m.PCM_C_Size
    model.pcm_c_init = pyo.Constraint(rule=pcm_c_init_rule)

    def pcm_h_term_rule(m):
        return m.InStoragePCM_H[m.NumTime - 1] == m.Intial_PCM_H_SOC * m.PCM_H_Size
    model.pcm_h_term = pyo.Constraint(rule=pcm_h_term_rule)

    def pcm_c_term_rule(m):
        return m.InStoragePCM_C[m.NumTime - 1] == m.Intial_PCM_C_SOC * m.PCM_C_Size
    model.pcm_c_term = pyo.Constraint(rule=pcm_c_term_rule)

    # Solve the model
    solver = pyo.SolverFactory('gurobi')  # Ensure Gurobi is installed or use another LP solver
    results = solver.solve(model, tee=True)  # tee=True for verbose output

    # Check if the solution is optimal
    if results.solver.status != pyo.SolverStatus.ok:
        raise RuntimeError("Solver did not exit normally")
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError("Solution is not optimal")

    # Retrieve results
    ObjValue = round(pyo.value(model.objective), 3)
    First_stage_cost = round(pyo.value(first_stage_cost), 3)
    Total_Cost = ObjValue + First_stage_cost

    return Total_Cost, First_stage_cost, ObjValue