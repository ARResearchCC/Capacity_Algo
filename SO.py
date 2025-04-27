import pyomo.environ as pyo
import numpy as np
import pandas as pd
import Input_Parameters

def SO_training(input_df_list, lossofloadcost, capacity_costs, scenario):
    """
    Two-stage stochastic optimization for capacity planning.
    
    Parameters:
    -----------
    input_df_list : list of DataFrames
        List of input DataFrames, one for each scenario/year.
    lossofloadcost : float
        Cost of loss of load.
        
    Returns:
    --------
    tuple
        (PV_Size, Battery_Size, PCM_Heating_Size, PCM_Cooling_Size, ObjValue, First_stage_cost, Second_stage_cost)
    """
    # Number of scenarios
    num_scenarios = len(input_df_list)
    
    # Extract time resolution from first scenario (assuming consistent across all scenarios)
    datetime_col = input_df_list[0]['DateTime']
    δt = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600  # Time resolution in hours
    NumTime = len(datetime_col)

    # Create Pyomo model
    model = pyo.ConcreteModel()
    
    # Define scenario set
    model.S = pyo.Set(initialize=range(num_scenarios))
    
    # Time index set
    model.T = pyo.Set(initialize=range(NumTime))
    model.NumTime = NumTime

    # First-stage variables (capacity decisions, constant across all scenarios)
    model.PVSize = pyo.Var(within=pyo.NonNegativeReals)
    model.BatterySize = pyo.Var(within=pyo.NonNegativeReals)
    model.PCM_H_Size = pyo.Var(within=pyo.NonNegativeReals)
    model.PCM_C_Size = pyo.Var(within=pyo.NonNegativeReals)
    
    # Constants as parameters
    if scenario == 'DC':           
        model.HPSize = pyo.Param(initialize=Input_Parameters.HPSize_DC)
    else:
        model.HPSize = pyo.Param(initialize=Input_Parameters.HPSize)
    model.C_PV = pyo.Param(initialize=capacity_costs[0])
    model.C_PV_OP = pyo.Param(initialize=capacity_costs[1])
    model.C_B = pyo.Param(initialize=capacity_costs[2])
    model.C_B_OP = pyo.Param(initialize=capacity_costs[3])
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
    model.HVAC_lol_cost = pyo.Param(initialize=Input_Parameters.HVAC_lol_cost)
    model.lossofloadcost = pyo.Param(initialize=lossofloadcost)
    model.η_PVIV = pyo.Param(initialize=0.94)
    model.Intial_B_SOC = pyo.Param(initialize=Input_Parameters.Intial_B_SOC)
    model.Intial_PCM_C_SOC = pyo.Param(initialize=Input_Parameters.Intial_PCM_C_SOC)
    model.Intial_PCM_H_SOC = pyo.Param(initialize=Input_Parameters.Intial_PCM_H_SOC)

    # Create scenario-specific parameter dictionaries
    PV_data = {}
    E_Load_data = {}
    Cooling_Load_data = {}
    Heating_Load_data = {}
    
    # Populate parameter dictionaries
    for s in range(num_scenarios):
        scenario_data = input_df_list[s]
        for t in range(NumTime):
            PV_data[s, t] = scenario_data['pv'].iloc[t]
            E_Load_data[s, t] = scenario_data['E_Load'].iloc[t]
            Cooling_Load_data[s, t] = scenario_data['Cooling_Load'].iloc[t]
            Heating_Load_data[s, t] = scenario_data['Heating_Load'].iloc[t]
    
    # Define scenario-specific parameters
    model.PV_param = pyo.Param(model.S, model.T, initialize=PV_data)
    model.E_Load_param = pyo.Param(model.S, model.T, initialize=E_Load_data)
    model.Cooling_Load_param = pyo.Param(model.S, model.T, initialize=Cooling_Load_data)
    model.Heating_Load_param = pyo.Param(model.S, model.T, initialize=Heating_Load_data)
    
    # Second-stage variables (specific to each scenario)
    # Power flows
    model.PV2H = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.PV2G = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.PV2B = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.B2H = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.H2HP = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.HP2H = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.H2C = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.C2H = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.G2H = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.HP2PCM_H = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.C2PCM_C = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.PCM_H2H = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.PCM_C2H = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.PV2E = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.B2E = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.G2E = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals) # critical electrical load loss of load
    
    # Storage states
    model.InStorageBattery = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.InStoragePCM_H = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
    model.InStoragePCM_C = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)

    # Objective Function
    # First-stage cost: capital cost + fixed O&M
    capital_cost = (
        model.C_PV * model.PVSize +
        model.C_B * model.BatterySize +
        model.C_HP * model.HPSize +
        model.C_PCM_H * model.PCM_H_Size +
        model.C_PCM_C * model.PCM_C_Size
    )

    fixed_OM_cost = (
        model.C_PV_OP * model.PVSize +
        model.C_B_OP * model.BatterySize +
        model.C_HP_OP * model.HPSize +
        model.C_PCM_H_OP * model.PCM_H_Size +
        model.C_PCM_C_OP * model.PCM_C_Size
    )

    first_stage_cost = (capital_cost * model.CRF + fixed_OM_cost)*(NumTime/8760)
    
    # Second-stage cost: average outage cost across all scenarios
    HVAC_cost = (1/num_scenarios) * sum(
        model.δt * model.HVAC_lol_cost * sum(model.G2H[s, t] for t in model.T)
        for s in model.S
    )

    critical_load_cost = (1/num_scenarios) * sum(
        model.δt * model.lossofloadcost * sum(model.G2E[s, t] for t in model.T)
        for s in model.S
    )

    second_stage_cost = HVAC_cost + critical_load_cost

    # Total objective: first_stage_cost + second_stage_cost
    model.objective = pyo.Objective(
        expr = first_stage_cost + second_stage_cost,
        sense = pyo.minimize
    )

    # Constraints for each scenario
    def HVAC_load_balance_rule(m, s, t):
        net_thermal_load = m.HP2H[s, t] - m.C2H[s, t] + m.PCM_H2H[s, t] - m.PCM_C2H[s, t]
        return (m.Cooling_Load_param[s, t] - m.Heating_Load_param[s, t] + net_thermal_load == 0)
    model.HVAC_load_balance = pyo.Constraint(model.S, model.T, rule=HVAC_load_balance_rule)

    # Battery storage initialization and termination for each scenario
    def battery_init_rule(m, s):
        return m.InStorageBattery[s, 0] == m.Intial_B_SOC * m.BatterySize
    model.battery_init = pyo.Constraint(model.S, rule=battery_init_rule)

    def battery_term_rule(m, s):
        return m.InStorageBattery[s, m.NumTime - 1] == m.Intial_B_SOC * m.BatterySize
    model.battery_term = pyo.Constraint(model.S, rule=battery_term_rule)

    # PV energy balance for each scenario
    def pv_energy_balance_rule(m, s, t):
        return m.PV_param[s, t] * m.PVSize == m.PV2B[s, t] + m.PV2H[s, t] + m.PV2G[s, t] + m.PV2E[s, t]
    model.pv_energy_balance = pyo.Constraint(model.S, model.T, rule=pv_energy_balance_rule)

    # House overall load balance
    def house_electricity_rule(m, s, t):
        return m.H2HP[s, t] + m.H2C[s, t] == m.PV2H[s, t] * m.η_PVIV + m.B2H[s, t] * m.η + m.G2H[s, t]
    model.house_electricity = pyo.Constraint(model.S, model.T, rule=house_electricity_rule)

    # House electricity load balance
    def house_crit_electricity_rule(m, s, t):
        return m.E_Load_param[s, t] == m.PV2E[s, t] * m.η_PVIV + m.B2E[s, t] * m.η + m.G2E[s, t]
    model.house_crit_electricity = pyo.Constraint(model.S, model.T, rule=house_crit_electricity_rule)

    # Battery storage dynamics for each scenario
    def battery_storage_balance_rule(m, s, t):
        if t < m.NumTime - 1:
            return m.InStorageBattery[s, t+1] == m.InStorageBattery[s, t] * (1 - m.BatteryLoss * m.δt) + m.δt * (m.PV2B[s, t] * m.η - m.B2H[s, t] - m.B2E[s, t])
        else:
            return pyo.Constraint.Skip
    model.battery_storage_balance = pyo.Constraint(model.S, model.T, rule=battery_storage_balance_rule)

    # Battery discharge constraint for each scenario
    def battery_discharge_rule(m, s, t):
        return m.δt * (m.B2H[s, t] + m.B2E[s, t]) <= m.InStorageBattery[s, t]
    model.battery_discharge = pyo.Constraint(model.S, model.T, rule=battery_discharge_rule)

    # Battery charging power constraint
    def battery_charging_power_rule(m, s, t):
        return m.PV2B[s, t] <= m.BatterySize * 0.25
    model.battery_charging_power = pyo.Constraint(model.S, model.T, rule=battery_charging_power_rule)

    # Battery discharging power constraint
    def battery_discharging_power_rule(m, s, t):
        return m.B2H[s, t] + m.B2E[s, t] <= m.BatterySize * 0.25
    model.battery_discharging_power = pyo.Constraint(model.S, model.T, rule=battery_discharging_power_rule)

    # Battery size constraint for each scenario
    def battery_size_rule(m, s, t):
        return m.InStorageBattery[s, t] <= m.BatterySize
    model.battery_size = pyo.Constraint(model.S, model.T, rule=battery_size_rule)

    # Battery max discharge constraint for each scenario
    def battery_max_discharge_rule(m, s, t):
        return m.InStorageBattery[s, t] >= m.BatterySize * (1 - m.MaxDischarge)
    model.battery_max_discharge = pyo.Constraint(model.S, model.T, rule=battery_max_discharge_rule)

    # Heating power balance for each scenario
    def heating_power_balance_rule(m, s, t):
        return m.H2HP[s, t] == (m.HP2PCM_H[s, t] + m.HP2H[s, t]) / m.COP_H
    model.heating_power_balance = pyo.Constraint(model.S, model.T, rule=heating_power_balance_rule)

    # Cooling power balance for each scenario
    def cooling_power_balance_rule(m, s, t):
        return m.H2C[s, t] == (m.C2PCM_C[s, t] + m.C2H[s, t]) / m.COP_C
    model.cooling_power_balance = pyo.Constraint(model.S, model.T, rule=cooling_power_balance_rule)

    # Heat pump heating and cooling capacity constraint for each scenario
    def heat_pump_heating_capacity_rule(m, s, t):
        return m.H2HP[s, t] + m.H2C[s, t] <= m.HPSize
    model.heat_pump_heating_capacity = pyo.Constraint(model.S, model.T, rule=heat_pump_heating_capacity_rule)

    # PCM heating storage dynamics for each scenario
    def pcm_h_storage_balance_rule(m, s, t):
        if t < m.NumTime - 1:
            return m.InStoragePCM_H[s, t+1] == m.InStoragePCM_H[s, t] + m.δt * (m.HP2PCM_H[s, t] - m.PCM_H2H[s, t])
        else:
            return pyo.Constraint.Skip
    model.pcm_h_storage_balance = pyo.Constraint(model.S, model.T, rule=pcm_h_storage_balance_rule)

    # PCM cooling storage dynamics for each scenario
    def pcm_c_storage_balance_rule(m, s, t):
        if t < m.NumTime - 1:
            return m.InStoragePCM_C[s, t+1] == m.InStoragePCM_C[s, t] + m.δt * (m.C2PCM_C[s, t] - m.PCM_C2H[s, t])
        else:
            return pyo.Constraint.Skip
    model.pcm_c_storage_balance = pyo.Constraint(model.S, model.T, rule=pcm_c_storage_balance_rule)

    # PCM heating discharge constraint for each scenario
    def pcm_h_discharge_rule(m, s, t):
        return m.δt * m.PCM_H2H[s, t] <= m.InStoragePCM_H[s, t]
    model.pcm_h_discharge = pyo.Constraint(model.S, model.T, rule=pcm_h_discharge_rule)

    # PCM cooling discharge constraint for each scenario
    def pcm_c_discharge_rule(m, s, t):
        return m.δt * m.PCM_C2H[s, t] <= m.InStoragePCM_C[s, t]
    model.pcm_c_discharge = pyo.Constraint(model.S, model.T, rule=pcm_c_discharge_rule)

    # PCM heating size constraint for each scenario
    def pcm_h_size_rule(m, s, t):
        return m.InStoragePCM_H[s, t] <= m.PCM_H_Size
    model.pcm_h_size = pyo.Constraint(model.S, model.T, rule=pcm_h_size_rule)

    # PCM cooling size constraint for each scenario
    def pcm_c_size_rule(m, s, t):
        return m.InStoragePCM_C[s, t] <= m.PCM_C_Size
    model.pcm_c_size = pyo.Constraint(model.S, model.T, rule=pcm_c_size_rule)

    # PCM heating initialization and termination for each scenario
    def pcm_h_init_rule(m, s):
        return m.InStoragePCM_H[s, 0] == m.Intial_PCM_H_SOC * m.PCM_H_Size
    model.pcm_h_init = pyo.Constraint(model.S, rule=pcm_h_init_rule)

    def pcm_c_init_rule(m, s):
        return m.InStoragePCM_C[s, 0] == m.Intial_PCM_C_SOC * m.PCM_C_Size
    model.pcm_c_init = pyo.Constraint(model.S, rule=pcm_c_init_rule)

    def pcm_h_term_rule(m, s):
        return m.InStoragePCM_H[s, m.NumTime - 1] == m.Intial_PCM_H_SOC * m.PCM_H_Size
    model.pcm_h_term = pyo.Constraint(model.S, rule=pcm_h_term_rule)

    def pcm_c_term_rule(m, s):
        return m.InStoragePCM_C[s, m.NumTime - 1] == m.Intial_PCM_C_SOC * m.PCM_C_Size
    model.pcm_c_term = pyo.Constraint(model.S, rule=pcm_c_term_rule)

    # Solve the model
    solver = pyo.SolverFactory('gurobi')  # Ensure Gurobi is installed or use another LP solver
    solver.options['Threads'] = 4 
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
    HVAC_Cost = round(pyo.value(HVAC_cost), 3)
    Critical_load_cost = round(pyo.value(critical_load_cost), 3)
    Second_stage_cost = round(pyo.value(second_stage_cost), 3)
    return PV_Size, Battery_Size, PCM_Heating_Size, PCM_Cooling_Size, ObjValue, First_stage_cost, Second_stage_cost, HVAC_Cost, Critical_load_cost
