import pyomo.environ as pyo
import numpy as np
import pandas as pd
import Input_Parameters

def RO_training(input_df_list, lossofloadcost, capacity_costs, scenario):
    """
    Robust optimization for capacity planning (min-max-min structure).
    
    Parameters:
    -----------
    input_df_list : list of DataFrames
        List of input DataFrames, one for each scenario/year.
    lossofloadcost : float
        Cost of loss of load.
        
    Returns:
    --------
    tuple
        (PV_Size, Battery_Size, PCM_Heating_Size, PCM_Cooling_Size, ObjValue, First_stage_cost, Second_stage_cost, HVAC_Cost, Critical_load_cost)
    """
    # Number of scenarios
    num_scenarios = len(input_df_list)
    
    # Extract time resolution from first scenario (assuming consistent across all scenarios)
    datetime_col = input_df_list[0]['DateTime']
    δt = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds() / 3600  # Time resolution in hours
    NumTime = len(datetime_col)
    
    # Create a single integrated model (following the reference approach)
    model = pyo.ConcreteModel()
    
    # Define scenario and time sets
    model.S = pyo.RangeSet(0, num_scenarios-1)
    model.T = pyo.RangeSet(0, NumTime-1)
    
    # First-stage variables (capacity decisions, constant across all scenarios)
    model.PVSize = pyo.Var(within=pyo.NonNegativeReals)
    model.BatterySize = pyo.Var(within=pyo.NonNegativeReals)
    model.PCM_H_Size = pyo.Var(within=pyo.NonNegativeReals)
    model.PCM_C_Size = pyo.Var(within=pyo.NonNegativeReals)
    model.hatQ = pyo.Var(within=pyo.NonNegativeReals)  # Worst-case operational cost
    
    # Define master problem parameters
    if scenario == 'DC':           
        model.HPSize = pyo.Param(initialize=Input_Parameters.HPSize_DC)
    else:
        model.HPSize = pyo.Param(initialize=Input_Parameters.HPSize)
    model.C_PV = pyo.Param(initialize=capacity_costs[0])
    model.C_PV_OP = pyo.Param(initialize=capacity_costs[1])
    model.C_B = pyo.Param(initialize=capacity_costs[2])
    model.C_B_OP = pyo.Param(initialize=capacity_costs[3])
    model.C_HP = pyo.Param(initialize=Input_Parameters.C_HP)
    model.C_HP_OP = pyo.Param(initialize=Input_Parameters.C_HP_OP)
    model.C_PCM_H = pyo.Param(initialize=Input_Parameters.C_PCM_H)
    model.C_PCM_H_OP = pyo.Param(initialize=Input_Parameters.C_PCM_H_OP)
    model.C_PCM_C = pyo.Param(initialize=Input_Parameters.C_PCM_C)
    model.C_PCM_C_OP = pyo.Param(initialize=Input_Parameters.C_PCM_C_OP)
    model.d = pyo.Param(initialize=Input_Parameters.d)
    model.CRF = pyo.Param(initialize=Input_Parameters.CRF)
    model.HVAC_lol_cost = pyo.Param(initialize=Input_Parameters.HVAC_lol_cost)
    
    # First-stage costs (capital + fixed O&M)
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
    
    model.first_stage_cost = (capital_cost * model.CRF + fixed_OM_cost)*(NumTime/8760)
    
    # Master objective: minimize first-stage cost + worst-case second-stage cost
    model.objective = pyo.Objective(
        expr=model.first_stage_cost + model.hatQ,
        sense=pyo.minimize
    )
    
    # Create scenario blocks with complete constraints (following reference approach)
    def scenario_block_rule(b, s):
        # Get this scenario's data
        scenario_data = input_df_list[s]
        
        # Define scenario-specific parameters
        b.PV = pyo.Param(model.T, initialize={t: scenario_data['pv'].iloc[t] for t in range(NumTime)})
        b.E_Load = pyo.Param(model.T, initialize={t: scenario_data['E_Load'].iloc[t] for t in range(NumTime)})
        b.Cooling_Load = pyo.Param(model.T, initialize={t: scenario_data['Cooling_Load'].iloc[t] for t in range(NumTime)})
        b.Heating_Load = pyo.Param(model.T, initialize={t: scenario_data['Heating_Load'].iloc[t] for t in range(NumTime)})
        
        # Parameters
        if scenario == 'DC':           
            b.HPSize = pyo.Param(initialize=Input_Parameters.HPSize_DC)
        else:
            b.HPSize = pyo.Param(initialize=Input_Parameters.HPSize)
        b.C_PV = pyo.Param(initialize=capacity_costs[0])
        b.C_PV_OP = pyo.Param(initialize=capacity_costs[1])
        b.C_B = pyo.Param(initialize=capacity_costs[2])
        b.C_B_OP = pyo.Param(initialize=capacity_costs[3])
        b.BatteryLoss = pyo.Param(initialize=Input_Parameters.BatteryLoss)
        b.MaxDischarge = pyo.Param(initialize=Input_Parameters.MaxDischarge)
        b.η = pyo.Param(initialize=Input_Parameters.η)
        b.C_HP = pyo.Param(initialize=Input_Parameters.C_HP)
        b.C_HP_OP = pyo.Param(initialize=Input_Parameters.C_HP_OP)
        b.COP_H = pyo.Param(initialize=Input_Parameters.COP_H)
        b.COP_C = pyo.Param(initialize=Input_Parameters.COP_C)
        b.C_PCM_H = pyo.Param(initialize=Input_Parameters.C_PCM_H)
        b.C_PCM_H_OP = pyo.Param(initialize=Input_Parameters.C_PCM_H_OP)
        b.C_PCM_C = pyo.Param(initialize=Input_Parameters.C_PCM_C)
        b.C_PCM_C_OP = pyo.Param(initialize=Input_Parameters.C_PCM_C_OP)
        b.d = pyo.Param(initialize=Input_Parameters.d)
        b.CRF = pyo.Param(initialize=Input_Parameters.CRF)
        b.δt = pyo.Param(initialize=δt)
        b.lossofloadcost = pyo.Param(initialize=lossofloadcost)
        b.HVAC_lol_cost = pyo.Param(initialize=Input_Parameters.HVAC_lol_cost)
        b.η_PVIV = pyo.Param(initialize=0.94)
        b.Intial_B_SOC = pyo.Param(initialize=Input_Parameters.Intial_B_SOC)
        b.Intial_PCM_C_SOC = pyo.Param(initialize=Input_Parameters.Intial_PCM_C_SOC)
        b.Intial_PCM_H_SOC = pyo.Param(initialize=Input_Parameters.Intial_PCM_H_SOC)
        
        # Second-stage variables (operational decisions)
        # Power flows
        b.PV2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.PV2G = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.PV2B = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.B2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.H2HP = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.HP2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.H2C = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.C2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.G2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.HP2PCM_H = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.C2PCM_C = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.PCM_H2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.PCM_C2H = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.PV2E = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.B2E = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.G2E = pyo.Var(model.T, within=pyo.NonNegativeReals) # critical electrical load loss of load

        # Storage states
        b.InStorageBattery = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.InStoragePCM_H = pyo.Var(model.T, within=pyo.NonNegativeReals)
        b.InStoragePCM_C = pyo.Var(model.T, within=pyo.NonNegativeReals)
        
        # Operational cost components for this scenario
        b.HVAC_cost = pyo.Expression(
            expr=b.δt * b.HVAC_lol_cost * sum(b.G2H[t] for t in model.T)
        )
        
        b.critical_load_cost = pyo.Expression(
            expr=b.δt * b.lossofloadcost * sum(b.G2E[t] for t in model.T)
        )
        
        # Total operational cost expression (sum of components)
        b.operational_cost = pyo.Expression(
            expr=b.HVAC_cost + b.critical_load_cost
        )
        
        # HVAC load balance
        def HVAC_load_balance_rule(b, t):
            net_thermal_load = b.HP2H[t] - b.C2H[t] + b.PCM_H2H[t] - b.PCM_C2H[t]
            return (b.Cooling_Load[t] - b.Heating_Load[t] + net_thermal_load == 0)
        b.HVAC_load_balance = pyo.Constraint(model.T, rule=HVAC_load_balance_rule)
        
        # Battery storage initialization and termination
        def battery_init_rule(b):
            return b.InStorageBattery[0] == b.Intial_B_SOC * model.BatterySize
        b.battery_init = pyo.Constraint(rule=battery_init_rule)
        
        def battery_term_rule(b):
            return b.InStorageBattery[model.T.last()] == b.Intial_B_SOC * model.BatterySize
        b.battery_term = pyo.Constraint(rule=battery_term_rule)
        
        # PV energy balance
        def pv_energy_balance_rule(b, t):
            return b.PV[t] * model.PVSize == b.PV2B[t] + b.PV2H[t] + b.PV2G[t] + b.PV2E[t]
        b.pv_energy_balance = pyo.Constraint(model.T, rule=pv_energy_balance_rule)
        
        # House overall load balance
        def house_electricity_rule(b, t):
            return b.H2HP[t] + b.H2C[t] == b.PV2H[t] * b.η_PVIV + b.B2H[t] * b.η + b.G2H[t]
        b.house_electricity = pyo.Constraint(model.T, rule=house_electricity_rule)

        # House electricity load balance
        def house_crit_electricity_rule(b, t):
            return b.E_Load[t] == b.PV2E[t] * b.η_PVIV + b.B2E[t] * b.η + b.G2E[t]
        b.house_crit_electricity = pyo.Constraint(model.T, rule=house_crit_electricity_rule)

        # Battery storage dynamics
        def battery_storage_balance_rule(b, t):
            if t < model.T.last():
                return b.InStorageBattery[t+1] == b.InStorageBattery[t] * (1 - b.BatteryLoss * b.δt) + b.δt * (b.PV2B[t] * b.η - b.B2H[t] - b.B2E[t])
            else:
                return pyo.Constraint.Skip
        b.battery_storage_balance = pyo.Constraint(model.T, rule=battery_storage_balance_rule)
        
        # Battery discharge constraint
        def battery_discharge_rule(b, t):
            return b.δt * (b.B2H[t] + b.B2E[t]) <= b.InStorageBattery[t]
        b.battery_discharge = pyo.Constraint(model.T, rule=battery_discharge_rule)
        
        # Battery charging power constraint
        def battery_charging_power_rule(b, t):
            return b.PV2B[t] <= model.BatterySize * 0.25
        b.battery_charging_power = pyo.Constraint(model.T, rule=battery_charging_power_rule)

        # Battery discharging power constraint
        def battery_discharging_power_rule(b, t):
            return b.B2H[t] + b.B2E[t] <= model.BatterySize * 0.25
        b.battery_discharging_power = pyo.Constraint(model.T, rule=battery_discharging_power_rule)
        
        # Battery size constraint
        def battery_size_rule(b, t):
            return b.InStorageBattery[t] <= model.BatterySize
        b.battery_size = pyo.Constraint(model.T, rule=battery_size_rule)
        
        # Battery max discharge constraint
        def battery_max_discharge_rule(b, t):
            return b.InStorageBattery[t] >= model.BatterySize * (1 - b.MaxDischarge)
        b.battery_max_discharge = pyo.Constraint(model.T, rule=battery_max_discharge_rule)
        
        # Heating power balance
        def heating_power_balance_rule(b, t):
            return b.H2HP[t] == (b.HP2PCM_H[t] + b.HP2H[t]) / b.COP_H
        b.heating_power_balance = pyo.Constraint(model.T, rule=heating_power_balance_rule)
        
        # Cooling power balance
        def cooling_power_balance_rule(b, t):
            return b.H2C[t] == (b.C2PCM_C[t] + b.C2H[t]) / b.COP_C
        b.cooling_power_balance = pyo.Constraint(model.T, rule=cooling_power_balance_rule)
        
        # Heat pump heating and cooling capacity constraint
        def heat_pump_heating_capacity_rule(b, t):
            return b.H2HP[t] + b.H2C[t] <= b.HPSize
        b.heat_pump_heating_capacity = pyo.Constraint(model.T, rule=heat_pump_heating_capacity_rule)
        
        # PCM heating storage dynamics
        def pcm_h_storage_balance_rule(b, t):
            if t < model.T.last():
                return b.InStoragePCM_H[t+1] == b.InStoragePCM_H[t] + b.δt * (b.HP2PCM_H[t] - b.PCM_H2H[t])
            else:
                return pyo.Constraint.Skip
        b.pcm_h_storage_balance = pyo.Constraint(model.T, rule=pcm_h_storage_balance_rule)
        
        # PCM cooling storage dynamics
        def pcm_c_storage_balance_rule(b, t):
            if t < model.T.last():
                return b.InStoragePCM_C[t+1] == b.InStoragePCM_C[t] + b.δt * (b.C2PCM_C[t] - b.PCM_C2H[t])
            else:
                return pyo.Constraint.Skip
        b.pcm_c_storage_balance = pyo.Constraint(model.T, rule=pcm_c_storage_balance_rule)
        
        # PCM heating discharge constraint
        def pcm_h_discharge_rule(b, t):
            return b.δt * b.PCM_H2H[t] <= b.InStoragePCM_H[t]
        b.pcm_h_discharge = pyo.Constraint(model.T, rule=pcm_h_discharge_rule)
        
        # PCM cooling discharge constraint
        def pcm_c_discharge_rule(b, t):
            return b.δt * b.PCM_C2H[t] <= b.InStoragePCM_C[t]
        b.pcm_c_discharge = pyo.Constraint(model.T, rule=pcm_c_discharge_rule)
        
        # PCM heating size constraint
        def pcm_h_size_rule(b, t):
            return b.InStoragePCM_H[t] <= model.PCM_H_Size
        b.pcm_h_size = pyo.Constraint(model.T, rule=pcm_h_size_rule)
        
        # PCM cooling size constraint
        def pcm_c_size_rule(b, t):
            return b.InStoragePCM_C[t] <= model.PCM_C_Size
        b.pcm_c_size = pyo.Constraint(model.T, rule=pcm_c_size_rule)
        
        # PCM heating initialization and termination
        def pcm_h_init_rule(b):
            return b.InStoragePCM_H[0] == b.Intial_PCM_H_SOC * model.PCM_H_Size
        b.pcm_h_init = pyo.Constraint(rule=pcm_h_init_rule)
        
        def pcm_c_init_rule(b):
            return b.InStoragePCM_C[0] == b.Intial_PCM_C_SOC * model.PCM_C_Size
        b.pcm_c_init = pyo.Constraint(rule=pcm_c_init_rule)
        
        def pcm_h_term_rule(b):
            return b.InStoragePCM_H[model.T.last()] == b.Intial_PCM_H_SOC * model.PCM_H_Size
        b.pcm_h_term = pyo.Constraint(rule=pcm_h_term_rule)
        
        def pcm_c_term_rule(b):
            return b.InStoragePCM_C[model.T.last()] == b.Intial_PCM_C_SOC * model.PCM_C_Size
        b.pcm_c_term = pyo.Constraint(rule=pcm_c_term_rule)
    
    # Create scenario blocks
    model.scenario = pyo.Block(model.S, rule=scenario_block_rule)
    
    # Constraint for worst-case cost
    model.hatQ_constraints = pyo.ConstraintList()
    
    # Variable to track the final worst-case scenario
    final_worst_scenario = -1
    
    # Implementation of decomposition algorithm
    def solve_model():
        nonlocal final_worst_scenario  # Allow updating the final_worst_scenario variable
        
        solver = pyo.SolverFactory('gurobi')
        solver.options['MIPGap'] = 1e-3
        # solver.options['NumericFocus'] = 3  # Higher focus on numerical accuracy
        
        # Algorithm parameters
        tolerance = 1e-3
        max_iterations = 20
        
        # Initial capacity values
        prev_caps = {
            'PV': 10.0,
            'Battery': 20.0,
            'PCM_H': 5.0,
            'PCM_C': 5.0
        }
        
        # Track which scenarios have been used for cuts
        worst_scenarios = set()
        
        # Track upper and lower bounds
        UB = float('inf')
        LB = float('-inf')
        
        print("Starting robust optimization algorithm...")
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration+1} ---")
            
            # Fix capacity variables for this iteration
            model.PVSize.fix(prev_caps['PV'])
            model.BatterySize.fix(prev_caps['Battery'])
            model.PCM_H_Size.fix(prev_caps['PCM_H'])
            model.PCM_C_Size.fix(prev_caps['PCM_C'])
            
            # Find the worst-case scenario by solving each scenario's operational problem
            worst_cost = -float('inf')
            worst_scenario = -1
            
            for s in model.S:
                # Create temporary objective for this scenario block
                model.scenario[s].temp_obj = pyo.Objective(
                    expr=model.scenario[s].operational_cost,
                    sense=pyo.minimize
                )
                
                # Set all other objectives to None
                model.objective.deactivate()
                for other_s in model.S:
                    if other_s != s and hasattr(model.scenario[other_s], 'temp_obj'):
                        model.scenario[other_s].temp_obj.deactivate()
                
                # Solve scenario subproblem
                results = solver.solve(model, tee=False)
                
                # Get scenario cost and compare to worst case
                if (results.solver.status == pyo.SolverStatus.ok and 
                    results.solver.termination_condition == pyo.TerminationCondition.optimal):
                    scenario_cost = pyo.value(model.scenario[s].operational_cost)
                    
                    if scenario_cost > worst_cost:
                        worst_cost = scenario_cost
                        worst_scenario = s
                
                # Clean up temporary objective
                del model.scenario[s].temp_obj
            
            # Reactivate main objective
            model.objective.activate()
            
            # Calculate upper bound for current solution
            first_stage_val = (
                (prev_caps['PV'] * Input_Parameters.C_PV * Input_Parameters.CRF + 
                prev_caps['PV'] * Input_Parameters.C_PV_OP +
                prev_caps['Battery'] * Input_Parameters.C_B * Input_Parameters.CRF + 
                prev_caps['Battery'] * Input_Parameters.C_B_OP +
                Input_Parameters.HPSize * Input_Parameters.C_HP * Input_Parameters.CRF + 
                Input_Parameters.HPSize * Input_Parameters.C_HP_OP +
                prev_caps['PCM_H'] * Input_Parameters.C_PCM_H * Input_Parameters.CRF + 
                prev_caps['PCM_H'] * Input_Parameters.C_PCM_H_OP +
                prev_caps['PCM_C'] * Input_Parameters.C_PCM_C * Input_Parameters.CRF + 
                prev_caps['PCM_C'] * Input_Parameters.C_PCM_C_OP)*(NumTime/8760)
            )
            
            current_UB = first_stage_val + worst_cost
            if current_UB < UB:
                UB = current_UB
            
            # Add cut for worst-case scenario
            if worst_scenario not in worst_scenarios:
                print(f"Adding cut for scenario {worst_scenario} with cost {worst_cost:.2f}")
                model.hatQ_constraints.add(
                    model.hatQ >= model.scenario[worst_scenario].operational_cost
                )
                worst_scenarios.add(worst_scenario)
                
                # Update the final worst-case scenario
                final_worst_scenario = worst_scenario
            
            # Unfix variables and solve master problem
            model.PVSize.unfix()
            model.BatterySize.unfix()
            model.PCM_H_Size.unfix()
            model.PCM_C_Size.unfix()
            
            # Solve master problem
            results = solver.solve(model, tee=False)
            
            if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
                LB = pyo.value(model.objective)
            
            # Get updated capacity values
            new_caps = {
                'PV': pyo.value(model.PVSize),
                'Battery': pyo.value(model.BatterySize),
                'PCM_H': pyo.value(model.PCM_H_Size),
                'PCM_C': pyo.value(model.PCM_C_Size)
            }
            
            # Report current solution
            print(f"Current solution:")
            print(f"  PV Size: {new_caps['PV']:.2f} kW")
            print(f"  Battery Size: {new_caps['Battery']:.2f} kWh")
            print(f"  PCM Heating Size: {new_caps['PCM_H']:.2f} kWh")
            print(f"  PCM Cooling Size: {new_caps['PCM_C']:.2f} kWh")
            print(f"  Worst-case operational cost: {pyo.value(model.hatQ):.2f}")
            print(f"  First-stage cost: {first_stage_val:.2f}")
            print(f"  Total cost: {pyo.value(model.objective):.2f}")
            print(f"  Lower Bound: {LB:.2f}, Upper Bound: {UB:.2f}")
            
            # Calculate gap
            if UB > 0 and LB > 0:
                gap = (UB - LB) / UB
                print(f"  Optimality gap: {gap * 100:.2f}%")
                
                if gap < tolerance:
                    print("\nAlgorithm converged based on optimality gap!")
                    break
            else:
                print("  Cannot calculate optimality gap - bounds are not both positive")
            
            # Check capacity change convergence as backup
            capacity_change = max(
                abs(new_caps['PV'] - prev_caps['PV']),
                abs(new_caps['Battery'] - prev_caps['Battery']),
                abs(new_caps['PCM_H'] - prev_caps['PCM_H']),
                abs(new_caps['PCM_C'] - prev_caps['PCM_C'])
            )
            
            print(f"  Maximum capacity change: {capacity_change:.6f}")
            
            if capacity_change < tolerance:
                print("\nAlgorithm converged based on capacity stability!")
                break
            
            # Update for next iteration
            prev_caps = new_caps
            
            if iteration == max_iterations - 1:
                print("\nWarning: Maximum iterations reached without convergence")
        
        return final_worst_scenario
    
    # Run the decomposition algorithm
    final_worst_scenario = solve_model()
    
    # Calculate final costs
    PV_Size = round(pyo.value(model.PVSize), 3)
    Battery_Size = round(pyo.value(model.BatterySize), 3)
    PCM_Heating_Size = round(pyo.value(model.PCM_H_Size), 3)
    PCM_Cooling_Size = round(pyo.value(model.PCM_C_Size), 3)
    
    # Calculate first-stage cost
    cap_cost = (
        Input_Parameters.C_PV * PV_Size +
        Input_Parameters.C_B * Battery_Size +
        Input_Parameters.C_HP * Input_Parameters.HPSize +
        Input_Parameters.C_PCM_H * PCM_Heating_Size +
        Input_Parameters.C_PCM_C * PCM_Cooling_Size
    )
    
    om_cost = (
        Input_Parameters.C_PV_OP * PV_Size +
        Input_Parameters.C_B_OP * Battery_Size +
        Input_Parameters.C_HP_OP * Input_Parameters.HPSize +
        Input_Parameters.C_PCM_H_OP * PCM_Heating_Size +
        Input_Parameters.C_PCM_C_OP * PCM_Cooling_Size
    )
    
    First_stage_cost = round(cap_cost * Input_Parameters.CRF + om_cost, 3)*(NumTime/8760)
    Second_stage_cost = round(pyo.value(model.hatQ), 3)
    
    # Get the broken-down cost components from the worst-case scenario
    if final_worst_scenario >= 0:  # Check if we have a valid worst-case scenario
        HVAC_Cost = round(pyo.value(model.scenario[final_worst_scenario].HVAC_cost), 3)
        Critical_load_cost = round(pyo.value(model.scenario[final_worst_scenario].critical_load_cost), 3)
    else:
        # Fallback if no worst-case scenario was identified (shouldn't happen in normal execution)
        HVAC_Cost = 0.0
        Critical_load_cost = Second_stage_cost
    
    ObjValue = round(First_stage_cost + Second_stage_cost, 3)
    
    return PV_Size, Battery_Size, PCM_Heating_Size, PCM_Cooling_Size, ObjValue, First_stage_cost, Second_stage_cost, HVAC_Cost, Critical_load_cost
