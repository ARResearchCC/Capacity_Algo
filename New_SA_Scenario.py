import Data_Conversion
import Input_Parameters 
import Data_Conversion
import Passive_Model
import Solar_Generation
import Electrical_Load

import Simulate
import Baseline_CO
import SO
import RO

import numpy as np
import pandas as pd
import os
from pathlib import Path

# Define the base data directory, list of locations, and the weather year.
data_dir = "Data"
# locations = ["HalfMoonBay", "Arizona", "Alaska", "Minnesota", "Florida"]
location = ["HalfMoonBay"]
scenarios = ["FOB", "DC", "RC"]
algorithms = ["LP", "SO", "RO"]

fold = 5 # testing data is 1998-2002, 2003-2007...
# Baseline capacity costs
capacity_costs = [Input_Parameters.C_PV, Input_Parameters.C_PV_OP, Input_Parameters.C_B, Input_Parameters.C_B_OP]

# 20 years of training data
weather_year_list = list(range(1998, 2023))  # The upper bound in range() is exclusive

# Set a random seed for consistency
# Define the number of rows (i) and columns (j)

j = len(weather_year_list)  # Number of columns

# Generate a sequential array of numbers starting from 1
total_cells = j  # Total number of cells in the DataFrame
sequential_numbers = np.arange(1, total_cells + 1)  # Generates 1, 2, 3, ..., i*j
random_seeds = sequential_numbers
# The base case should always be the first row of the random seed matrix, for sensitivity analysis on locations, scenarios, and capacity costs.

# Get the list of latitude, longitude, and timezones for all locations
lats, lons, timezones = Data_Conversion.get_timezones(data_dir, location)

# Create the nested dictionary to store all input data
nested_dict = {scenario: {year: {} for year in weather_year_list} for scenario in scenarios}

# Get all input data for each of the 25 years
for i in range(len(scenarios)):
    for j in range(len(weather_year_list)):

        scenario = scenarios[i]
        year = weather_year_list[j]
        
        # Get a unique random seed number
        random_seed = random_seeds[j]

        # Read NSRDB weather data of the given location of the given year
        # NSRDB_raw_weather = Data_Conversion.read_NSRDB(data_dir, location, year).head(24)
        NSRDB_raw_weather = Data_Conversion.read_NSRDB(data_dir, location[0], year).head(24)
        
        # Prepare weather data file using NSRDB data
        weather_data = Data_Conversion.prepare_NSRDB(NSRDB_raw_weather, lats[0], lons[0], timezones[0])

        # Prepare heating and cooling load using weather data and passive model
        NetHeatTransfers = Passive_Model.passive_model(Input_Parameters.calibration_file_path, weather_data, Input_Parameters.T_indoor_constant, lats[0])

        # Prepare solar PV capacity factor using weather data
        pv_cf = Solar_Generation.generate_pv(weather_data, lats[0])

        # Prepare occupancy and electrical load schedule using for a specific random seed number for a specific year at a specific scenario
        load_sched = Electrical_Load.generate_schedules(scenario, weather_data, random_seed)
        
        # Combine all relative input data as input_df, which will be the input of the capacity optimization algorithm
        input_df = Data_Conversion.combine_input_NSRDB(weather_data, load_sched, pv_cf, NetHeatTransfers)
        
        # Add input_df to list
        nested_dict[scenario][year] = input_df


# Define folder name
folder_name = "SA_Scenarios"
# Create folder if it doesn't already exist
os.makedirs(folder_name, exist_ok=True)

# Iterate through scenarios
for i in range(len(scenarios)):
    
    # Get current scenario
    scenario = scenarios[i]
    if scenario == 'DC':
        lolc = Input_Parameters.lossofloadcost_DC
    else:
        lolc = Input_Parameters.lossofloadcost

    # Make a excel workbook file
    output_file = os.path.join(folder_name, f"SA_Scenarios_{scenario}.xlsx")

    # Create the nested dictionary to store training and testing results
    training_results = {fold_iteration: {algo: {} for algo in algorithms} for fold_iteration in range(fold)}
    testing_results = {fold_iteration: {algo: {} for algo in algorithms} for fold_iteration in range(fold)}
    
    # Iterate through folds
    for k in range(fold):
        start_idx = k * 5
        testing_year_list = weather_year_list[start_idx : start_idx + 5]
        training_year_list = [y for y in weather_year_list if y not in testing_year_list]

        num_train_years = len(training_year_list)
        num_test_years = len(testing_year_list)

        # Create the temporary placeholder dictionary to store LP training results
        training_results_temporary_lp = {year: {} for year in training_year_list}
        
        ################################### LP ###################################
        # Training
        for y in range(len(training_year_list)):
            
            # Get current year
            year = training_year_list[y]            
            # Fetch input_df
            input_df = nested_dict[scenario][year]
            # Run LP function
            PV_Size, Battery_Size, PCM_Heating_Size, PCM_Cooling_Size, ObjValue, First_stage_cost, Second_stage_cost, HVAC_Cost, Critical_load_cost = Baseline_CO.Cap_Baseline_V1(input_df, lolc, capacity_costs, scenario)
            # Store the variables in the temporary placeholder dictionary for LP training results
            training_results_temporary_lp[year] = {
                'PV_Size': PV_Size,
                'Battery_Size': Battery_Size,
                'PCM_Heating_Size': PCM_Heating_Size,
                'PCM_Cooling_Size': PCM_Cooling_Size,
                'Training Total Cost': ObjValue,
                'Training Capital Cost': First_stage_cost,
                'Training Operation Cost': Second_stage_cost,
                'Training HVAC Cost': HVAC_Cost,
                'Training Critical Load Cost': Critical_load_cost
            }

        # Initialize dictionary to store the average results for LP training
        training_results_averaged_lp = {
            'PV_Size': 0,
            'Battery_Size': 0,
            'PCM_Heating_Size': 0,
            'PCM_Cooling_Size': 0,
            'Training Total Cost': 0,
            'Training Capital Cost': 0,
            'Training Operation Cost': 0,
            'Training HVAC Cost': 0,
            'Training Critical Load Cost': 0
        }

        # Sum the values across years
        for result in training_results_temporary_lp.values():
            for key in training_results_averaged_lp:
                training_results_averaged_lp[key] += result[key]

        # Divide by number of years to get average
        for key in training_results_averaged_lp:
            training_results_averaged_lp[key] /= num_train_years

        # Store LP training output formally
        training_results[k]["LP"] = training_results_averaged_lp

        # Testing
        test_capacities = [training_results[k]["LP"]['PV_Size'], training_results[k]["LP"]['Battery_Size'], training_results[k]["LP"]['PCM_Heating_Size'], training_results[k]["LP"]['PCM_Cooling_Size']]

        # Initialize temporary space to store LP testing result
        testing_results_temporary_lp = {year: {} for year in testing_year_list}

        for y in range(len(testing_year_list)):
            
            # Get current year
            year = testing_year_list[y] 
            # Fetch input_df
            input_df = nested_dict[scenario][year]
            # Simulate with LP result
            ObjValue, First_stage_cost, Second_stage_cost, HVAC_Cost, Critical_load_cost = Simulate.simulate(input_df, lolc, test_capacities, capacity_costs, scenario)
            
            # Store the variables in the temporary placeholder dictionary for LP testing results
            testing_results_temporary_lp[year] = {
                'Testing Total Cost': ObjValue,
                'Testing Capital Cost': First_stage_cost,
                'Testing Operation Cost': Second_stage_cost,
                'Testing HVAC Cost': HVAC_Cost,
                'Testing Critical Load Cost': Critical_load_cost
            }

        # Initialize dictionary to store the average results for LP testing
        testing_results_averaged_lp = {
            'Testing Total Cost': 0,
            'Testing Capital Cost': 0,
            'Testing Operation Cost': 0,
            'Testing HVAC Cost': 0,
            'Testing Critical Load Cost': 0
        }

        # Sum the values across years
        for result in testing_results_temporary_lp.values():
            for key in testing_results_averaged_lp:
                testing_results_averaged_lp[key] += result[key]

        # Divide by number of years to get average
        for key in testing_results_averaged_lp:
            testing_results_averaged_lp[key] /= num_test_years

        # Store LP testing output formally
        testing_results[k]["LP"] = testing_results_averaged_lp
    
        ################################### SO ###################################
        
        # Initialize storing space to store all input_dfs for SO and RO
        input_df_list_train = []
        
        # Training
        for y in range(len(training_year_list)):
            
            # Get current year
            year = training_year_list[y]

            # Fetch input_df
            input_df = nested_dict[scenario][year]

            # Add input_df to list
            input_df_list_train.append(input_df)

        # Run SO function 
        PV_Size, Battery_Size, PCM_Heating_Size, PCM_Cooling_Size, ObjValue, First_stage_cost, Second_stage_cost, HVAC_Cost, Critical_load_cost = SO.SO_training(input_df_list_train, lolc, capacity_costs, scenario)
        
        # Store SO training output formally
        training_results[k]["SO"] = {
            'PV_Size': PV_Size,
            'Battery_Size': Battery_Size,
            'PCM_Heating_Size': PCM_Heating_Size,
            'PCM_Cooling_Size': PCM_Cooling_Size,
            'Training Total Cost': ObjValue,
            'Training Capital Cost': First_stage_cost,
            'Training Operation Cost': Second_stage_cost,
            'Training HVAC Cost': HVAC_Cost,
            'Training Critical Load Cost': Critical_load_cost
        }

        # Testing

        test_capacities = [training_results[k]["SO"]['PV_Size'], training_results[k]["SO"]['Battery_Size'], training_results[k]["SO"]['PCM_Heating_Size'], training_results[k]["SO"]['PCM_Cooling_Size']]

        # Initialize temporary space to store SO testing result
        testing_results_temporary_so = {year: {} for year in testing_year_list}

        for y in range(len(testing_year_list)):
            
            # Get current year
            year = testing_year_list[y] 
            # Fetch input_df
            input_df = nested_dict[scenario][year]
            # Simulate with SO result            
            ObjValue, First_stage_cost, Second_stage_cost, HVAC_Cost, Critical_load_cost = Simulate.simulate(input_df, lolc, test_capacities, capacity_costs, scenario)
            
            # Store the variables in the temporary placeholder dictionary for SO testing results
            testing_results_temporary_so[year] = {
                'Testing Total Cost': ObjValue,
                'Testing Capital Cost': First_stage_cost,
                'Testing Operation Cost': Second_stage_cost,
                'Testing HVAC Cost': HVAC_Cost,
                'Testing Critical Load Cost': Critical_load_cost
            }

        # Initialize dictionary to store the average results for SO testing
        testing_results_averaged_SO = {
            'Testing Total Cost': 0,
            'Testing Capital Cost': 0,
            'Testing Operation Cost': 0,
            'Testing HVAC Cost': 0,
            'Testing Critical Load Cost': 0
        }

        # Sum the values across years
        for result in testing_results_temporary_so.values():
            for key in testing_results_averaged_SO:
                testing_results_averaged_SO[key] += result[key]

        # Divide by number of years to get average
        for key in testing_results_averaged_SO:
            testing_results_averaged_SO[key] /= num_test_years

        # Store SO testing output formally
        testing_results[k]["SO"] = testing_results_averaged_SO

        ################################### RO ###################################

        # Training

        # Run RO function 
        PV_Size, Battery_Size, PCM_Heating_Size, PCM_Cooling_Size, ObjValue, First_stage_cost, Second_stage_cost, HVAC_Cost, Critical_load_cost = RO.RO_training(input_df_list_train, lolc, capacity_costs, scenario)
        
        # Store RO training output formally
        training_results[k]["RO"] = {
            'PV_Size': PV_Size,
            'Battery_Size': Battery_Size,
            'PCM_Heating_Size': PCM_Heating_Size,
            'PCM_Cooling_Size': PCM_Cooling_Size,
            'Training Total Cost': ObjValue,
            'Training Capital Cost': First_stage_cost,
            'Training Operation Cost': Second_stage_cost,
            'Training HVAC Cost': HVAC_Cost,
            'Training Critical Load Cost': Critical_load_cost
        }

        # Testing

        test_capacities = [training_results[k]["RO"]['PV_Size'], training_results[k]["RO"]['Battery_Size'], training_results[k]["RO"]['PCM_Heating_Size'], training_results[k]["RO"]['PCM_Cooling_Size']]

        # Initialize temporary space to store RO testing result 
        testing_results_temporary_ro = {year: {} for year in testing_year_list}

        for y in range(len(testing_year_list)):
            
            # Get current year
            year = testing_year_list[y] 
            # Fetch input_df
            input_df = nested_dict[scenario][year]
            # Simulate with RO result
            ObjValue, First_stage_cost, Second_stage_cost, HVAC_Cost, Critical_load_cost = Simulate.simulate(input_df, lolc, test_capacities, capacity_costs, scenario)
             
            # Store the variables in the temporary placeholder dictionary for RO testing results
            testing_results_temporary_ro[year] = {
                'Testing Total Cost': ObjValue,
                'Testing Capital Cost': First_stage_cost,
                'Testing Operation Cost': Second_stage_cost,
                'Testing HVAC Cost': HVAC_Cost,
                'Testing Critical Load Cost': Critical_load_cost
            }

        # Initialize dictionary to store the average results for RO testing
        testing_results_averaged_RO = {
            'Testing Total Cost': 0,
            'Testing Capital Cost': 0,
            'Testing Operation Cost': 0,
            'Testing HVAC Cost': 0,
            'Testing Critical Load Cost': 0
        }

        # Sum the values across years
        for result in testing_results_temporary_ro.values():
            for key in testing_results_averaged_RO:
                testing_results_averaged_RO[key] += result[key]

        # Divide by number of years to get average
        for key in testing_results_averaged_RO:
            testing_results_averaged_RO[key] /= num_test_years

        # Store RO testing output formally
        testing_results[k]["RO"] = testing_results_averaged_RO

    # Reporting (export to .xlsx file)
    # In the SA_Scenario folder, there should be three files, one for each scenario. In each .xlsx file, there should be 5 sheets, one for each fold. In each sheet, there should be a dataframe
    # with 3 rows (LP, SO, RO), and the column names are the capacities, training costs, and testing costs.
    with pd.ExcelWriter(output_file) as writer:
        for k in training_results:
            combined_dict = {}

            for algo in algorithms:
                # Combine training and testing dicts directly (keys already have labels)
                combined = {}
                combined.update(training_results[k][algo])
                combined.update(testing_results[k][algo])

                combined_dict[algo] = combined

            # Create DataFrame and write to Excel
            df = pd.DataFrame.from_dict(combined_dict, orient='index')
            df.index.name = "Algorithm"
            df.to_excel(writer, sheet_name=f"Fold_{k+1}")
