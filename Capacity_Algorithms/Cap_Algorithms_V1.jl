######## Capacity Optimization Algorithms ########

# Baseline/Control Algorithm

function Cap_Baseline_V1(stepsize, Weather, Schedules; plotting::Bool=false, plot_range::Union{Nothing, Tuple{Int64, Int64}}=nothing)
    ########## Instructions  ##########
    begin
        #=
        Optimize Function takes the following inputs: stepsize(in minutes), weather forecasts(temperature, wind speed, PV production etc.), lighting/plugs/occupancy schedules.
        the target parameter of sensitivity analysis (in this case infiltration parameter added thermal capacitance), and an option to turn on the plotting option.
        The plots here include the stacked timely energy plot and the energy breakdown pie chart. The last input parameter "plot_range" is optional. 
        When it is given as a tuple, it will update the x range of the stacked energy plot. If the plotting option is on, the plots will be saved in 
        the created results folder automatically.
        
        Optimize Function optimizes for the optimal capacity of compliances (lowest in levelized cost over lifetime) in order to satisfy demand of the Berg Structure 
        for the entire optimization horizon. It will also perform an energy conservation test, and it will report error if fails.
        
        Optimize Function returns the optimal capacity sizes of compliances, the Objective Value (total cost over lifetime), and other scalar values
        (such as total energy curtailment etc.) and time series data (such as power flows in the energy network).
        =#
    end
    ########## Data Preparations  ##########  
    begin
        # Directly use the function DataPreparation
        NumTime, TemperatureAmbient, TemperatureAmbientC, PercentLighting, PercentPlug, PercentOccupied, PVGeneration, RadHeatGain, RadCooling, CFM = DataPreparation(Weather, Schedules)
    end  
    ########## Declare model  ##########
    begin
        # Define the model name and solver. In this case, model name is "m"
        # m = Model(Clp.Optimizer)
        # m = Model(Ipopt.Optimizer)
        # For Gurobi (note that sometimes Clp and Gurobi give slightly different results)
        begin
            # Set path to license (for those using Gurobi)
            ENV["GRB_LICENSE_FILE"] = "C:\\Users\\Fred\\.julia\\environments\\v1.7\\gurobi.lic"
            m = Model(Gurobi.Optimizer)
        end
    end
    ######## Decision variables ########
    begin
        @variable(m, PV2H[1:NumTime] >= 0); # [kW] electrical power transfer from PV to home (Berg)

        @variable(m, PV2G[1:NumTime] >= 0); # [kW] electrical power transfer from PV to ground (curtailment)

        @variable(m, PV2B[1:NumTime] >= 0); # [kW] electrical power transfer from PV to battery

        @variable(m, B2H[1:NumTime] >= 0); # [kW] electrical power transfer from battery to home (Berg)

        @variable(m, H2HP[1:NumTime] >= 0); # [kW] electrical power transfer from home (Berg) to heat pump heating mode

        @variable(m, HP2H[1:NumTime] >= 0); # [BTU/hr] heating power transfer from heat pump heating mode to home (Berg)

        @variable(m, H2C[1:NumTime] >= 0); # [kW] electrical power transfer from home (Berg) to heat pump cooling unit

        @variable(m, C2H[1:NumTime] >= 0); # [BTU/hr] cooling power transfer from heat pump cooling mode to home (Berg)

        @variable(m, G2H[1:NumTime] >= 0); # [kW] Loss of Load
        
        @variable(m, HP_OPTION[1:NumTime], Bin) # Heating Mode = 1, Cooling Mode = 0

        @variable(m, HP2PCM_H[1:NumTime] >= 0); # [BTU/hr] heating power transfer from heat pump heating mode to PCM heating storage

        @variable(m, C2PCM_C[1:NumTime] >= 0); # [BTU/hr] cooling power transfer from heat pump cooling mode to PCM cooling storage

        @variable(m, PCM_H2H[1:NumTime] >= 0); # [BTU/hr] heating power transfer from PCM heating storage to home (Berg)

        @variable(m, PCM_C2H[1:NumTime] >= 0); # [BTU/hr] cooling power transfer from PCM cooling storage to home (Berg)

        @variable(m, BatterySize >= 0); # [kWh] Battery Energy Capacity

        @variable(m, PVSize >= 0); # [kW] PV DC Power Capacity

        @variable(m, PCM_H_Size >= 0); # [BTU] PCM Heating Storage Energy Capacity

        @variable(m, PCM_C_Size >= 0); # [BTU] PCM Cooling Storage Energy Capacity

        # @variable(m, HPSize >= 0); # [kW] Heat Pump Power Capacity

        @variable(m, InStorageBattery[1:NumTime] >= 0); # [kWh] Battery Remaining Charge

        @variable(m, InStoragePCM_H[1:NumTime] >= 0); # [BTU] PCM Heating Remaining Charge

        @variable(m, InStoragePCM_C[1:NumTime] >= 0); # [BTU] PCM Cooling Remaining Charge

        @variable(m, TemperatureIndoor[1:NumTime] >= 0); # [°F] Indoor Air Temperature
    end
    ############ Objective Functions #############
    begin
        # Set single objective for minimizing annual total cost

        # Calculate Total Capital(Capacity) Cost [$]
        @expression(m, capital_cost, C_PV * PVSize + 2 * C_B * BatterySize + C_HP * HPSize + C_PCM_H * PCM_H_Size + C_PCM_C * PCM_C_Size + C_IV) # cost at year 10
        
        # Calculate Yearly Fixed Operational Cost [$/YR]
        @expression(m, fixed_OM_cost, C_PV_OP * PVSize + C_B_OP * BatterySize + C_HP_OP * HPSize + C_PCM_H_OP * PCM_H_Size + C_PCM_C_OP * PCM_C_Size)
        
        # Calculate Yearly Short Run Marginal Cost [$/YR]
        @expression(m, short_run_marginal_cost, δt * sum(B2H[t] * C_B_OPV for t=1:NumTime))
        
        # Penalty for curtailment when battery is empty
        @expression(m, penalty, 0.0001*sum(BatterySize - InStorageBattery[t] for t = 1:NumTime-1))

        # Penalty loss of load (outages)
        @expression(m, outage_cost, δt * lossofloadcost * sum(G2H[t] for t = 1:NumTime))

        # Levelized Cost over Lifetime [$/YR]
        @objective(m, Min, capital_cost*CRF + fixed_OM_cost + short_run_marginal_cost + penalty + outage_cost);
        
        # Read more about strange energy curtailment behavior
        begin
            #=
            Currently, the algorithm often chooses energy curtailment over charging battery even when battery has vacancy. This is due to the variable operational cost
            of the battery. If more energy is cycled through the battery, the variable operational cost increases.
            
            When the algorithm is given a perfect forecast for the entire optimization horizon, there will be many timesteps when the algorithm has 100% 
            confidence that additional energy from PV will never be used while guaranteeing demand satisfaction for the future. When the PV produces a lot of additional 
            energy, instead of charging it into the battery for the extra "energy security measures" that the algorithm determined to be useless, it will curtail it so that 
            less energy will be cycled through the battery, which will reduce the over all cost of the optimization horizon.
            
            Realistically, the actual system-operating algorithm will be different than the capacity optimization algorithm. The actual system-operating algorithm will
            be given a limited period of imperfect forecast with uncertainty, therefore, it should always prefer charging additional generated energy into battery with vacancy
            instead of curtailment. But for the current algorithm, there are several tricks to discourage the energy curtailment behavior:
                
                1. Add an arbitrary incentive to have full battery in the objective function.
                2. Reduce or eliminate the variable operational battery cost in the objective function.  
                3. Shorten the length of the perfect forecast horizon so the algorithm must store additional energy into battery for security.
                4. Add additional energy storage methods such as EV or hydrogen (or even other energy loads).
            =#    
        end    
    end
    ############# Expressions ############
    begin
        # DeltaTemp 
        @expression(m, TempDelta[t=1:NumTime], TemperatureAmbient[t] - TemperatureIndoor[t]); # [°F]

        # DeltaTemp 
        @expression(m, TempDelta_C[t=1:NumTime], Weather[t, 9] - (TemperatureIndoor[t] * 1.8 + 32)); # [°C]

        # Electricity usage from lighting 
        @expression(m, E_Lighting[t=1:NumTime], PeakLighting * PercentLighting[t]); # [kW]

        # Electricity usage from plugs
        @expression(m, E_Plugs[t=1:NumTime], PeakPlugLoad * PercentPlug[t]); # [kW]

        # Total electricity usage
        @expression(m, E_total[t=1:NumTime], E_Lighting[t] + E_Plugs[t]); # [kW]

        # Calculate Ventilation
        @expression(m, CFMVen[t=1:NumTime], min(Rp * PercentOccupied[t] * MaxOccupancy + Ra * Area, PercentOccupied[t] * MaxOccupancy * Ventilation)); # [ft^3/min]

        # Heat gain from lighting
        @expression(m, Q_Lighting[t=1:NumTime], PeakLighting * PercentLighting[t] * 3412.14); # [BTU/hr]

        # Heat gain from plugs
        @expression(m, Q_Plugs[t=1:NumTime], PeakPlugLoad * PercentPlug[t] * 3412.14); # [BTU/hr]

        # Heat gain from occupancy
        @expression(m, Q_Occupancy[t=1:NumTime], MaxOccupancy * PercentOccupied[t] * TotalPersonHeat); # [BTU/hr]

        # Heat gain from infiltration
        # 1.08 = specific heat capacity of air at STP:0.24 [BTU/(lb*°F)] * air density at STP:0.075 [lb/ft^3] * 60 [min/hr] 
        @expression(m, Q_Infiltration[t=1:NumTime], 1.08 * TempDelta[t] * CFM[t]); # [BTU/hr]

        # Heat gain from ventilation
        @expression(m, Q_Ventilation[t=1:NumTime], 1.08 * TempDelta[t] * CFMVen[t]); # [BTU/hr]

        # Heat gain from lighting, plugs, occupancy, ventilation, infiltration
        @expression(m, Q_Others[t=1:NumTime], Q_Infiltration[t] + Q_Ventilation[t] + Q_Occupancy[t] + Q_Lighting[t] + Q_Plugs[t]); # [BTU/hr]

        # Heat gain through structural evenlope
        @expression(m, Q_Envelope[t=1:NumTime], UA * TempDelta[t]); # [BTU/hr]

        # Heat gain from solar radiation
        @expression(m, Q_Rad[t=1:NumTime], SHGC * RadHeatGain[t]); # [BTU/hr]

        # Net thermal load (positive for heating, negative for cooling)
        @expression(m, NetThermalLoad[t=1:NumTime], HP2H[t] - C2H[t] + PCM_H2H[t] - PCM_C2H[t]); # [BTU/hr]

        # Detailed COP of heating and cooling (linearized using TemperatureIndoor = 22 [°C])
        # COP (Heating Home)
        @expression(m, COP_Heating_H[t=1:NumTime], HP_a + HP_b * (T_indoor_constant - TemperatureAmbientC[t]) + HP_c * (T_indoor_constant - TemperatureAmbientC[t])^2);
        
        # COP (Heating PCM H)
        @expression(m, COP_Heating_PCM[t=1:NumTime], HP_a + HP_b * (48 - TemperatureAmbientC[t]) + HP_c * (48 -TemperatureAmbientC[t])^2); 

        # COP (Cooling Home)
        @expression(m, COP_Cooling_H[t=1:NumTime], HP_a + HP_b * (TemperatureAmbientC[t] - T_indoor_constant) + HP_c * (TemperatureAmbientC[t] - T_indoor_constant)^2);
        
        # COP (Cooling PCM C)
        @expression(m, COP_Cooling_PCM[t=1:NumTime], HP_a + HP_b * (TemperatureAmbientC[t] - 11) + HP_c * (TemperatureAmbientC[t] - 11)^2); 
    end
    ############# Constraints ############
    begin
        # Set point temperature range constraints
        @constraint(m, [t=1:NumTime], TemperatureIndoor[t] <= SetPointT_High); # [°F]

        @constraint(m, [t=1:NumTime], TemperatureIndoor[t] >= SetPointT_Low); # [°F]

        # Indoor Temperature initialization constraint
        @constraint(m, TemperatureIndoor[1] == 72); # [°F]

        # Internal temperature balance evolution constraint
        @constraint(m, [t=1:NumTime-1], TemperatureIndoor[t+1] == TemperatureIndoor[t] + 
        (δt/TC)*(Q_Others[t] + Q_Envelope[t] + Q_Rad[t] + HP2H[t] - C2H[t] + PCM_H2H[t] - PCM_C2H[t])); # [°F]

        # Battery storage initialization constraint
        @constraint(m, InStorageBattery[1] == 0.5 * BatterySize); # [kWh] half full battery to start with
        
        @constraint(m, InStorageBattery[NumTime] == 0.5 * BatterySize); # [kWh] half full battery to end with
        
        # PV energy balance constraint, node at PV
        @constraint(m, [t=1:NumTime], PVGeneration[t] * PVSize ==  PV2B[t] + PV2H[t] + PV2G[t]); # [kW]
    
        # House electricity load constraint, node at house, battery efficiency modeled
        @constraint(m, [t=1:NumTime], E_total[t] + H2HP[t] + H2C[t] == PV2H[t] * η_PVIV + B2H[t] * η + G2H[t]); # [kW]

        # Battery storage balance constraint, node at battery, battery leakage modeled, battery efficiency modeled
        @constraint(m, [t=1:NumTime-1], InStorageBattery[t+1] == InStorageBattery[t] * δt * (1 - BatteryLoss) +  δt * (PV2B[t] * η - B2H[t])); # [kWh]

        # Battery discharging constraint, node at battery 
        @constraint(m, [t=1:NumTime], δt * B2H[t] <= InStorageBattery[t]); # [kWh]
        
        # Battery power inverter constraint, node at battery (inverter power constraint) 
        @constraint(m, [t=1:NumTime], B2H[t] + PV2B[t] <= InverterSize); # [kW]
        
        # Battery storage size constraint, node at battery
        @constraint(m, [t=1:NumTime], InStorageBattery[t] <= BatterySize); # [kWh]
        
        # Battery storage max discharge constraint, node at battery, always at least 20% full 
        @constraint(m, [t=1:NumTime], InStorageBattery[t] >= BatterySize * (1-MaxDischarge)); # [kWh]
        
        # Detailed COP of heating and cooling (requires nonlinear optimization solver)
        # Heating (from kW to BTU/hr), node at heat pump heating option
        @constraint(m, [t=1:NumTime], H2HP[t] * 3412.14 == HP2PCM_H[t]/COP_Heating_PCM[t] + HP2H[t]/COP_Heating_H[t]);
        
        # Cooling (from kWh to BTU), node at heat pump cooling option
        @constraint(m, [t=1:NumTime], H2C[t] * 3412.14 == C2PCM_C[t]/COP_Cooling_PCM[t] + C2H[t]/COP_Cooling_H[t]);

        # If HP_OPTION = 1, heating mode on, heating only
        @constraint(m, [t=1:NumTime], H2HP[t] <= M * HP_OPTION[t]) # [kW]          
        
        # If HP_OPTION = 0, cooling mode on, cooling only
        @constraint(m, [t=1:NumTime], H2C[t] <= M * (1 - HP_OPTION[t])) # [kW]    

        # Heat pump power capacity constraint, node at heat pump
        @constraint(m, [t=1:NumTime], H2HP[t] + H2C[t] <= HPSize); # [kW]

        # PCM heating storage balance constraint, node at PCM heating storage
        @constraint(m, [t=1:NumTime-1], InStoragePCM_H[t+1] == InStoragePCM_H[t] + δt * (HP2PCM_H[t] - PCM_H2H[t])); # [BTU]
        
        # PCM cooling storage balance constraint, node at PCM cooling storage
        @constraint(m, [t=1:NumTime-1], InStoragePCM_C[t+1] == InStoragePCM_C[t] + δt * (C2PCM_C[t] - PCM_C2H[t])); # [BTU]
        
        # PCM heating storage discharging constraint, node at PCM heating storage
        @constraint(m, [t=1:NumTime], δt * PCM_H2H[t] <= InStoragePCM_H[t]); # [BTU]
        
        # PCM cooling storage discharging constraint, node at PCM cooling storage
        @constraint(m, [t=1:NumTime], δt * PCM_C2H[t] <= InStoragePCM_C[t]); # [BTU]
        
        # PCM heating storage size constraint, node at PCM heating storage
        @constraint(m, [t=1:NumTime], InStoragePCM_H[t] <= PCM_H_Size); # [BTU]

        # PCM cooling storage size constraint, node at PCM cooling storage
        @constraint(m, [t=1:NumTime], InStoragePCM_C[t] <= PCM_C_Size); # [BTU]
        
        # PCM heating storage initialization constraint, node at PCM heating storage
        @constraint(m, InStoragePCM_H[1] == 0); # [BTU]

        # PCM cooling storage initialization constraint, node at PCM cooling storage
        @constraint(m, InStoragePCM_C[1] == 0); # [BTU]
    end
    ########### Solve  ##########
    optimize!(m); 
    ########### Model Results  ##########
    begin
        # Return system sizes and other scalar and time series data

        PV_Size = round(value.(PVSize), digits=2); # [kW]

        Battery_Size = round(value.(BatterySize), digits=2); # [kWh]

        HP_Size = round(value.(HPSize), digits=2); # [kW]

        PCM_Heating_Size = round(value.(PCM_H_Size)/3412.14, digits=2); # [kWh]

        PCM_Cooling_Size = round(value.(PCM_C_Size)/3412.14, digits=2); # [kWh]
        
        ObjValue = objective_value(m) - value.(penalty); # [$] Levelized cost of system over lifetime

        Solar_Production = value.(PVSize)*PVGeneration # [kW] Electrical power produced from PV 

        Curtailment = value.(PV2G); # [kW] Curtailed power (wasted to ground)

        Instorage = value.(InStorageBattery); # [kWh] Battery state (how much energy is in battery)
        
        Instorage_H = value.(InStoragePCM_H)/3412.14; # [kWh] Remaining heating energy in PCM heating storage
        
        Instorage_C = value.(InStoragePCM_C)/3412.14; # [kWh] Remaining cooling energy in PCM cooling storage

        BatteryLeakage = value.(InStorageBattery[t] for t = 1:NumTime-1) * BatteryLoss * δt; # [kWh] Battery leakage
        push!(BatteryLeakage, 0) # No battery leakage at the very last timestep, this is to make the length of the array consistent with the others

        RoundTripLoss = value.((PV2B + B2H) * (1-η) + PV2H * (1-η_PVIV)) # [kW] Electrical power loss from battery roundtrip efficiency and PV inverter efficiency

        totalcharge = sum(value.(PV2B)) * δt # [kWh] Total energy charged from PV to battery

        totaldischarge = sum(value.(B2H)) * δt # [kWh] Total energy discharged from battery to the Berg

        HouseConsumption = value.(PV2H * η_PVIV + B2H * η); # [kW] Total electrical power consumption (thermal + electrical) of the Berg

        total_C = sum(Curtailment[t] for t = 1:NumTime) * δt # [kWh] Total curtailed energy (wasted to ground) over the entire optimization horizon

        total_L = sum(HouseConsumption[t] for t = 1:NumTime) * δt # [kWh] Total energy consumption (thermal + electrical) of the Berg over the entire optimization horizon

        total_LL = sum(value.(G2H)) * δt # [kWh] Total load curtailment of the Berg over the entire optimization horizon

        HeatPumpUsage = value.(H2HP); # [kW] Electrical power consumption used at heat pump for heating 

        ChillerUsage = value.(H2C); # [kW] Electrical power consumption used at heat pump for cooling 

        HeatCharge = value.(HP2PCM_H)/3412.14; # [kW] Heating power charged into PCM heating storage

        CoolCharge = value.(C2PCM_C)/3412.14; # [kW] Cooling power charged into PCM cooling storage

        HeatDischarge = value.(PCM_H2H)/3412.14; # [kW] Heating power discharged from PCM heating storage

        CoolDischarge = value.(PCM_C2H)/3412.14; # [kW] Cooling power discharged from PCM cooling storage
        
        IndoorTemp = value.(TemperatureIndoor) # [°F] Indoor Temperature

        AmbientTemp = value.(TemperatureAmbient) # [°F] Ambient Temperature

        Net_ThermalLoad = value.(NetThermalLoad) # [BTU/hr] Net Thermal Load

        EnergyBreakdown = [total_C, sum(BatteryLeakage), sum(RoundTripLoss) * δt, sum(HeatPumpUsage) * δt, sum(ChillerUsage) * δt, sum(E_Lighting) * δt, sum(E_Plugs) * δt] # [kWh] Total energy consumption of each sector over the entire optimization horizon     
    end
    ########### Optional Plots  ##########
    begin
        TIME = Weather[:,"datetime"];
        if plotting
            # Option to display the stacked energy plot
            begin
                # Determine preferred range for the stacked energy plot
                if plot_range === nothing
                    # Default range for the stacked energy plot is the "TIME" defined at the start of the function
                    plot_range_start: TimeStart
                    plot_range_end = TimeEnd
                else
                    plot_range_start = plot_range[1]
                    plot_range_end = plot_range[2]
                end    
                println()
                println("Plotting: stacked energy plot")
                load = PlotlyJS.scatter(; x=TIME[plot_range_start:plot_range_end],
                    y=HouseConsumption[plot_range_start:plot_range_end] * δt,
                    mode="lines", # one line
                    line_color="black",
                    name="load")
                discharge = PlotlyJS.scatter(; x=TIME[plot_range_start:plot_range_end],
                    y=value.(B2H[plot_range_start:plot_range_end] * η) * δt,
                    mode="none", # no line (just shaded fill)
                    fill="tonexty", # fill to next variable
                    line_color="green",
                    stackgroup=1, # make different stacked groups of variables
                    fillcolor="rgba(1,220,0,1)", # last digit is opacity 0-1
                    name="discharge")
                solar = PlotlyJS.scatter(; x=TIME[plot_range_start:plot_range_end],
                    y=value.(PV2H[plot_range_start:plot_range_end]*η_PVIV) * δt,
                    mode="none", # no line (just shaded fill)
                    fill="tonexty", # fill to next variable
                    line_color="yellow",
                    stackgroup=1, # make different stacked groups of variables
                    fillcolor="rgba(225,220,0,1)", # last digit is opacity 0-1
                    name="solar")
                lossofload = PlotlyJS.scatter(; x=TIME[plot_range_start:plot_range_end],
                    y=value.(G2H[plot_range_start:plot_range_end]) * δt,
                    mode="none", # no line (just shaded fill)
                    fill="tonexty", # fill to next variable
                    line_color="purple",
                    stackgroup=1, # make different stacked groups of variables
                    fillcolor="purple", # last digit is opacity 0-1
                    name="loss of load")
                curtail = PlotlyJS.scatter(; x=TIME[plot_range_start:plot_range_end],
                    y=value.(PV2G[plot_range_start:plot_range_end]) * δt,
                    mode="none", # no line (just shaded fill)
                    fill="tonexty", # fill to next variable
                    line_color="red",
                    stackgroup=1, # make different stacked groups of variables
                    fillcolor="rgba(225,0,0,1)", # last digit is opacity 0-1
                    name="curtailment")
                RTloss = PlotlyJS.scatter(; x=TIME[plot_range_start:plot_range_end],
                    y=value.(RoundTripLoss[plot_range_start:plot_range_end]) * δt,
                    mode="none", # no line (just shaded fill)
                    fill="tonexty", # fill to next variable
                    line_color="skyblue",
                    stackgroup=1, # make different stacked groups of variables
                    fillcolor="rgba(66,239,245,1)", # last digit is opacity 0-1
                    name="round trip loss")
                leak = PlotlyJS.scatter(; x=TIME[plot_range_start:plot_range_end],
                    y=value.(BatteryLeakage[plot_range_start:plot_range_end]),
                    mode="none", # no line (just shaded fill)
                    fill="tonexty", # fill to next variable
                    line_color="orange",
                    stackgroup=1, # make different stacked groups of variables
                    fillcolor="rgba(225,110,0,1)", # last digit is opacity 0-1
                    name="battery leakage")          
                dataseries = [discharge, solar, lossofload, load, leak, RTloss, curtail]
                layout = PlotlyJS.Layout(title="Stacked Energy Plot ($code_name.V$version $SA_parameter = $SA_unit)",
                    #framestyle = "box",
                    showlegend = true,
                    legend = attr(x = 100, y = 1),
                    xaxis_title="Time",
                    yaxis_title="Energy (kWh)",
                    # xaxis_range=[plot_range_start, plot_range_end], 
                    yaxis_range=AUTOMATIC,
                    xaxis_showgrid=true,
                    yaxis_showgrid=true,
                    xaxis_showline=true,
                    xaxis_automargin=true,
                    xaxis_nticks=24,
                    xaxis_tickformat="%H:%M %b %d", # If we add datetime, then use this line to format x-axis
                    yaxis_showline=true,
                    xaxis_mirror="ticks",
                    yaxis_mirror="ticks",
                    xaxis_ticks="inside",
                    xaxis_tickangle=45,
                    yaxis_ticks="inside",
                    xaxis_rangemode = "tozero",
                    yaxis_rangemode = "nonnegative",
                    xaxis_titlefont_family = "Arial, sans-serif",
                    xaxis_titlefont_size = 18,
                    xaxis_titlefont_color = "black",
                    xaxis_tickfont_family = "Arial, sans-serif",
                    xaxis_tickfont_size = 16,
                    xaxis_tickfont_color = "black",
                    yaxis_titlefont_family = "Arial, sans-serif",
                    yaxis_titlefont_size = 18,
                    yaxis_titlefont_color = "black",
                    yaxis_tickfont_family = "Arial, sans-serif",
                    yaxis_tickfont_size = 16,
                    yaxis_tickfont_color = "black",
                    legend_font_family = "Arial, sans-serif",
                    legend_font_size = 18,
                    legend_font_color = "black"
                    )
                SEPplot = PlotlyJS.plot(dataseries,layout) 
                display(SEPplot)
                save_plot(SEPplot, folder_path, "Stacked_Energy_Plot_$SA_parameter=($code_name.V$version)_$today_date.$plot_range_start.to.$plot_range_end", "png")
            end
            # Option to display the energy breakdown pie chart
            begin
                println()
                println("Plotting: energy breakdown pie chart")
                labels = ["Curtailment", "Battery Leakage", "RoundTrip Loss", "Heating", "Cooling", "Lighting", "Plugs"]
                ebp = plot(pie(
                    values=round.(EnergyBreakdown, digits = 2),
                    labels=labels,
                    mode="text",
                    text=round.(EnergyBreakdown, digits = 2),
                    textposition="inside",
                    hoverinfo = "labels + values",
                    title="Energy Breakdown Pie Chart (kWh) ($code_name.V$version $SA_parameter = $SA_unit)"
                    ))
                display(ebp)
                save_plot(ebp, folder_path, "Energy_Breakdown_Pie_Chart_$SA_parameter=($code_name.V$version)_$today_date", "png")
            end   
        end        
    end
    ########### Model Testings  ##########
    begin
        # Energy Conservation Test
        begin
            # The total energy in the system should be conserved across the optimization horizon following the equation below:

            # total energy input + total initial energy given = total energy output + total energy remained
            
            # total energy input = total energy generated by PV panels + load curtailment (imaginary electricity imported from "grid")
            # total initial energy given = initial energy stored in Berg battery storage, PCM heating storage, and PCM cooling storage
            # total energy output = total curtailed energy + total load from Berg + total battery leakage + total energy lost in roundtrip and inverter
            # total energy remained = remaining energy stored in Berg battery storage, PCM heating storage, and PCM cooling storage

            # The system will display the inconsistency value in energy conservation if the energy conservation test failed, and a test passed message otherwise.

            TotalEnergyInput = sum(PVGeneration[:]*value.(PVSize)) * δt + total_LL # [kWh]
            TotalInitialEnergy = Instorage[TimeStart] + Instorage_C[TimeStart] + Instorage_H[TimeStart] # [kWh]
            TotalEnergyOutput =  total_C + total_L + sum(Instorage[1:TimeEnd-1]) * BatteryLoss * δt + (totalcharge + totaldischarge)*(1-η) + sum(value.(PV2H[:])*(1-η_PVIV)) * δt # [kWh]
            TotalEnergyRemained = Instorage[TimeEnd] + Instorage_C[TimeEnd] + Instorage_H[TimeEnd] # [kWh]
            
            conservation = TotalEnergyInput + TotalInitialEnergy - TotalEnergyOutput - TotalEnergyRemained 
            if conservation > -0.5 && conservation < 0.5
                println()
                println("Energy conservation test passed.")    
            else
                println()
                println("Energy conservation test failed, there is an additional $conservation kWh energy found in the system")
            end
        end
        # Testing if there's strange Heat Pump behavior such as simutaneous heating and cooling
        begin
            #=
            Normally, there should not be simutaneous heating and cooling energy consumption from HP at any given moment, therefore, 
            min(HeatPumpUsage[i], ChillerUsage[i]) for i = TimeStart:TimeEnd should always be zero.
            We plot this value across the entire optimization horizon, and if the line stays at zero, then it is working well.

            If there is no simutaneous heating and cooling, then no plot will be displayed; instead, the following message will appear:
            "HP test passed, no simutaneous heating and cooling."
            =#
            testline = zeros(TimeEnd)
            for i = TimeStart:TimeEnd
                testline[i] = min(HeatPumpUsage[i], ChillerUsage[i])
            end    
            s_test = scatter(
                x = Weather[TimeStart:TimeEnd, "datetime"],  
                y = testline,
                name="test (kWh)"
            )
            if sum(testline[:]) > 0
                p_test = plot([s_test], Layout(title="HP Energy Usage Test ($code_name.V$version)", xaxis_title="Time", yaxis_title="Energy(kWh)"))
                display(p_test)
            else
                println()
                println("HP test passed, no simutaneous heating and cooling.")    
            end        
        end    
    end    
    return PV_Size, Battery_Size, HP_Size, PCM_Heating_Size, PCM_Cooling_Size, round(ObjValue, digits=2), round(total_C, digits=2), round(total_L, digits=2), round(total_LL, digits=2), Solar_Production, Curtailment, value.(PV2H), value.(PV2B), value.(B2H), value.(G2H),
        Instorage, Instorage_H, Instorage_C, HouseConsumption, HeatPumpUsage, HeatCharge, HeatDischarge, ChillerUsage, CoolCharge, CoolDischarge, BatteryLeakage, RoundTripLoss, 
        IndoorTemp, AmbientTemp, Net_ThermalLoad
end
