from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import curve_fit
from multiprocessing import Pool

import time # BORRAR

def func_time(IL):
    # Function that decides until what time to simulate to get to the steady state
    IL = -abs(np.log10(IL)) + 16
    if IL > 1e-11:
        IL = 5
    return 10**(abs(IL))/(1e5),10**(abs(IL))

def simulate(df_sim_data, num_cores, model_func):
    # Funtion that splits a simulation dataset and parallelizes each simualtion data row 
    param_parallel = []
    for i in range(0,len(df_sim_data.index)):
        param_parallel.append([df_sim_data.loc[[i]]])
    with Pool(processes=num_cores) as pool:
        out_vec = pool.map(model_func, param_parallel)  
    return pd.concat(out_vec, axis=0)

def set_simulation(df_bind, df_sim_data, df_IC_data, df_linear_models, num_sim):
    # Function that generates t
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","A0","R10","Tf","dT","Result"])
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        A_vec = 10**(np.linspace(np.log10(df_plot["A"].min()), np.log10(df_plot["A"].max()), num_sim))
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            df_var = df_plot.loc[df_plot["Variant"] == variant]
            for i in range(0,num_sim):
                A0 = A_vec[i]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log10 Prot. count"].values[0]):
                    R10 = df_linear_models.loc[df_linear_models["Gene"]=="IL1R1_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL1R1_HUMAN"]["Slope"].values[0]
                else:
                    R10 = df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log10 Prot. count"].values[0]
                dT, Tf = func_time(A0)
                df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [df_plot["IL"].values[0],A0,R10,Tf,dT,np.nan]    
    return df_sim

def hill_func(inp,K):
    out=100/(1+inp/K)
    return out

def calculate_RSS(df_sim, df_EC50_AMP, w_EC50):
    for plot_num in list(dict.fromkeys(df_sim["Plot"])):
        df_plot = df_sim.loc[df_sim["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        i = 0
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            if i == 0:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                max_WT = df_var["Result"].max() 
                fit, cov = curve_fit(hill_func, df_var["A0"], df_var["Result"].values*100/(max_WT), bounds = ([df_var["A0"].min()/10], [df_var["A0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[0])
            else:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                fit, cov = curve_fit(hill_func, df_var["A0"], df_var["Result"].values*100/(max_WT), bounds = ([df_var["A0"].min()/10], [df_var["A0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[0])
            i += 1
    df_EC50_AMP["EC50_e"] = (df_EC50_AMP["EC50_t"] - df_EC50_AMP["EC50_m"])**2/w_EC50
    df_EC50_AMP["Error"] = df_EC50_AMP["EC50_e"]
    
    print("Mean EC50 error: "+str((df_EC50_AMP["EC50_e"].mean()*w_EC50)**(1/2))+" RSS: "+str(df_EC50_AMP["Error"].sum()))
    
    return df_EC50_AMP["EC50_e"].sum()

def update_df_bind(df_bind, param_list, param_mut, df_fit_data, param_depen):
    # Create copy of binding dataframe
    df_bind_fit = df_bind.copy()

    # Update df_bind with new parameter values
    i=0
    for variant in list(dict.fromkeys(df_fit_data["Variant"])):
        if variant == "WT":
            for param in df_fit_data.loc[df_fit_data["Variant"] == variant]["Parameter"]:
                # Copy new parameter to dataframe
                df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param] = param_list[i]
                # Parameter dependancies
                if param in param_depen.keys():
                    for param_d in param_depen[param]:
                        param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                        param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                        df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param_d] = param_list[i]/param1_0*param2_0
                # Parameter which are the same for WT and othe variants
                for variant_WT_mut in list(param_mut.keys())[1:]:
                    if param not in param_mut[variant_WT_mut]:
                        df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).Variant, :].index[0], param] = param_list[i]
                        # Parameter dependancies
                        if param in param_depen.keys():
                            for param_d in param_depen[param]:
                                param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                                param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                                df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param_d] = param_list[i]/param1_0*param2_0
                i += 1
        else:
            for param in df_fit_data.loc[df_fit_data["Variant"] == variant]["Parameter"]:
                df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param] = param_list[i]
                # Parameter dependancies
                if param in param_depen.keys():
                    for param_d in param_depen[param]:
                        param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                        param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                        df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param_d] = param_list[i]/param1_0*param2_0
                i += 1
    return df_bind_fit

def set_simulation_fit(df_bind, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim):
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","A0","R10","Tf","dT","Result"])
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            df_var = df_plot.loc[df_plot["Variant"] == variant]
            df_var_EC50 = df_plot_EC50.loc[df_plot_EC50["Variant"] == variant]
            A_vec =  10**(np.linspace(df_var_EC50["EC50_t"].values[0]-3, df_var_EC50["EC50_t"].values[0]+4, num_IL_sim))
            for i in range(0,num_IL_sim):
                A0 = A_vec[i]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log10 Prot. count"].values[0]):
                    R10 = df_linear_models.loc[df_linear_models["Gene"]=="IL1R1_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL1R1_HUMAN"]["Slope"].values[0]
                else:
                    R10 = df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log10 Prot. count"].values[0]
                dT, Tf = func_time(A0)            
                df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [df_var["IL"].values[0],A0,R10,Tf,dT,np.nan]
    return df_sim

def fit_data(df_bind, df_sim_data, df_EC50_AMP, df_linear_models, df_IC_data, df_fit_data, num_IL_sim, num_cores, model_function, param_mut, param_depen, num_iter, exit_cond):
     
    def error_func(param_list):
        # Update parameters dataframe
        df_bind_fit = update_df_bind(df_bind, param_list, param_mut, df_fit_data, param_depen)
        
        # Create dataset of simulations
        df_sim = set_simulation_fit(df_bind_fit, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim)
        
        df_sim["R10"]=df_sim["R10"]/1.2 # IL1R1 expression is overestimated for HEK IL1B cells (only data available is from HEK-TE from CCLE)
        
        # Simulate
        df_res = simulate(df_sim, num_cores, model_function)
        
        # Calculate error
        error = calculate_RSS(df_res, df_EC50_AMP, w_EC50)
        
        print(param_list)
        return error
    
    # Create first simulation dataframe
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","A0","R10","Tf","dT","Result"])
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        A_vec = 10**(np.linspace(np.log10(df_plot["A"].min()), np.log10(df_plot["A"].max()), num_IL_sim))
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            df_var = df_plot.loc[df_plot["Variant"] == variant]
            for i in range(0,num_IL_sim):
                A0 = A_vec[i]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log10 Prot. count"].values[0]):
                    R10 = df_linear_models.loc[df_linear_models["Gene"]=="IL1R1_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL1R1_HUMAN"]["Slope"].values[0]
                else:
                    R10 = df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log10 Prot. count"].values[0]
                dT, Tf = func_time(A0)
                df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [df_var["IL"].values[0],A0,R10,Tf,dT,np.nan]
                
    # Perform simulations with raw parameters
    df_sim = simulate(df_sim, num_cores, model_function)
    
    # Calculate EC50 and Amplitude error to add weight
    for plot_num in list(dict.fromkeys(df_sim["Plot"])):
        df_plot = df_sim.loc[df_sim["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        i = 0
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            if i == 0:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                max_WT = df_var["Result"].max()
                fit, cov = curve_fit(hill_func, df_var["A0"], df_var["Result"].values*100/(max_WT), bounds = ([df_var["A0"].min()/10], [df_var["A0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[0])
            else:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                fit, cov = curve_fit(hill_func, df_var["A0"], df_var["Result"].values*100/(max_WT), bounds = ([df_var["A0"].min()/10], [df_var["A0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[0])
            i += 1
    df_EC50_AMP["EC50_e"] = (df_EC50_AMP["EC50_t"] - df_EC50_AMP["EC50_m"])**2
    w_EC50 = df_EC50_AMP["EC50_e"].mean()
    print("EC50 mean error: " + str(w_EC50**(1/2)))
    
    
    # Initial values and bounds for the parameters to fit
    list_initial = []
    list_bounds = []
    for variant in list(dict.fromkeys(df_fit_data["Variant"])):
        df_fit_variant = df_fit_data.loc[df_fit_data["Variant"] == variant]
        for param in df_fit_variant["Parameter"]:
            list_initial.append(df_fit_variant.loc[df_fit_variant["Parameter"] == param]["Initial_val"].values[0])
            list_bounds.append((df_fit_variant.loc[df_fit_variant["Parameter"] == param]["Lower_bound"].values[0],df_fit_variant.loc[df_fit_variant["Parameter"] == param]["Upper_bound"].values[0]))
            
    res = optimize.minimize(
        error_func,
        list_initial,
        bounds=list_bounds,
        method="Nelder-Mead",
        options={ 'maxiter':num_iter, 'fatol':exit_cond}
    )
    print("Finalized fitting of parameters")
    
    return df_sim, df_EC50_AMP, res
    
def set_simulation_fit_new_mut(df_bind, df_linear_models, df_IC_data, num_IL_sim):
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","A0","R10","Tf","dT","Result"])
    plot_num = 0
    for cell in ["ACH-000049","ACH-000552"]:
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==cell]
        for variant in ["93:60","Mut"]:
            A_vec =  10**(np.linspace(-13, -7, num_IL_sim))
            for i in range(0,num_IL_sim):
                A0 = A_vec[i]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log10 Prot. count"].values[0]):
                    R10 = df_linear_models.loc[df_linear_models["Gene"]=="IL1R1_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL1R1_HUMAN"]["Slope"].values[0]
                else:
                    R10 = df_cell.loc[df_cell["Gene"]=="IL1R1_HUMAN"]["Log10 Prot. count"].values[0]
                dT, Tf = func_time(A0)            
                df_sim.loc[len(df_sim.index)] = [plot_num,cell,variant,"NO_STAT"] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [6e-12,A0,R10,Tf,dT,np.nan]
        plot_num += 1
    return df_sim
    
def hill_func_new_mut(inp,K,amp):
    out=amp/(1+inp/K)
    return out

def error_new_mut(df_sim):
    df_HEK = df_sim.loc[df_sim["Cell_type"]=="ACH-000049"]
    df_HT29 = df_sim.loc[df_sim["Cell_type"]=="ACH-000552"]
    
    df_var = df_HEK.loc[df_HEK["Variant"] == "93:60"]
    max_WT = df_var["Result"].max()
    fit, cov = curve_fit(hill_func_new_mut, df_var["A0"], df_var["Result"].values*100/(max_WT), bounds = ([df_var["A0"].min()/100,max_WT-17], [df_var["A0"].max()*100,max_WT+17]))
    Isun_HEK_IC50 = fit[0]

    df_var = df_HEK.loc[df_HEK["Variant"] == "Mut"]
    max_WT = df_var["Result"].max()
    fit, cov = curve_fit(hill_func_new_mut, df_var["A0"], df_var["Result"].values*100/(max_WT), bounds = ([df_var["A0"].min()/100,max_WT-17], [df_var["A0"].max()*100,max_WT+17]))
    Mut_HEK_IC50 = fit[0]

    error_HEK = Isun_HEK_IC50/Mut_HEK_IC50 - 6.5
    print("Error HEK: "+str(error_HEK))

    df_var = df_HT29.loc[df_HT29["Variant"] == "93:60"]
    max_WT = df_var["Result"].max()
    fit, cov = curve_fit(hill_func_new_mut, df_var["A0"], df_var["Result"].values*100/(max_WT), bounds = ([df_var["A0"].min()/100,max_WT-17], [df_var["A0"].max()*100,max_WT+17]))
    Isun_HT29_IC50 = fit[0]

    df_var = df_HT29.loc[df_HT29["Variant"] == "Mut"]
    max_WT = df_var["Result"].max()
    fit, cov = curve_fit(hill_func_new_mut, df_var["A0"], df_var["Result"].values*100/(max_WT), bounds = ([df_var["A0"].min()/100,max_WT-17], [df_var["A0"].max()*100,max_WT+17]))
    Mut_HT29_IC50 = fit[0]

    error_HT29 = Isun_HT29_IC50/Mut_HT29_IC50 - 22
    print("Error HT29: "+str(error_HT29))

    return abs(error_HEK)+abs(error_HT29)

def fit_data_new_mut(df_bind, df_linear_models, df_IC_data, num_IL_sim, num_cores, model_function, num_iter, exit_cond):
     
    def error_func(param_list):
        # Update parameters dataframe
        df_bind_fit = df_bind.copy()
        df_bind_fit.loc[1,"k_A_R1_f"] = param_list[0]
        print(param_list)
        
        # Create dataset of simulations
        df_sim = set_simulation_fit_new_mut(df_bind_fit, df_linear_models, df_IC_data, num_IL_sim)
        df_sim["R10"]=df_sim["R10"]/1.2 # IL1R1 expression is overestimated for HEK IL1B cells (only data available is from HEK-TE from CCLE)
        # Simulate
        df_res = simulate(df_sim, num_cores, model_function)
        # Calculate error
        error = error_new_mut(df_res)
        
        
        return error
    
    # Initial values and bounds for the parameters to fit
    list_initial = [df_bind.loc[0,"k_A_R1_f"]*3]
    list_bounds = [(df_bind.loc[0,"k_A_R1_f"]/10,df_bind.loc[0,"k_A_R1_f"]*1000)]
            
    res = optimize.minimize(
        error_func,
        list_initial,
        bounds=list_bounds,
        method="Nelder-Mead",
        options={ 'maxiter':num_iter, 'fatol':exit_cond}
    )
    print("Finalized fitting of parameters")
    
    return df_sim, res            
            
    
        
    
