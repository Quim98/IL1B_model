from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import curve_fit
from multiprocessing import Pool

def func_time(IL):
    if IL > 1e-8:
        return 0.1,1e4
    else:
        IL = abs(np.log10(IL)) - 4
        return 10**(IL)/(1e5),10**(IL)

def simulate(df_sim_data, num_cores, model_func):
    param_parallel = []
    for i in range(0,len(df_sim_data.index)):
        param_parallel.append([df_sim_data.loc[[i]]])
    with Pool(processes=num_cores) as pool:
        out_vec = pool.map(model_func, param_parallel)  
    return pd.concat(out_vec, axis=0)

def set_simulation(df_bind, df_sim_data, df_IC_data, df_linear_models, num_sim):
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        IL_vec = np.exp(np.linspace(np.log(df_plot["IL"].min()), np.log(df_plot["IL"].max()), num_sim))
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            df_var = df_plot.loc[df_plot["Variant"] == variant]
            for i in range(0,num_sim):
                IL0 = IL_vec[i]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log10 Prot. count"].values[0]):
                    RA0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RA_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RA_HUMAN"]["Slope"].values[0]
                else:
                    RA0 = df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log10 Prot. count"].values[0]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log10 Prot. count"].values[0]):
                    RB0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RB_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RB_HUMAN"]["Slope"].values[0]
                else:
                    RB0 = df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log10 Prot. count"].values[0]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log10 Prot. count"].values[0]):
                    RG0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RG_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RG_HUMAN"]["Slope"].values[0]
                else:
                    RG0 = df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log10 Prot. count"].values[0]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log10 Prot. count"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log10 Prot. count"].values[0]):
                    STAT50 = np.log10(10**(df_linear_models.loc[df_linear_models["Gene"]=="STA5A_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="STA5A_HUMAN"]["Slope"].values[0]) + 10**(df_linear_models.loc[df_linear_models["Gene"]=="STA5B_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="STA5B_HUMAN"]["Slope"].values[0]))
                else:
                    STAT50 = np.log10(10**(df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log10 Prot. count"].values[0]) + 10**(df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log10 Prot. count"].values[0]))
                dT, Tf = func_time(IL0)
                df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
    return df_sim

def hill_func(inp,B,K):
    out=B*inp/(K+inp)
    return out

def sensitivity_analysis(df_bind, df_EC50_data, df_IC_data, df_linear_models, num_IL_sim, num_var_sim, vec_param, cell_sim):
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Var. parameter","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    param_vec = list(np.exp(np.linspace(np.log(0.001), np.log(1), int(num_var_sim/2)+1))) + list(np.exp(np.linspace(np.log(1), np.log(1000), int(num_var_sim/2)+1)))[1:]
    plot_num = 0
    for parameter in vec_param:
        for cell in cell_sim:
            df_cell = df_IC_data.loc[df_IC_data["Cell"]==cell]
            # Higher IL2 concentrations since model overestiates EC50
            IL_vec = np.exp(np.linspace(np.log(df_EC50_data.loc[df_EC50_data["Cell_type"] == cell]["EC50_t"].values[0]/1e3), np.log(df_EC50_data.loc[df_EC50_data["Cell_type"] == cell]["EC50_t"].values[0]*1e5), num_IL_sim))
            for j in range(0,num_var_sim):
                df_WT = df_bind.loc[df_bind["Variant"] == "WT"]
                df_WT[parameter] = df_WT[parameter]*param_vec[j]
                for i in range(0,num_IL_sim):
                    IL0 = IL_vec[i]
                    if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log10 Prot. count"].values[0]):
                            RA0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RA_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RA_HUMAN"]["Slope"].values[0]
                    else:
                        RA0 = df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log10 Prot. count"].values[0]
                    if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log10 Prot. count"].values[0]):
                        RB0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RB_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RB_HUMAN"]["Slope"].values[0]
                    else:
                        RB0 = df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log10 Prot. count"].values[0]
                    if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log10 Prot. count"].values[0]):
                        RG0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RG_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RG_HUMAN"]["Slope"].values[0]
                    else:
                        RG0 = df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log10 Prot. count"].values[0]
                    if np.isnan(df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log10 Prot. count"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log10 Prot. count"].values[0]):
                        STAT50 = np.log10(10**(df_linear_models.loc[df_linear_models["Gene"]=="STA5A_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="STA5A_HUMAN"]["Slope"].values[0]) + 10**(df_linear_models.loc[df_linear_models["Gene"]=="STA5B_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="STA5B_HUMAN"]["Slope"].values[0]))
                    else:
                        STAT50 = np.log10(10**(df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log10 Prot. count"].values[0]) + 10**(df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log10 Prot. count"].values[0]))
                    dT, Tf = func_time(IL0)
                    df_sim.loc[len(df_sim.index)] = [plot_num, cell, parameter, "pSTAT5"] + df_WT.values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
                plot_num += 1
    return df_sim

def profile_likelihood():
    return 0

def calculate_RSS(df_sim, df_EC50_AMP, w_EC50, w_AMP):
    for plot_num in list(dict.fromkeys(df_sim["Plot"])):
        df_plot = df_sim.loc[df_sim["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        i = 0
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            if i == 0:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                max_WT = df_var["Result"].max()
                #fit, cov = curve_fit(hill_func, df_var["IL0"], np.multiply(df_var["Result"].values,100/(max_WT)), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max(), df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = 100 #fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(1e-10) #np.log10(fit[1])
            else:
                #fit, cov = curve_fit(hill_func, df_var["IL0"], np.multiply(df_var["Result"].values,100/(max_WT)), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max(), df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = 120 #fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(1e-8) #np.log10(fit[1])
            i += 1
    df_EC50_AMP["EC50_e"] = (df_EC50_AMP["EC50_t"] - df_EC50_AMP["EC50_m"])**2/w_EC50
    df_EC50_AMP["Amp_e"] = (df_EC50_AMP["Amp_t"] - df_EC50_AMP["Amp_m"])**2/w_AMP
    df_EC50_AMP["Error"] = df_EC50_AMP["EC50_e"] + df_EC50_AMP["Amp_e"]
    return df_EC50_AMP["EC50_e"].mean(), df_EC50_AMP["Amp_e"].mean(), df_EC50_AMP["Error"].sum()
    

def fit_data(df_bind, df_sim_data, df_EC50_AMP, df_linear_model, df_IC_data, num_IL_sim, num_cores,):
    # Create first simulation dataframe
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            df_var = df_plot.loc[df_plot["Variant"] == variant]
            df_var_EC50 = df_plot_EC50.loc[df_plot_EC50["Variant"] == variant]
            IL_vec = np.exp(np.linspace(np.log(df_var_EC50["EC50_t"].values[0]/1e3), np.log(df_var_EC50["EC50_t"].values[0]*1e5), num_IL_sim))
            for i in range(0,num_IL_sim):
                IL0 = IL_vec[i]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log10 Prot. count"].values[0]):
                    RA0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RA_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RA_HUMAN"]["Slope"].values[0]
                else:
                    RA0 = df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log10 Prot. count"].values[0]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log10 Prot. count"].values[0]):
                    RB0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RB_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RB_HUMAN"]["Slope"].values[0]
                else:
                    RB0 = df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log10 Prot. count"].values[0]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log10 Prot. count"].values[0]):
                    RG0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RG_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RG_HUMAN"]["Slope"].values[0]
                else:
                    RG0 = df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log10 Prot. count"].values[0]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log10 Prot. count"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log10 Prot. count"].values[0]):
                    STAT50 = np.log10(10**(df_linear_models.loc[df_linear_models["Gene"]=="STA5A_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="STA5A_HUMAN"]["Slope"].values[0]) + 10**(df_linear_models.loc[df_linear_models["Gene"]=="STA5B_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="STA5B_HUMAN"]["Slope"].values[0]))
                else:
                    STAT50 = np.log10(10**(df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log10 Prot. count"].values[0]) + 10**(df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log10 Prot. count"].values[0]))
                dT, Tf = func_time(IL0)
                df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
                
    # Perform simulations with raw parameters
    df_sim = simulate(df_sim_data, num_cores, model_func)
    
    # Calculate EC50 and Amplitude error to add weight
    for plot_num in list(dict.fromkeys(df_sim["Plot"])):
        df_plot = df_sim.loc[df_sim["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        i = 0
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            if i == 0:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                max_WT = df_var["Result"].max()
                #fit, cov = curve_fit(hill_func, df_var["IL0"], np.multiply(df_var["Result"].values,100/(max_WT)), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max(), df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = 100 #fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(1e-10) #np.log10(fit[1])
            else:
                #fit, cov = curve_fit(hill_func, df_var["IL0"], np.multiply(df_var["Result"].values,100/(max_WT)), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max(), df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = 120 #fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(1e-8) #np.log10(fit[1])
            i += 1
    df_EC50_AMP["EC50_e"] = (df_EC50_AMP["EC50_t"] - df_EC50_AMP["EC50_m"])**2
    df_EC50_AMP["Amp_e"] = (df_EC50_AMP["Amp_t"] - df_EC50_AMP["Amp_m"])**2
    w_EC50 = df_EC50_AMP["EC50_e"].mean()
    w_AMP = df_EC50_AMP["Amp_e"].mean()
    print("EC50 mean error: " + str(w_EC50**(1/2)))
    print("Amplitude mean error: " + str(w_AMP**(1/2)))
    
    # Fitting of parameters
    
    

    def error_func(params_list):
        return 0
    
    return 1
    
            
            
            
    
        
    
