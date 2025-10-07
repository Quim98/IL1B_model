from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from src.models.IL1B_parallel import model_function
import pandas as pd
from statistics import median
from sys import exit
from scipy.optimize import curve_fit
from multiprocessing import Pool
import matplotlib.pyplot as plt
from src.simulate_func import set_simulation,simulate,func_time,hill_func

# Import datasets
df_IC_data = pd.read_csv('data_IL1B/IL1B_dataset.csv')
df_linear_models = pd.read_csv('data_IL1B/linear_models_IL1B.csv')
df_sim_data = pd.read_csv('data_IL1B/IL1B_data.csv')
df_bind = pd.read_csv('data_IL1B/IL1B_data_param_fit.csv')

# Create dataset of simulations
num_sim = 12
num_heat = 20
num_cores = 20

# Take only WT and only 93:60
df_sim_data = df_sim_data.loc[df_sim_data["Variant"] == "93:60"]

# dataset to save data
df_fin = pd.DataFrame(columns=["IL","R1","dEC50","EC50_WT","EC50_Mut","Amp"])
df_hek = df_IC_data.loc[df_IC_data["Cell"] == "ACH-000049"]
R1_hek = (df_linear_models.loc[df_linear_models["Gene"] == "IL1R1_HUMAN","Intercept"].values[0]+df_linear_models.loc[df_linear_models["Gene"] == "IL1R1_HUMAN","Slope"].values[0]*df_hek.loc[df_hek["Gene"]== "IL1R1_HUMAN","Log2 TPM"].values[0])/1.2

# Vectors for IL and R1
IL_vec = list(10**(np.linspace(np.log10(6e-12/100), np.log10(6e-12), int(num_heat/2)))) + list(10**(np.linspace(np.log10(6e-12), np.log10(6e-12*1000), int(num_heat/2+1))))[1:]
R1_vec = list(np.linspace(np.log10(10**(R1_hek)/100), np.log10(10**(R1_hek)), int(num_heat/2))) + list(np.linspace(np.log10(10**(R1_hek)), np.log10(10**(R1_hek)*100), int(num_heat/2+1)))[1:]

for IL0 in IL_vec:
    for R10 in R1_vec:
        df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[2:]+["IL0","A0","R10","Tf","dT","Result"])
        for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
            df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
            A_vec = 10**(np.linspace(np.log10(df_plot["A"].min()/10), np.log10(df_plot["A"].max()), num_sim))
            df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
            for variant in list(dict.fromkeys(df_plot["Variant"])):
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                for i in range(0,num_sim):
                    A0 = A_vec[i]
                    dT, Tf = func_time(A0)
                    df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[2:] + [IL0,A0,R10,Tf,dT,np.nan]
                    df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-4] + ["Mut", 6e-12] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[2:] + [IL0,A0,R10,Tf,dT,np.nan]
                    df_sim.loc[len(df_sim.index)-1,"k_A_R1_f"] = df_sim.loc[len(df_sim.index)-1,"k_A_R1_f"]*(10)
                    df_sim.loc[len(df_sim.index)-1,"k_A_R1_b"] = df_sim.loc[len(df_sim.index)-1,"k_A_R1_b"]/(10)
        df_sim["R10"] = df_sim["R10"]/1.2 # IL1R1 expression is overestimated for HEK IL1B cells (only data available is from HEK-TE from CCLE)
        # Simulate
        df_res = simulate(df_sim, num_cores, model_function)
        df_isu = df_res.loc[df_res["Variant"] == "93:60"]
        max_isu = df_isu["Result"].max()
        fit_isu, cov = curve_fit(hill_func, df_isu["A0"], df_isu["Result"].values*100/(max_isu), bounds = ([df_isu["A0"].min()/10], [df_isu["A0"].max()*10]))
        df_mut = df_res.loc[df_res["Variant"] == "Mut"]
        fit_mut, cov = curve_fit(hill_func, df_mut["A0"], df_mut["Result"].values*100/(max_isu), bounds = ([df_mut["A0"].min()/10], [df_mut["A0"].max()*10]))     
        df_fin.loc[len(df_fin.index)] = [IL0, R10, abs(np.log10(fit_isu[0])-np.log10(fit_mut[0])), fit_isu[0], fit_mut[0] ,max_isu-df_isu["Result"].min()]
df_fin.to_csv("data/heat_map_isu.csv")