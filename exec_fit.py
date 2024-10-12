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
from src.simulate_func import set_simulation_fit,simulate,func_time,calculate_RSS,update_df_bind,fit_data
from multiprocessing import Pool

import time # BORRAR

# Import datasets
df_bind = pd.read_csv('data_IL1B/IL1B_data_param.csv')
df_sim_data = pd.read_csv('data_IL1B/IL1B_data.csv')
df_EC50_AMP = pd.read_csv('data_IL1B/IL1B_data_hill.csv')
df_linear_models = pd.read_csv('data_IL1B/linear_models_IL1B.csv')
df_IC_data = pd.read_csv('data_IL1B/IL1B_dataset.csv')
df_fit_data = pd.read_csv('data_IL1B/IL1B_data_fit.csv') # Parameters to fit
num_IL_sim = 12
num_cores = 16

# Parameters data
param_mut = {"WT": df_bind.columns[1:],"39:113":["k_A_R1_b"],"54:98":["k_A_R1_b"],"74:79":["k_A_R1_b"],"87:66":["k_A_R1_b"],"93:60":["k_A_R1_b"]}
param_depen = {}

# Fitting settings
num_iter = 200
exit_cond = 1e-6 # Minimum error between two iterations to exit

# Execute fitting
df_sim, df_EC50_AMP, res = fit_data(df_bind, df_sim_data, df_EC50_AMP, df_linear_models, df_IC_data, df_fit_data, num_IL_sim, num_cores, model_function, param_mut, param_depen, num_iter, exit_cond)

# Save fitting results
df_bind_fit = update_df_bind(df_bind, res.x, param_mut, df_fit_data, param_depen)
df_bind_fit.to_csv('../Data_analysis/data_IL1B/IL1B_data_param_fit.csv')
df_fit_data = df_fit_data[["Variant","Parameter"]]
df_fit_data["Final_val"] = res.x
df_bind_fit.to_csv('../Data_analysis/data_IL1B/IL1B_data_fit_result.csv')