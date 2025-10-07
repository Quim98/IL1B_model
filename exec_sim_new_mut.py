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
from src.simulate_func import set_simulation_fit_new_mut,simulate,func_time,error_new_mut

# Import datasets
df_IC_data = pd.read_csv('data_IL1B/IL1B_dataset.csv')
df_linear_models = pd.read_csv('data_IL1B/linear_models_IL1B.csv')
df_bind = pd.read_csv('data_IL1B/IL1B_data_param_fit.csv')

# Chnage df_bind to add the new mutant to fit
df_bind = df_bind.loc[[5]][df_bind.columns[1:]]
df_bind = df_bind.reset_index(drop=True)
df_bind.loc[1] = df_bind.loc[0]
df_bind.loc[1,"Variant"] = "Mut"
df_bind.loc[1,"k_A_R1_f"] = 3794328.37281239

# Create dataset of simulations
num_sim = 20
num_cores = 16
df_sim = set_simulation_fit_new_mut(df_bind, df_linear_models, df_IC_data, num_sim)
df_sim["R10"] = df_sim["R10"]/1.2 # IL1R1 expression is overestimated for HEK IL1B cells (only data available is from HEK-TE from CCLE)
df_res = simulate(df_sim, num_cores, model_function)
df_res.to_csv("data/simulations_New_mut.csv")

