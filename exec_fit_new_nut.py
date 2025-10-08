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
from src.simulate_func import set_simulation_fit_new_mut,simulate,func_time,error_new_mut,fit_data_new_mut
from multiprocessing import Pool
<<<<<<< HEAD
=======
<<<<<<< HEAD

=======
>>>>>>> 88251ed (New plot, README and requirements files)
>>>>>>> 80a5968 (Figs in .svg)
import time # BORRAR

# Import datasets
df_bind = pd.read_csv('data_IL1B/IL1B_data_param_fit.csv')
df_linear_models = pd.read_csv('data_IL1B/linear_models_IL1B.csv')
df_IC_data = pd.read_csv('data_IL1B/IL1B_dataset.csv')
num_IL_sim = 20
num_cores = 16

# Chnage df_bind to add the new mutant to fit
df_bind = df_bind.loc[[5]][df_bind.columns[1:]]
df_bind = df_bind.reset_index(drop=True)
df_bind.loc[1] = df_bind.loc[0]
df_bind.loc[1,"Variant"] = "Mut"

# Fitting settings
num_iter = 200
exit_cond = 1e-6 # Minimum error between two iterations to exit

print(1)
# Execute fitting
df_sim, res = fit_data_new_mut(df_bind, df_linear_models, df_IC_data, num_IL_sim, num_cores, model_function, num_iter, exit_cond)

# Save fitting results
df_bind_fit = df_bind.copy()
df_bind_fit.loc[1,"k_A_R1_b"] = param_list[0]
df_bind_fit.to_csv('../Data_analysis/data_IL1B/IL1B_data_param_fit_new_mut.csv')
df_fit_data = df_fit_data[["Variant","Parameter"]]
df_fit_data["Final_val"] = res.x
df_bind_fit.to_csv('../Data_analysis/data_IL1B/IL1B_data_fit_result_new_mut.csv')