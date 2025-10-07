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
from src.simulate_func import set_simulation,simulate,func_time

# Import datasets
df_IC_data = pd.read_csv('data_IL1B/IL1B_dataset.csv')
df_linear_models = pd.read_csv('data_IL1B/linear_models_IL1B.csv')
df_sim_data = pd.read_csv('data_IL1B/IL1B_data.csv')
df_bind = pd.read_csv('data_IL1B/IL1B_data_param_fit.csv')

# Create dataset of simulations
num_sim = 20
df_sim = set_simulation(df_bind, df_sim_data, df_IC_data, df_linear_models, num_sim)
df_sim["R10"] = df_sim["R10"]/1.2 # IL1R1 expression is overestimated for HEK IL1B cells (only data available is from HEK-TE from CCLE)

# Simulate
num_cores = 10
df_res = simulate(df_sim, num_cores, model_function)
df_res.to_csv("data/simulations.csv")