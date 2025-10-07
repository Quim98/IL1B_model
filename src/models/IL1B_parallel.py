from pysb import Model, Parameter, Compartment, Monomer, Parameter, Rule, Initial, Observable
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import cython

def model_function(PA):
    PA=PA[0]
    Na=6.023e23
    Width_PM = 1e-7 # In dm
    # Initialize model
    Model()
    
    # Compartment
    Compartment(name='EC', parent=None, dimension=3, size=None)
    Compartment(name='Cell_PM', parent=EC, dimension=2, size=None)
    
    # Monomers
    Monomer('IL', ['b'])
    Monomer('R1', ['b1'])
    Monomer('A', ['bA']) 

    # Parameters
    Parameter('K_IL_R1_f', PA["k_IL_R1_f"].values[0]/(Na*PA["vol_EC"].values[0]))
    Parameter('K_IL_R1_b', PA["k_IL_R1_b"].values[0])
    Parameter('K_A_R1_f', PA["k_A_R1_f"].values[0]/(Na*PA["vol_EC"].values[0]))
    Parameter('K_A_R1_b', PA["k_A_R1_b"].values[0])
    
    # Complexes and reactions (binding of receptors) 
    Rule('IL_R1_binding', IL(b=None) + R1(b1=None) | IL(b=1) % R1(b1=1), *[K_IL_R1_f, K_IL_R1_b])
    Rule('A_R1_binding', A(bA=None) + R1(b1=None) | A(bA=2) % R1(b1=2), *[K_A_R1_f, K_A_R1_b])
    
    # Initial conditions
    Parameter('IL_0', PA["IL0"].values[0]*Na*PA["vol_EC"])
    Initial(IL(b=None) ** EC, IL_0)
    Parameter('A_0', PA["A0"].values[0]*Na*PA["vol_EC"])
    Initial(A(bA=None) ** EC, A_0)  
    Parameter('R1_0',10**(PA["R10"].values[0]))
    Initial(R1(b1=None) ** Cell_PM, R1_0)

    # Simulation
    Observable('IL_R1_complex', IL(b=1) % R1(b1=1))
    Observable('A_R1_complex', A(bA=2) % R1(b1=2))
    
    t = np.arange(0,PA["Tf"].values[0],PA["dT"].values[0])
    simulator = ScipyOdeSimulator(model, tspan=t, compiler='python', integrator = 'lsoda').run()
    PA["Result"] = simulator.all['IL_R1_complex'][-1]
    return PA

