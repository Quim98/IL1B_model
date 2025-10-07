from pysb import Model, Parameter, Compartment, Monomer, Parameter, Rule, Initial, Observable
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import cython

def model_function(PA, df_out):
    Na=6.023e23
    Width_PM = 1e-7 # In dm
    # Initialize model
    Model()
    
    # Compartment
    Compartment(name='EC', parent=None, dimension=3, size=None)
    Compartment(name='Cell_PM', parent=EC, dimension=2, size=None)
    Compartment(name='Cell_CP', parent=Cell_PM, dimension=3, size=None)
    
    # Monomers
    Monomer('IL', ['bA', 'bB', 'bG','B_t'], {'B_t': ['B', 'U']})
    Monomer('RA', ['bA'])
    Monomer('RB', ['bB', 'bSTAT5', 'bSTAT1_3']) 
    Monomer('RG', ['bG'])
    #Monomer('STAT1', ['b1','P'], {'P': ['u', 'p']})
    #Monomer('STAT3', ['b3','P'], {'P': ['u', 'p']})
    Monomer('STAT5', ['b5','P'], {'P': ['u', 'p']})
    
    # Parameters
    
    Parameter('K_IL_RA_f', PA["k_IL_RA_f"][0]/(Na*PA["vol_EC"][0]))
    Parameter('K_IL_RA_b', PA["k_IL_RA_b"][0])
    Parameter('K_IL_RB_f', PA["k_IL_RB_f"][0]/(Na*PA["vol_EC"][0]))
    Parameter('K_IL_RB_b', PA["k_IL_RB_b"][0])
    Parameter('K_IL_RG_f', PA["k_IL_RG_f"][0]/(Na*PA["vol_EC"][0]))
    Parameter('K_IL_RG_b', PA["k_IL_RG_b"][0])
    
    Parameter('K_IL_RA2_f', PA["k_IL_RA_f"][0]/(Na*PA["surf_cell"][0]*Width_PM))
    Parameter('K_IL_RA2_b', PA["k_IL_RA_b"][0])
    Parameter('K_IL_RB2_f', PA["k_IL_RB_f"][0]/(Na*PA["surf_cell"][0]*Width_PM))
    Parameter('K_IL_RB2_b', PA["k_IL_RB_b"][0])
    Parameter('K_IL_RG2_f', PA["k_IL_RG_f"][0]/(Na*PA["surf_cell"][0]*Width_PM))
    Parameter('K_IL_RG2_b', PA["k_IL_RG_b"][0])
    
    Parameter('K_IL_RA_RB_f', PA["k_IL_RA_RB_f"][0]/(Na*PA["surf_cell"][0]*Width_PM))
    Parameter('K_IL_RA_RB_b', PA["k_IL_RA_RB_b"][0])
    Parameter('K_IL_RB_RG_f', PA["k_IL_RB_RG_f"][0]/(Na*PA["surf_cell"][0]*Width_PM))
    Parameter('K_IL_RB_RG_b', PA["k_IL_RB_RG_b"][0])
    Parameter('K_IL_RA_RB_RG_f', PA["k_IL_RA_RB_RG_f"][0]/(Na*PA["surf_cell"][0]*Width_PM))
    Parameter('K_IL_RA_RB_RG_b', PA["k_IL_RA_RB_RG_b"][0])
    
    
    #Parameter('K_STAT1_f', PA["k_STAT1_f"]/(Na*PA["vol_cell"]))
    #Parameter('K_STAT1_b', PA["k_STAT1_b"])
    #Parameter('K_STAT3_f', PA["k_STAT3_f"]/(Na*PA["vol_cell"]))
    #Parameter('K_STAT3_b', PA["k_STAT3_b"])
    Parameter('K_STAT5_f', PA["k_STAT5_f"][0]/(Na*PA["vol_cell"][0]))
    Parameter('K_STAT5_b', PA["k_STAT5_b"][0])
    
    Parameter('K_PHOS', PA["k_PHOS"][0])   
    Parameter('K_DEPHOS', PA["k_DEPHOS"][0])
    Parameter('K_DEPHOS_R', PA["k_DEPHOS_R"][0])
    
    # Complexes and reactions (binding of receptors) 
    IL_free = IL(bA=None, bB=None, bG=None, B_t='U')
    IL_RA = IL(bA=1, bB=None, bG=None, B_t='B') % RA(bA=1)
    Rule('IL_RA_binding', IL_free + RA(bA=None)| IL_RA, *[K_IL_RA_f, K_IL_RA_b])
    IL_RB = IL(bA=None, bB=2, bG=None, B_t='B') % RA(bA=2)
    Rule('IL_RB_binding', IL_free + RB(bB=None)| IL_RB, *[K_IL_RB_f, K_IL_RB_b])
    IL_RG = IL(bA=None, bB=None, bG=3, B_t='B') % RG(bG=3)
    Rule('IL_RG_binding', IL_free + RG(bG=None)| IL_RG, *[K_IL_RG_f, K_IL_RG_b])
    
    IL_RA_RB = IL(bA=1, bB=2, bG=None, B_t='B') % RA(bA=1) % RB(bB=2)
    Rule('IL_RA_RB_binding', IL_RA + RB(bA=None)| IL_RA_RB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
    IL_RA_RG = IL(bA=1, bB=None, bG=3, B_t='B') % RA(bA=1) % RG(bG=3)
    Rule('IL_RA_RG_binding', IL_RA + RG(bA=None)| IL_RA_RG, *[K_IL_RG2_f, K_IL_RG2_b])
    Rule('IL_RA_RB_binding_2', IL_RB + RA(bA=None)| IL_RA_RB, *[K_IL_RA2_f, K_IL_RA2_b])
    IL_RB_RG = IL(bA=None, bB=2, bG=3, B_t='B') % RB(bB=2) % RG(bG=3)
    Rule('IL_RB_RG_binding', IL_RB + RG(bG=None)| IL_RB_RG, *[K_IL_RB_RG_f, K_IL_RB_RG_b])
    Rule('IL_RA_RG_binding_2', IL_RG + RA(bA=None)| IL_RA_RG, *[K_IL_RA2_f, K_IL_RA2_b])
    Rule('IL_RB_RG_binding_2', IL_RG + RB(bB=None)| IL_RB_RG, *[K_IL_RB2_f, K_IL_RB2_b])
    
    IL_RA_RB_RG = IL(bA=1, bB=2, bG=3, B_t='B') % RA(bA=1) % RB(bB=2) % RG(bG=3)
    Rule('IL_RA_RB_RG_binding_1', IL_RA_RB + RG(bG=None)| IL_RA_RB_RG, *[K_IL_RA_RB_RG_f, K_IL_RA_RB_RG_b])
    Rule('IL_RA_RB_RG_binding_2', IL_RA_RG + RB(bB=None)| IL_RA_RB_RG, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
    Rule('IL_RA_RB_RG_binding_3', IL_RB_RG + RA(bB=None)| IL_RA_RB_RG, *[K_IL_RA2_f, K_IL_RA2_b])
    
    
    # Complexes and reactions (binding of STAT)
    IL_RB_RG_noSTAT5 = IL(bB=2, bG=3, B_t='B') % RB(bB=2, bSTAT5=None) % RG(bG=3)
    IL_RB_RG_STAT5 = IL(bB=2, bG=3, B_t='B') % RB(bB=2, bSTAT5=4) % RG(bG=3) % STAT5(b5=4, P='u')
    Rule('IL_RB_RG_STAT5_binding', IL_RB_RG_noSTAT5 + STAT5(b5=None, P='u') | IL_RB_RG_STAT5, *[K_STAT5_f, K_STAT5_b])
    #RB_noSTAT1_3 = RB(bSTAT1_3=None)
    #RB_STAT1 = RB(bSTAT1_3=5) % STAT1(b1=5, P='u')
    #Rule('RB_STAT1_binding', RB_noSTAT1_3 + STAT1(b1=None, P='u') | RB_STAT1, *[K_STAT1_f, K_STAT1_b])
    #RB_STAT3 = RB(bSTAT1_3=6) % STAT3(b1=6, P='u')
    #Rule('RB_STAT3_binding', RB_noSTAT1_3 + STAT3(b3=None, P='u') | RB_STAT3, *[K_STAT3_f, K_STAT3_b])
    
    
    # Reactions (STAT phosphorylation)
    Rule('IL_RB_RG_STAT5_phosphorylation', IL_RB_RG_STAT5 >> IL_RB_RG_noSTAT5 + STAT5(b5=None, P='p'), K_PHOS)
    #IL_RB_RG_STAT1 = IL(bB=2, bG=3, B_t='B') % RB(bB=2, bSTAT1_3=5) % RG(bG=3) % STAT1(b1=5, P='u')
    #IL_RB_RG_STAT3 = IL(bB=2, bG=3, B_t='B') % RB(bB=2, bSTAT1_3=6) % RG(bG=3) % STAT3(b1=6, P='u')
    #IL_RB_RG_noSTAT1_3 = IL(bB=2, bG=3, B_t='B') % RB(bB=2, bSTAT1_3=None) % RG(bG=3)
    #Rule('IL_RB_RG_STAT1_phosphorylation', IL_RB_RG_STAT1 >> IL_RB_RG_noSTAT1_3 + STAT1(b1=None, P='p'), K_PHOS)
    #Rule('IL_RB_RG_STAT3_phosphorylation', IL_RB_RG_STAT3 >> IL_RB_RG_noSTAT1_3 + STAT3(b3=None, P='p'), K_PHOS)
    
    
    # Reactions (STAT dephosphorylation)
    #Rule('STAT1_dephosphorylation', STAT1(b1=None, P='p') >> STAT1(b1=None, P='u'), K_DEPHOS)
    #Rule('STAT3_dephosphorylation', STAT3(b3=None, P='p') >> STAT3(b3=None, P='u'), K_DEPHOS)
    Rule('STAT5_dephosphorylation', STAT5(b5=None, P='p') >> STAT5(b5=None, P='u'), K_DEPHOS)
    
    
    # Initial conditions
    Parameter('IL_0', PA["IL0"][0])
    Initial(IL(bA=None, bB=None, bG=None, B_t='U') ** EC, IL_0)
    
    Parameter('RA_0',PA["RA0"][0])
    Parameter('RB_0',PA["RB0"][0])
    Parameter('RG_0',PA["RG0"][0])
    Initial(RA(bA=None) ** Cell_PM, RA_0)
    Initial(RB(bB=None, bSTAT3=None, bSTAT1_5=None) ** Cell_PM, RB_0)
    Initial(RG(bG=None) ** Cell_PM, RG_0)
    
    #Parameter('STAT_1_0', IC["STAT10"])
    #Parameter('STAT_3_0', IC["STAT30"])
    Parameter('STAT_5_0', PA["STAT50"][0])
    #Initial(STAT1(b1=None, P='u') ** Cell_CP, STAT_1_0)
    #Initial(STAT3(b3=None, P='u') ** Cell_CP, STAT_3_0)
    Initial(STAT5(b5=None, P='u') ** Cell_CP, STAT_5_0)
    
    # Simulation
    #Observable('STAT1P', STAT1(b1=None, P='p') ** Cell_CP)
    #Observable('STAT3P', STAT3(b3=None, P='p') ** Cell_CP)
    Observable('STAT5P', STAT5(b5=None, P='p') ** Cell_CP)
    
    t = np.arange(0,PA["Tf"][0],PA["dT"][0])
    simulator = ScipyOdeSimulator(model, tspan=t, compiler='python', integrator = 'lsoda').run()
    
    if df_out["STAT_type"][0] == "pSTAT1":
        return simulator.all['STAT1P'][-1]
    elif df_out["STAT_type"][0] == "pSTAT3":
        return simulator.all['STAT3P'][-1]
    elif df_out["STAT_type"][0] == "pSTAT5":
        return simulator.all['STAT5P'][-1]

