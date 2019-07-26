
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyop2.profiling import timed_stage
from firedrake.petsc import PETSc

parameters["pyop2_options"]["lazy_evaluation"] = False


cfl_list=[0.1]#cfl number
order=1#space dimension
RE=1#reynolds number
N=6#5#fe number (space discretisation)
TMAX=0.01
XLEN=2*pi
bc_type="dirichlet"
output=False

# cfl number restrics size of dt
dt=0.1/order**2*XLEN/(2*2**N)
print(dt)
T=TMAX/dt#(pi/2)/dt
t_params=[dt,T]

#dx defined over element number & space dimensions
dx=XLEN/2**N


PETSc.Log.begin()

#with timed_stage("taylorgreen"):
with PETSc.Log.Stage("taylorgreen"):
    _,err_u,err_p,_ = taylorgreen(dx,order,t_params,RE,False,output)

PETSc.Log.Stage("TG").push()


tg = PETSc.Log.Event("TG").getPerfInfo()

print(tg)