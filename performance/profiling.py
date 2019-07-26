
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
from mpi4py import MPI


parameters["pyop2_options"]["lazy_evaluation"] = False

cfl_list=[0.1]#cfl number
order=1#space dimension
RE=1#reynolds number
N=6#5#fe number (space discretisation)
TMAX=0.1
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

#initiate the logging
PETSc.Log.begin()

with PETSc.Log.Event("taylorgreen"):
    _,err_u,err_p,_,comm = taylorgreen(dx,order,t_params,RE,False,output)

#gather all time information
time_data={
    "taylorgreen":PETSc.Log.Event("taylorgreen").getPerfInfo()["time"],
    "configuration":PETSc.Log.Event("configuration").getPerfInfo()["time"],
    "spcs":PETSc.Log.Event("spcs").getPerfInfo()["time"],
    "spcs configuration":PETSc.Log.Event("spcs configuration").getPerfInfo()["time"],
    "initial values":PETSc.Log.Event("initial values").getPerfInfo()["time"],
    "build forms":PETSc.Log.Event("initial values").getPerfInfo()["time"],
    "build problems and solvers":PETSc.Log.Event("build problems and solvers").getPerfInfo()["time"],
    "predictor":PETSc.Log.Event("predictor").getPerfInfo()["time"],
    "update":PETSc.Log.Event("update").getPerfInfo()["time"],
    "corrector":PETSc.Log.Event("corrector").getPerfInfo(),
    "time progressing":PETSc.Log.Event("time progressing").getPerfInfo()["time"],
    "picard iteration":PETSc.Log.Event("picard iteration").getPerfInfo()["time"],
    "predictor solve":PETSc.Log.Event("predictor solve").getPerfInfo()["time"],
    "update solve":PETSc.Log.Event("update solve").getPerfInfo()["time"],
    "corrector solve":PETSc.Log.Event("corrector solve").getPerfInfo()["time"],
    "postprocessing":PETSc.Log.Event("postprocessing").getPerfInfo()["time"]
}

#write out data to .csv
datafile= pd.DataFrame(time_data, index=[0])
result="results/timedata_taylorgreen_CFL%d_RE%d_TMAX%d_XLEN%d_N%d_BC%s.csv"%(cfl_list[0],RE,TMAX,XLEN,N,bc_type)
if not os.path.exists(os.path.dirname('results/')):
        os.makedirs(os.path.dirname('results/'))
datafile.to_csv(result, index=False,mode="w", header=True)