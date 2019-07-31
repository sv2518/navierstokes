
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

cfl=20#cfl number
order_list=[1,2,3,4]#space dimension
RE=1#reynolds number
N_list=[5,6,7,8,9]#5#fe number (space discretisation)
TMAX=1
XLEN=2*pi
bc_type="dirichlet"
output=False




for order in order_list:
    c=0
    tas_data_rows=[]
    for N in N_list:
        #dx defined over element number & space dimensions
        dx=XLEN/2**N

        # cfl number restrics size of dt
        dt=cfl/order**2*XLEN/(2*2**N)
        print(dt)
        T=TMAX/dt
        t_params=[dt,T]
        
        #initiate the logging
        PETSc.Log.begin()

        #solve problem for timings
        with PETSc.Log.Event("taylorgreen"):
            w,err_u,err_p,_,comm = taylorgreen(dx,order,t_params,RE,False,output)

        tas_data={"order": order,
              "N": N,
              "dx":dx}

        #gather all time information
        time_data={
            "taylorgreen":PETSc.Log.Event("taylorgreen").getPerfInfo()["time"],
            "configuration":PETSc.Log.Event("configuration").getPerfInfo()["time"],
            "spcs":PETSc.Log.Event("spcs").getPerfInfo()["time"],
            "spcs configuration":PETSc.Log.Event("spcs configuration").getPerfInfo()["time"],
            "initial values":PETSc.Log.Event("initial values").getPerfInfo()["time"],
            "build forms":PETSc.Log.Event("build forms").getPerfInfo()["time"],
            "build problems and solvers":PETSc.Log.Event("build problems and solvers").getPerfInfo()["time"],
            "predictor":PETSc.Log.Event("predictor").getPerfInfo()["time"],
            "update":PETSc.Log.Event("update").getPerfInfo()["time"],
            "corrector":PETSc.Log.Event("corrector").getPerfInfo()["time"],
            "time progressing":PETSc.Log.Event("time progressing").getPerfInfo()["time"],
            "picard iteration":PETSc.Log.Event("picard iteration").getPerfInfo()["time"],
            "predictor solve":PETSc.Log.Event("predictor solve").getPerfInfo()["time"],
            "update solve":PETSc.Log.Event("update solve").getPerfInfo()["time"],
            "corrector solve":PETSc.Log.Event("corrector solve").getPerfInfo()["time"],
            "postprocessing":PETSc.Log.Event("postprocessing").getPerfInfo()["time"],
            "dt": dt
        }
        tas_data.update(time_data)

        #seperate times 
        #for predictor
        PETSc.Log.Stage("predictor solve").push()
        snes=PETSc.Log.Event("SNESSolve").getPerfInfo()
        ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
        residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()
        internal_time_data={
            #Scalable Nonlinear Equations Solvers
            "snes_time_pred" : comm.allreduce(snes["time"], op=MPI.SUM) / comm.size,
            #scalable linear equations solvers
            "ksp_time_pred": comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size,
            "pc_setup_time_pred" : comm.allreduce(pcsetup["time"], op=MPI.SUM) / comm.size,
            "pc_apply_time_pred": comm.allreduce(pcapply["time"], op=MPI.SUM) / comm.size,
            "jac_eval_time_pred" :comm.allreduce(jac_eval["time"], op=MPI.SUM) / comm.size,
            "res_eval_time_pred" : comm.allreduce(residual["time"], op=MPI.SUM) / comm.size
        }
        PETSc.Log.Stage("predictor solve").pop()
        tas_data.update(internal_time_data)


        #seperate times 
        #for update
        PETSc.Log.Stage("update solve").push()
        snes=PETSc.Log.Event("SNESSolve").getPerfInfo()
        ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
        residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()
        internal_time_data={
            #Scalable Nonlinear Equations Solvers
            "snes_time_upd" : comm.allreduce(snes["time"], op=MPI.SUM) / comm.size,
            #scalable linear equations solvers
            "ksp_time_upd": comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size,
            "pc_setup_time_upd" : comm.allreduce(pcsetup["time"], op=MPI.SUM) / comm.size,
            "pc_apply_time_upd": comm.allreduce(pcapply["time"], op=MPI.SUM) / comm.size,
            "jac_eval_time_upd" :comm.allreduce(jac_eval["time"], op=MPI.SUM) / comm.size,
            "res_eval_time_upd" : comm.allreduce(residual["time"], op=MPI.SUM) / comm.size
        }
        PETSc.Log.Stage("update solve").pop()
        tas_data.update(internal_time_data)


        #seperate times 
        #for corrector
        PETSc.Log.Stage("corrector solve").push()
        snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
        ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
        residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()
        internal_time_data={
            #Scalable Nonlinear Equations Solvers
            "snes_time_corr" : comm.allreduce(snes["time"], op=MPI.SUM) / comm.size,
            #scalable linear equations solvers
            "ksp_time_corr": comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size,
            "pc_setup_time_corr" : comm.allreduce(pcsetup["time"], op=MPI.SUM) / comm.size,
            "pc_apply_time_corr": comm.allreduce(pcapply["time"], op=MPI.SUM) / comm.size,
            "jac_eval_time_corr" :comm.allreduce(jac_eval["time"], op=MPI.SUM) / comm.size,
            "res_eval_time_corr" : comm.allreduce(residual["time"], op=MPI.SUM) / comm.size
        }
        PETSc.Log.Stage("corrector solve").pop()


        tas_data.update(internal_time_data)

        #gather dofs
        u,p=w.split()
        u_dofs=u.dof_dset.layout_vec.getSize() 
        p_dofs=p.dof_dset.layout_vec.getSize()


        size_data={
            "velo dofs": u_dofs,
            "pres dofs": p_dofs,
            "sum dofs": u_dofs+p_dofs
        }

        tas_data.update(size_data)

        accuracy_data={
                    "LinfPres": err_p[0],
                    "LinfVelo": err_u[0],
                    "L2Pres": err_p[1],
                    "L2Velo": err_u[1],
                    "H1Pres": err_p[2],
                    "HDivVelo": err_u[2],  
        }

        tas_data.update(accuracy_data)

        tas_data_rows.append(tas_data)


    #write out data to .csv
    datafile = pd.DataFrame(tas_data_rows)  
    result="results/timedata_taylorgreen_ORDER%d_CFL%d_RE%d_TMAX%d_XLEN%d_BC%s.csv"%(order,cfl,RE,TMAX,XLEN,bc_type)
    if not os.path.exists(os.path.dirname('results/')):
            os.makedirs(os.path.dirname('results/'))
    datafile.to_csv(result, index=False,mode="w", header=True)