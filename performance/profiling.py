
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


#get internal solver specific times of predictor,update,corrector
#state can be cold or warm
def get_internal_timedata(state):
    internal_timedata={}

    #PREDICTOR
    PETSc.Log.Stage("predictor solve").push()
    snes=PETSc.Log.Event("SNESSolve").getPerfInfo()["time"]#general solve time(preconditioner application, linear solvers, assembly)
    ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()["time"]#solely Krylov solve
    pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()["time"]
    pcapply = PETSc.Log.Event("PCApply").getPerfInfo()["time"]#solve
    jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()["time"]#assembly
    residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()["time"]
    PETSc.Log.Stage("predictor solve").pop()

    internal_timedata.update({
            #Scalable Nonlinear Equations Solvers
            "snes_time_pred"+state : comm.allreduce(snes, op=MPI.SUM) / comm.size,
            #scalable linear equations solvers
            "ksp_time_pred"+state: comm.allreduce(ksp, op=MPI.SUM) / comm.size,
            "pc_setup_time_pred"+state : comm.allreduce(pcsetup, op=MPI.SUM) / comm.size,
            "pc_apply_time_pred"+state: comm.allreduce(pcapply, op=MPI.SUM) / comm.size,
            "jac_eval_time_pred"+state:comm.allreduce(jac_eval, op=MPI.SUM) / comm.size,
            "res_eval_time_pred"+state: comm.allreduce(residual, op=MPI.SUM) / comm.size
    })

    #UPDATE
    PETSc.Log.Stage("update solve").push()
    snes=PETSc.Log.Event("SNESSolve").getPerfInfo()["time"]
    ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()["time"]
    pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()["time"]
    pcapply = PETSc.Log.Event("PCApply").getPerfInfo()["time"]#solving time
    jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()["time"]
    residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()["time"]
    elim= PETSc.Log.Event("SCForwardElim").getPerfInfo()["time"]#elimination time (rhs)
    trace = PETSc.Log.Event("SCSolve").getPerfInfo()["time"]#trace solve time
    full_recon = PETSc.Log.Event("SCBackSub").getPerfInfo()["time"]#backsub time
    hybridassembly= PETSc.Log.Event("HybridOperatorAssembly").getPerfInfo()["time"]#assembly time (inside init)    
    hybridinit = PETSc.Log.Event("HybridInit").getPerfInfo()["time"]
    hybridupdate = PETSc.Log.Event("HybridUpdate").getPerfInfo()["time"]
    PETSc.Log.Stage("update solve").pop()

    internal_timedata.update({
            #Scalable Nonlinear Equations Solvers
            "snes_time_upd"+state: comm.allreduce(snes, op=MPI.SUM) / comm.size,
            #scalable linear equations solvers
            "ksp_time_upd"+state: comm.allreduce(ksp, op=MPI.SUM) / comm.size,
            "pc_setup_time_upd"+state: comm.allreduce(pcsetup, op=MPI.SUM) / comm.size,
            "pc_apply_time_upd"+state: comm.allreduce(pcapply, op=MPI.SUM) / comm.size,
            "jac_eval_time_upd"+state:comm.allreduce(jac_eval, op=MPI.SUM) / comm.size,
            "res_eval_time_upd"+state: comm.allreduce(residual, op=MPI.SUM) / comm.size,
            "HDGInit"+state: comm.allreduce(hybridinit, op=MPI.SUM) / comm.size,
            "HDGAssembly"+state: comm.allreduce(hybridassembly, op=MPI.SUM) / comm.size,
            "HDGUpdate"+state: comm.allreduce(hybridupdate, op=MPI.SUM) / comm.size,
            "HDGRhs"+state: comm.allreduce(elim, op=MPI.SUM) / comm.size,
            "HDGRecover"+state: comm.allreduce(full_recon, op=MPI.SUM) / comm.size,
            "HDGTraceSolve"+state: comm.allreduce(trace, op=MPI.SUM) / comm.size,
            "HDGTotal"+state: hybridinit+hybridupdate+ elim+full_recon+trace        
            })
    
    #CORRECTOR
    PETSc.Log.Stage("corrector solve").push()
    snes = PETSc.Log.Event("SNESSolve").getPerfInfo()["time"]
    ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()["time"]
    pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()["time"]
    pcapply = PETSc.Log.Event("PCApply").getPerfInfo()["time"]
    jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()["time"]
    residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()["time"]
    PETSc.Log.Stage("corrector solve").pop()
    
    internal_timedata.update({
            #Scalable Nonlinear Equations Solvers
            "snes_time_corr"+state: comm.allreduce(snes, op=MPI.SUM) / comm.size,
            #scalable linear equations solvers
            "ksp_time_corr"+state: comm.allreduce(ksp, op=MPI.SUM) / comm.size,
            "pc_setup_time_corr"+state: comm.allreduce(pcsetup, op=MPI.SUM) / comm.size,
            "pc_apply_time_corr"+state: comm.allreduce(pcapply, op=MPI.SUM) / comm.size,
            "jac_eval_time_corr"+state:comm.allreduce(jac_eval, op=MPI.SUM) / comm.size,
            "res_eval_time_corr"+state: comm.allreduce(residual, op=MPI.SUM) / comm.size
        })
    
    return internal_timedata

#general time spend on different parts of the whole run
def get_external_timedata():
    #gather all time information
    time_data={
            "warm up":PETSc.Log.Event("warm up").getPerfInfo()["time"],
            "second solve":PETSc.Log.Event("second solve").getPerfInfo()["time"],
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
    return time_data

######################################
##############   MAIN   ##############
######################################

parameters["pyop2_options"]["lazy_evaluation"] = False

cfl=10#cfl number
order_list=[1,2,3,4,5,6,7,8]#space dimension
RE=1#reynolds number
#N_list=[9]#,6]#,7,8,9]#5#fe number (space discretisation)
TMAX=1
XLEN=2*pi
bc_type="periodic"
output=False
splitstates=False
dofcount_list=[25000,50000,75000,100000,150000,200000,300000,400000,600000,800000]


dofpercell=0
for order in order_list:
    c=0
    tas_data_rows=[]
    for dofcount in dofcount_list:
        #dx defined over element number & space dimensions
        print(dofpercell)
        dx=math.sqrt(XLEN**2*(order**2+0+2*order+dofpercell)/dofcount)#/2**N
        PETSc.Sys.Print("!!!!dx is:",dx) 

        #cfl number restrics size of dt for stability
        #scaled by order**2 (some people say it should be to the power of 1, some 1.5)
        #chose two bc better too small than too large
        #divided by two bc 2d
        dt=cfl/order**2*dx/2
        PETSc.Sys.Print(dt)

        #number of timesteps
        T=TMAX/dt
        t_params=[dt,T]
        
        #initiate the logging
        PETSc.Log.begin()
        tas_data={}

########get internal time data of solvers
        if splitstates:
        ###warm up solver
            with PETSc.Log.Event("warm up"):
                w,err_u,err_p,_,comm = taylorgreen(dx,order,t_params,RE,XLEN,None,False,output)
                internal_timedata_cold=get_internal_timedata("cold")
                temp_internal_timedata_cold=get_internal_timedata("warm")#temp needed for subtraction 

            tas_data.update(internal_timedata_cold)

        ###get timings for solving without assembly
            with PETSc.Log.Event("second solve"):
                w,err_u,err_p,_,comm = taylorgreen(dx,order,t_params,RE,XLEN,None,False,output)
                temp_internal_timedata_warm=get_internal_timedata("warm")

            internal_timedata_warm={key: temp_internal_timedata_warm[key] - temp_internal_timedata_cold.get(key, 0) for key in temp_internal_timedata_warm.keys()}
            tas_data.update(internal_timedata_warm)

        else:
        ###get timings for solving one run
            with PETSc.Log.Event("taylorgreen"):
                w,err_u,err_p,_,comm = taylorgreen(dx,order,t_params,RE,XLEN,None,False,output)
                internal_timedata=get_internal_timedata("")

            tas_data.update(internal_timedata)

########add general times spend on different parts
        external_timedata=get_external_timedata()
        tas_data.update(external_timedata)

########add further information
        #spatial setup information
        tas_data.update({"order": order,
              "dx":dx})
              #   "N": N,

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

        #gather errrors
        accuracy_data={
                    "LinfPres": err_p[0],
                    "LinfVelo": err_u[0],
                    "L2Pres": err_p[1],
                    "L2Velo": err_u[1],
                    "H1Pres": err_p[2],
                    "HDivVelo": err_u[2],  
        }

        tas_data.update(accuracy_data)

########update rows
#        tas_data_rows.append(tas_data)


        #write out data to .csv
        #datafile = pd.DataFrame(tas_data_rows) 
        datafile = pd.DataFrame(tas_data,index=[0])   
        result="results/timedata_taylorgreen_ORDER%d_CFL%d_RE%d_TMAX%d_XLEN%d_BC%s_DOFS%d.csv"%(order,cfl,RE,TMAX,XLEN,bc_type,dofcount)
        if not os.path.exists(os.path.dirname('results/')):
                os.makedirs(os.path.dirname('results/'))
        datafile.to_csv(result, index=False,mode="w", header=True)

    dofpercell+=(order)*4###TAKE CARE, this is based on the fact that my orderlist goes from 1 to 4
       