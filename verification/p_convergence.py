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


def p_convergence(cfl_list,order_list,RE,TMAX,XLEN,N,bc_type,output,name):
    #various erros for velocity and pressure
    linf_error_pres_list=[]
    l2_error_pres_list=[]
    h1_error_pres_list=[]
    linf_error_velo_list=[]
    l2_error_velo_list=[]
    hdiv_error_velo_list=[]

    #dx defined over element number & space dimensions
    dx=XLEN/2**N

    for cfl in cfl_list:
        print("cfl number: ",cfl)

        linf_error_velo=[]
        linf_error_pres=[]
        l2_error_velo=[]
        l2_error_pres=[]
        hdiv_error_velo=[]
        h1_error_pres=[]
        dt_list=[]

        for D in order_list:
            print("order is: ",D)

            # cfl number restrics size of dt
            dt=cfl/D**2*XLEN/(2*2**N)
            print(dt)
            T=TMAX/dt#(pi/2)/dt
            t_params=[dt,T]
        
            #solve
            if bc_type=="periodic":
                _,err_u,err_p,_,_ = taylorgreen(dx,D,t_params,RE,True,output)
            elif bc_type=="dirichlet":
                _,err_u,err_p,_,_ = taylorgreen(dx,D,t_params,RE,False,output)

            #update list
            linf_error_velo.append(err_u[0])
            linf_error_pres.append(err_p[0])
            l2_error_velo.append(err_u[1])
            l2_error_pres.append(err_p[1])
            hdiv_error_velo.append(err_u[2])
            h1_error_pres.append(err_p[2])
            dt_list.append(dt)

        #save for all cfl for convergence plots
        linf_error_velo_list.append(linf_error_velo)
        l2_error_velo_list.append(l2_error_velo)
        hdiv_error_velo_list.append(hdiv_error_velo)
        linf_error_pres_list.append(linf_error_pres)
        l2_error_pres_list.append(l2_error_pres)
        h1_error_pres_list.append(h1_error_pres)


        #outputting
        print("Linf error pressure:",linf_error_pres_list)
        print("Linf error velocity:",linf_error_velo_list)
        print("L2 error pressure:",l2_error_pres_list)
        print("L2 error velocity:",l2_error_velo_list)
        print("H1 error pressure:",h1_error_pres_list)
        print("Hdiv error velocity:",hdiv_error_velo_list)

    
        data={
            "CFL":cfl,
            "Order":order_list,
            "dt": dt_list,
            "LinfPres": linf_error_pres,
            "LinfVelo": linf_error_velo,
            "L2Pres": l2_error_pres,
            "L2Velo": l2_error_velo,
            "H1Pres": h1_error_pres,
            "HDivVelo": hdiv_error_velo,    
        }

        #write convergence rates to csv file
        data_file=pd.DataFrame(data)
        result="verification/results/taylorgreen_%s_CFL%d_RE%d_TMAX%d_XLEN%d_N%d_BC%s.csv"%(name,cfl,RE,TMAX,XLEN,N,bc_type)
        data_file.to_csv(result, index=False, mode="w")
