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


def run_p_convergence(cfl_list,order_list,RE,TMAX,XLEN,N,bc_type,output):
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
                _,err_u,err_p,_ = taylorgreen(dx,D,t_params,RE,periodic=True,output)
            elif bc_type=="dirichlet":
                _,err_u,err_p,_ = taylorgreen(dx,D,t_params,RE,False,output)

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
        result="results/taylorgreen_CFL%d_RE%d_TMAX%d_XLEN%d_N%d_BC%s.csv"%(cfl,RE,TMAX,XLEN,N,bc_type)
        data_file.to_csv(result, index=False, mode="w")

    #plot convergence rate
    value=1#constant for scaling the reference orders
    xlabel="p"
    fig_velo = plt.figure(1)
    fig_pres= plt.figure(2)
    axis_velo = fig_velo.gca()
    axis_pres= fig_pres.gca()
    axis_velo.semilogy(order_list,value*np.power(order_list[::-1],2),'b-',label="$\propto$ ("+str(xlabel)+"$^2$)")
    axis_pres.semilogy(order_list,value*np.power(order_list[::-1],1),'b-',label="$\propto$ ("+str(xlabel)+"$^2)$")
    axis_velo.set_xlabel(xlabel)
    axis_velo.set_ylabel('$Error$')
    axis_pres.set_xlabel(xlabel)
    axis_pres.set_ylabel('$Error$')
    
    for cc in range(0,len(cfl_list)):
        label="cfl= "+str(cfl_list[cc])
        axis_velo.semilogy(order_list,linf_error_velo_list[cc],"*-",label=label)
        axis_pres.semilogy(order_list,linf_error_pres_list[cc],"*-",label=label)

    axis_velo.legend()
    axis_pres.legend()
    plt.show()
    plt.savefig("convergence.pdf", dpi=150)





#####################MAIN#############################
#plottin spatial convergence for taylor green vortices


cfl_list=[20,10,8,4,2,1,0.1]#cfl number
order_list=range(1,5)#space dimension
RE=1#reynolds number
N=6#5#fe number (space discretisation)
TMAX=1
XLEN=2*pi
bc_type="dirichlet"
output=False

###various re
tmp0=[0.004199928234239094, 0.0006203590189461528, 0.00027357106048593594, 0.00014612394787934246]#1.5,1.5/4..
tmp=[0.12506863431237716, 0.0015602447017334254, 0.0003541209172361233, 0.0002269698665969197]
plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.0001,"cfl=1.5,N=5,T=1s")
tmp0=[0.0018910704375055933, 0.00020123617502074143, 8.327579420473914e-05, 2.2441518410518287e-05]#n=6, cfl=2
plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.0001,"cfl=2,N=6,T=1s")
tmp0=[0.010373937727772505, 0.00168355950512189, 0.0006003826762952247, 0.0002101329615306069]#n=5,cfl=3
plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.0001,"cfl=3,N=5,T=1s")
tmp0=[0.00668929374320992, 0.00337689363074189, 0.0004213895415813296, 0.0003333310949201508]#n=5, cfl=4
plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.0001,"cfl=4,N=5,T=1s")
tmp0=[0.006913096228617827, 0.0007296063034354094, 0.00023802938457741403, 8.561598912988459e-05]
plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.0001,"cfl=2,N=5,T=1s")
plt.show()


#right  cfl
tmp0=[0.004575733232865732, 6.511989489434156e-05, 2.6288564754922694e-05, 1.4757693163211516e-05]
plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.0001,"cfl=0.2,N=6,T=0.01s")
plt.show()

run_p_convergence(cfl_list,order_list,RE,TMAX,XLEN,N,bc_type,output)


#############################################################################
