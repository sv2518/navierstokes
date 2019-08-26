import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres
import numpy as np

#[0.0009466337695318103, 0.0001208373843608699, 5.963569863700561e-05]
#[[[0.022744069969931904, 0.006653512482597822, 0.0033215440908135915], [0.07202624215452187, 0.008396837848853936, 0.00716315496114191], [0.09817477042468103, 0.04908738521234052, 0.02454369260617026]], [[0.0025971656304338113, 0.000560527024551765, 8.302800973850576e-05], [0.016358350838374558, 0.0171376522920607, 0.01795696796275882], [0.09817477042468103, 0.04908738521234052, 0.02454369260617026]], [[0.0009466337695318103, 0.0001208373843608699, 5.963569863700561e-05], [0.01729082512947986, 0.01806049256342366, 0.01800318934514998], [0.09817477042468103, 0.04908738521234052, 0.02454369260617026]]]



def convergence_rate(error_list,dof_list):
    conv_rate=[]
    for i,error in enumerate(error_list):
        if i<len(error_list)-1:
            conv_rate.append(np.log10(error_list[i]/error_list[i+1])/np.log10((dof_list[i+1])/(dof_list[i])))

    print(conv_rate)

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices

error_velo=[]
error_pres=[]
list_dx=[]

refin=range(5,10)#fe number (space discretisation)
order_list=[1,2,3]#space dimension
RE=1#reynolds number
cfl=10
XLEN=2*pi
TMAX=0.000000001
output=False

errorList=[]
#for various order
for order in order_list:
    nue=1
    u=1
    error_velo=[]
    error_pres=[]
    list_dx=[]
    #increasing spatial refinement (number of elements)

    for N in refin:
        #dt=min(peclet/(2*nue*(2**N)**2),cfl/(2*u*2**N))
        dx=XLEN/2**N
        dt=TMAX/1#cfl*dx/(2*order**2)#time step restricted by mesh size
        print(dt)
        T=TMAX/dt
        t_params=[dt,T]

        #solve
        w,err_u,err_p,_,comm= taylorgreen(dx,order,t_params,RE,XLEN,False,False,output)
        error_velo.append(err_u[1])
        error_pres.append(err_p[1])
        list_dx.append(dx)
        print(error_velo)
    errorList.append([error_velo,error_pres,list_dx])

#convergence plots
print(errorList)
fig = plt.figure(1)
axis = fig.gca()

for cc in range(0,len(order_list)): 
    axis.loglog(errorList[cc][2],errorList[cc][0],"*",label=cc)
    convergence_rate(errorList[cc][0],errorList[cc][2])
    
value=0.001
xlabel="x"
list_tmp=errorList[cc][2]
axis.loglog(list_tmp,value*np.power(list_tmp,2),'b-',label="$\propto$ ("+str(xlabel)+"$^2$)")
axis.loglog(list_tmp,value*np.power(list_tmp,1),'r-',label="$\propto$ ("+str(xlabel))
axis.set_xlabel(xlabel)
axis.set_ylabel('$Error$')
fig.savefig("verification_velo_periodic.pdf", dpi=150)


    
