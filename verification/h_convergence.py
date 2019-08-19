import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres
import numpy as np

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices

error_velo=[]
error_pres=[]
list_dx=[]

refin=range(6,9)#fe number (space discretisation)
order_list=[1,2,3]#space dimension
RE=1#reynolds number
cfl=10
XLEN=2*pi
TMAX=1
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
        dt=cfl*dx/(2*order**2)#time step restricted by mesh size
        print(dt)
        T=TMAX/dt
        t_params=[dt,T]

        #solve
        w,err_u,err_p,_,comm= taylorgreen(dx,order,t_params,RE,XLEN,"linear_order_scaled",True,output)
        error_velo.append(err_u[0])
        error_pres.append(err_p[0])
        list_dx.append(dx)
        print(error_velo)
    errorList.append([error_velo,error_pres,list_dx])

#convergence plots
print(errorList)
fig = plt.figure(1)
axis = fig.gca()

for cc in range(0,len(order_list)): 
    axis.loglog(errorList[cc][2],errorList[cc][1],"*",label=cc)
    
value=0.1
xlabel="x"
list_tmp=errorList[cc][2]
axis.loglog(list_tmp,value*np.power(list_tmp,2),'b-',label="$\propto$ ("+str(xlabel)+"$^2$)")
axis.loglog(list_tmp,value*np.power(list_tmp,1),'r-',label="$\propto$ ("+str(xlabel))
axis.set_xlabel(xlabel)
axis.set_ylabel('$Error$')
axis.legend()
plt.show()

    
