from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices

error_velo=[]
error_pres=[]
list_dx=[]

refin=range(6,9)#fe number (space discretisation)
DList=[1,2]#space dimension
RE=1#reynolds number


errorList=[[[0.0006498782703349346, 0.000327180498933672, 0.0001637269956571083], [0.11517428595013965, 0.2714662788522166, 0.31766490281901133], [0.04908738521234052, 0.02454369260617026, 0.01227184630308513]], [[1.7979317189789623e-05, 4.499860648786436e-06, 1.1244696398876597e-06], [0.0020302807268155476, 0.0009733351551158287, 0.00044698987432173487], [0.04908738521234052, 0.02454369260617026, 0.01227184630308513]]]
#convergence plots
print(errorList)
for cc in range(0,len(DList)): 
    plot_convergence_velo_pres(errorList[cc][0],errorList[cc][1],errorList[cc][2],0.01,"p="+str(DList[cc]))
plt.show()

errorList=[]
#for various order
for D in DList:
    cfl=0.1/D**2
    nue=1
    u=1
    error_velo=[]
    error_pres=[]
    list_dx=[]
    #increasing spatial refinement (number of elements)
    for N in refin:
        #dt=min(peclet/(2*nue*(2**N)**2),cfl/(2*u*2**N))
        dt=cfl*pi/(2*u*2**N)#time step restricted by mesh size
        print(dt)
        T=0.0015/dt
        t=[dt,T]

        #solve
        w,err_u,err_p,dx = taylorgreen(pi/2**N, D,t,RE)
        error_velo.append(err_u)
        error_pres.append(err_p)
        list_dx.append(dx)
        print(error_velo)
    errorList.append([error_velo,error_pres,list_dx])

#convergence plots
print(errorList)
for cc in range(0,len(DList)): 
    plot_convergence_velo_pres(errorList[cc][0],errorList[cc][1],errorList[cc][2],1,"dim")
plt.show()

    
