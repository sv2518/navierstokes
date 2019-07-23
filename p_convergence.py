from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres
import numpy as np
import matplotlib.pyplot as plt

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices


list_D=[]

refin=range(1,4)#space dimension
RE=1#reynolds number
N=6#5#fe number (space discretisation)
#cflList=[[8,4,8/3,2],[4,2,4/3,1],[2,1,2/3,0.5],[1,0.5,1/3,0.25]]
cflList=[[0.2,0.2/4,0.2/9,0.2/16,0.2/25]]


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


#increasing spatial refinement (number of elements)[]
errorList=[]
for cfl in cflList:
    print(cfl)
    c=0
    error_velo=[]
    error_pres=[]
    for D in refin:
        print(D)

        # cfl number restrics size of dt
        dt=cfl[c]*pi/(2*2**N)
        print(dt)
        T=0.01/dt#(pi/2)/dt
        t=[dt,T]
    
        #solve
        w,err_u,err_p,dx = taylorgreen(pi/2**N, D,t,RE,periodic=False)
        u,p=w.split()
        error_velo.append(err_u)
        error_pres.append(err_p)
        list_D.append(D)
        print(error_velo)
        print(error_pres)
        c+=1
    errorList.append([error_velo,error_pres,list_D])


#convergence plots
print(error_velo)
print(error_pres)
print(list_D)
print("\n\n")
print(errorList)

for cfl in cflList:
    plot_convergence_velo_pres(cfl[0],cfl[1],cfl[2],0.0001)

matplotlib.pyplot.show()

    

