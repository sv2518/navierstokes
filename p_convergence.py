from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres
import numpy as np
import matplotlib.pyplot as plt

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices


list_D=[]

refin=range(1,5)#space dimension
RE=1#reynolds number
N=5#5#fe number (space discretisation)
#cflList=[[8,4,8/3,2],[4,2,4/3,1],[2,1,2/3,0.5],[1,0.5,1/3,0.25]]
cflList=[[4,4/4,4/9,4/16],[3,3/4,3/9,3/16],[2,2/4,2/9,2/16,2/25]]



tmp0=[3.87989176e-03, 3.30565673e-04, 1.66330960e-04, 4.48729506e-05]#1,1/4,..
tmp=[0.12506863431237716, 0.0015602447017334254, 0.0003541209172361233, 0.0002269698665969197]

#plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.0001,"cfl=1,N=5,T=1s")
#tmp0=[0.0033105083627187485, 8.571531463291183e-05,5.840376149353517e-05,0]#0.5,..
tmp0=[0.004199928234239094, 0.0006203590189461528, 0.00027357106048593594, 0.00014612394787934246]#1.5,1.5/4..
tmp=[0.12506863431237716, 0.0015602447017334254, 0.0003541209172361233, 0.0002269698665969197]
#plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.0001,"cfl=1.5,N=5,T=1s")
tmp0=[0.0018910704375055933, 0.00020123617502074143, 8.327579420473914e-05, 2.2441518410518287e-05]#n=6
#plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.0001,"cfl=2,N=6,T=1s")
#plt.show()
####0.0018910704375055933, 0.000201236175020741430, 8.327579420473914e-05
#0.08802591193613027, 0.007365619748701712


#increasing spatial refinement (number of elements)
errorList=[]
for cfl in cflList:
    c=0
    error_velo=[]
    error_pres=[]
    for D in refin:
        #time stepping
        # cfl number 4 restrics size of dt
        # due to accuracy considerations rather than stability
        print(D)

        dt=cfl[c]*pi/(2*2**N)
        print(dt)
        T=1/dt#(pi/2)/dt
        t=[dt,T]
    
        #solve
        w,err_u,err_p,dx = taylorgreen(N, D,t,RE)
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

    



#plot actual errors in L2
#plt.plot(refin,np.array([3.9,1.2,0.33,0.093,0.026]),label="paper")
#plt.plot(refin,np.array(error_pres)/100,label="mine/100")
#plt.legend()
#plt.title("L2 error pressure")
#plt.show()

#plt.plot(refin,np.array([0.23,0.062,0.016,0.0042,0.0011]),label="paper")
#plt.plot(refin,error_velo,label="mine")
#plt.legend()
#plt.title("L2 error velocity")
#plt.show()
