from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices

error_velo=[]
error_pres=[]
list_D=[]

refin=range(1,5)#space dimension
RE=1#reynolds number
N=5#5#fe number (space discretisation)
u=exp(-2*1/RE)
print(u)
cflList=[[4,2,1,0.5],[2,1,0.5,0.25],[1,0.5,0.25,0.125],
[0.4,0.2,0.1,0.05],[0.2,0.1,0.05,0.025],[0.1,0.05,0.025,0.0125]]

#0.0018998908567869968, 4.344594410393178e-05, 4.254379857171822e-05

#0.006135873317419445
#2.2213806666824714
tmp0=[7.59925505679146e-05, 7.57161819435765e-05,7.570697664648768e-05]
tmp=[0.00022842084251824615, 0.0001528619428370617,0.00015190719944913793]
plot_convergence_velo_pres(tmp0,tmp,[2,3,4],0.000001)




#increasing spatial refinement (number of elements)
errorList=[]
for cfl in cflList:
    c=0
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
        errorList.append([error_velo,error_pres,list_D])
        print(error_velo)
        print(error_pres)
        c+=1


#convergence plots
print(error_velo)
print(error_pres)
print(list_D)

print(errorList)

for cfl in cflList:
    plot_convergence_velo_pres(error_velo,error_pres,list_D,0.0001)

    



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
