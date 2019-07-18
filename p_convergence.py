from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices

error_velo=[]
error_pres=[]
list_D=[]

refin=range(2,7)#space dimension
RE=1#reynolds number
N=9#5#fe number (space discretisation)


#0.0018998908567869968, 4.344594410393178e-05, 4.254379857171822e-05

#0.006135873317419445
#2.2213806666824714
tmp0=[0.004276849913502535, 0.00022663213303872332, 0.00022079643162701418,0]
tmp=[1.1584581347037219e-05, 3.5735907263755316e-05, 8.047774788336951e-05,0]
plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.000001)

#increasing spatial refinement (number of elements)
for D in refin:
    #time stepping
    # cfl number 4 restrics size of dt
    # due to accuracy considerations rather than stability
    print(D)
   
    dt=0.06*pi*pi/(2*2**N)#0.001#pi/(2 ** N)*0.6
    print(dt)
    T=2#(pi/2)/dt
    t=[dt,T]
    #solve
    w,err_u,err_p,dx = taylorgreen(N, D,t,RE)
    u,p=w.split()
    error_velo.append(err_u)
    error_pres.append(err_p)
    list_D.append(D)
    print(error_velo)
    print(error_pres)


#convergence plots
print(error_velo)
print(error_pres)
print(list_D)
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
