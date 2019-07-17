from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices

error_velo=[]
error_pres=[]
list_dx=[]

refin=range(5,9)#fe number (space discretisation)
D=1#space dimension
RE=1#reynolds number

#increasing spatial refinement (number of elements)
for N in refin:
    #time stepping
    dt=(0.5/(2 ** mesh_size)**2)
    T=100
    t=[dt,T]
    #solve
    w,err_u,err_p,dx = taylorgreen(N, D,t,RE)
    u,p=w.split()
    error_velo.append(err_u)
    error_pres.append(err_p)
    list_dx.append(dx)

#convergence plots
print(error_velo)
print(error_pres)
print(list_dx)
plot_convergence_velo_pres(error_velo,error_pres,list_dx,10)

    



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
