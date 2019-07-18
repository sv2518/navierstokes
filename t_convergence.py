from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres
import math

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices

error_velo=[]
error_pres=[]
list_dx=[]


RE=1
refin=[0.5,0.25,0.1,0.05,0.025,0.01]
cfl=4
#tmp0=[0.018664174119308548, 0.004345185588971723, 0.0007046155441139323]
#tmp=[0.018664174119308548, 0.004345185588971723, 0.0007046155441139323]

D=1#space dimension

#RE=100
#refin=[0.2,0.1,0.05,0.025,0.0125]
#cfl=4
#tmp0=[0.008037579298779657, 0.0037023249497532343, 0.0017633803060275594, 0.0008581454468339789, 0.00042299888477377763]
#tmp=[0.0685551391772734, 0.07180374671677864, 0.0726675791593635, 0.07289509741975841, 0.07295713256079672]
#plot_convergence_velo_pres(tmp0,tmp,[0.2,0.1,0.05,0.025,0.0125],1)
#cfl=4

#re1 cfl4
#tmp0=[0.03636248052614443, 0.008123268566361015, 0.001247392864111384, 0.00040031105464151916, 0.00018844227829679352]
#tmp=[0.03636248052614443, 0.008123268566361015, 0.001247392864111384, 0.00040031105464151916, 0.00018844227829679352]

#plot_convergence_velo_pres(tmp0,tmp,refin,0.1)

#increasing spatial refinement (number of elements)
for dt in refin:
    #time stepping
    u=exp(-2*1/RE)
    print(u)
    N=math.floor(math.log(cfl*pi/(2*u*dt))/math.log(2))
    print(N)
    T=1/dt
    t=[dt,T]
    #solve
    w,err_u,err_p,dx = taylorgreen(N, D,t,RE)
    u,p=w.split()
    error_velo.append(err_u)
    error_pres.append(err_p)
    list_dx.append(dx)
    print(error_velo)

#convergence plots
print(error_velo)
print(error_pres)
print(list_dx)
plot_convergence_velo_pres(error_velo,error_pres,list_dx,1)

    



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
