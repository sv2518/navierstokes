from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices


list_D=[]

refin=range(1,5)#space dimension
RE=1#reynolds number
N=5#5#fe number (space discretisation)
u=exp(-2*1/RE)
print(u)
cflList=[[8,4,8/3,2],[4,2,4/3,1],[2,1,2/3,0.5],[1,0.5,1/3,0.25]]
#cflList=[[2,2/4,2/9,2/16]]
#0.0018998908567869968, 4.344594410393178e-05, 4.254379857171822e-05

#0.006135873317419445
#2.2213806666824714
#tmp0=[0.006913096228617835, 0.00337689362447338, 0.001984438532709451, 0.00080584031028125772]#[7.59925505679146e-05, 7.57161819435765e-05,7.570697664648768e-05]
#0.006689293745712409, 0.009262522265189903, 0.0025201446727629055, 0.003776870855408938
tmp0=[0.006689293745712407, 0.0033768936244733764, 0.000421389541237124, 0.0003333310948737938]
tmp=[0.12506863431237716, 0.0015602447017334254, 0.0003541209172361233, 0.0002269698665969197]
plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.000001)

tmp0=[0.006689293745712407, 0.0033768936244733764, 0.000421389541237124, 0.0003333310948737938]
tmp=[0.12506863431237716, 0.0015602447017334254, 0.0003541209172361233, 0.0002269698665969197]
plot_convergence_velo_pres(tmp0,tmp,[1,2,3,4],0.000001)




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
        errorList.append([error_velo,error_pres,list_D])
        print(error_velo)
        print(error_pres)
        c+=1


#convergence plots
print(error_velo)
print(error_pres)
print(list_D)
print("\n\n")
print(errorList)

for cfl in cflList:
    plot_convergence_velo_pres(cfl[0],cfl[1],cfl[2],0.0001)

    



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
