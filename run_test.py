from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers import plot_convergence_velo_pres

#####################MAIN##########################
error_velo=[]
error_pres=[]
list_N=[]

#fe number and order
refin=range(5,10)
D=1

for n in refin:#increasing element number
    
    #solve
    w,err_u,err_p,N = taylorgreen(n, D)
    u,p=w.split()
    error_velo.append(err_u)
    error_pres.append(err_p)
    list_N.append(N)


print(err_u)
print(err_p)

plot_convergence_velo_pres(error_velo,error_pres,list_N)

    



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
