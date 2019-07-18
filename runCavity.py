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
    error_pres.append(N)



plot_convergence_velo_pres(error_velo,error_pres,list_N)

    


