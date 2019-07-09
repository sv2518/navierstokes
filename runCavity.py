from tests.cavity import *

#####################MAIN##########################
error_velo=[]
error_pres=[]

#fe number and order
refin=range(6,7)
D=1

for n in refin:#increasing element number
    
    #solve
    w,err_u,err_p,N = cavity(n, D)
    u,p=w.split()
    error_velo.append(err_u)
    error_pres.append(err_p)

    



