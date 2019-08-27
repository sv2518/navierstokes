from tests.taylorgreen import *
from tests.cavity import *
import matplotlib.pyplot as plt
import numpy as np
import math

#####################MAIN##########################

#solver params
output=True
scaling=None

#flow params
cfl=1#cfl number
RE=100#reynolds number
U_inf=1

#space params
dim=1
XLEN=1
dofcount=20000
dx=math.sqrt(XLEN**2*(dim**2+0+2*dim+0)/dofcount)
print(dx)
bc_type="dirichlet"
if bc_type=="dirichlet":
    bc_type_periodic=False
else:
    bc_type_periodic=True

#time params
TMAX=1200
dt=cfl/dim**2*dx/2
T=TMAX/dt
time_params=[dt,T]

######run taylorgreen OR cavity flow

#sol,[linf_err_u,l2_err_u,hdiv_err_u],[linf_err_p,l2_err_p,h1_err_p],dx_size,mesh.comm =  taylorgreen(dx,dim,time_params,RE,XLEN,scaling,bc_type_periodic,output)
    
sol,mesh.comm =  cavity(U_inf,dx,dim,time_params,RE,XLEN,scaling,bc_type_periodic,output)
    
#solutions get saved in output folder!

    

