from verification.p_convergence import p_convergence
from firedrake import *

cfl_list=[20,15,10,8,4,2,1,0.1]#cfl number
order_list=range(1,5)#space dimension
RE=1#reynolds number
N=6#5#fe number (space discretisation)
TMAX=pi
XLEN=2*pi
bc_type="dirichlet"
output=True

#IP_stabilityparam_type="order_unscaled"
#p_convergence(cfl_list,order_list,RE,TMAX,XLEN,N,bc_type,output,"newstabs")
IP_stabilityparam_type="linear_order_scaled"
p_convergence(cfl_list,order_list,RE,TMAX,XLEN,N,bc_type,output,IP_stabilityparam_type)
IP_stabilityparam_type="quadratic_order_scaled"
p_convergence(cfl_list,order_list,RE,TMAX,XLEN,N,bc_type,output,IP_stabilityparam_type)
