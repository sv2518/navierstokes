from verification.p_convergence import p_convergence
from firedrake import *
<<<<<<< HEAD
cfl_list=[20,10,15,8,4,2,1,0.1]
#cfl_list=[20]
=======

cfl_list=[0.1]
>>>>>>> b171637ce76c968cb2a38e91a229a4b883f4ef08
#cfl_list=[20,15,10,8,4,2,1,0.1]#cfl number
order_list=range(1,5)#space dimension
RE=1#reynolds number
N=6#5#fe number (space discretisation)
TMAX=1
XLEN=2*pi
bc_type="dirichlet"
output=True

p_convergence(cfl_list,order_list,RE,TMAX,XLEN,N,bc_type,output,"newstabs")

<<<<<<< HEAD
p_convergence(cfl_list,order_list,RE,TMAX,XLEN,N,bc_type,output,"newstabs2")
=======
>>>>>>> b171637ce76c968cb2a38e91a229a4b883f4ef08
