from firedrake import *
from solver.parameters import *
from solver.spcs import *
import math


def cavity(U_inf,dx_size,dimension,time_params,RE,XLEN,IP_stabilityparam_type=None,periodic=False,output=False):
    outfile=File("./output/cavity/cavity.pvd")

    #generate mesh
    LX=XLEN
    LY=LX
    mesh = RectangleMesh(math.floor(LX/dx_size),math.floor(LX/dx_size),Lx=LX,Ly=LY,quadrilateral=True)
   
    #function spaces
    U = FunctionSpace(mesh, "RTCF",dimension)
    P = FunctionSpace(mesh, "DG", dimension-1)
    W = U*P    
    
    #1/reynolds number,time stepping params
    NU=1/RE
    nue=Constant(NU) 
    [dt,T]=time_params

    #normal boundary conditions
    if periodic:
        bc_norm=None
    else:
        bc_norm=[]
        bc0=DirichletBC(W.sub(0),Constant((0,0)),1)#plane x=0
        bc_norm.append(bc0)
        bc1=DirichletBC(W.sub(0),Constant((0,0)),2)#plane x=L
        bc_norm.append(bc1)
        bc2=DirichletBC(W.sub(0),Constant((0,0)),3)#plane y=0
        bc_norm.append(bc2)
        bc3=DirichletBC(W.sub(0),Constant((0,0)),4)#plane y=L
        bc_norm.append(bc3)

    #tangential boundary conditions
    t=Constant(1)
    x, y = SpatialCoordinate(mesh)
    bc_expr=as_vector((-x*100*(x-1)/25*U_inf*(t)*dt,0))
    bc_expr_list=[]
    bc_expr_list.append(Constant((0,0)))
    bc_expr_list.append(Constant((0,0)))
    bc_expr_list.append(Constant((0,0)))
    bc_expr_list.append(bc_expr)

    bc_tang=[]
    if (not periodic):
        bc_tang.append([Function(U).project(bc_expr_list[0]),1])
        bc_tang.append([Function(U).project(bc_expr_list[1]),2])
        bc_tang.append([Function(U).project(bc_expr_list[2]),3])
        bc_tang.append([Function(U).project(bc_expr_list[3]),4])


    #gather bcs
    bc=[bc_norm,bc_tang,bc_expr_list]

    #run standard pressure correction scheme to solve Navier Stokes equations
    sol=spcs(W,mesh,nue,bc,0,t,dt,T,outfile,dimension,IP_stabilityparam_type,None,None,output)

    return  sol,mesh.comm