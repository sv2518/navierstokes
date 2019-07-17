from firedrake import *
from solver.parameters import *
from solver.spcs import *


def cavity(mesh_size,dimension,time_params,re):
    outfile=File("./output/cavity/cavity.pvd")

    #generate mesh
    LX=1.0
    LY=1.0
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
   
    #function spaces
    U = FunctionSpace(mesh, "RTCF",dimension)
    P = FunctionSpace(mesh, "DG", dimension-1)
    W = U*P
    
    
    #1/reynolds number,time stepping params
    NU=1/re
    nue=Constant(NU) 
    [dt,T]=time_params
    print("dt is: ",dt)

    #normal boundary conditions
    bc_norm=[]
    bc0=DirichletBC(W.sub(0),Constant((0.0,0.0)),1)
    bc_norm.append(bc0)
    bc1=DirichletBC(W.sub(0),Constant((0.0,0.0)),2)#plane x=0
    bc_norm.append(bc1)
    bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
    bc_norm.append(bc2)
    bc3=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
    bc_norm.append(bc3)

    #tangential boundary conditions
    t=Constant(1)
    x, y = SpatialCoordinate(mesh)
    bc_expr=as_vector((-x*100*(x-1)/25*U_inf*(t)*dt,0))

    bc_tang=[]
    bc_tang.append([Function(U).project(bc_expr),4])
    bc_tang.append([Function(U),1])
    bc_tang.append([Function(U),2])
    bc_tang.append([Function(U),3])

    #gather bcs
    bc=[bc_norm,bc_tang,bc_expr]

    #run standard pressure correction scheme to solve Navier Stokes equations
    sol=spcs(W,mesh,nue,bc,U_inf,dt,T+1,outfile,bc_expr)

    N=2 ** mesh_size
    return sol,0,0,N