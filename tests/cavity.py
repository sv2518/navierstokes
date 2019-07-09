from firedrake import *
from solver.parameters import *
from solver.spcs import *


def cavity(mesh_size,dimension):
    outfile=File("./output/cavity.pvd")

    #generate mesh
    LX=1.0
    LY=1.0
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    U_in=1
    U_inf=Constant(U_in)
   
    #function spaces
    U = FunctionSpace(mesh, "RTCF",dimension)
    P = FunctionSpace(mesh, "DG", dimension-1)
    W = U*P
    
    #1/reynolds number,time stepping params
    NU=0.01
    nue=Constant(NU) 
    dt=1/(U_in*(2 ** mesh_size)) #with cfl number
    T=1500
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
    bc_tang=[]
    bc_tang.append([Function(U).project(as_vector([-x*100*(x-1)/25*U_inf*(t)*dt,0])),4])
    bc_tang.append([Function(U),1])
    bc_tang.append([Function(U),2])
    bc_tang.append([Function(U),3])

    #gather bcs
    bc=[bc_norm,bc_tang]

    #run standard pressure correction scheme to solve Navier Stokes equations
    sol=spcs(W,mesh,nue,bc,U_inf,dt,T,outfile)

    N=2 ** mesh_size
    return sol,0,0,N