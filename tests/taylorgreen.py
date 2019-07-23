from firedrake import *
from solver.parameters import *
from solver.spcs import *
import matplotlib.pyplot as plt
import math



def taylorgreen(dx_size,dimension,time_params,re,periodic=False):
    outfile=File("./output/taylorgreen/taylorgreen.pvd")

    #generate mesh
    LX=pi
    LY=pi
    if periodic:
        mesh = PeriodicRectangleMesh(math.floor(LX/dx_size),math.floor(LX/dx_size),Lx=LX,Ly=LY,quadrilateral=True)
    else:
        mesh = RectangleMesh(math.floor(LX/dx_size),math.floor(LX/dx_size),Lx=LX,Ly=LY,quadrilateral=True)
    mesh.coordinates.dat.data[:,0]-=pi/2
    mesh.coordinates.dat.data[:,1]-=pi/2
   
    #function spaces
    U = FunctionSpace(mesh, "RTCF",dimension)
    P = FunctionSpace(mesh, "DG", dimension-1)
    W = U*P
    test= FunctionSpace(mesh, "CG",dimension)
    
    #1/reynolds number,time stepping params
    NU=1/re
    nue=Constant(NU) 
    [dt,T]=time_params
    print("dt is: ",dt)

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
    t=Constant(0)
    x, y = SpatialCoordinate(mesh)
    ux=-cos(x)*sin(y)*exp(-2*t*nue*dt)
    uy=sin(x)*cos(y)*exp(-2*t*nue*dt)
    bc_expr=as_vector((ux,uy))

    bc_tang=[]
    if (not periodic):
        bc_tang.append([Function(U).project(bc_expr),4])
        bc_tang.append([Function(U).project(bc_expr),1])
        bc_tang.append([Function(U).project(bc_expr),2])
        bc_tang.append([Function(U).project(bc_expr),3])


    #gather bcs
    bc=[bc_norm,bc_tang,bc_expr]


    p_exact=-1/4*(cos(2*x)+cos(2*y))*exp(-4*(T+0.5)*dt*nue)

    #run standard pressure correction scheme to solve Navier Stokes equations
    sol=spcs(W,mesh,nue,bc,0,t,dt,T,outfile,bc_expr,p_exact)

    #return errors
    t=Constant(T)
    u_exact=as_vector((ux,uy))

    #L2 error
    err_u=errornorm(sol.sub(0),Function(U).project(u_exact),"L2")
    err_p=errornorm(sol.sub(1),Function(P).project(p_exact),"L2")
    print("L_inf error of velo",max(abs(assemble(sol.sub(0)-Function(U).project(u_exact)).dat.data)))
    print("L_inf error of pres",max(abs(assemble(sol.sub(1)-Function(P).project(p_exact)).dat.data)))
    
    err_u_inf=max(abs(assemble(sol.sub(0)-Function(U).project(u_exact)).dat.data))
    err_p_inf=max(abs(assemble(sol.sub(1)-Function(P).project(p_exact)).dat.data))


    #plot(Function(U).project(sol.sub(0)))
    #plt.show()

    #plot(Function(U).project(u_exact))
    #plt.show()

   # plot(Function(U).project(sol.sub(0)-Function(U).project(u_exact)))
   # plt.show()
    #N=LX/2 ** mesh_size
    return sol,err_u_inf,err_p_inf,dx_size