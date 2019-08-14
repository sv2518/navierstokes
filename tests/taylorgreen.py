from firedrake import *
from solver.parameters import *
from solver.spcs import *
import matplotlib.pyplot as plt
import math
from firedrake.petsc import PETSc


def taylorgreen(dx_size,dimension,time_params,RE,XLEN,IP_stabilityparam_type=None,periodic=False,output=False):
    with PETSc.Log.Event("configuration"):
        outfile=File("./output/taylorgreen/taylorgreen.pvd")

        #generate mesh
        LX=XLEN
        LY=XLEN
        if periodic:
            mesh = PeriodicRectangleMesh(math.floor(LX/dx_size),math.floor(LX/dx_size),Lx=LX,Ly=LY,quadrilateral=True)
        else:
            mesh = RectangleMesh(math.floor(LX/dx_size),math.floor(LX/dx_size),Lx=LX,Ly=LY,quadrilateral=True)
        mesh.coordinates.dat.data[:,0]-=pi
        mesh.coordinates.dat.data[:,1]-=pi
    
        #function spaces
        U = FunctionSpace(mesh, "RTCF",dimension)
        P = FunctionSpace(mesh, "DG", dimension-1)
        W = U*P
        
        #1/reynolds number,time stepping params
        NU=1/RE
        nue=Constant(NU) 
        [dt,T]=time_params
        PETSc.Sys.Print("dt is: ",dt)

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
        k=2*pi/LX
        ux=cos(k*y)*sin(k*x)*exp(-2*k**2*t*dt)
        uy=-sin(k*y)*cos(k*x)*exp(-2*k**2*t*dt)
        bc_expr=as_vector((ux,uy))

        bc_tang=[]
        if (not periodic):
            bc_tang.append([Function(U).project(bc_expr),4])
            bc_tang.append([Function(U).project(bc_expr),1])
            bc_tang.append([Function(U).project(bc_expr),2])
            bc_tang.append([Function(U).project(bc_expr),3])


        #gather bcs
        bc=[bc_norm,bc_tang,bc_expr]


        p_exact=-1/4*(cos(2*k*x)+cos(2*k*y))*exp(-4*k**2*(t+0.5)*dt*nue)

    with PETSc.Log.Event("spcs"):
        #run standard pressure correction scheme to solve Navier Stokes equations
        sol=spcs(W,mesh,nue,bc,0,t,dt,T,outfile,dimension,IP_stabilityparam_type,bc_expr,None,output)

    with PETSc.Log.Event("postprocessing"):
        #return errors
        t.assign(T)#+1/dt)
        u_exact=as_vector((ux,uy))

        #various errors
        linf_err_u=max(abs(assemble(sol.sub(0)-Function(U).project(u_exact)).dat.data))
        linf_err_p=max(abs(assemble(sol.sub(1)-Function(P).project(p_exact)).dat.data))
        l2_err_u=errornorm(sol.sub(0),Function(U).project(u_exact),"L2")
        l2_err_p=errornorm(sol.sub(1),Function(P).project(p_exact),"L2")
        hdiv_err_u=errornorm(sol.sub(0),Function(U).project(u_exact),"hdiv")
        h1_err_p=errornorm(sol.sub(1),Function(P).project(p_exact),"H1")


        #plot(Function(U).project(sol.sub(0)))
        #plt.show()

        #plot(Function(U).project(u_exact))
        #plt.show()

        #plot(Function(U).project(sol.sub(0)-Function(U).project(u_exact)))
        #plt.show()
        #N=LX/2 ** mesh_size
    return sol,[linf_err_u,l2_err_u,hdiv_err_u],[linf_err_p,l2_err_p,h1_err_p],dx_size,mesh.comm

