from firedrake import *
from solver.picard import *
from helpers.plot_velo_pres import *
import numpy as np
import matplotlib.pyplot as plt

def kovasznay(mesh_size, dimension):
    outfile=File("./output/kovasznay.pvd")

    #generate mesh
    LX=2.0
    LY=2.0
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    mesh.coordinates.dat.data[:,0]-=0.5

    #function spaces
    U = FunctionSpace(mesh, "RTCF",dimension)
    P = FunctionSpace(mesh, "DG", dimension-1)
    W = U*P

    #1/reynolds number
    nue=Constant(0.4)#re=40

    #specify inflow/solution
    x,y=SpatialCoordinate(mesh)
    lam=(-8.*pi**2/(nue**(-1)+sqrt(nue**(-2)+16*pi**2))) #from paper 
    #lam=(-2*pi**2/(nue**(1)+sqrt(nue**(2)+16*pi**2)))
    #lam=-2*pi
    #lam=nue/2-sqrt(nue**2/4+4*pi**2) #from old book
    #lam=1/(2*nue)-sqrt(1/(4*nue**2)+4*pi**2) #from new book
    #lam= 1/(2*nue)-sqrt(1/(4*nue**2)+4*pi**2) #from nektar
    ux=1-exp(lam*x)*cos(2*pi*y)
    uy=lam/(2*pi)*exp(lam*x)*sin(2*pi*y)
    inflow=Function(U).project(as_vector((ux,uy)))
    
    #normal boundaries
    bc_norm=[]
    bc0=DirichletBC(W.sub(0),Constant((0.0,0.0)),2)
    bc_norm.append(bc0)
    bc1=DirichletBC(W.sub(0),inflow,1)#plane x=0
    bc_norm.append(bc1)
    bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
    bc_norm.append(bc2)
    bc3=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
    bc_norm.append(bc3)

    #tangential boundaries
    x, y = SpatialCoordinate(mesh)
    bc_tang=[]
    bc_tang.append([Function(U),4])
    bc_tang.append([Function(U),1])
    bc_tang.append([Function(U),2])
    bc_tang.append([Function(U),3])

    #gather bcs
    bc=[bc_norm,bc_tang]
  
    #exact solutions
    u_exact=as_vector((ux,uy))
    p_exact=-1./2*exp(2*lam*x)
    p_sol=Function(P).project(p_exact)
    u_sol=Function(U).project(u_exact)

    #picard iteration
    w=picard(W,mesh,nue,bc,outfile,inflow)

    plot_velo_pres(Function(U).project(u1-u_exact),Function(P).project(p1-p_exact),"Error")
   

    #L2 error of divergence
    err_u=errornorm(w.sub(0),Function(U).project(u_exact))
    err_p=errornorm(w.sub(1),Function(P).project(p_exact))
    print("L2 error of divergence",errornorm(Function(P).project(div(w.sub(0))),Function(P)))
    print("L_inf error of velo",max(abs(assemble(w.sub(0)-Function(U).project(u_exact)).dat.data)))
    print("L_inf error of pres",max(abs(assemble(w.sub(1)-Function(P).project(p_exact)).dat.data)))
    print("L_2 error of velo", err_u)
    print("L_2 error of pres", err_p)
    print("Hdiv error of velo", errornorm(w.sub(0),Function(U).project(u_exact),"Hdiv"))
    print("Hdiv error of pres",  errornorm(w.sub(1),Function(P).project(p_exact),"Hdiv"))
    print("H1 error of velo", errornorm(w.sub(0),Function(U).project(u_exact),"H1"))
    print("H1 error of pres",  errornorm(w.sub(1),Function(P).project(p_exact),"H1"))
    #L2 and HDiv the same...why?
    N=2 ** mesh_size
    return w,err_u,err_p,N
