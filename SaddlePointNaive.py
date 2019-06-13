from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

def both(expr):
    return expr('+') + expr('-')

def plot_velo_pres(u,p,title):
    plot(u)
    plt.title(str(title+" Velocity"))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    plot(p)
    plt.title(str(title+" Pressure"))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plot_convergence_velo_pres(error_velo,error_pres,list_N):
    # velocity convergence plot
    fig = plt.figure()
    axis = fig.gca()
    linear=error_velo[::-1]
    axis.loglog(list_N,linear,label='$||e_u||_{\infty}$')
    axis.loglog(list_N,0.0001*np.power(list_N,2),'r*',label="second order")
    axis.loglog(list_N,0.0001*np.power(list_N,1),'g*',label="first order")
    axis.set_xlabel('$2**Level$')
    axis.set_ylabel('$Error$')
    axis.legend()
    plt.show()

    #pressure convergence plot
    fig = plt.figure()
    axis = fig.gca()
    linear=error_pres[::-1]
    axis.loglog(list_N,linear,label='$||e_p||_{\infty}$')
    axis.loglog(list_N,0.1*np.power(list_N,2),'r*',label="second order")
    axis.loglog(list_N,0.1*np.power(list_N,1),'g*',label="first order")
    axis.set_xlabel('$2**Level$')
    axis.set_ylabel('$Error$')
    axis.legend()
    plt.show()

def solve_problem(mesh_size, parameters, aP=None, block_matrix=False):
    #generate mesh
    LX=2.0
    LY=2.0
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    mesh.coordinates.dat.data[:,0]-=0.5

    #function spaces
    U = FunctionSpace(mesh, "RTCF",1)
    P = FunctionSpace(mesh, "DG", 0)
    W = U*P

    #functions
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    f =Function(U)
	
    #building the operators
    n=FacetNormal(W.mesh())
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
    u_exact=as_vector((ux,uy))
    p_exact=-1./2*exp(2*lam*x)
    p_sol=Function(P).project(p_exact)
    u_sol=Function(U).project(u_exact)
    inflow=Function(U).project(as_vector((ux,uy)))
    #inflow_uniform=Function(U).project(Constant((1.0,0.0)))  
    

    #Picard iteration
    u_linear=Function(U).assign(inflow)
    counter=0
    while(True):

        #Laplacian
        alpha=Constant(1000.)#interior
        gamma=Constant(1000.) #exterior
        h=CellVolume(mesh)/FacetArea(mesh)  
        havg=avg(CellVolume(mesh))/FacetArea(mesh)
        kappa1=nue*alpha/havg
        kappa2=nue*gamma/h

        g=Function(U).assign(inflow) 
        #g=Constant((0.0,0.0))

        a_dg=(nue*inner(grad(u),grad(v))*dx
            -inner(outer(v,n),nue*grad(u))*ds 
            -inner(outer(u-g,n),nue*grad(v))*ds 
            +kappa2*inner(v,u-g)*ds 
            -inner(nue*avg(grad(v)),both(outer(u-g,n)))*dS
            -inner(both(outer(v,n)),nue*avg(grad(u)))*dS
            +kappa1*inner(both(outer(u-g,n)),both(outer(v,n)))*dS)
         
        #Advection
        un = 0.5*(dot(u_linear, n)+sign(dot(u_linear, n))*dot(g,n)+abs(dot(g,n))+ abs(dot(u_linear, n)))
        adv_dg=(dot(u_linear,div(outer(v,u)))*dx#like paper
            -inner(v,u*un)*ds#similar to matt piggots
            -dot((v('+')-v('-')),(un('+')*(u('+')) - un('-')*(u('-'))))*dS)#like in the tutorial
    
        #form
        eq = a_dg+Constant(-1.)*adv_dg-div(v)*p*dx-div(u)*q*dx
        f=dot(Function(U),v)*dx   
        eq +=f
        a=lhs(eq)
        L=rhs(eq)

        #preconditioning(not used here)
        if aP is not None:
            aP = aP(W)
        if block_matrix:
            mat_type = 'nest'
        else:
            mat_type = 'aij'

        #boundary conditions
        bc_1=[]
      #  bc0=DirichletBC(W.sub(1),p_sol,1)
       # bc_1.append(bc0)
        bc1=DirichletBC(W.sub(0),inflow,1)#plane x=0
        bc_1.append(bc1)
        bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
        bc_1.append(bc2)
        bc3=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
        bc_1.append(bc3)

        #add nullspace bc all boundaries are specified
        #nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])

        #build problem and solver
        w = Function(W)
        problem = LinearVariationalProblem(a, L, w, bc_1)
        solver = LinearVariationalSolver(problem, solver_parameters=parameters)
        solver.solve()
        u1,p1=w.split()

        #convergence criterion
        eps=errornorm(u1,u_linear)#l2 by default
        counter+=1
        print("Picard iteration change in approximation",eps,", counter: ",counter)
        if(eps<10**(-12)):
            print("Picard iteration converged")  
            break          
        else:
            u_linear.assign(u1)
            #one comp has to be set to 0?


        # plot error fields
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

#####################MAIN##########################
parameters={
    #"ksp_type": "fgmres",
    #"ksp_rtol": 1e-8,
   # "pc_type": "fieldsplit",
  #  "pc_fieldsplit_type": "schur",
   # "pc_fieldsplit_schur_fact_type": "full",
   # "fieldsplit_0_ksp_type": "cg",
   # "fieldsplit_0_pc_type": "ilu",
    #"fieldsplit_0_ksp_rtol": 1e-8,
    #"fieldsplit_1_ksp_type": "cg",
    #"fieldsplit_1_ksp_rtol": 1e-8,
    #"pc_fieldsplit_schur_precondition": "selfp",
    #"fieldsplit_1_pc_type": "hypre"
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
   "ksp_gmres_restart":100,
    "ksp_rtol":1e-12,
    "pc_type":"lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_type":"aij"
}

error_velo=[]
error_pres=[]
refin=range(4,9)
list_N=[]
for n in refin:#increasing element number
    
    #solve
    w,err_u,err_p,N = solve_problem(n, parameters,aP=None, block_matrix=False)
    u,p=w.split()
    error_velo.append(err_u)
    error_pres.append(err_p)
    list_N.append(N)

    
    #plot solutions
    File("poisson_mixed_velocity_.pvd").write(u)
    File("poisson_mixed_pressure_.pvd").write(p)
    try:
        import matplotlib.pyplot as plt
    except:
        warning("Matplotlib not imported")

    #plot solutions
    try:
        plot_velo_pres(u,p)
    except:
        warning("Cannot show figure")

#plot convergence
plot_convergence_velo_pres(error_velo,error_pres,list_N)

#plot actual errors in L2
plt.plot(refin,np.array([3.9,1.2,0.33,0.093,0.026]),label="paper")
plt.plot(refin,np.array(error_pres)/100,label="mine/100")
plt.legend()
plt.title("L2 error pressure")
plt.show()

plt.plot(refin,np.array([0.23,0.062,0.016,0.0042,0.0011]),label="paper")
plt.plot(refin,error_velo,label="mine")
plt.legend()
plt.title("L2 error velocity")
plt.show()