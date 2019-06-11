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
    axis.loglog(list_N,0.001*np.power(list_N,2),'r*',label="second order")
    axis.loglog(list_N,0.001*np.power(list_N,1),'g*',label="first order")
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
    LX=100
    LY=1
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    
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
    nue=Constant(0.1)#viscosity

    #specify inflow/initial solution
    x,y=SpatialCoordinate(mesh)
    u_exact=as_vector((-0.5*(y-1)*y,0.0*y))
    p_exact=LX-x#factor of pressure gradient is double of factor of velocity

    inflow=Function(U).project(u_exact)

    #Picard iteration
    u_linear=Function(U).assign(inflow)
    counter=0
    while(True):

        #Laplacian
        alpha=Constant(10.)
        gamma=Constant(10.) 
        h=CellVolume(mesh)/FacetArea(mesh)
        havg=avg(CellVolume(mesh))/FacetArea(mesh)
        kappa1=nue*alpha/havg
        kappa2=nue*gamma/h

        a_dg=(nue*inner(grad(u),grad(v))*dx
            -inner(outer(v,n),nue*grad(u))*ds 
            -inner(outer(u,n),nue*grad(v))*ds 
            +kappa2*inner(v,u)*ds 
            -inner(nue*avg(grad(v)),both(outer(u,n)))*dS
            -inner(both(outer(v,n)),nue*avg(grad(u)))*dS
            +kappa1*inner(both(outer(u,n)),both(outer(v,n)))*dS)
         
        #Advection
        un = 0.5*(dot(u_linear, n) + abs(dot(u_linear, n)))
        adv_dg=(dot(u_linear,div(outer(v,u)))*dx#like paper
            -inner(v,(u*dot(u_linear,n)))*ds#similar to matt piggots
            -dot((v('+')-v('-')),(un('+')*u('+') - un('-')*u('-')))*dS)#like in the tutorial
    
        #form
        eq = a_dg+Constant(-1.)*adv_dg-div(v)*p*dx-div(u)*q*dx

        #MMS
        strongform1=Function(U).project(div(grad(u_exact))-grad(dot(u_exact,u_exact))-0.5*dot(u_exact,grad(u_exact))-grad(p_exact))
        strongform2=Function(P).project(div(u_exact))

        #->plot corrector force
        #plot_velo_pres(strongform1,strongform2,"Corrector Force")

        #->manufacture equations
        f=dot(strongform1,v)*dx+dot(strongform2,q)*dx
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
        bc1=DirichletBC(W.sub(0),inflow,1)#plane x=0
        bc_1.append(bc1)
        bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
        bc_1.append(bc2)
        bc3=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
        bc_1.append(bc3)

        #build problem and solver
        w = Function(W)
        #nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
        problem = LinearVariationalProblem(a, L, w, bc_1)
        appctx = {"Re": 1, "velocity_space": 0}
        solver = LinearVariationalSolver(problem, solver_parameters=parameters,appctx=appctx)
        solver.solve()
        u1,p1=w.split()

        #convergence criterion
        eps=errornorm(u1,u_linear)#l2 by default
        counter+=1
        print("Picard iteration error",eps,", counter: ",counter)
        if(eps<10**(-8)):
            print("Picard iteration converged")  
            break          
        else:
            u_linear.assign(u1)


    # plot error fields
    #plot_velo_pres(Function(U).project(u1-u_exact),Function(P).project(p1-p_exact),"Error")
   

    #L2 error of divergence
    err_u=errornorm(w.sub(0),Function(U).project(u_exact))
    err_p=errornorm(w.sub(1),Function(P).project(p_exact))
    print("L2 error of divergence",errornorm(Function(P).project(div(w.sub(0))),Function(P)))
    print("L_inf error of velo",max(abs(assemble(w.sub(0)-Function(U).project(u_exact)).dat.data)))
    print("L_inf error of pres",max(abs(assemble(w.sub(1)-Function(P).project(p_exact)).dat.data)))
    print("L_2 error of velo", err_u)
    print("L_2 error of pres", err_p)
    # print("Hdiv error of velo", errornorm(w.sub(0),Function(U).project(u_exact),"Hdiv"))
    # print("Hdiv error of pres",  errornorm(w.sub(1),Function(P).project(p_exact),"Hdiv"))
    #L2 and HDiv the same...why?
    N=2 ** mesh_size
    return w,err_u,err_p,N

#####################MAIN##########################
parameters={
   # "ksp_type": "fgmres",
   # "ksp_rtol": 1e-8,
   # "pc_type": "fieldsplit",
   # "pc_fieldsplit_type": "schur",
   # "pc_fieldsplit_schur_fact_type": "full",
   # "fieldsplit_0_ksp_type": "cg",
   # "fieldsplit_0_pc_type": "ilu",
   # "fieldsplit_0_ksp_rtol": 1e-8,
   # "fieldsplit_1_ksp_type": "cg",
   # "fieldsplit_1_ksp_rtol": 1e-8,
   # "pc_fieldsplit_schur_precondition": "selfp",
   # "fieldsplit_1_pc_type": "hypre"
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
refin=range(3,9)
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