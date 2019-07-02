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

    #INITAL VALUES: solve the following from for initial values
def initialVelocity(W,U,P,mesh,nue,bc,U_inf,parameters,dt,bc_tang):
    print("....Solving Stokes problem for initial velocity ....")

    #Functions and parameters
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    n=FacetNormal(W.mesh())


    #build form
     #Laplacian
    alpha=Constant(10)#interior
    gamma=Constant(10) #exterior
    h=CellVolume(mesh)/FacetArea(mesh)  
    havg=avg(CellVolume(mesh))/FacetArea(mesh)
    kappa1=nue*alpha/havg
    kappa2=nue*gamma/h
    
    x, y = SpatialCoordinate(mesh)

    #a_dg for other sides
    lapl_dg=(nue*inner(grad(u),grad(v))*dx
          -inner(outer(v,n),nue*grad(u))*ds((1,2,3)) 
          -inner(outer(u,n),nue*grad(v))*ds((1,2,3))
          +kappa2*inner(v,u)*ds((1,2,3))
          -inner(nue*avg(grad(v)),both(outer(u,n)))*dS
          -inner(both(outer(v,n)),nue*avg(grad(u)))*dS
          +kappa1*inner(both(outer(u,n)),both(outer(v,n)))*dS)
    
    #a_dg for lid
    lapl_dg+=(
        -inner(outer(v,n),nue*grad(u-bc_tang))*ds((4))
        -inner(outer(u-bc_tang,n),nue*grad(v))*ds((4)) 
        +kappa2*inner(v,u-bc_tang)*ds((4))
    )


    eq_init=-div(u)*q*dx+div(v)*p*dx-lapl_dg#dt somewhere in here??
 #########include gpos somewhere else her??????

    #solve
    w_init= Function(W)
    init = LinearVariationalProblem(lhs(eq_init),rhs(eq_init), w_init,bc)
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    solver = LinearVariationalSolver(init, solver_parameters=parameters)
    solver.solve()
    u_init_sol,p_init_sol=w_init.split()
    
    return u_init_sol

def initialPressure(W,U,P,mesh,nue,bc,u_init,U_inf,parameters_velo,parameters_pres,dt,bc_tang):
    print("....Solving problem for initial pressure ....")

    print(".........part1: solve for some non-divergence free velocity field")
    F,p=TrialFunctions(W)
    v,q=TestFunctions(W)
    n=FacetNormal(W.mesh())

    #build form
    #Laplacian
    u=Function(U).assign(u_init)

    alpha=Constant(10)#interior
    gamma=Constant(10) #exterior
    h=CellVolume(mesh)/FacetArea(mesh)  
    havg=avg(CellVolume(mesh))/FacetArea(mesh)
    kappa1=nue*alpha/havg
    kappa2=nue*gamma/h
    
    #laplacian for other sides
    a_dg=(
        nue*inner(grad(u),grad(v))*dx
          -inner(outer(v,n),nue*grad(u))*ds((1,2,3)) 
         -inner(outer(u,n),nue*grad(v))*ds((1,2,3))
          +kappa2*inner(v,u)*ds((1,2,3))
         -inner(nue*avg(grad(v)),both(outer(u,n)))*dS
         -inner(both(outer(v,n)),nue*avg(grad(u)))*dS
         +kappa1*inner(both(outer(u,n)),both(outer(v,n)))*dS
        )
    
    #laplacian for lid
    a_dg+=(
        -inner(outer(v,n),nue*grad(u-bc_tang))*ds((4))
        -inner(outer(u-bc_tang,n),nue*grad(v))*ds((4)) 
        +kappa2*inner(v,u-bc_tang)*ds((4))
    )

    #Advection
    #for lid
    u_linear=u_init
    un_pos = 0.5*(dot(u_linear, n)+sign(dot(u_linear, n))*dot(bc_tang,n)+abs(dot(bc_tang,n))+ abs(dot(u_linear, n)))
    un = 0.5*(dot(u_linear, n)+ abs(dot(u_linear, n)))    
    adv_dg=(
        dot(u_linear,div(outer(v,u)))*dx#like paper
            -dot(v,u*un_pos)*ds(4)#similar to matt piggots
            -dot((v('+')-v('-')),(un('+')*(u('+')) - un('-')*(u('-'))))*dS
        )#like in the tutorial
    
    #for other thingis
    adv_dg+=-inner(v,u*un)*ds((1,2,3)) #similar to matt piggots

    eq_init=dot(v,F)*dx-adv_dg+a_dg####do we need to include divergence here??

    #solve
    w_init= Function(W)
    init = LinearVariationalProblem(lhs(eq_init),rhs(eq_init), w_init,bc)
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    solver = LinearVariationalSolver(init, nullspace=nullspace,solver_parameters=parameters_velo)
    solver.solve()
    F_init_sol,p_init_sol=w_init.split()

    print("........part2: solve mixed possion problem for initial pressure ")
    
    w,beta = TrialFunctions(W)
    divtest=Function(P).project(div(F_init_sol))
    print("Div error of initial velocity",errornorm(divtest,Function(P)))
    f_pres=Function(P).project(div(F_init_sol)) 
    force_dg_pres=dot(f_pres,q)*dx#sign right?
    incomp_dg_pres=div(w)*q*dx
    pres_dg_pres=div(v)*beta*dx
    eq_pres=dot(w,v)*dx+force_dg_pres+incomp_dg_pres+pres_dg_pres #dt somewhere in here??


    w_init= Function(W)
    init = LinearVariationalProblem(lhs(eq_pres),rhs(eq_pres), w_init,bc)
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    solver = LinearVariationalSolver(init, nullspace=nullspace,solver_parameters=parameters_pres)
    solver.solve()
    u_init_sol,p_init_sol=w_init.split()
 
    return p_init_sol

def buildPredictorForm(p_init_sol,u_init_sol,nue,mesh,U,P,W,U_inf,dt,bc_tang,v_k,u_n,p_n):
    print("....build predictor")

    v_knew=TrialFunction(U)
    v=TestFunction(U)
    n=FacetNormal(U.mesh())

    #functions
    ubar_k=Constant(0.5)*(u_n+v_k) #init old midstep
    ubar_knew=Constant(0.5)*(u_n+v_knew) #init new midstep
    
    #Laplacian a_dg
    alpha=Constant(10)#interior
    gamma=Constant(10) #exterior
    h=CellVolume(mesh)/FacetArea(mesh)  
    havg=avg(CellVolume(mesh))/FacetArea(mesh)
    kappa1=nue*alpha/havg
    kappa2=nue*gamma/h
    
    #lapl_dg for other sides
    lapl_dg=(nue*inner(grad(ubar_knew),grad(v))*dx
          -inner(outer(v,n),nue*grad(ubar_knew))*ds((1,2,3)) 
          -inner(outer(ubar_knew,n),nue*grad(v))*ds((1,2,3))
          +kappa2*inner(v,ubar_knew)*ds((1,2,3))
          -inner(nue*avg(grad(v)),both(outer(ubar_knew,n)))*dS
          -inner(both(outer(v,n)),nue*avg(grad(ubar_knew)))*dS
          +kappa1*inner(both(outer(ubar_knew,n)),both(outer(v,n)))*dS)
    
    #lapl_dg for lid
    lapl_dg+=(
        -inner(outer(v,n),nue*grad(ubar_knew-bc_tang))*ds((4))
        -inner(outer(ubar_knew-bc_tang,n),nue*grad(v))*ds((4)) 
        +kappa2*inner(v,ubar_knew-bc_tang)*ds((4))
    )
    
    #Advection adv_dg
    #for Picard iteration
    un = 0.5*(dot(ubar_k, n)+ abs(dot(ubar_k, n))) 
    un_pos = 0.5*(dot(ubar_k, n)+sign(dot(ubar_k, n))*dot(bc_tang,n)+abs(dot(bc_tang,n))+ abs(dot(ubar_k, n)))
   
    #for lid
    adv_dg=(dot(ubar_k,div(outer(v,ubar_knew)))*dx#like paper
            -dot(v,ubar_knew*un_pos)*ds(4)#similar to matt piggots
            -dot((v('+')-v('-')),(un('+')*(ubar_knew('+')) - un('-')*(ubar_knew('-'))))*dS)#like in the tutorial
    
    #for other thingis
    adv_dg+=-inner(v,ubar_knew*un)*ds((1,2,3)) #similar to matt piggots

    #Time derivative
    time=1/Constant(dt)*inner(v_knew-u_n,v)*dx

    #TODO: FORMS------------------------------------------- 
    eq=time-adv_dg+lapl_dg

    #form for predictor
    pres_dg_pred=dot(div(v),p_n)*dx#negative bc integration by parts!
    eq_pred=eq+pres_dg_pred

    return eq_pred

def buildPressureForm(W,U,P,dt,mesh,U_inf,bc_tang,div_old):
    print("....build pressure update")

    w,beta = TrialFunctions(W)
    v,q = TestFunctions(W)

    force_dg_pres=dot(div_old/dt ,q)*dx#sign right?
    incomp_dg_pres=div(w)*q*dx
    pres_dg_pres=div(v)*beta*dx
    
    eq_pres=dot(w,v)*dx+force_dg_pres-incomp_dg_pres+pres_dg_pres 

    #bctang

    return eq_pres

def buildCorrectorForm(W,U,P,dt,mesh,U_inf,v_knew_hat,beta):
    print("....build corrector")
    v_knew=TrialFunction(U)
    v=TestFunction(U)

    eq_corr=(
            dot(v_knew,v)*dx
            -dot(v_knew_hat,v)*dx
           # -div(v_knew)*q*dx #no need to be included because used before?
            -dt*beta*div(v)*dx
    )

    return eq_corr

def solve_problem(mesh_size,parameters_corr, parameters_pres,parameters_velo_initial, parameters_velo,parameters_4, aP=None, block_matrix=False):
    outfile=File("cavity.pvd")

    #generate mesh
    LX=1.0
    LY=1.0
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    U_inf=Constant(1)
   
    

    #function spaces
    U = FunctionSpace(mesh, "RTCF",1)
    P = FunctionSpace(mesh, "DG", 0)
    W = U*P
    
    #functions
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    f =Function(U)
	
    #normal and essentially reynolds number
    n=FacetNormal(W.mesh())
    nu=1
    nue=Constant(nu) 
    dt=0.000000001#0.0000001/(nu*(2 ** mesh_size)**2) #if higher dt predictor crashes
    T=20
    print("dt is: ",dt)

    #boundary conditions
    bc=[]
    bc0=DirichletBC(W.sub(0),Constant((0.0,0.0)),1)
    bc.append(bc0)
    bc1=DirichletBC(W.sub(0),Constant((0.0,0.0)),2)#plane x=0
    bc.append(bc1)
    bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
    bc.append(bc2)
    bc3=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
    bc.append(bc3)

    
    t=1
    x, y = SpatialCoordinate(mesh)
    bc_tang=Function(U).project(as_vector([-x*100*(x-1)/25*U_inf*(t),0]))

    print("\nCALCULATE INITIAL VALUES")########################################################

    #calculate inital value for pressure with potential flow
    u_init_sol=initialVelocity(W,U,P,mesh,nue,bc,U_inf,parameters_velo_initial,dt,bc_tang)

    #with that initial value calculate intial pressure 
    # with Poission euqation including some non-divergence free velocity
    p_init_sol=initialPressure(W,U,P,mesh,nue,bc,u_init_sol,U_inf,parameters_4,parameters_pres,dt,bc_tang)


    print("\nBUILD FORMS")#####################################################################
    v_k=Function(U)
    u_n=Function(U)
    p_n=Function(P)   
    div_old=Function(P)
    v_knew_hat=Function(U)
    beta=Function(P)
    eq_pred=buildPredictorForm(p_init_sol,u_init_sol,nue,mesh,U,P,W,U_inf,dt,bc_tang,v_k,u_n,p_n)
    eq_pres=buildPressureForm(W,U,P,dt,mesh,U_inf,bc_tang,div_old)
    eq_corr=buildCorrectorForm(W,U,P,dt,mesh,U_inf,v_knew_hat,beta)

    u_n.assign(u_init_sol)
    p_n.assign(p_init_sol)

    plot(u_init_sol)
    plt.show()
    plot(p_init_sol)
    plt.show()
    outfile.write(u_n,p_n,time=0)

    divtest=Function(P).project(div(u_n))
    print("Div error of initial velocity",errornorm(divtest,Function(P)))


    print("\nBUILD PROBLEM AND SOLVERS")########################################################
    #predictor only contains tangential BC
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    def nullspace_basis(T):
        return VectorSpaceBasis(constant=True)
    appctx = {'trace_nullspace': nullspace_basis}

    w_pred = Function(U)
    predictor = LinearVariationalProblem(lhs(eq_pred),rhs(eq_pred), w_pred)
    solver_pred = LinearVariationalSolver(predictor, solver_parameters=parameters_velo)

    #pressure
    w_pres = Function(W)
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    pressure= LinearVariationalProblem(lhs(eq_pres),rhs(eq_pres),w_pres,bc)#BC RIGHT???
    solver_pres = LinearVariationalSolver(pressure,solver_parameters=parameters_pres,appctx=appctx)
        
    #corrector
    w_corr = Function(U)
    corrector= LinearVariationalProblem(lhs(eq_corr),rhs(eq_corr), w_corr,bc)
    solver_corr = LinearVariationalSolver(corrector,solver_parameters=parameters_corr)
        
   

    print("\nTIME PROGRESSING")################################################################
    #outerloop for time progress
    t = 1
    while t < T :

        inflow=-x*100*(x-1)/25*U_inf*((1+t)*dt)### why is my inflow getting lower
        
        bc_tang.project(as_vector([inflow,0]))
        
        #innerloop for progressing Picard iteration       
        counter=0
        v_k.assign(u_n)


        print("\n1)PREDICTOR")##################################################################
        while(True):  
            
            
            solver_pred.solve()

            #convergence criterion
            eps=errornorm(v_k,w_pred)#l2 by default          
            counter+=1
            print("Picard iteration counter: ",counter)
            if(counter>5):
                print("Picard iteration converged")  
                break      
            else:
                v_k.assign(w_pred)# is this actually enough??
        

        
        plot(w_pred)
        plt.show()

        print("\n2) PRESSURE UPDATE")#########################################################
        #first modify pressure solve
        #pressure has only normal boundary=0 bc only tangential component of the lid 
        div_old_temp=Function(P).project(div(w_pred))
        div_old.assign(div_old_temp)
        print("Div error of predictor velocity",errornorm(div_old,Function(P)))
        solver_pres.solve()
        wsol,betasol=w_pres.split()
        print(assemble(betasol).dat.data)
        p_knew=Function(P).assign(p_n+betasol)

        plot(p_knew)
        plt.show()


        print("\n3) CORRECTOR")##############################################################
        #first update corrector form        
        v_knew_hat.assign(w_pred)
        beta.assign(betasol)
        
        solver_corr.solve()
        usol=Function(U).assign(w_corr)
        plot(usol)
        plt.title("Velocity")
       ## plt.xlabel("x")
       # plt.ylabel("y")
        plt.show()


        divtest=Function(P).project(div(usol))
        print("Div error of corrector velocity",errornorm(divtest,Function(P)))

        #update for next time step
        u_n.assign(usol)
        p_n.assign(p_knew)
        usol=Function(U)
        psol=Function(P)
        wsol=Function(U)
        betasol=Function(P)

        outfile.write(u_n,p_n,time=t)

        t += 1     

    sol=Function(W)
    sol.sub(0).assign(u_n)
    sol.sub(1).assign(p_n)

    divtest=Function(P)
    divtest.project(div(u_n))
    print("Div error of final solution",errornorm(divtest,Function(P)))

    N=2 ** mesh_size
    return sol,0,0,N


     
#####################MAIN##########################
#parameters_1={
#    "ksp_type": "preonly",
#   "pc_type": "mg",
#   "pc_mg_type": "full",
#   "mg_levels_ksp_type": "chebyshev",
#   "mg_levels_ksp_max_it": 2,
#   "mg_levels_pc_type": "jacobi"
#}

parameters_velo={'pc_type': 'sor',
                'ksp_type': 'gmres',
                'ksp_rtol': 1.0e-7,
                'pc_sor_symmetric': True
}


parameters_pres = { 'mat_type': 'matfree',
                    'ksp_type': 'preonly',
                    'pc_type': 'python',
                    'pc_python_type': 'firedrake.HybridizationPC',
                    'hybridization': {'ksp_type': 'preonly',
                                      'pc_type': 'lu'}
                  }

parameters_pres_better={
                     'mat_type': 'matfree',
                      'ksp_type': 'preonly',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'none',
                                        'ksp_rtol': 1e-8,
                                        'mat_type': 'matfree'}
}

parameters_2={   "ksp_type": "gmres",
                "ksp_converged_reason": None,
                 "ksp_gmres_restart":100,
                "ksp_rtol":1e-4,
                 "pc_type":"lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_type":"aij",
                "mat_mumps_icntl_14":200
}

parameters_velo_initial={
"mat_type": "matfree",
"ksp_type": "gmres",
"ksp_monitor_true_residual": None,
"ksp_view": None,
"pc_type": "fieldsplit",
"pc_fieldsplit_type": "schur",
"pc_fieldsplit_schur_fact_type": "diag",
"fieldsplit_0_ksp_type": "preonly",
"fieldsplit_0_pc_type": "python",
"fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
"fieldsplit_0_assembled_pc_type": "hypre",
"fieldsplit_1_ksp_type": "preonly",
"fieldsplit_1_pc_type": "python",
"fieldsplit_1_pc_python_type": "firedrake.MassInvPC", 
"fieldsplit_1_Mp_ksp_type": "preonly",
"fieldsplit_1_Mp_pc_type": "ilu"
}

parameters_4={
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "cg",
   "fieldsplit_0_pc_type": "ilu",
    "fieldsplit_0_ksp_rtol": 1e-8,
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_ksp_rtol": 1e-8,
    "pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_1_pc_type": "hypre"
}


parameters_corr={"ksp_type": "cg",
        'pc_type': 'ilu'
}


error_velo=[]
error_pres=[]
refin=range(4,5)
list_N=[]
for n in refin:#increasing element number
    
    #solve
    w,err_u,err_p,N = solve_problem(n, parameters_corr,parameters_pres_better,parameters_velo_initial,parameters_velo,parameters_4,aP=None, block_matrix=False)
    u,p=w.split()
    error_velo.append(err_u)
    error_pres.append(err_p)
    list_N.append(N)

    
    try:
        import matplotlib.pyplot as plt
    except:
        warning("Matplotlib not imported")

    #try:
    #    plot_velo_pres(u,p)
    #except:
    #    warning("Cannot show figure")


