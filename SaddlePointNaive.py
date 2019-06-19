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
def initialVelocity(W,U,P,mesh,nue,bc,U_inf,parameters,dt):
    print("....Solving Stokes problem for initial velocity ....")

    #Functions and parameters
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    n=FacetNormal(W.mesh())

    v_knew_hat=Function(U)
    f_pres=Function(P)

    alpha=Constant(10.)
    gamma=Constant(10.) 
    h=CellVolume(mesh)/FacetArea(mesh)
    havg=avg(CellVolume(mesh))/FacetArea(mesh)
    kappa1=nue * alpha/havg
    kappa2=nue * gamma/h

    #build form
     #Laplacian
    alpha=Constant(10)#interior
    gamma=Constant(10) #exterior
    h=CellVolume(mesh)/FacetArea(mesh)  
    havg=avg(CellVolume(mesh))/FacetArea(mesh)
    kappa1=nue*alpha/havg
    kappa2=nue*gamma/h
    
    x, y = SpatialCoordinate(mesh)
    g_pos=Function(U).project(as_vector([-x*100*(x-1)/25*U_inf,0]))
    
    #a_dg for other sides
    a_dg=(nue*inner(grad(u),grad(v))*dx
          -inner(outer(v,n),nue*grad(u))*ds((1,2,3)) 
          -inner(outer(u,n),nue*grad(v))*ds((1,2,3))
          +kappa2*inner(v,u)*ds((1,2,3))
          -inner(nue*avg(grad(v)),both(outer(u,n)))*dS
          -inner(both(outer(v,n)),nue*avg(grad(u)))*dS
          +kappa1*inner(both(outer(u,n)),both(outer(v,n)))*dS)
    
    #a_dg for lid
    a_dg+=(
        -inner(outer(v,n),nue*grad(u-g_pos))*ds((4))
        -inner(outer(u-g_pos,n),nue*grad(v))*ds((4)) 
        +kappa2*inner(v,u-g_pos)*ds((4))
    )


    eq_init=dot(u,v)*dx-dot(f_pres,q)*dx-1/dt*div(u)*q*dx-div(v)*p*dx+a_dg#dt somewhere in here??
 #########include gpos somewhere else her??????

    #solve
    w_init= Function(W)
    init = LinearVariationalProblem(lhs(eq_init),rhs(eq_init), w_init,bc)
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    solver = LinearVariationalSolver(init, solver_parameters=parameters)
    solver.solve()
    u_init_sol,p_init_sol=w_init.split()
    
    return u_init_sol

def initialPressure(W,U,P,mesh,nue,bc,u_init,U_inf,parameters,dt):
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
    
    x, y = SpatialCoordinate(mesh)
    g_pos=Function(U).project(as_vector([-x*100*(x-1)/25*U_inf,0]))
    
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
        -inner(outer(v,n),nue*grad(u-g_pos))*ds((4))
        -inner(outer(u-g_pos,n),nue*grad(v))*ds((4)) 
        +kappa2*inner(v,u-g_pos)*ds((4))
    )

    #Advection
    #for lid
    u_linear=Function(U).assign(u_init)
    un_pos = 0.5*(dot(u_linear, n)+sign(dot(u_linear, n))*dot(g_pos,n)+abs(dot(g_pos,n))+ abs(dot(u_linear, n)))
    un = 0.5*(dot(u_linear, n)+ abs(dot(u_linear, n)))    
    adv_dg=(
        dot(u_linear,div(outer(v,u)))*dx#like paper
            -dot(v,u*un_pos)*ds(4)#similar to matt piggots
            -dot((v('+')-v('-')),(un('+')*(u('+')) - un('-')*(u('-'))))*dS
        )#like in the tutorial
    
    #for other thingis
    adv_dg+=-inner(v,u*un)*ds((1,2,3)) #similar to matt piggots

    eq_init=dot(v,F)*dx-adv_dg+a_dg#-div(F)*q*dx+dot(Function(U).project(Constant((1,0))),v)*dx#dt somewhere in here??

    #solve
    w_init= Function(W)
    init = LinearVariationalProblem(lhs(eq_init),rhs(eq_init), w_init,bc)
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    solver = LinearVariationalSolver(init, nullspace=nullspace,solver_parameters=parameters)
    solver.solve()
    F_init_sol,p_init_sol=w_init.split()

    print("........part2: solve mixed possion problem for initial pressure ")
    
    w,beta = TrialFunctions(W)
    f_pres=Function(P).project(div(F_init_sol)) 
    force_dg_pres=dot(f_pres,q)*dx#sign right?
    incomp_dg_pres=div(w)*q*dx
    pres_dg_pres=div(v)*beta*dx    
    eq_pres=dot(w,v)*dx-force_dg_pres-1/dt*incomp_dg_pres+pres_dg_pres #dt somewhere in here??


    w_init= Function(W)
    init = LinearVariationalProblem(lhs(eq_pres),rhs(eq_pres), w_init,bc)
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    solver = LinearVariationalSolver(init, nullspace=nullspace,solver_parameters=parameters)
    solver.solve()
    u_init_sol,p_init_sol=w_init.split()
 
    return p_init_sol

def buildPredictorForm(p_init_sol,u_init_sol,nue,mesh,U,P,W,U_inf,dt):
    print("....build predictor")

    v_knew,pk_new=TrialFunctions(W)
    v,q=TestFunctions(W)
    n=FacetNormal(W.mesh())

    #function
    p_n= Function(P).assign(p_init_sol)#pres for init time step #??????
    u_n=Function(U).assign(u_init_sol) #velo for init time step
    v_k=Function(U).assign(u_n)#init Picard value vk=un
    p_k=Function(P).assign(p_n)#init Picard value vk=un

    ubar_k=Constant(0.5)*(u_n+v_k) #init old midstep
    ubar_knew=Constant(0.5)*(u_n+v_knew) #init new midstep
    
    #Laplacian a_dg
    alpha=Constant(10)#interior
    gamma=Constant(10) #exterior
    h=CellVolume(mesh)/FacetArea(mesh)  
    havg=avg(CellVolume(mesh))/FacetArea(mesh)
    kappa1=nue*alpha/havg
    kappa2=nue*gamma/h
    
    x, y = SpatialCoordinate(mesh)
    g_pos=Function(U).project(Constant((0.0,0.00)))
    
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
        -inner(outer(v,n),nue*grad(ubar_knew-g_pos))*ds((4))
        -inner(outer(ubar_knew-g_pos,n),nue*grad(v))*ds((4)) 
        +kappa2*inner(v,ubar_knew-g_pos)*ds((4))
    )
    
    #Advection adv_dg
    #for Picard iteration
    u_linear=Function(U).assign(ubar_k)
    un = 0.5*(dot(u_linear, n)+ abs(dot(u_linear, n))) 
    un_pos = 0.5*(dot(u_linear, n)+sign(dot(u_linear, n))*dot(g_pos,n)+abs(dot(g_pos,n))+ abs(dot(u_linear, n)))
   
    #for lid
    adv_dg=(dot(u_linear,div(outer(v,ubar_knew)))*dx#like paper
            -dot(v,ubar_knew*un_pos)*ds(4)#similar to matt piggots
            -dot((v('+')-v('-')),(un('+')*(ubar_knew('+')) - un('-')*(ubar_knew('-'))))*dS)#like in the tutorial
    
    #for other thingis
    adv_dg+=-inner(v,ubar_knew*un)*ds((1,2,3)) #similar to matt piggots

    ###

    #Time derivative
    time=1/Constant(dt)*inner(v_knew-u_n,v)*dx

    #Incompressibility
    incomp_dg=div(v_knew)*q*dx
    
    #Body Force 
    f=Function(U)
    force_dg =dot(f,v)*dx

    #TODO: FORMS------------------------------------------- 
    eq=time+adv_dg-lapl_dg-force_dg-incomp_dg

    #form for predictor
    pres_dg_pred=div(v)*p_k*dx
    eq_pred=eq+pres_dg_pred

    return eq_pred,v_k,u_n,p_n

def buildPressureForm(W,U,P,dt):
    print("....build pressure update")
    w,beta = TrialFunctions(W)
    v,q = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)
    g_pos=Function(U).project(as_vector([-x*100*(x-1)/25*U_inf,0]))


    v_knew_hat=Function(U)
    f_pres=Function(P).project(div(v_knew_hat)) 
    force_dg_pres=dot(f_pres,q)*dx#sign right?
    incomp_dg_pres=div(w)*q*dx
    pres_dg_pres=div(v)*beta*dx
    
    eq_pres=dot(w,v)*dx-dot(w-g_pos,v)*ds-force_dg_pres-1/dt*incomp_dg_pres+pres_dg_pres #dt somewhere in here??

    return eq_pres,v_knew_hat

def buildCorrectorForm(W,U,P,dt):
    print("....build corrector")
    v_knew,p_knew=TrialFunctions(W)
    v,q=TestFunctions(W)

    v_knew_hat2=Function(U)
    beta=Function(P)

    eq_corr=(dot(v_knew,v)-dot(v_knew_hat2,v)-1/dt*div(v_knew_hat2)*q+dt*dot(grad(beta),v))*dx

    return eq_corr,v_knew_hat2,beta


def solve_problem(mesh_size, parameters_1,parameters_2, aP=None, block_matrix=False):
    #generate mesh
    LX=1.0
    LY=1.0
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    U_inf=Constant(1.0)
    
    #max dx is 0.05 min dx is 0.001
    dt=0.00000001 #for lower Reynoldnumber lower dt??
    T=0.00000002
    

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
    nue=Constant(1)#re=40

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

    print("\nCALCULATE INITIAL VALUES")########################################################
    #calculate inital value for pressure with potential flow
    u_init_sol=initialVelocity(W,U,P,mesh,nue,bc,U_inf,parameters_2,dt)

    #with that initial value calculate intial pressure 
    # with Poission euqation including some non-divergence free velocity
    p_init_sol=initialPressure(W,U,P,mesh,nue,bc,u_init_sol,U_inf,parameters_1,dt)
    p_n=Function(P).assign(p_init_sol)

    print("\nBUILD FORMS")#####################################################################
    eq_pred,v_k,u_n,p_n=buildPredictorForm(p_init_sol,u_init_sol,nue,mesh,U,P,W,U_inf,dt)
    eq_pres,v_knew_hat=buildPressureForm(W,U,P,dt)
    eq_corr,v_knew_hat2,beta=buildCorrectorForm(W,U,P,dt)


    print("\nTIME PROGRESSING")################################################################
    #outerloop for time progress
    t = dt
    while t < T :
        
        #innerloop for progressing Picard iteration
        nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])

        #build problem and solver
        w_pred = Function(W)
        predictor = LinearVariationalProblem(lhs(eq_pred),rhs(eq_pred), w_pred)
        solver = LinearVariationalSolver(predictor, solver_parameters=parameters_1)
       
        counter=0

        print("\n1)PREDICTOR")##################################################################
        while(True):
            
            solver.solve()
            usolhat,psolhat=w_pred.split() 

            #convergence criterion
            eps=errornorm(v_k,usolhat)#l2 by default          
            counter+=1
            print("Picard iteration error",eps,", counter: ",counter)
            if(counter>1):
                print("Picard iteration converged")  
                break      
            else:
                v_k.assign(usolhat)# is this actually enough??
                usolhat.assign(0.)
                psolhat.assign(0.)    
            
        plot(usolhat)
        plt.show()

        divtest=Function(P)
        divtest.project(div(usolhat))
        print("Div error of predictor velocity",errornorm(divtest,Function(P)))
        
    
        print("\n2) PRESSURE UPDATE")#########################################################
        #first modify pressure solve
        v_knew_hat.assign(usolhat)
        w_pres = Function(W)
        nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
        bc_1=[]
        bc0=DirichletBC(W.sub(0),Constant((0.0,0.0)),1)
        bc_1.append(bc0)
        bc1=DirichletBC(W.sub(0),Constant((0.0,0.0)),2)#plane x=0
        bc_1.append(bc1)
        bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
        bc_1.append(bc2)
        bc3=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
        bc_1.append(bc3)
        #bc4=DirichletBC(W.sub(1),Constant(0.0),3)
        #bc_1.append(bc4)
        pressure= LinearVariationalProblem(lhs(eq_pres),rhs(eq_pres),w_pres,bc_1)#BC RIGHT???
        solver = LinearVariationalSolver(pressure,nullspace=nullspace,solver_parameters=parameters_2)
        solver.solve()
        wsol,betasol=w_pres.split()
        print(assemble(betasol).dat.data)
        p_knew=Function(P).project(p_n+betasol/dt)

        plot(p_knew)
        plt.show()


        print("\n3) CORRECTOR")##############################################################
        #first update corrector form        
        v_knew_hat.assign(usolhat)
        beta.assign(betasol)
        
        w_corr = Function(W)
        corrector= LinearVariationalProblem(lhs(eq_corr),rhs(eq_corr), w_corr,bc)
        solver = LinearVariationalSolver(corrector, nullspace=nullspace,solver_parameters=parameters_1)
        solver.solve()
        usol,psol=w_corr.split()

        plot(usol)
        plt.title("Velocity")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plot(p_knew)
        plt.title("Pressure")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        #update for next time step
        u_n.assign(usol)
        p_n.assign(p_knew)

        t += dt      

    sol=Function(W)
    sol.sub(0).assign(u_n)
    sol.sub(1).assign(p_n)

    divtest=Function(P)
    divtest.project(div(u_n))
    print("Div error of final solution",errornorm(divtest,Function(P)))

    N=2 ** mesh_size
    return sol,0,0,N


     
#####################MAIN##########################
parameters_1={
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


parameters_2={   "ksp_type": "gmres",
    "ksp_converged_reason": None,
   "ksp_gmres_restart":100,
  "ksp_rtol":1e-12,
    "pc_type":"lu",
  "pc_factor_mat_solver_type": "mumps",
  "mat_type":"aij"
}

error_velo=[]
error_pres=[]
refin=range(5,6)
list_N=[]
for n in refin:#increasing element number
    
    #solve
    w,err_u,err_p,N = solve_problem(n, parameters_1,parameters_2,aP=None, block_matrix=False)
    u,p=w.split()
    error_velo.append(err_u)
    error_pres.append(err_p)
    list_N.append(N)

    
    #plot solutions
    File("cavity.pvd").write(u,p)
    try:
        import matplotlib.pyplot as plt
    except:
        warning("Matplotlib not imported")

    #try:
    #    plot_velo_pres(u,p)
    #except:
    #    warning("Cannot show figure")


