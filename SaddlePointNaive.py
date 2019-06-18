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
    
    dt_max=0.001
    dt=0.001 #for lower Reynoldnumber lower dt??
    T=0.002
    theta=1
    LX=1.0
    LY=1.0
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    U_inf=1.0
    
    #max dx is 0.05 min dx is 0.001

    #function spaces
    U = FunctionSpace(mesh, "RT",1)
    P = FunctionSpace(mesh, "DG", 0)
    W = U*P
    
    #functions
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    f =Function(U)
	
    #normal and essentially reynolds number
    n=FacetNormal(W.mesh())
    nue=Constant(0.59)#re=40

    #specify inflow/solution
    x,y=SpatialCoordinate(mesh)
    t=0.0
    inflow_uniform=Function(U).project(Constant((0.0,0.0))) 

    #boundary conditions
    bc_1=[]
    bc0=DirichletBC(W.sub(0),Constant((0.0,0.0)),1)
    bc_1.append(bc0)
    bc1=DirichletBC(W.sub(0),Constant((0.0,0.0)),2)#plane x=0
    bc_1.append(bc1)
    bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
    bc_1.append(bc2)
    bc3=DirichletBC(W.sub(0),Constant((0.0,.0)),4)#plane y=L
    bc_1.append(bc3)

    #intial values
    p_n= Function(P).project(-8*1.5*(2-x))#pres for init time step #??????
    u_n=Function(U).assign(inflow) #velo for init time step
    v_k=Function(U).assign(u_n)#init Picard value vk=un


    #solve the following problem for initial values
    u_init,p_init = TrialFunctions(W)
    v_knew_hat=Function(U)
    f_pres=Function(P)
    lapl_dg_init=(nue*inner(grad(u_init),grad(v))*dx
        -inner(outer(v,n),nue*grad(u_init))*ds 
        -inner(outer(u_init,n),nue*grad(v))*ds 
        +kappa2*inner(v,u_init)*ds 
        -inner(nue*avg(grad(v)),both(outer(u_init,n)))*dS
        -inner(both(outer(v,n)),nue*avg(grad(u_init)))*dS
        +kappa1*inner(both(outer(u_init,n)),both(outer(v,n)))*dS)


    init=dot(w,v)*dx-dot(f_pres,q)*dx-div(u_init)*q*dx-div(v)*p_init*dx-lapl_dg#dt somewhere in here??

################

    p_k=Function(P).assign(p_n)#init Picard value vk=un

    ubar_k=Constant(0.5)*(u_n+v_k) #init old midstep
    v_knew,pk_new=TrialFunctions(W)
    ubar_knew=Constant(0.5)*(u_n+v_knew) #init new midstep
    
    #TODO: OPERATORS---------------------------------------
     #Picard iteration
    u_linear=Function(U).assign(inflow_uniform)

    #Laplacian
    alpha=Constant(10)#interior
    gamma=Constant(10) #exterior
    h=CellVolume(mesh)/FacetArea(mesh)  
    havg=avg(CellVolume(mesh))/FacetArea(mesh)
    kappa1=nue*alpha/havg
    kappa2=nue*gamma/h
    
    x, y = SpatialCoordinate(mesh)
    g_neg=Function(U).project(Constant((0.0,0.0)))
    g_pos=Function(U).project(as_vector([-x*100*(x-1)/25*U_inf,0]))
    #g=Constant((0.0,0.0))
    
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
    
    #Advection
    #for lid
    un = 0.5*(dot(u_linear, n)+ abs(dot(u_linear, n)))
    
    adv_dg=(dot(u_linear,div(outer(v,u)))*dx#like paper
            -dot(v,u*un)*ds(4)#similar to matt piggots
            -dot((v('+')-v('-')),(un('+')*(u('+')) - un('-')*(u('-'))))*dS)#like in the tutorial
    
    #for other thingis
    #un_neg = 0.5*(dot(u_linear, n)+sign(dot(u_linear, n))*dot(g_neg,n)+abs(dot(g_neg,n))+ abs(dot(u_linear, n)))
    adv_dg+=-inner(v,u*un)*ds((1,2,3)) #similar to matt piggots
    ############

    #Time derivative
    time=1/Constant(dt)*inner(v_knew-u_n,v)*dx

    #Incompressibility
    incomp_dg=div(v_knew)*q*dx
    
    #Body Force 
    f=Function(U)
    force_dg =dot(f,v)*dx#sign right?

    #TODO: FORMS------------------------------------------- 
    eq=time+adv_dg-lapl_dg-force_dg-incomp_dg

    #form for predictor
    pres_dg_pred=div(v)*p_k*dx
    eq_pred=eq+pres_dg_pred

    #Form for pressure correction
    w,beta = TrialFunctions(W)
    v_knew_hat=Function(U)
    f_pres=Function(P).project(div(v_knew_hat)) 
    force_dg_pres=dot(f_pres,q)*dx#sign right?
    incomp_dg_pres=div(w)*q*dx
    pres_dg_pres=div(v)*beta*dx
    
    eq_pres=dot(w,v)*dx+force_dg_pres+1/dt*incomp_dg_pres-pres_dg_pres #dt somewhere in here??

    #Form for corrector
    p_k_update=Function(P)
    pres_dg_corr=div(v)*p_k_update*dx

    eq_corr=eq+dt*pres_dg_corr

    #TODO: LOOPS------------------------------------------------------------

    #outerloop for time progress
    t = dt
    while t < T :

        #for time dependent inflow
       # inflow_expr=as_vector((-100*(y-1)*(y),0.0*y))
       # inflow=Function(U).project(inflow_expr)#changed to time dependent
       # infl=DirichletBC(W.sub(0),inflow,2)
        
        #innerloop for progressing Picard iteration
        nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])

        #build problem and solver
        # predictor
        print("\n....predictor solve\n")
        w_pred = Function(W)
        predictor = LinearVariationalProblem(lhs(eq_pred),rhs(eq_pred), w_pred,[infl,noslip])
        solver = LinearVariationalSolver(predictor, solver_parameters=parameters)
       
        counter=0
        dt=theta*dt_max
        while(True):
            
            solver.solve()
            usolhat,psolhat=w_pred.split() 
            
            # plot solutionfields
            plot_velo_pres(Function(U).project(usolhat),Function(P).project(psolhat),"Solution")

            #convergence criterion
            eps=errornorm(v_k,usolhat)#l2 by default
            v_k.assign(usolhat)
            p_k.assign(p_n)
            counter+=1
            print("Picard iteration error",eps,", counter: ",counter)
            if(counter>2):
                print("Picard iteration converged")  
                break      
            else:
                #?????????
                u_linear.project(u1)
                u1.assign(0.)
                p1.assign(0.)    
            
        
        dt=dt_max
        #PRESSURE UPDATE
        print("\n....update solve\n")
        #first modify pressure solve
        eq_pres_new=replace(eq_pres,{v_knew_hat:v_k})
        #amg as preconditioner?
        w_pres = Function(W)
        nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
        pressure= LinearVariationalProblem(lhs(eq_pres_new),rhs(eq_pres_new),w_pres,[infl,noslip,outfl])#BC RIGHT???
        solver = LinearVariationalSolver(pressure,solver_parameters=parameters)
        solver.solve()
        wsol,betasol=w_pres.split()
        print(assemble(betasol).dat.data)
        p_knew=Function(P).project(p_n+betasol/dt)#or pk??????????

        #v_knew=Function(U).project(usolhat+grad(betasol))

        #VELOCITY CORRECTION
        print("\n.....corrector solve\ns")
        #first update corrector form
        p_k_update.assign(p_knew)
        #v_k already updated
        w_corr = Function(W)
        corrector= LinearVariationalProblem(lhs(eq_corr),rhs(eq_corr), w_corr,[infl,noslip])
        solver = LinearVariationalSolver(corrector, solver_parameters=parameters)
        solver.solve()
        usol,psol=w_corr.split()

        plot(usol)
        plt.title("Velocity")
        plt.xlabel("x")
        plt.ylabel("y")
       # plt.show()
        plot(p_knew)
        plt.title("Pressure")
        plt.xlabel("x")
        plt.ylabel("y")
        #plt.show()

        u_n.assign(usol)
        p_n.assign(p_knew)

        t += dt      

    sol=Function(W)
    sol.sub(0).assign(u_n)
    sol.sub(1).assign(p_n)

    divtest=Function(P)
    divtest.project(div(u_n))
    print("Div error",errornorm(divtest,Function(P)))

    N=2 ** mesh_size
    return w,0,0,N


     
#####################MAIN##########################
parameters={
   # "ksp_type": "fgmres",
   # "ksp_rtol": 1e-8,
   # "pc_type": "fieldsplit",
   # "pc_fieldsplit_type": "schur",
   # "pc_fieldsplit_schur_fact_type": "full",
   # "fieldsplit_0_ksp_type": "cg",
    #"fieldsplit_0_pc_type": "ilu",
   # "fieldsplit_0_ksp_rtol": 1e-8,
   # "fieldsplit_1_ksp_type": "cg",
   # "fieldsplit_1_ksp_rtol": 1e-8,
   # "pc_fieldsplit_schur_precondition": "selfp",
   # "fieldsplit_1_pc_type": "hypre"
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
   "ksp_gmres_restart":500,
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
    w,err_u,err_p,N = solve_problem(n, parameters,aP=None, block_matrix=False)
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


