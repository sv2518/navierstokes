from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

def both(expr):
    return expr('+') + expr('-')


def solve_problem(mesh_size, parameters, aP=None, block_matrix=False):
    #generate mesh
    LX=100
    LY=1
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    dt=0.5
    T=5
    
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
    nue=Constant(1)

    #specify inflow/initial solution
    x,y=SpatialCoordinate(mesh)
    inflow=Function(U).project(as_vector((-0.5*(y-1)*(y),0.0*y)))
    inflow_uniform=Function(U).project(Constant((1.0,0.0)))  

    #boundary conditions
    bc=[]
    infl=DirichletBC(W.sub(0),inflow,1)#plane x=0
    bc.append(infl)
    noslip_bottom=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
    bc.append(noslip_bottom)
    noslip_top=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
    bc.append(noslip_top)

    #TODO: OPERATORS---------------------------------------------------------    
    #Laplacian operator
    #stability params
    alpha=Constant(10.)
    gamma=Constant(10.) 
    kappa1=nue * alpha/Constant(LX/2**mesh_size)
    kappa2=nue * gamma/Constant(LX/2**mesh_size)
    lapl_dg=(nue*inner(grad(u),grad(v))*dx
        -inner(outer(v,n),nue*grad(u))*ds 
        -inner(outer(u,n),nue*grad(v))*ds 
        +kappa2*inner(v,u)*ds 
        -inner(nue*avg(grad(v)),both(outer(u,n)))*dS
        -inner(both(outer(v,n)),nue*avg(grad(u)))*dS
        +kappa1*inner(both(outer(u,n)),both(outer(v,n)))*dS)
         
    #Advection operator
    #conditional for upwind discretisation
    u_linear=Function(U)
    un = 0.5*(dot(u_linear, n) + abs(dot(u_linear, n))) 
    adv_dg=-(dot(u_linear,div(outer(v,u)))*dx#like paper
        -inner(v,(u*dot(u_linear,n)))*ds#similar to matt piggots
        -dot((v('+')-v('-')),(un('+')*u('+') - un('-')*u('-')))*dS)#like in the tutorial

    #Pressure Gradient
    pres_dg=div(v)*p*dx
    
    #Incompressibility
    incomp_dg=div(u)*q*dx
    
    #Body Force 
    f=Function(U).project(Constant((0.0, 0.0)))
    force_dg =-dot(f,v)*dx

    #Time derivative
    u0=Function(U)
    time=Constant(dt)*inner((u-u0),v)*dx

    #TODO: FORMS-------------------------------------------------------------
    p= Function(P).assign(Constant(1.0))#pres for init time step #??????
    u_n=Function(U).assign(inflow) #velo for init time step
    v_k=Function(U).assign(u_n)#init Picard value vk=un
    ubar_k=Constant(0.5)*(u_n+v_k) #init midstep
    v_knew,pk_new=TrialFunctions(W)
    ubar_knew=Constant(0.5)*(u_n+v_knew) #init midstep
    
    #Form for predictor
    lapl_dg_pred=replace(lapl_dg, {u: ubar_knew})
    adv_dg_pred=replace(adv_dg, {u: ubar_knew})
    adv_dg_pred=replace(adv_dg, {u_linear: ubar_k})
    incomp_dg_pred=replace(incomp_dg,{u:v_knew})
    time_pred=replace(time,{u:v_knew})
    time_pred=replace(time_pred,{u0:u_n})

    eq_pred=time_pred+adv_dg_pred+pres_dg+lapl_dg_pred+force_dg+incomp_dg_pred
    predictor=lhs(eq_pred)-rhs(eq_pred)

    #Form for pressure correction
    w,beta = TrialFunctions(W)
    f_pres=div(v_knew)
    force_dg_pres=replace(force_dg,{f:f_pres})
    incomp_dg_pres=replace(incomp_dg,{u:w})
    pres_dg_pres=replace(pres_dg,{p:beta})
    
    eq_pres=dot(w,v)+force_dg_pres+incomp_dg_pres+pres_dg_pres
    pressure=lhs(eq_pres)-rhs(eq_pres)

    #Form for corrector
    v_knew,p_knew= TrialFunctions(W)
    pres_dg_corr=replace(pres_dg,{p:p_knew})

    eq_corr=time_pred+adv_dg_pred+pres_dg_corr+lapl_dg_pred+force_dg+incomp_dg_pred
    corrector=lhs(eq_corr)-rhs(eq_corr)

    #TODO: LOOPS------------------------------------------------------------

    #outerloop for time progress
    t = 0.0
    while t < T - 0.5*dt:
        
        #innerloop for progressing Picard iteration 
        counter=0
        while(True):

            #evaluate  predictor 
            #build problem and solver (maybe also outside??)
            nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
            w_pred = Function(W)
            problem = LinearVariationalProblem(a, L, w_pred, bc)
            solver = LinearVariationalSolver(predictor, nullspace=nullspace,solver_parameters=parameters)
            solver.solve()
            usolhat,psolhat=w_pred.split()
            v_knew.assign(usolhat)

            #evaluate pressure update
            #amg as preconditioner?
            nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
            w_pres = Function(W)
            problem = LinearVariationalProblem(a, L, w_pres, bc)
            solver = LinearVariationalSolver(predictor, nullspace=nullspace,solver_parameters=parameters)
            solver.solve()
            wsol,betasol=w_pres.split()
            p_knew.assign(p+betasol/dt)

            #evaluate velocity correction
            nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
            w_corr = Function(W)
            problem = LinearVariationalProblem(a, L, w_corr, bc)
            solver = LinearVariationalSolver(predictor, nullspace=nullspace,solver_parameters=parameters)
            solver.solve()
            usol,psol=w_corr.split()

            #convergence criterion
            #eps=errornorm(u1,u_linear)#l2 by default
            #counter+=1
            #print("Picard iteration error",eps,", counter: ",counter)
            if(eps<10**(-5)):
                print("Picard iteration converged")  
                break          
            else:
                v_k.assign(u1)
                p.assign(psol)

        un.assign(usol)
        p.assign(psol)
        t += dt
            
        

        

        

    #method of manufactured solutions
    #test=Function(W)
    #p_sol=Function(P).project(-(x-50)/50*4)
    #test.sub(0).assign(inflow)
    #test.sub(1).assign(p_sol)
    # plt.plot((assemble(action(a-L,w),bcs=bc_1).dat.data[0]))
    #plt.plot((assemble(action(a-L-action(a-L,test),test),bcs=bc_1).dat.data[0]))
    #plt.show()

    #conv=max(abs(assemble(action(a-L-action(a,test),test),bcs=bc_1).dat.data[0]))
    #d_x=LX/2**mesh_size
    return w#,conv,d_x

#
parameters={
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_gmres_restart":100,
    "ksp_rtol":1e-12,
    "pc_type":"lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_type":"aij"}
print("Channel Flow")
print("Cell number","IterationNumber")

convergence=[]
refin=range(4,9)
delta_x=[]
for n in refin:#increasing element number
    
    #solve
    w,conv,d_x = solve_problem(n, parameters,aP=None, block_matrix=False)
    u,p=w.split()
    convergence.append(conv)
    delta_x.append(d_x)

    
    #plot solutions
    File("poisson_mixed_velocity_.pvd").write(u)
    File("poisson_mixed_pressure_.pvd").write(p)
    try:
        import matplotlib.pyplot as plt
    except:
        warning("Matplotlib not imported")

    #plot solutions
    try:
        plot(u)
        plt.title("Velocity")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plot(p)
        plt.title("Pressure")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    except:
        warning("Cannot show figure")

print("max error in velocity",convergence)

#convergence plot
fig = plt.figure()
axis = fig.gca()
linear=convergence
axis.loglog(refin,linear)
axis.plot(refin,refin[::-1],'r*')
axis.set_xlabel('$Level$')
axis.set_ylabel('$||e||_{\infty}$')
plt.show()