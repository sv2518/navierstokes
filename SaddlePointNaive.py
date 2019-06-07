from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

def both(expr):
    return expr('+') + expr('-')


def solve_problem(mesh_size, parameters, aP=None, block_matrix=False):
    #generate mesh
    LX=10
    LY=1
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    dt=0.1
    T=3
    
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

    #intial values
    p_n= Function(P).project(x)#pres for init time step #??????
    u_n=Function(U).assign(inflow) #velo for init time step
    v_k=Function(U).assign(u_n)#init Picard value vk=un
    p_k=Function(P).assign(p_n)#init Picard value vk=un

    ubar_k=Constant(0.5)*(u_n+v_k) #init old midstep
    v_knew,pk_new=TrialFunctions(W)
    ubar_knew=Constant(0.5)*(u_n+v_knew) #init new midstep
    
    #TODO: OPERATORS---------------------------------------
    #Advection operator
    un = 0.5*(dot(ubar_k, n) + abs(dot(ubar_k, n)))#conditional for upwind discretisation
    adv_dg=-(dot(ubar_k,div(outer(v,ubar_knew)))*dx#like paper
        -inner(v,(ubar_knew*dot(ubar_k,n)))*ds#similar to matt piggots
        -dot((v('+')-v('-')),(un('+')*ubar_knew('+') - un('-')*ubar_knew('-')))*dS)#like in the tutorial

    #Laplacian operator
    alpha=Constant(10.)
    gamma=Constant(10.) 
    kappa1=nue * alpha/Constant(LX/2**mesh_size)
    kappa2=nue * gamma/Constant(LX/2**mesh_size)
    lapl_dg=(nue*inner(grad(ubar_knew),grad(v))*dx
        -inner(outer(v,n),nue*grad(ubar_knew))*ds 
        -inner(outer(ubar_knew,n),nue*grad(v))*ds 
        +kappa2*inner(v,ubar_knew)*ds 
        -inner(nue*avg(grad(v)),both(outer(ubar_knew,n)))*dS
        -inner(both(outer(v,n)),nue*avg(grad(ubar_knew)))*dS
        +kappa1*inner(both(outer(ubar_knew,n)),both(outer(v,n)))*dS)

    #Time derivative
    time=-1/Constant(dt)*inner(v_knew-u_n,v)*dx

    #Incompressibility
    incomp_dg=div(v_knew)*q*dx
    
    #Body Force 
    f=Function(U)
    force_dg =-dot(f,v)*dx#sign right?

    #TODO: FORMS------------------------------------------- 
    eq=time+adv_dg+lapl_dg+force_dg+incomp_dg

    #form for predictor
    pres_dg_pred=-div(v)*p_k*dx
    eq_pred=eq+pres_dg_pred

    #Form for pressure correction
    w,beta = TrialFunctions(W)
    v_knew_sol=Function(U)
    f_pres=div(v_knew_sol) 
    force_dg_pres=dot(f_pres,q)*dx#sign right?
    incomp_dg_pres=div(w)*q*dx
    pres_dg_pres=div(v)*beta*dx
    
    eq_pres=dot(w,v)*dx+force_dg_pres+incomp_dg_pres+pres_dg_pres #dt somewhere in here??

    #Form for corrector
    p_k_update=Function(P)
    pres_dg_corr=div(v)*p_k_update*dx

    eq_corr=eq+pres_dg_corr

    #TODO: LOOPS------------------------------------------------------------

    #outerloop for time progress
    t = 0.0
    while t < T - 0.5*dt:
        
        #innerloop for progressing Picard iteration 
        counter=0
        while(True):

            #PREDICTOR
            #build problem and solver (maybe also outside??)
            print("\n....predictor solve\n")
            w_pred = Function(W)
            predictor = LinearVariationalProblem(lhs(eq_pred),rhs(eq_pred), w_pred, [noslip_bottom,noslip_top])
            solver = LinearVariationalSolver(predictor, solver_parameters=parameters)
            solver.solve()
            usolhat,psolhat=w_pred.split()
              
            
            #PRESSURE UPDATE
            print("\n....update solve\n")
            #first modify pressure solve
            eq_pres=replace(eq_pres,{v_knew_sol:usolhat})
            #amg as preconditioner?
            nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
            w_pres = Function(W)
            pressure= LinearVariationalProblem(lhs(eq_pres),rhs(eq_pres),w_pres, [noslip_bottom,noslip_top])
            solver = LinearVariationalSolver(pressure, nullspace=nullspace,solver_parameters=parameters)
            solver.solve()
            wsol,betasol=w_pres.split()
            p_knew=Function(P).project(p_n+betasol/dt)
            v_knew=Function(U).project(usolhat+grad(betasol))

            #VELOCITY CORRECTION
            print("\n.....corrector solve\ns")
            #first update corrector form
            eq_corr=replace(eq_pres,{p_k_update:p_knew})

            nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
            w_corr = Function(W)
            corrector= LinearVariationalProblem(lhs(eq_corr),rhs(eq_corr), w_corr, bc)
            solver = LinearVariationalSolver(corrector, nullspace=nullspace,solver_parameters=parameters)
            solver.solve()
            usol,psol=w_corr.split()

            #convergence criterion
            #eps=errornorm(u1,u_linear)#l2 by default
            counter+=1
            print("Picard iteration error",counter,", counter: ",counter)
            if(counter>(0)):
                print("Picard iteration converged")  
                break          
            else:
                v_k.assign(usol)
                p_k.assign(psol)

        u_n.assign(usol)
        p_n.assign(psol)
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
    return w_corr#,conv,d_x

#
parameters={
    "ksp_type": "gmres",
    "ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "mg",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "__main__.Mass",
    "fieldsplit_1_aux_pc_type": "bjacobi",
    "fieldsplit_1_aux_sub_pc_type": "icc"
    }
print("Channel Flow")
print("Cell number","IterationNumber")

convergence=[]
refin=range(4,5)
delta_x=[]
for n in refin:#increasing element number
    
    #solve
    w= solve_problem(n, parameters,aP=None, block_matrix=False)
    u,p=w.split()
    #convergence.append(conv)
    #delta_x.append(d_x)

    
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

#print("max error in velocity",convergence)

#convergence plot
#fig = plt.figure()
#axis = fig.gca()
#linear=convergence
#axis.loglog(refin,linear)
#axis.plot(refin,refin[::-1],'r*')
#axis.set_xlabel('$Level$')
#axis.set_ylabel('$||e||_{\infty}$')
#plt.show()