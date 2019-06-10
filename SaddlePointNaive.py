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
    dt_max=0.001
    dt=0.001 #for lower Reynoldnumber lower dt??
    T=0.005
    theta=0.25
    
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
    nue=Constant(0.01)

    #specify inflow/initial solution
    x,y=SpatialCoordinate(mesh)
    t=0.0
    inflow=Function(U).project(as_vector((-sin(3.0*(1-t))*(y-1)*(y),0.0*y)))#changed to time dependent
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
    p_n= Function(P).project(Constant(0))#pres for init time step #??????
    u_n=Function(U).assign(inflow) #velo for init time step
    v_k=Function(U).assign(u_n)#init Picard value vk=un
    p_k=Function(P).assign(p_n)#init Picard value vk=un

    ubar_k=Constant(0.5)*(u_n+v_k) #init old midstep
    v_knew,pk_new=TrialFunctions(W)
    ubar_knew=Constant(0.5)*(u_n+v_knew) #init new midstep
    
    #TODO: OPERATORS---------------------------------------
    #Advection operator
    un = 0.5*(dot(ubar_k, n) + abs(dot(ubar_k, n)))#conditional for upwind discretisation
    adv_dg=(dot(ubar_k,div(outer(v,ubar_knew)))*dx#like paper
        -inner(v,(ubar_knew*dot(ubar_k,n)))*ds#similar to matt piggots
        -dot((v('+')-v('-')),(un('+')*ubar_knew('+') - un('-')*ubar_knew('-')))*dS)#like in the tutorial

    #Laplacian operator
    alpha=Constant(10.)
    gamma=Constant(10.) 
    h=CellVolume(mesh)/FacetArea(mesh)
    havg=avg(CellVolume(mesh))/FacetArea(mesh)
    kappa1=nue * alpha/havg
    kappa2=nue * gamma/h
    lapl_dg=(nue*inner(grad(ubar_knew),grad(v))*dx
        -inner(outer(v,n),nue*grad(ubar_knew))*ds 
        -inner(outer(ubar_knew,n),nue*grad(v))*ds 
        +kappa2*inner(v,ubar_knew)*ds 
        -inner(nue*avg(grad(v)),both(outer(ubar_knew,n)))*dS
        -inner(both(outer(v,n)),nue*avg(grad(ubar_knew)))*dS
        +kappa1*inner(both(outer(ubar_knew,n)),both(outer(v,n)))*dS)

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
    
    eq_pres=dot(w,v)*dx+force_dg_pres+1/dt*incomp_dg_pres+pres_dg_pres #dt somewhere in here??

    #Form for corrector
    p_k_update=Function(P)
    pres_dg_corr=div(v)*p_k_update*dx

    eq_corr=eq+pres_dg_corr

    #TODO: LOOPS------------------------------------------------------------

    #outerloop for time progress
    t = 0.0
    while t < T :
        
        inflow.t=t

        #innerloop for progressing Picard iteration DO WE NEED THIS?
        counter=0
        dt=theta*dt_max
        while(True):

            #PREDICTOR
            #build problem and solver (maybe also outside??)
            print("\n....predictor solve\n")
            w_pred = Function(W)
            predictor = LinearVariationalProblem(lhs(eq_pred),rhs(eq_pred), w_pred,[noslip_bottom,noslip_top])
            solver = LinearVariationalSolver(predictor, solver_parameters=parameters)
            solver.solve()
            usolhat,psolhat=w_pred.split()
              
          #  plot(psolhat)
          #  plt.title("Pressure")
          ##  plt.xlabel("x")
          #  plt.ylabel("y")
          #  plt.show()

            #convergence criterion
            eps=errornorm(v_k,usolhat)#l2 by default
            v_k.assign(usolhat)
            p_k.assign(psolhat)
            counter+=1
            print("Picard iteration error",eps,", counter: ",counter)
            if(counter>dt_max/dt):
                print("Picard iteration converged")  
                break          
            
        
        dt=dt_max
        #PRESSURE UPDATE
        print("\n....update solve\n")
        #first modify pressure solve
        v_knew_hat.assign(v_k)
        #amg as preconditioner?
        w_pres = Function(W)
        pressure= LinearVariationalProblem(lhs(eq_pres),rhs(eq_pres),w_pres,[infl,DirichletBC(W.sub(1),Constant((0.0)),2)])#BC RIGHT???
        solver = LinearVariationalSolver(pressure, solver_parameters=parameters)
        solver.solve()
        wsol,betasol=w_pres.split()
        print(assemble(betasol).dat.data)
        p_knew=Function(P).project(p_k+betasol/dt)
        #v_knew=Function(U).project(usolhat+grad(betasol))

        #VELOCITY CORRECTION
        print("\n.....corrector solve\ns")
        #first update corrector form
        p_k_update.assign(p_knew)
        #v_k already updated
        w_corr = Function(W)
        corrector= LinearVariationalProblem(lhs(eq_corr),rhs(eq_corr), w_corr, [noslip_bottom,noslip_top])
        solver = LinearVariationalSolver(corrector, solver_parameters=parameters)
        solver.solve()
        usol,psol=w_corr.split()

       # plot(usol)
       # plt.title("Velocity")
       # plt.xlabel("x")
      #  plt.ylabel("y")
       # plt.show()
        plot(p_knew)
        plt.title("Pressure")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        u_n.assign(usol)
        p_n.assign(p_knew)
        t += dt
        
        

        

        

    sol=Function(W)
    sol.sub(0).assign(u_n)
    sol.sub(1).assign(p_n)

    divtest=Function(P)
    divtest.project(div(u_n))
    print("Div error",errornorm(divtest,Function(P)))

    return sol#,conv,d_x

#
parameters={    
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_1_pc_type": "hypre"
    }
print("Channel Flow")
print("Cell number","IterationNumber")

convergence=[]
refin=range(6,7)
delta_x=[]
for n in refin:#increasing element number
    
    #solve
    sol= solve_problem(n, parameters,aP=None, block_matrix=False)
    u,p=sol.split()
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