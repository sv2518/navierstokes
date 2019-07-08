from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from helpers.both import *
from forms.Operators import *
from forms.SPCSForms import *
from logic.InitialValues import *


def solve_problem(mesh_size,parameters_corr, parameters_pres,parameters_velo_initial, parameters_velo,parameters_4, aP=None, block_matrix=False):
    outfile=File("./output/cavity.pvd")

    #generate mesh
    LX=1.0
    LY=1.0
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    U_in=1
    U_inf=Constant(U_in)
   
    #function spaces
    U = FunctionSpace(mesh, "RTCF",1)
    P = FunctionSpace(mesh, "DG", 0)
    W = U*P
    
    #functions
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    f =Function(U)
	
    #outwards pointing normal,1/reynolds number,time stepping params
    n=FacetNormal(W.mesh())
    nu=0.01
    nue=Constant(nu) 
    dt=1/(U_in*(2 ** mesh_size)) #with cfl number
    T=1500
    print("dt is: ",dt)

    #normal boundary conditions
    bc=[]
    bc0=DirichletBC(W.sub(0),Constant((0.0,0.0)),1)
    bc.append(bc0)
    bc1=DirichletBC(W.sub(0),Constant((0.0,0.0)),2)#plane x=0
    bc.append(bc1)
    bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
    bc.append(bc2)
    bc3=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
    bc.append(bc3)

    #tangential boundary conditions
    t=1
    x, y = SpatialCoordinate(mesh)
    bc_tang=[]
    bc_tang.append([Function(U).project(as_vector([-x*100*(x-1)/25*U_inf*(t)*dt,0])),4])
    bc_tang.append([Function(U),1])
    bc_tang.append([Function(U),2])
    bc_tang.append([Function(U),3])

    print("\nCALCULATE INITIAL VALUES")########################################################
    #calculate inital value for pressure with potential flow
    u_init_sol=initialVelocity(W,U,P,mesh,nue,bc,U_inf,parameters_velo_initial,dt,bc_tang)

    #with that initial value calculate intial pressure 
    # with Poission euqation including some non-divergence free velocity
    p_init_sol=initialPressure(W,U,P,mesh,nue,bc,u_init_sol,U_inf,parameters_corr,parameters_pres,dt,bc_tang)


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

    #initialise time stepping
    u_n.assign(u_init_sol)
    p_n.assign(p_init_sol)

    #save & divtest
    outfile.write(u_n,p_n,time=0)
    divtest=Function(P).project(div(u_n))
    print("Div error of initial velocity",errornorm(divtest,Function(P)))


    print("\nBUILD PROBLEM AND SOLVERS")########################################################
    #predictor
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    def nullspace_basis(T):
        return VectorSpaceBasis(constant=True)
    appctx = {'trace_nullspace': nullspace_basis}
    w_pred = Function(U)
    predictor = LinearVariationalProblem(lhs(eq_pred),rhs(eq_pred), w_pred,bc)
    solver_pred = LinearVariationalSolver(predictor, solver_parameters=parameters_velo)

    #pressure update
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

        #update time-dependent boundary
        print("t is: ",t*dt)
        print("n is: ",t)
        bc_tang[0]=[bc_tang[0][0].project(as_vector([-x*100*(x-1)/25*U_inf*((1+t)*dt),0])),4]

        #update picard iteration       
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
                v_k.assign(w_pred)
        

        print("\n2) PRESSURE UPDATE")#########################################################
        #first modify pressure solve
        div_old_temp=Function(P).project(div(w_pred))
        div_old.assign(div_old_temp)
        print("Div error of predictor velocity",errornorm(div_old,Function(P)))

        #solve update equation
        solver_pres.solve()
        wsol,betasol=w_pres.split()
        print(assemble(betasol).dat.data)

        #update pressure
        p_knew=Function(P).assign(p_n+betasol)


        print("\n3) CORRECTOR")##############################################################
        #first update corrector form        
        v_knew_hat.assign(w_pred)
        beta.assign(betasol)
        
        #then solve 
        solver_corr.solve()
        usol=Function(U).assign(w_corr)

        #divtest
        divtest=Function(P).project(div(usol))
        print("Div error of corrector velocity",errornorm(divtest,Function(P)))

        #update for next time step
        u_n.assign(usol)
        p_n.assign(p_knew)
        usol=Function(U)
        psol=Function(P)
        wsol=Function(U)
        betasol=Function(P)

        #time stepping
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
parameters_velo={'pc_type': 'sor',
                'ksp_type': 'gmres',
                'ksp_rtol': 1.0e-7
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
        "ksp_rtol": 1e-8,
        'pc_type': 'ilu'
}


error_velo=[]
error_pres=[]
refin=range(6,7)
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


