from firedrake import *
from forms.spcsforms import *
from solver.initialvalues import *
from solver.parameters import *
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc

#standard pressure correction scheme
def spcs(W,mesh,nue,bc,U_inf,t,dt,T,outfile,u_init=None,p_init=None,output=False):

    with PETSc.Log.Event("spcs configuration"):
        #functions and normal
        U=W.sub(0)
        P=W.sub(1)
        u,p = TrialFunctions(W)
        v,q = TestFunctions(W)
        f =Function(U)

        #get solver parameters
        [parameters_corr,parameters_pres,parameters_pres_better,parameters_velo,parameters_velo_initial,nn]=defineSolverParameters()

        #split up boundary conditions
        [bc_norm,bc_tang,bc_expr]=bc

    with PETSc.Log.Event("initial values"):
        print("\nCALCULATE INITIAL VALUES")########################################################
        #check if analytical initial condition is given
        if(u_init):
            u_init_sol=Function(U).project(u_init)
        else:
            #calculate inital velocity with potential flow
            u_init_sol=initial_velocity(W,dt,mesh,bc,nue)

        
        divtest=Function(P).project(div(u_init_sol))
        print("Div error of initial velocity",errornorm(divtest,Function(P)))

        #check in analytical solutions is given
        #if p_init:
            #x,y=SpatialCoordinate(W.mesh())
            #p_init_sol=Function(P).project(p_init)
        #else:
            #with that initial value calculate intial pressure 
            # with Poission euqation including some non-divergence free velocity
        p_init_sol=initial_pressure(W,dt,mesh,nue,bc,u_init_sol)
    
    with PETSc.Log.Event("build forms"):
        print("\nBUILD FORMS")#####################################################################
        v_k=Function(U)
        u_n=Function(U)
        p_n=Function(P)   
        div_old=Function(P)
        v_knew_hat=Function(U)
        beta=Function(P)
        eq_pred=build_predictor_form(W,dt,mesh,nue,bc_tang,v_k,u_n,p_n)
        eq_upd=build_update_form(W,dt,mesh,bc_tang,div_old)
        eq_corr=build_corrector_form(W,dt,mesh,v_knew_hat,beta)

    #initialise time stepping
    u_n.assign(u_init_sol)
    p_n.assign(p_init_sol)

    #save & divtest
    if output:
        outfile.write(u_n,p_n,time=0)

    divtest=Function(P).project(div(u_n))
    print("Div error of initial velocity",errornorm(divtest,Function(P)))

    with PETSc.Log.Event("build problems and solvers"):
        print("\nBUILD PROBLEM AND SOLVERS")########################################################
        
        nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
        def nullspace_basis(T):
            return VectorSpaceBasis(constant=True)
        appctx = {'trace_nullspace': nullspace_basis}

        with PETSc.Log.Event("predictor"):
            w_pred = Function(U)
            predictor = LinearVariationalProblem(lhs(eq_pred),rhs(eq_pred), w_pred,bc_norm)
            solver_pred = LinearVariationalSolver(predictor, solver_parameters=parameters_velo)

        with PETSc.Log.Event("update"):
            w_upd = Function(W)
            nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
            update= LinearVariationalProblem(lhs(eq_upd),rhs(eq_upd),w_upd,bc_norm)#BC RIGHT???
            solver_upd= LinearVariationalSolver(update,solver_parameters=parameters_pres,appctx=appctx)
            
        with PETSc.Log.Event("corrector"):
            w_corr = Function(U)
            corrector= LinearVariationalProblem(lhs(eq_corr),rhs(eq_corr), w_corr,bc_norm)
            solver_corr = LinearVariationalSolver(corrector,solver_parameters=parameters_corr)

    with PETSc.Log.Event("time progressing"):
        print("\nTIME PROGRESSING")################################################################
        #outerloop for time progress
        n = 1
        while n < (T+1) :
            #update time-dependent boundary
            t.assign(n)
            print("t is: ",n*dt)
            print("n is: ",n)

            if bc_tang:
                bc_tang[0]=[bc_tang[0][0].project(bc_expr),4]
                bc_tang[1]=[bc_tang[1][0].project(bc_expr),1]
                bc_tang[2]=[bc_tang[2][0].project(bc_expr),2]
                bc_tang[3]=[bc_tang[3][0].project(bc_expr),3]

        
            #update start value of picard iteration       
            counter=0
            v_k.assign(u_n)

            with PETSc.Log.Event("picard iteration"):
                print("\n1)PREDICTOR")##################################################################
                #loop for non-linearity
                while(True):  

                    with PETSc.Log.Event("predictor solve"):
                        solver_pred.solve()

                    #convergence criterion
                    eps=errornorm(v_k,w_pred)#l2 by default          
                    counter+=1
                    print("Picard iteration counter: ",counter,"Picard iteration norm: ",eps)
                    if(counter>6):#eps<10**(-8)):
                        print("Picard iteration converged")  
                        break      
                    else:
                        v_k.assign(w_pred)
            
                print("\n2) PRESSURE UPDATE")#########################################################
                #first modify update form
                div_old_temp=Function(P).project(div(w_pred))
                div_old.assign(div_old_temp)
                print("Div error of predictor velocity",errornorm(div_old,Function(P)))

                #solve update equation
                with PETSc.Log.Event("update solve"):
                    solver_upd.solve()
                wsol,betasol=w_upd.split()

                #update pressure
                p_knew=Function(P).assign(p_n+betasol)

                print("\n3) CORRECTOR")##############################################################
                #first modify corrector form        
                v_knew_hat.assign(w_pred)
                beta.assign(betasol)
                
                #then solve 
                with PETSc.Log.Event("corrector solve"):
                    solver_corr.solve()
                usol=Function(U).assign(w_corr)

            #divtest
            divtest=Function(P).project(div(usol))
            print("Div error of corrector velocity",errornorm(divtest,Function(P)))

            #update for next time step
            u_n.assign(usol)
            p_n.assign(p_knew)

            #write in vtk file
            if output:
                outfile.write(u_n,p_n,time=n)
            n += 1     


    #final time step solution
    sol=Function(W)
    sol.sub(0).assign(u_n)
    sol.sub(1).assign(p_n)

    return sol