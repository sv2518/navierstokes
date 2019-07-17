from firedrake import *
from solver.parameters import *
from forms.picardforms import *

def picard(W,mesh,nue,bc,outfile,u_init):

    #functions
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    U=W.sub(0)
    P=W.sub(1)
    n=FacetNormal(W.mesh())
    f =Function(U)

    #get solver parameters
    parameters=defineSolverParameters()[5]

    #split up boundary conditions
    [bc_norm,bc_tang]=bc

    #start value for picard
    u_linear=Function(U).assign(u_init)

    print("\nBUILD FORMS")#####################################################################
    eq=build_picard_form(W,mesh,bc_tang,nue,u_linear)

    print("\nBUILD PROBLEM AND SOLVERS")########################################################
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    w = Function(W)
    problem = LinearVariationalProblem(lhs(eq),rhs(eq),w, bc_norm)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters)
    

    #Picard iteration
    counter=0
    while(True):
 
        solver.solve()
        u1,p1=w.split()

        #convergence criterion
        eps=errornorm(u1,u_linear)#l2 by default
        counter+=1
        print("Picard iteration change in approximation",eps,", counter: ",counter)
        if(eps<10**(-6)):
            print("Picard iteration converged")  
            break          
        else:
            u_linear.assign(u1)
            #one comp has to be set to 0?

    return w