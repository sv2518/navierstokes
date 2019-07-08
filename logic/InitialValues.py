from firedrake import *
from forms.operators import *


#INITAL VALUES: solve the following from for initial values
def initialVelocity(W,U,P,mesh,nue,bc,U_inf,parameters,dt,bc_tang):
    print("....Solving Stokes problem for initial velocity ....")

    #Functions and parameters
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    n=FacetNormal(W.mesh())

    #Form for Stokes problem
    lapl_dg=DiffusionOperator(nue,u,v,n,bc_tang,mesh)    
    incomp_dg=Product(u,q)
    pres_dg=Product(v,p)
    eq_init=incomp_dg-pres_dg+lapl_dg

    #solve
    w_init= Function(W)
    init = LinearVariationalProblem(lhs(eq_init),rhs(eq_init), w_init,bc)
    solver = LinearVariationalSolver(init, solver_parameters=parameters)
    solver.solve()
    u_init_sol,p_init_sol=w_init.split()
    
    return u_init_sol

def initialPressure(W,U,P,mesh,nue,bc,u_init,U_inf,parameters_velo,parameters_pres,dt,bc_tang):
    print("....Solving problem for initial pressure ....")


    ############################################################################
    print(".........part1: solve for some non-divergence free velocity field")
    #functions
    F=TrialFunction(U)
    v=TestFunction(U)
    n=FacetNormal(U.mesh())

    #reuse initial velocity
    u_linear=Function(U).assign(u_init)
    
    #projection form
    lapl_dg=DiffusionOperator(nue,u_linear,v,n,bc_tang,mesh)
    adv_dg=AdvectionOperator(u_linear,u_linear,v,n,bc_tang)
    eq_init=dot(v,F)*dx-adv_dg+lapl_dg

    #solve
    w_init= Function(U)
    init = LinearVariationalProblem(lhs(eq_init),rhs(eq_init), w_init,bc)
    solver = LinearVariationalSolver(init, solver_parameters=parameters_velo)
    solver.solve()
    F_init_sol=Function(U).assign(w_init)

    ############################################################################
    print("........part2: solve mixed possion problem for initial pressure ")
    
    #functions
    w,beta = TrialFunctions(W)
    v,q=TestFunctions(W)
    n=FacetNormal(W.mesh())

    #correction form for initial pressure
    f_pres=Function(P).project(div(F_init_sol))#old divergence acts like forcing 
    print("Div error of projected velocity",errornorm(f_pres,Function(P)))
    force_dg_pres=Product(f_pres,q)
    incomp_dg_pres=Product(div(w),q)
    pres_dg_pres=Product(div(v),beta)
    eq_pres=dot(w,v)*dx-force_dg_pres-incomp_dg_pres-pres_dg_pres 

    #solve
    w_init= Function(W)
    init = LinearVariationalProblem(lhs(eq_pres),rhs(eq_pres), w_init,bc)
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    def nullspace_basis(T):
        return VectorSpaceBasis(constant=True)
    appctx = {'trace_nullspace': nullspace_basis}
    solver = LinearVariationalSolver(init, solver_parameters=parameters_pres,appctx=appctx)
    solver.solve()
    u_init_sol,p_init_sol=w_init.split()
 
    return p_init_sol