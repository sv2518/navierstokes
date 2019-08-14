from firedrake import *
from forms.operators import *
from solver.parameters import *
from firedrake.petsc import PETSc


#INITAL VALUES: solve the following from for initial values
def initial_velocity(W,dt,mesh,bc,nue,order,IP_stabilityparam_type=None):
    PETSc.Sys.Print("....Solving Stokes problem for initial velocity ....")
    parameters_velo_initial=defineSolverParameters()[2][0]#initial,velo

    #extract bc and parameters
    [bc_norm,bc_tang,bc_expr]=bc

    #Functions and parameters
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    n=FacetNormal(W.mesh())

    #Form for Stokes problem
    lapl_dg=diffusion_operator(nue,u,v,n,bc_tang,mesh,10.,order,IP_stabilityparam_type)    
    incomp_dg=ibp_product(div(u),q)
    pres_dg=ibp_product(div(v),p)
    eq_init=incomp_dg-pres_dg+lapl_dg

    #solve
    w_init= Function(W)
    init = LinearVariationalProblem(lhs(eq_init),rhs(eq_init), w_init,bc_norm)
    solver = LinearVariationalSolver(init, solver_parameters=parameters_velo_initial)
    solver.solve()
    u_init_sol,p_init_sol=w_init.split()
    
    return u_init_sol

def initial_pressure(W,dt,mesh,nue,bc,u_init,order,IP_stabilityparam_type=None):
    PETSc.Sys.Print("....Solving problem for initial pressure ....")

    #extract bcs, parameters and subspace
    [bc_norm,bc_tang,bc_expr]=bc
    parameters_pres=defineSolverParameters()[0][1]#direct,pres
    parameters_velo=defineSolverParameters()[1][0]#iterative,velo
    U=W.sub(0)
    P=W.sub(1)

    ############################################################################
    PETSc.Sys.Print(".........part1: solve for some non-divergence free velocity field")
    #functions
    F=TrialFunction(U)
    v=TestFunction(U)
    n=FacetNormal(U.mesh())

    #reuse initial velocity
    u_linear=Function(U).assign(u_init)
    
    #projection form
    lapl_dg=diffusion_operator(nue,u_linear,v,n,bc_tang,mesh,10.,order,IP_stabilityparam_type)
    adv_dg=advection_operator(u_linear,u_linear,v,n,bc_tang)
    eq_init=dot(v,F)*dx-adv_dg+lapl_dg

    #solve
    w_init= Function(U)
    init = LinearVariationalProblem(lhs(eq_init),rhs(eq_init), w_init,bc_norm)
    solver = LinearVariationalSolver(init, solver_parameters=parameters_velo)
    solver.solve()
    F_init_sol=Function(U).assign(w_init)

    ############################################################################
    PETSc.Sys.Print("........part2: solve mixed possion problem for initial pressure ")
    
    #functions
    w,beta = TrialFunctions(W)
    v,q=TestFunctions(W)
    n=FacetNormal(W.mesh())

    #correction form for initial pressure
    #divergence of projected velocity acts like forcing 
    f_pres=Function(P).project(div(F_init_sol))
    PETSc.Sys.Print("Div error of projected velocity",errornorm(f_pres,Function(P)))
    force_dg_pres=ibp_product(f_pres,q)
    incomp_dg_pres=ibp_product(div(w),q)
    pres_dg_pres=ibp_product(div(v),beta)
    eq_pres=dot(w,v)*dx-force_dg_pres-incomp_dg_pres-pres_dg_pres 

    #solve
    w_init= Function(W)
    init = LinearVariationalProblem(lhs(eq_pres),rhs(eq_pres), w_init,bc_norm)
    nullspace=MixedVectorSpaceBasis(W,[W.sub(0),VectorSpaceBasis(constant=True)])
    def nullspace_basis(T):
        return VectorSpaceBasis(constant=True)
    appctx = {'trace_nullspace': nullspace_basis}
    solver = LinearVariationalSolver(init, solver_parameters=parameters_pres,appctx=appctx)
    solver.solve()
    u_init_sol,p_init_sol=w_init.split()
 
    return p_init_sol