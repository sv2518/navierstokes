from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

def both(expr):
    return expr('+') + expr('-')


def solve_problem(mesh_size, parameters, aP=None, block_matrix=False):
    #generate mesh
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=2,Ly=2,quadrilateral=True)
    
    #function spaces
    U = FunctionSpace(mesh, "RTCF",1)
    P = FunctionSpace(mesh, "DG", 0)
    W = U*P

    #functions
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    f =Function(U)
	
    #building the operators
    n=FacetNormal(W.mesh())
    nue=Constant(1.0)#viscosity

    #specify inflow/initial solution
    x,y=SpatialCoordinate(mesh)
    lam=(-8.*pi**2/(nue**(-1)+sqrt(nue**(-2)+16*pi**2)))
    inflow=Function(U).project(as_vector((1-exp(lam*x)*cos(2*pi*y),0.0)))
    #inflow_uniform=Function(U).project(Constant((1.0,0.0)))  
    
    #Picard iteration
    u_linear=Function(U).assign(inflow)
    counter=0
    while(True):

        #Laplacian
        alpha=Constant(10.)
        gamma=Constant(10.) 
        kappa1=nue * alpha/Constant(mesh_size)
        kappa2=nue * gamma/Constant(mesh_size)
      #  g=Function(U).project(as_vector((0.0,lam/(2*pi)*exp(lam*x)*sin(2*pi*y))))
        g=Constant((0.0,0.0))
        a_dg=(nue*inner(grad(u),grad(v))*dx
            -inner(outer(v,n),nue*grad(u))*ds 
            -inner(outer(u-g,n),nue*grad(v))*ds 
            +kappa2*inner(v,u-g)*ds 
            -inner(nue*avg(grad(v)),both(outer(u,n)))*dS
            -inner(both(outer(v,n)),nue*avg(grad(u)))*dS
            +kappa1*inner(both(outer(u,n)),both(outer(v,n)))*dS)
         
        #Advection
        un = 0.5*(dot(u_linear, n) + abs(dot(u_linear, n)))
        adv_dg=(dot(u_linear,div(outer(v,u)))*dx#like paper
            -inner(v,(u*dot(u_linear,n)))*ds#similar to matt piggots
            -dot((v('+')-v('-')),(un('+')*u('+') - un('-')*u('-')))*dS)#like in the tutorial
    
        #form
        eq = a_dg+Constant(-1.)*adv_dg-div(v)*p*dx+div(u)*q*dx
        eq -= dot(Constant((0.0, 0.0)),v)*dx
        a=lhs(eq)
        L=rhs(eq)

        #preconditioning(not used here)
        if aP is not None:
            aP = aP(W)
        if block_matrix:
            mat_type = 'nest'
        else:
            mat_type = 'aij'

        #boundary conditions
        bc_1=[]
        bc1=DirichletBC(W.sub(0),inflow,1)#plane x=0
        bc_1.append(bc1)
        bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
        bc_1.append(bc2)
        bc3=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
        bc_1.append(bc3)

        #build problem and solver
        w = Function(W)
        problem = LinearVariationalProblem(a, L, w, bc_1)
        solver = LinearVariationalSolver(problem, solver_parameters=parameters)
        solver.solve()
        u1,p1=w.split()

        #convergence criterion
        eps=errornorm(u1,u_linear)#l2 by default
        counter+=1
        print("Picard iteration change in approximation",eps,", counter: ",counter)
        if(eps<10**(-8)):
            print("Picard iteration converged")  
            break          
        else:
            u_linear.assign(u1)

        #method of manufactured solutions
        test=Function(W)
        p_sol=Function(P).project(-1./2*exp(2*lam*x))
        u_sol=Function(U).project(as_vector((1-exp(lam*x)*cos(2*pi*y),lam/(2*pi)*exp(lam*x)*sin(2*pi*y))))
        test.sub(0).assign(u_sol)
        test.sub(1).assign(p_sol)#why does it not matter if I take this in or not?
        #plt.plot((assemble(action(a-L,test),bcs=bc_1).dat.data[0]))#maxnorm
        #plt.show()

        print("velo error",errornorm(u1,u_sol),", pres error: ",errornorm(p1,p_sol))
        

    return w


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

for n in range(6,7):#increasing element number
    
    #solve
    w = solve_problem(n, parameters,aP=None, block_matrix=False)
    u,p=w.split()
    
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

