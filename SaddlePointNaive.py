from firedrake import *
import numpy as np

def both(expr):
    return expr('+') + expr('-')


def build_problem(mesh_size, parameters, aP=None, block_matrix=False):
    #generate and plot mesh
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)
    #plot(mesh)
    import matplotlib.pyplot as plt
    plt.show()

    #function spaces
    U = FunctionSpace(mesh, "RT",1)
    P = FunctionSpace(mesh, "DG", 0)
    W = U*P

    #functions
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    f =Function(U)
    #fvector = f.vector()
    #a=np.random.uniform(size=fvector.local_size())
    #fvector.set_local(np.array(-0*a/a))
	
    #laplacian
    n=FacetNormal(W.mesh())
    nue=0.1#viscosity
    h=CellSize(W.mesh())
    h_avg=(h('+')+h('-'))/2
    alpha=Constant(10.)
    gamma=Constant(10.0) 
    kappa1=nue * alpha*4.
    kappa2=nue * gamma*4
    #excluding exterior facets stuff: slip-BC
    g=Constant((0.0,0.0))
    a_dg=(nue*inner(grad(u-g),grad(v))*dx
           -inner(outer(v,n),nue*grad(u-g))*ds 
           -inner(outer(u,n),nue*grad(v-g))*ds 
           +kappa2*inner(v,u)*ds 
           -inner(nue*avg(grad(v-g)),both(outer(u,n)))*dS
           -inner(both(outer(v,n)),nue*avg(grad(u-g)))*dS
           +kappa1*inner(both(outer(u,n)),both(outer(v,n)))*dS)
 
    #forms
    a = a_dg-div(v)*p*dx+div(u)*q*dx
    L = dot(Constant((0.0, 0.0)),v)*dx

    #preconditioning(not used here)
    if aP is not None:
        aP = aP(W)
    if block_matrix:
        mat_type = 'nest'
    else:
        mat_type = 'aij'

    #boundary conditions on A
    x,y=SpatialCoordinate(mesh)
    inflow=Function(U).project(as_vector(((y-0.5)**2,0.0*y)))
    bc_1=[]
    bc1=DirichletBC(W.sub(0),inflow,1)#plane x=0
    bc_1.append(bc1)
    bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
    bc_1.append(bc2)
    bc3=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
    bc_1.append(bc3)

    w = Function(W)
    problem = LinearVariationalProblem(a, L, w, bc_1)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters)
   

    # check the mesh ordering
    # print(mesh.coordinates.dat.data[:,1])
    # print(mesh.coordinates.dat.data[:,0])
    
    return solver, w, a, L, bc_1


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

for n in range(4,6):
    #solve with linear solve
    solver, w, a, L, bc = build_problem(n, parameters,aP=None, block_matrix=False)
    solver.solve()
    #print(w.function_space().mesh().num_cells(), solver.ksp.getIterationNumber())
    u,p=w.split()

    
    #plot solutions
    File("poisson_mixed_velocity_.pvd").write(u)
    File("poisson_mixed_pressure_.pvd").write(p)
    try:
        import matplotlib.pyplot as plt
    except:
        warning("Matplotlib not imported")

    #print the velocity solutions vector
    print(assemble(u).dat.data)


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

