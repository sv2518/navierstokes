from firedrake import *
import numpy as np


def build_problem(mesh_size, parameters, aP=None, block_matrix=False):
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)

    #function spaces
    U = FunctionSpace(mesh, "RT",1)
    P = FunctionSpace(mesh, "DG", 0)
    W = U*P

    #functions
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    f =Function(U)
    fvector = f.vector()
    a=np.random.uniform(size=fvector.local_size())
    fvector.set_local(np.array(-0*a/a))
	
    #laplacian
    n=FacetNormal(W.mesh())
    nue=1.0
    h=CellSize(W.mesh())
    h_avg=(h('+')+h('-'))/2
    alpha=Constant(5)
    gamma=Constant(5) 
    kappa1=alpha/h_avg
    kappa2=gamma/h
    a_dg=nue*inner(grad(u),grad(v))*dx \
         -inner(nue*grad(v),outer(u,n))*ds \
         -inner(outer(v,n),nue*grad(u))*ds \
	 +kappa2*inner(v,u)*ds \
         -inner(avg(nue*grad(v)),jump(outer(u,n)))*dS \
         -inner(jump(outer(v,n)),nue*avg(grad(u)))*dS \
	 +kappa1*inner(jump(outer(u,n)),jump(outer(v,n)))*dS 

    #linear forms
    a = a_dg-div(v)*p*dx+div(u)*q*dx
    L = dot(f,v)*dx

    #preconditioning
    if aP is not None:
        aP = aP(W)
    
    #assembling
    if block_matrix:
        mat_type = 'nest'
    else:
        mat_type = 'aij'

    #boundary conditions
    bc_1=[]
    bc2=DirichletBC(W.sub(0),Constant((0.0,0.0)),3)#plane y=0
    bc_1.append(bc2)
    bc3=DirichletBC(W.sub(0),Constant((0.0,0.0)),4)#plane y=L
    bc_1.append(bc3)
    bc1=DirichletBC(W.sub(0),Constant((1.0,0.0)),1)#plane x=0
    bc_1.append(bc1)

    #assembling for solving with linear solver
    A = assemble(a, mat_type=mat_type,bcs=bc_1)
    if aP is not None:
        P = assemble(aP, mat_type=mat_type)
    else:
        P = None
    solver = LinearSolver(A, P=P, solver_parameters=parameters)
    w = Function(W)
    b = assemble(L)
    
    return solver, w,b,a,L,bc_1


parameters={
    "ksp_type":"gmres",
    "ksp_gmres_restart":100,
    "ksp_rtol":1e-8,
    "pc_type":"ilu"}

print("Channel Flow")
print("Cell number","IterationNumber")

for n in range(6):
    #solve with linear solve
    solver, w,b,a,L,bc= build_problem(n, parameters,aP=None, block_matrix=False)
    solver.solve(w, b)
    print(w.function_space().mesh().num_cells(), solver.ksp.getIterationNumber())
    u,p=w.split()
    
    #plot solutions
    File("poisson_mixed_velocity.pvd").write(u)
    File("poisson_mixed_pressure.pvd").write(p)
    try:
        import matplotlib.pyplot as plt
    except:
        warning("Matplotlib not imported")

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

    #check and plot residuals
    res_u=assemble(action(a,w)-L,bcs=bc).dat.data[0]
    res_p=assemble(action(a,w)-L,bcs=bc).dat.data[1]
    print("L_infty norm of u:%d",res_u.max())
    print("L_infty norm of p:%d",res_p.max())
    plt.plot(res_u)
    plt.title("Velocity Residual")
    plt.xlabel("Facet")
    plt.ylabel("r")
    plt.show()
    plt.plot(res_p)
    plt.title("Pressure Residual")
    plt.xlabel("Node")
    plt.ylabel("r")
    plt.show()
