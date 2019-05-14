from firedrake import *
import numpy as np

def laplacian(W):
    #laplacian
    p,u = TrialFunctions(W)
    q,v = TestFunctions(W)
    n=FacetNormal(W.mesh())
    nue=1
    kappa=Constant(0.5)
    a_dg=inner(grad(u),grad(v))*dx \
         -inner(grad(v),outer(u,n))*ds \
         -inner(outer(v,n),grad(u))*ds \
	 +kappa*inner(v,u)*ds \
         -inner(avg(grad(v)),jump(outer(u,n)))*dS \
         -inner(jump(outer(v,n)),avg(grad(u)))*dS \
	 +kappa*inner(jump(outer(u,n)),jump(outer(v,n)))*dS 
    return a_dg

def build_problem(mesh_size, parameters, aP=None, block_matrix=False):
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)

    #function spaces
    U = FunctionSpace(mesh, "RT", 1)
    P = FunctionSpace(mesh, "DG", 0)
    W = U*P

    #functions
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    f = Function(U)
    fvector = f.vector()
    fvector.set_local(np.random.uniform(size=fvector.local_size()))
	
    #laplacian
    n=FacetNormal(W.mesh())
    nue=1
    kappa=Constant(0.5)
    a_dg=inner(grad(u),grad(v))*dx \
         -inner(grad(v),outer(u,n))*ds \
         -inner(outer(v,n),grad(u))*ds \
	 +kappa*inner(v,u)*ds \
         -inner(avg(grad(v)),jump(outer(u,n)))*dS \
         -inner(jump(outer(v,n)),avg(grad(u)))*dS \
	 +kappa*inner(jump(outer(u,n)),jump(outer(v,n)))*dS 

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
    A = assemble(a, mat_type=mat_type)
    if aP is not None:
        P = assemble(aP, mat_type=mat_type)
    else:
        P = None

    solver = LinearSolver(A, P=P, solver_parameters=parameters)
    w = Function(W)
    b = assemble(L)

    return solver, w, b


#parameters = {
#    "ksp_type": "fgmres",
#    "ksp_rtol": 1e-2,
#    "pc_type": "fieldsplit",
#    "pc_fieldsplit_type": "schur",
#    "pc_fieldsplit_schur_fact_type": "full",
#    "fieldsplit_0_ksp_type": "cg",
#    "fieldsplit_0_pc_type": "ilu",
#    "fieldsplit_0_ksp_rtol": 1e-12,
#    "fieldsplit_1_ksp_type": "cg",
#    "fieldsplit_1_pc_type": "none",
#    "fieldsplit_1_ksp_rtol": 1e-12}

parameters={
    "ksp_type":"gmres",
    "ksp_gmres_restart":100,
    "ksp_rtol":1e-2,
    "pc_type":"ilu"}

print("Channel Flow")
print("Cell number","IterationNumber")

for n in range(8):
    solver, w, b = build_problem(n, parameters,aP=None, block_matrix=False)
    solver.solve(w, b)
    print(w.function_space().mesh().num_cells(), solver.ksp.getIterationNumber())
    p,u=w.split()
    File("poisson_mixed.pvd").write(u)
