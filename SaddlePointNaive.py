from firedrake import *

def build_problem(mesh_size, parameters, aP=None, block_matrix=False):
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)

    Sigma = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    W = Sigma * 
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    f = Function(V)
    import numpy as np
    fvector = f.vector()
    fvector.set_local(np.random.uniform(size=fvector.local_size()))
    a = dot(sigma, tau)*dx + div(tau)*u*dx + div(sigma)*v*dx
    L = -f*v*dx
    if aP is not None:
        aP = aP(W)
        
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
    parameters = {
    "ksp_type": "gmres",
    "ksp_gmres_restart": 100,"ksp_rtol": 1e-8,"pc_type": "ilu",
    }
    print("Naive preconditioning")
    for n in range(8):
        solver, w, b = build_problem(n, parameters, block_matrix=False)
        solver.solve(w, b)
    print(w.function_space().mesh().num_cells(), solver.ksp.getIterationNumber())