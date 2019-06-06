from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

def both(expr):
    return expr('+') + expr('-')


def solve_problem(mesh_size, parameters, aP=None, block_matrix=False):
    #generate mesh
    LX=200
    LY=1
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size,Lx=LX,Ly=LY,quadrilateral=True)
    
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
    nue=Constant(1.)#viscosity

    #specify inflow/initial solution
    x,y=SpatialCoordinate(mesh)
    u_exact=as_vector((-0.5*(y-1)*y,0.0*y))
    p_exact=-1*(x-LX)

    inflow=Function(U).project(u_exact)

    #Picard iteration
    u_linear=Function(U).assign(inflow)
    counter=0
    while(True):

        #Laplacian
        alpha=Constant(10.)
        gamma=Constant(10.) 
        h=CellVolume(mesh)/FacetArea(mesh)
        havg=avg(CellVolume(mesh))/FacetArea(mesh)
        kappa1=nue*alpha/havg
        kappa2=nue*gamma/h

        a_dg=(nue*inner(grad(u),grad(v))*dx
            -inner(outer(v,n),nue*grad(u))*ds 
            -inner(outer(u,n),nue*grad(v))*ds 
            +kappa2*inner(v,u)*ds 
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

        #method of manufactured solutions
        strongform1=Function(U).project(div(grad(u_exact))-dot(u_exact,grad(u_exact))-grad(p_exact)+f)
        strongform2=Function(P).project(div(u_exact))
        eq -=(dot(strongform1,v)*dx+dot(strongform2,q)*dx)
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
        print("Picard iteration error",eps,", counter: ",counter)
        if(eps<10**(-12)):
            print("Picard iteration converged")  
            break          
        else:
            u_linear.assign(u1)

    plt.plot(assemble(w.sub(0)-Function(U).project(u_exact)).dat.data)
    plt.show()

    plt.plot(assemble(w.sub(1)-Function(P).project(p_exact)).dat.data)
    plt.show()

    #L2 error of divergence
    print("L2 error of divergence",errornorm(Function(P).project(div(w.sub(0))),Function(P)))
    print("L_inf error of velo",max(abs(assemble(w.sub(0)).dat.data)))
    print("L_inf error of pres",max(abs(assemble(w.sub(1)).dat.data)))
    print("L_2 error of velo", errornorm(w.sub(0),Function(U)))
    print("L_2 error of pres",  errornorm(w.sub(1),Function(P)))
    print("Hdiv error of velo", errornorm(w.sub(0),Function(U),"Hdiv"))
    print("Hdiv error of pres",  errornorm(w.sub(1),Function(P),"Hdiv"))

    conv=max(abs(assemble(w.sub(0)).dat.data))
    d_x=LX/2**mesh_size
    return w,conv,d_x

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

convergence=[]
refin=range(4,9)
delta_x=[]
for n in refin:#increasing element number
    
    #solve
    w,conv,d_x = solve_problem(n, parameters,aP=None, block_matrix=False)
    u,p=w.split()
    convergence.append(conv)
    delta_x.append(d_x)

    
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

print("max error in velocity",convergence)

#convergence plot
fig = plt.figure()
axis = fig.gca()
linear=convergence
axis.loglog(refin,linear)
#axis.plot(refin,refin[::-1],'r*')
axis.set_xlabel('$Level$')
axis.set_ylabel('$||e||_{\infty}$')
plt.show()