from firedrake import *
from forms.Operators import *

def buildPredictorForm(p_init_sol,u_init_sol,nue,mesh,U,P,W,U_inf,dt,bc_tang,v_k,u_n,p_n):
    print("....build predictor")

    #functions
    v_knew=TrialFunction(U)
    v=TestFunction(U)
    n=FacetNormal(U.mesh())

    #implicit midpoint rule
    ubar_k=Constant(0.5)*(u_n+v_k) #init old midstep
    ubar_knew=Constant(0.5)*(u_n+v_knew) #init new midstep  
    
    lapl_dg=DiffusionOperator(nue,ubar_knew,v,n,bc_tang,mesh)
    adv_dg=AdvectionOperator(ubar_k,ubar_knew,v,n,bc_tang)
    pres_dg=Product(div(v),p_n)
   
    #Time derivative
    time=1/Constant(dt)*inner(v_knew-u_n,v)*dx

    eq_pred=time+adv_dg-lapl_dg+pres_dg

    return eq_pred

def buildPressureForm(W,U,P,dt,mesh,U_inf,bc_tang,div_old):
    print("....build pressure update")

    w,beta = TrialFunctions(W)
    v,q = TestFunctions(W)

    force_dg_pres=Product(div_old/dt ,q)
    incomp_dg_pres=Product(div(w) ,q)
    pres_dg_pres=Product(div(v),beta)
    
    eq_pres=dot(w,v)*dx+force_dg_pres-incomp_dg_pres-pres_dg_pres 

    return eq_pres

def buildCorrectorForm(W,U,P,dt,mesh,U_inf,v_knew_hat,beta):
    print("....build corrector")
    v_knew=TrialFunction(U)
    v=TestFunction(U)

    eq_corr=(
            dot(v_knew,v)*dx
            -dot(v_knew_hat,v)*dx
           # -div(v_knew)*q*dx #no need to be included because used before?
            -dt*beta*div(v)*dx
    )

    return eq_corr