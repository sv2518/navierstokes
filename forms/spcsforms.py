from firedrake import *
from forms.operators import *

def build_predictor_form(W,dt,mesh,nue,bc_tang,v_k,u_n,p_n,order,IP_stabilityparam_type=None):
    print("....build predictor")

    #subspaces
    U=W.sub(0)
    P=W.sub(1)

    #functions
    v_knew=TrialFunction(U)
    v=TestFunction(U)
    n=FacetNormal(U.mesh())

    #implicit midpoint rule
    ubar_k=Constant(0.5)*(u_n+v_k) #init old midstep
    ubar_knew=Constant(0.5)*(u_n+v_knew) #init new midstep  
    
    lapl_dg=diffusion_operator(nue,ubar_knew,v,n,bc_tang,mesh,10.,order,IP_stabilityparam_type)
    adv_dg=advection_operator(ubar_k,ubar_knew,v,n,bc_tang)
    pres_dg=ibp_product(div(v),p_n)
    time=1/Constant(dt)*inner(v_knew-u_n,v)*dx

    eq_pred=time+adv_dg-lapl_dg+pres_dg

    return eq_pred

def build_update_form(W,dt,mesh,bc_tang,div_old):
    print("....build pressure update")

    #subspaces
    U=W.sub(0)
    P=W.sub(1)

    #functions
    w,beta = TrialFunctions(W)
    v,q = TestFunctions(W)

    force_dg_upd=ibp_product(div_old/dt ,q)
    incomp_dg_upd=ibp_product(div(w) ,q)
    pres_dg_upd=ibp_product(div(v),beta)
    
    eq_upd=dot(w,v)*dx+force_dg_upd-incomp_dg_upd-pres_dg_upd

    return eq_upd

def build_corrector_form(W,dt,mesh,v_knew_hat,beta):
    print("....build corrector")
    
    #subspaces
    U=W.sub(0)
    P=W.sub(1)

    #functions
    v_knew=TrialFunction(U)
    v=TestFunction(U)

    eq_corr=(
            dot(v_knew,v)*dx
            -dot(v_knew_hat,v)*dx
           # -div(v_knew)*q*dx #no need to be included because used before?
            -dt*beta*div(v)*dx
    )

    return eq_corr