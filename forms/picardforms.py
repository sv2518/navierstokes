from firedrake import *
from forms.operators import *

def build_picard_form(W,mesh,bc_tang,nue,u_linear):

    #functions
    u,p = TrialFunctions(W)
    v,q = TestFunctions(W)
    n=FacetNormal(W.mesh())

    #Laplacian
    lapl_dg=diffusion_operator(nue,u,v,n,bc_tang,mesh,10000.)         
    adv_dg=advection_operator(u_linear,u,v,n,bc_tang)
    eq = -lapl_dg+adv_dg+ibp_product(div(v),p)+ibp_product(div(u),q)

    return eq