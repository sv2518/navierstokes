from firedrake import *
from helpers.both import both 

def diffusion_operator(nue,u,v,n,bc_tang,mesh,stab,order):

    #Stability params for Laplacian
    alpha=Constant(stab)*order**1#interior
    gamma=Constant(stab)*order**1 #exterior
    h=CellVolume(mesh)/FacetArea(mesh)  
    havg=avg(CellVolume(mesh))/FacetArea(mesh)
    kappa1=nue*alpha/havg
    kappa2=nue*gamma/h

    #laplacian for interior domain and interior facets
    lapl_dg=(
            nue*inner(grad(u),grad(v))*dx
            -inner(nue*avg(grad(v)),both(outer(u,n)))*dS
            -inner(both(outer(v,n)),nue*avg(grad(u)))*dS
            +kappa1*inner(both(outer(u,n)),both(outer(v,n)))*dS
    )

    for [bc_t,m] in bc_tang:
        #laplacian for exterior facets
        lapl_dg+=(
            -inner(outer(v,n),nue*grad(u-bc_t))*ds(m) 
            -inner(outer(u-bc_t,n),nue*grad(v))*ds(m)
            +kappa2*inner(v,u-bc_t)*ds(m)
        )
    
    return -lapl_dg

def advection_operator(u_linear,u,v,n,bc_tang):
    #interior flux
    u_flux_int = 0.5*(dot(u_linear, n)+ abs(dot(u_linear, n))) 

    #advection in for interior domain and interior facets
    adv_dg=(
        dot(u_linear,div(outer(v,u)))*dx 
        -dot((v('+')-v('-')),(u_flux_int('+')*(u('+')) - u_flux_int('-')*(u('-'))))*dS
    )

    for [bc_t,m] in bc_tang:
        #advection for exterior facets 
        u_flux_ext = 0.5*(dot(u_linear,n)*u+abs(dot(u_linear,n))*u+ dot(u_linear,n)*bc_t-abs(dot(u_linear,n))*bc_t) 
        adv_dg+=(-dot(v,u_flux_ext)*ds(m))

    return -adv_dg

#integration by parts product 
def ibp_product(v,p):
    #NOTE: v and p can Test-, Trial- or just a Function
    #called for pressure, forcing, and incompressibility
    return -dot(v,p)*dx