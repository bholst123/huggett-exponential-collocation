"""Core numerical routines for the Huggett exponential-collocation solver."""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
from scipy.sparse.linalg import expm_multiply

@dataclass(frozen=True)
class Params:
    rho: float = 0.05
    gamma: float = 2.0
    z1: float = 0.10
    z2: float = 0.20
    lambda1: float = 0.60
    lambda2: float = 0.50
    amin: float = -0.15
    amax: float = 4.0
    I: int = 80
    Delta_stationary: float = 100.0
    stationary_tol: float = 1e-9
    stationary_maxit: int = 250

    @property
    def z(self):
        return np.array([self.z1, self.z2], dtype=float)


def utility(c, gamma):
    c = np.maximum(c, 1e-13)
    if abs(gamma - 1.0) < 1e-14:
        return np.log(c)
    return c**(1.0-gamma)/(1.0-gamma)


def grid(p):
    a = np.linspace(p.amin, p.amax, p.I)
    return a, a[1]-a[0]


def _assemble_generator(X, Y, Z, p):
    blocks=[]
    for j in range(2):
        blocks.append(sp.diags([X[1:,j],Y[:,j],Z[:-1,j]],[-1,0,1],shape=(p.I,p.I),format='csc'))
    E=sp.eye(p.I,format='csc')
    Asw=sp.bmat([[-p.lambda1*E, p.lambda1*E],
                 [ p.lambda2*E,-p.lambda2*E]],format='csc')
    return sp.block_diag(blocks,format='csc')+Asw


def build_generator(v, a, r, p):
    """Monotone upwind generator A(v,r); A acts on values and A.T on masses."""
    I=p.I; da=a[1]-a[0]
    V=np.column_stack((v[:I],v[I:]))
    aa=np.column_stack((a,a)); zz=np.ones((I,1))*p.z[None,:]
    dVf=np.empty_like(V); dVb=np.empty_like(V)
    dVf[:-1,:]=(V[1:,:]-V[:-1,:])/da
    dVb[1:,:]=(V[1:,:]-V[:-1,:])/da
    dVf[-1,:]=np.maximum(np.abs(p.z+r*a[-1]),1e-12)**(-p.gamma)
    dVb[0,:]=np.maximum(np.abs(p.z+r*a[0]),1e-12)**(-p.gamma)
    dVf_safe=np.maximum(dVf,1e-13); dVb_safe=np.maximum(dVb,1e-13)
    cf=dVf_safe**(-1.0/p.gamma); cb=dVb_safe**(-1.0/p.gamma)
    sf=zz+r*aa-cf; sb=zz+r*aa-cb
    cash=zz+r*aa
    dV0=np.maximum(np.abs(cash),1e-12)**(-p.gamma)
    use_f=sf>0; use_b=sb<0
    both=use_f & use_b
    if np.any(both):
        Hf=utility(cf,p.gamma)+dVf_safe*sf
        Hb=utility(cb,p.gamma)+dVb_safe*sb
        choose_f=Hf>=Hb
        use_f[both]=choose_f[both]; use_b[both]=~choose_f[both]
    use_0=~(use_f|use_b)
    dVu=dVf_safe*use_f+dVb_safe*use_b+dV0*use_0
    c=np.maximum(dVu,1e-13)**(-1.0/p.gamma)
    u=utility(c,p.gamma)
    X=(-np.minimum(sb,0.0)/da)*use_b
    Z=(np.maximum(sf,0.0)/da)*use_f
    Y=-(X+Z)
    A=_assemble_generator(X,Y,Z,p)
    rows=np.asarray(A.sum(axis=1)).ravel()
    scale=max(1.0,float(np.max(np.abs(A.data))) if A.nnz else 1.0)
    if np.max(np.abs(rows)) > 1e-8*scale:
        raise RuntimeError('generator row sums lost conservation')
    return A,np.r_[u[:,0],u[:,1]],np.r_[c[:,0],c[:,1]]


def solve_stationary(r,p,vinit=None):
    a,_=grid(p); N=2*p.I; eye=sp.eye(N,format='csc')
    if vinit is None:
        V0=np.column_stack([(np.maximum(p.z[j]+r*a,1e-10))**(1-p.gamma)/(1-p.gamma)/p.rho for j in range(2)])
        v=np.r_[V0[:,0],V0[:,1]]
    else:
        v=np.asarray(vinit).copy()
    for it in range(p.stationary_maxit):
        A,u,_=build_generator(v,a,r,p)
        vn=spla.spsolve((p.rho+1/p.Delta_stationary)*eye-A,u+v/p.Delta_stationary)
        if np.max(np.abs(vn-v)) < p.stationary_tol:
            v=vn; break
        v=vn
    A,u,c=build_generator(v,a,r,p)
    AT=A.T.tolil(); rhs=np.zeros(N); AT[0,:]=np.ones(N); rhs[0]=1.0
    g=spla.spsolve(AT.tocsc(),rhs)
    if g.min() < -1e-8:
        raise RuntimeError(f'stationary negative mass {g.min():g}')
    g=np.maximum(g,0); g/=g.sum()
    return {'r':r,'a':a,'v':v,'A':A,'g':g,'assets':float(np.r_[a,a]@g),'iterations':it+1}


def stationary_equilibrium(p,rlo=0.005,rhi=0.045):
    r=brentq(lambda rr: solve_stationary(rr,p)['assets'],rlo,rhi,xtol=2e-11,rtol=1e-11)
    return solve_stationary(r,p)


def affine_exp_step(A,u,rho,v0,h):
    N=len(v0); L=A-rho*sp.eye(N,format='csc')
    w=spla.spsolve(L,u)
    return expm_multiply(L*h,v0+w)-w


def build_generator_with_tangent(v, a, r, p, dv, dr):
    """A,u,c plus active-set directional derivatives dA,du,dc."""
    I=p.I; da=a[1]-a[0]
    V=np.column_stack((v[:I],v[I:])); dV=np.column_stack((dv[:I],dv[I:]))
    aa=np.column_stack((a,a)); zz=np.ones((I,1))*p.z[None,:]
    dVf=np.empty_like(V); dVb=np.empty_like(V); ddVf=np.empty_like(V); ddVb=np.empty_like(V)
    dVf[:-1,:]=(V[1:,:]-V[:-1,:])/da; dVb[1:,:]=(V[1:,:]-V[:-1,:])/da
    ddVf[:-1,:]=(dV[1:,:]-dV[:-1,:])/da; ddVb[1:,:]=(dV[1:,:]-dV[:-1,:])/da

    def pow_abs_and_tangent(x,dx):
        ax=np.maximum(np.abs(x),1e-12); y=ax**(-p.gamma)
        dy=-p.gamma*ax**(-p.gamma-1.0)*np.sign(x)*dx
        dy[np.abs(x)<=1e-12]=0.0
        return y,dy

    dVf[-1,:],ddVf[-1,:]=pow_abs_and_tangent(p.z+r*a[-1],a[-1]*dr*np.ones(2))
    dVb[0,:],ddVb[0,:]=pow_abs_and_tangent(p.z+r*a[0],a[0]*dr*np.ones(2))
    dVf_safe=np.maximum(dVf,1e-13); dVb_safe=np.maximum(dVb,1e-13)
    ddVf_safe=ddVf*(dVf>1e-13); ddVb_safe=ddVb*(dVb>1e-13)
    cf=dVf_safe**(-1.0/p.gamma); cb=dVb_safe**(-1.0/p.gamma)
    dcf=(-1.0/p.gamma)*dVf_safe**(-1.0/p.gamma-1.0)*ddVf_safe
    dcb=(-1.0/p.gamma)*dVb_safe**(-1.0/p.gamma-1.0)*ddVb_safe
    sf=zz+r*aa-cf; sb=zz+r*aa-cb
    dsf=dr*aa-dcf; dsb=dr*aa-dcb
    dV0,ddV0=pow_abs_and_tangent(zz+r*aa,dr*aa)
    use_f=sf>0; use_b=sb<0
    both=use_f & use_b
    if np.any(both):
        Hf=utility(cf,p.gamma)+dVf_safe*sf; Hb=utility(cb,p.gamma)+dVb_safe*sb
        choose_f=Hf>=Hb; use_f[both]=choose_f[both]; use_b[both]=~choose_f[both]
    use_0=~(use_f|use_b)
    dVu=dVf_safe*use_f+dVb_safe*use_b+dV0*use_0
    ddVu=ddVf_safe*use_f+ddVb_safe*use_b+ddV0*use_0
    dVu_safe=np.maximum(dVu,1e-13); ddVu_safe=ddVu*(dVu>1e-13)
    c=dVu_safe**(-1.0/p.gamma)
    dc=(-1.0/p.gamma)*dVu_safe**(-1.0/p.gamma-1.0)*ddVu_safe
    u=utility(c,p.gamma); du=c**(-p.gamma)*dc
    X=(-np.minimum(sb,0.0)/da)*use_b; Z=(np.maximum(sf,0.0)/da)*use_f; Y=-(X+Z)
    dX=(-dsb/da)*use_b; dZ=(dsf/da)*use_f; dY=-(dX+dZ)
    A=_assemble_generator(X,Y,Z,p)
    dblocks=[]
    for j in range(2):
        dblocks.append(sp.diags([dX[1:,j],dY[:,j],dZ[:-1,j]],[-1,0,1],shape=(I,I),format='csc'))
    dA=sp.block_diag(dblocks,format='csc')
    return A,np.r_[u[:,0],u[:,1]],np.r_[c[:,0],c[:,1]],dA,np.r_[du[:,0],du[:,1]],np.r_[dc[:,0],dc[:,1]]


def frechet_expm_action(M,E,q):
    n=M.shape[0]; Z=sp.csc_matrix((n,n)); B=sp.bmat([[M,E],[Z,M]],format='csc')
    rhs=np.r_[np.zeros(n),np.asarray(q)]
    return expm_multiply(B,rhs)[:n]


def affine_exp_step_tangent(A,u,rho,v0,h,dA,du,dv0):
    N=len(v0); L=A-rho*sp.eye(N,format='csc')
    w=spla.spsolve(L,u); dw=spla.spsolve(L,du-dA@w)
    q=v0+w; dq=dv0+dw; M=L*h; E=dA*h
    return frechet_expm_action(M,E,q)+expm_multiply(M,dq)-dw
