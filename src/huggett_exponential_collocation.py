"""Publication-candidate stage-centered exponential HJB-KFE DAE solver.

Key cleanup relative to v8:
  * the nonlinear HJB corrector is iterated to a fixed-point tolerance;
  * the exact same converged stage generator Abar is used in both the HJB
    affine exponential update and the forward Kolmogorov exponential update;
  * the pathwise tangent/JVP differentiates the converged fixed-point map by
    a corresponding tangent fixed-point iteration.

The equilibrium rate is an algebraic stage variable r_{n+1/2}.
"""
from dataclasses import replace
from time import perf_counter
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import expm_multiply

try:
    import huggett_nesvm_core as core
except ImportError:
    import importlib.util
    _core_path = Path(__file__).with_name('huggett_nesvm_core.py')
    spec = importlib.util.spec_from_file_location('huggett_nesvm_core', _core_path)
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
Params = core.Params


def smoothstep_lambda2_mid(N,T,l0=0.5,l1=0.8):
    tm=(np.arange(N)+0.5)*T/N
    s=tm/T
    q=3*s*s-2*s*s*s
    return l0+(l1-l0)*q


def hjb_shared_stage_step(vnext,r,p,a,h,tol=2e-11,maxit=30):
    """Backward HJB step with one converged stage generator shared with KFE.

    The returned vleft is *exactly* the affine exponential update produced by
    the returned Abar, ubar. Abar itself is based on the preceding Picard
    iterate; at convergence the fixed-point mismatch is below `tol`.
    """
    A_R,u_R,_ = core.build_generator(vnext,a,r,p)
    v = core.affine_exp_step(A_R,u_R,p.rho,vnext,h)
    scale=max(1.0,float(np.max(np.abs(v))))
    last_err=np.inf
    for k in range(1,maxit+1):
        A_L,u_L,_ = core.build_generator(v,a,r,p)
        Abar=0.5*(A_R+A_L); ubar=0.5*(u_R+u_L)
        vnew=core.affine_exp_step(Abar,ubar,p.rho,vnext,h)
        scale=max(1.0,float(np.max(np.abs(vnew))))
        err=float(np.max(np.abs(vnew-v))/scale)
        v=vnew; last_err=err
        if err <= tol:
            return v,Abar,ubar,k,err
    raise RuntimeError(f'HJB shared-stage Picard did not converge: err={last_err:.3e}, r={r:.6g}, h={h:.6g}')


def hjb_shared_stage_step_tangent(vnext,r,p,a,h,dvnext,dr,primal=None,
                                  tol=2e-10,maxit=50):
    """Active-set directional derivative of the converged shared-stage step."""
    if primal is None:
        vleft,Abar,ubar,_,_=hjb_shared_stage_step(vnext,r,p,a,h)
    else:
        vleft,Abar,ubar=primal[:3]

    A_R,u_R,_,dA_R,du_R,_ = core.build_generator_with_tangent(
        vnext,a,r,p,dvnext,dr)

    dv = core.affine_exp_step_tangent(A_R,u_R,p.rho,vnext,h,dA_R,du_R,dvnext)
    last_err=np.inf
    dAbar=None
    for k in range(1,maxit+1):
        A_L,u_L,_,dA_L,du_L,_ = core.build_generator_with_tangent(
            vleft,a,r,p,dv,dr)
        dAbar=0.5*(dA_R+dA_L); dubar=0.5*(du_R+du_L)
        dvnew=core.affine_exp_step_tangent(
            Abar,ubar,p.rho,vnext,h,dAbar,dubar,dvnext)
        scale=max(1.0,float(np.max(np.abs(dvnew))))
        err=float(np.max(np.abs(dvnew-dv))/scale)
        dv=dvnew; last_err=err
        if err <= tol:
            return dv,dAbar,k,err
    raise RuntimeError(f'HJB tangent fixed point did not converge: err={last_err:.3e}')


def transition_stage_shared(rmid,p,g0,vT,a,T,lambda2_mid=None,
                            corrector_tol=2e-11):
    rmid=np.asarray(rmid,float); N=len(rmid); h=T/N; ns=len(vT)
    if lambda2_mid is None: lambda2_mid=np.full(N,p.lambda2)
    lambda2_mid=np.asarray(lambda2_mid,float)
    V=np.empty((ns,N+1)); V[:,-1]=vT
    ops=[None]*N; ubars=[None]*N; picard_iters=np.zeros(N,dtype=int); picard_err=np.zeros(N)
    for n in range(N-1,-1,-1):
        pn=replace(p,lambda2=float(lambda2_mid[n]))
        vleft,Abar,ubar,ki,er=hjb_shared_stage_step(
            V[:,n+1],float(rmid[n]),pn,a,h,tol=corrector_tol)
        V[:,n]=vleft; ops[n]=Abar; ubars[n]=ubar; picard_iters[n]=ki; picard_err[n]=er

    G=np.empty((ns,N+1)); G[:,0]=g0
    avec=np.r_[a,a]; assets=np.empty(N+1); assets[0]=avec@g0
    for n in range(N):
        G[:,n+1]=expm_multiply(ops[n].T*h,G[:,n])
        assets[n+1]=avec@G[:,n+1]
    return {
        'V':V,'G':G,'assets':assets,'operators':ops,'ubars':ubars,
        'mass_error':float(np.max(np.abs(G.sum(axis=0)-1))),
        'min_mass':float(G.min()),
        'picard_iters':picard_iters,'picard_error':picard_err,
    }


def residual_stage_shared(rmid,p,e0,eT,T,lambda2_mid=None,corrector_tol=2e-11):
    out=transition_stage_shared(rmid,p,e0['g'],eT['v'],eT['a'],T,lambda2_mid,corrector_tol)
    return np.diff(out['assets']),out


def residual_stage_jvp_shared(rmid,drmid,p,e0,eT,T,lambda2_mid=None,
                              base=None,corrector_tol=2e-11):
    rmid=np.asarray(rmid,float); drmid=np.asarray(drmid,float); N=len(rmid); h=T/N
    if lambda2_mid is None: lambda2_mid=np.full(N,p.lambda2)
    lambda2_mid=np.asarray(lambda2_mid,float)
    if base is None:
        _,base=residual_stage_shared(rmid,p,e0,eT,T,lambda2_mid,corrector_tol)
    V=base['V']; G=base['G']; ops=base['operators']; ubars=base['ubars']
    dV=np.zeros_like(V); dops=[None]*N
    for n in range(N-1,-1,-1):
        pn=replace(p,lambda2=float(lambda2_mid[n]))
        primal=(V[:,n],ops[n],ubars[n])
        dvn,dAbar,_,_=hjb_shared_stage_step_tangent(
            V[:,n+1],float(rmid[n]),pn,eT['a'],h,dV[:,n+1],float(drmid[n]),
            primal=primal,tol=max(10*corrector_tol,2e-10))
        dV[:,n]=dvn; dops[n]=dAbar
    dG=np.zeros_like(G); avec=np.r_[eT['a'],eT['a']]; da=np.zeros(N+1)
    for n in range(N):
        M=ops[n].T*h; E=dops[n].T*h
        dG[:,n+1]=expm_multiply(M,dG[:,n])+core.frechet_expm_action(M,E,G[:,n])
        da[n+1]=avec@dG[:,n+1]
    return np.diff(da)


def solve_stage_shared(p0,pT,T,N,x0=None,lambda2_mid=None,tol=1e-9,
                       max_newton=8,gmres_rtol=2e-5,verbose=True,
                       corrector_tol=2e-11):
    e0=core.stationary_equilibrium(p0); eT=core.stationary_equilibrium(pT)
    tmid=(np.arange(N)+0.5)*T/N
    x=np.full(N,eT['r']) if x0 is None else np.asarray(x0,float).copy()
    hist=[]; gh=[]; start=perf_counter()
    for it in range(max_newton+1):
        f,base=residual_stage_shared(x,pT,e0,eT,T,lambda2_mid,corrector_tol)
        er=float(np.max(np.abs(f))); hist.append(er)
        if verbose:
            print(f'Shared N={N} Newton {it}: max|dS|={er:.3e}, r=[{x.min():.5f},{x.max():.5f}], '
                  f'Picard max/mean={base["picard_iters"].max()}/{base["picard_iters"].mean():.2f}',flush=True)
        if er<tol or it==max_newton: break
        Lop=spla.LinearOperator((N,N),matvec=lambda d: residual_stage_jvp_shared(
            x,np.asarray(d),pT,e0,eT,T,lambda2_mid,base=base,corrector_tol=corrector_tol),dtype=float)
        cnt={'n':0}
        def cb(_): cnt['n']+=1
        dx,info=spla.gmres(Lop,-f,rtol=gmres_rtol,atol=0,restart=min(30,N),maxiter=100,
                           callback=cb,callback_type='pr_norm')
        gh.append((cnt['n'],int(info)))
        if verbose: print('  GMRES',cnt['n'],'info',info,'|dx|inf',float(np.max(np.abs(dx))),flush=True)
        best=None
        for al in (1.,.75,.5,.25,.1,.05,.02):
            xt=x+al*dx
            if xt.min()<-.30 or xt.max()>.10: continue
            ft,_=residual_stage_shared(xt,pT,e0,eT,T,lambda2_mid,corrector_tol)
            ee=float(np.max(np.abs(ft)))
            if best is None or ee<best[0]: best=(ee,xt,al)
            if ee<er: break
        if best is None: raise RuntimeError('no admissible Newton step')
        if verbose: print('  alpha',best[2],'->',best[0],flush=True)
        x=best[1]
    f,out=residual_stage_shared(x,pT,e0,eT,T,lambda2_mid,corrector_tol)
    return {
        'tmid':tmid,'r':x,'residual':f,'transition':out,'eq0':e0,'eqT':eT,
        'history':np.asarray(hist),'gmres_history':gh,'seconds':perf_counter()-start,
        'converged':float(np.max(np.abs(f)))<tol,
    }


def interpolate_stage_path(old_t,old_r,new_t):
    return np.interp(new_t,old_t,old_r,left=old_r[0],right=old_r[-1])


if __name__=='__main__':
    I=50; T=20.; N=20
    p0=Params(I=I,lambda2=.5); pT=replace(p0,lambda2=.8)
    lam=smoothstep_lambda2_mid(N,T)
    sol=solve_stage_shared(p0,pT,T,N,lambda2_mid=lam,tol=1e-9)
    pd.DataFrame({'time_mid_year':sol['tmid'],'r_stage':sol['r'],
                  'residual':sol['residual'],'lambda2_mid':lam}).to_csv(
                  str(Path(__file__).resolve().parents[1] / 'results' / 'huggett_shared_demo.csv'),index=False)
