"""End-to-end simulation of test-time adaptation. Inject each station's real estimated test shift
(seasonal offset) and dispersion ratio into CV windows, then run estimate->correct->predict (ridge_seas)."""
import numpy as np, linear, adapt
from common import *
raw,doy_tr=load_train(); S,T,_=raw.shape
coefs0=fit_climatology(raw,doy_tr,None,4); A0=to_anom(raw,doy_tr,coefs0)
ca=adapt.adapted_coefs(coefs0); _,st_te,Xte,dte,_=load_test(ca)
beta_true=(ca-coefs0)[:, :3]/SCALE[None,None,:]                 # (S,3,4) true seasonal offset (k=1)
r_true=adapt.dispersion_ratio(A0,doy_tr,Xte,dte,st_te)          # (S,4)
rng=np.random.default_rng(1)
def off(s,d): return harmonics(d,1)@beta_true[s]
def estimate(Xp,dp,sp,A,doy,k,deadband):
    din=dp[:,None]+np.arange(-9,1); beta={}
    for s in range(S):
        m=sp==s; H=harmonics(din[m].reshape(-1),k)
        beta[s]=np.linalg.solve(H.T@H+1e-2*np.eye(H.shape[1]),H.T@Xp[m].reshape(-1,4))
    Xc=np.stack([Xp[i]-harmonics(din[i],k)@beta[sp[i]] for i in range(len(Xp))])
    r=adapt.dispersion_ratio(A,doy,Xc,dp,sp)
    if deadband: r=np.where((r>0.8)&(r<1.25),1.0,r)
    return beta,r
modes=[("none",None),("k0 +scale",(0,True)),("k1 no scale",(1,False)),("k1 +scale",(1,True)),("k1 +scale deadband",(1,"db")),("k2 +scale deadband",(2,"db")),("oracle",'o')]
res={m[0]:[] for m in modes}
for lo,hi in [cv_folds(T,5)[i] for i in (1,3)]:
    mask=np.ones((S,T),bool); mask[:,lo:hi]=False
    A=to_anom(raw,doy_tr,fit_climatology(raw,doy_tr,mask,4))
    spans=[(a,b) for a,b in [(0,lo),(hi,T)] if b-a>15]
    models={s:linear.station_windows(A[s],doy_tr[s],spans) for s in range(S)}
    for rep in range(2):
        Xp,Yp,dp,sp=[],[],[],[]
        for s in range(S):
            X,Y,d=make_windows(A[s],doy_tr[s],lo,hi); idx=rng.choice(len(X),89,replace=False); X,Y,d=X[idx],Y[idx],d[idx]
            din=d[:,None]+np.arange(-9,1); dout=d[:,None]+np.arange(1,6)
            Xp.append(X*r_true[s]+off(s,din.reshape(-1)).reshape(-1,10,4)); Yp.append(Y*r_true[s]+off(s,dout.reshape(-1)).reshape(-1,5,4))
            dp.append(d); sp.append(np.full(len(d),s))
        Xp,Yp,dp,sp=map(np.concatenate,(Xp,Yp,dp,sp))
        din=dp[:,None]+np.arange(-9,1); dout=dp[:,None]+np.arange(1,6)
        for name,cfg in modes:
            if cfg is None: Xin,rr,o_out=Xp,np.ones((S,4)),0
            elif cfg=='o':
                Xin=np.stack([(Xp[i]-off(sp[i],din[i]))/r_true[sp[i]] for i in range(len(Xp))]); rr=r_true
                o_out=np.stack([off(sp[i],dout[i]) for i in range(len(Xp))])
            else:
                k,sc=cfg; beta,r=estimate(Xp,dp,sp,A0,doy_tr,k,sc=="db"); rr=r if sc else np.ones((S,4))
                Xin=np.stack([(Xp[i]-harmonics(din[i],k)@beta[sp[i]])/rr[sp[i]] for i in range(len(Xp))])
                o_out=np.stack([harmonics(dout[i],k)@beta[sp[i]] for i in range(len(Xp))])
            P=np.zeros_like(Yp)
            for s in range(S):
                m=sp==s; P[m]=linear.fit_predict("ridge_seas",models[s],(Xin[m],dp[m]))*rr[s]
            res[name].append(np.abs(P+o_out-Yp).mean((0,1)))
print(f"{'mode':22s} {'mean':>7s}  tmax  tmin    rh  wind")
for name,_ in modes:
    r=np.mean(res[name],0); print(f"{name:22s} {r.mean():7.4f} "+" ".join(f"{x:5.3f}" for x in r))
