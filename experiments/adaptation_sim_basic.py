"""Simulate test-time climatology adaptation on CV folds (ridge_seas), with/without an injected station shift."""
import numpy as np, linear
from common import *
raw,doy=load_train(); S,T,_=raw.shape
rng=np.random.default_rng(0)
def offset_fit(Xin, din, k, lam=1e-2):
    # Xin (n,10,4) anomalies, din (n,10) doy per input day -> beta (2k+1,4)
    H=harmonics(din.reshape(-1),k); y=Xin.reshape(-1,4)
    return np.linalg.solve(H.T@H+lam*len(H)*np.eye(H.shape[1])*0+lam*np.eye(H.shape[1]),H.T@y)
shift_true=np.array([1.8,3.4,-2.0,0.9])
res={}
for lo,hi in cv_folds(T,5)[::2]:
    mask=np.ones((S,T),bool); mask[:,lo:hi]=False
    A=to_anom(raw,doy,fit_climatology(raw,doy,mask,4))
    spans=[(a,b) for a,b in [(0,lo),(hi,T)] if b-a>15]
    for s in range(S):
        tr=linear.station_windows(A[s],doy[s],spans)
        Xv,Yv,dv=make_windows(A[s],doy[s],lo,hi)
        idx=rng.choice(len(Xv),89,replace=False); Xv,Yv,dv=Xv[idx],Yv[idx],dv[idx]
        # per-day doy of inputs
        din=dv[:,None]+np.arange(-9,1)
        for shift_name,sh in [("none",0*shift_true),("S8-like",shift_true*(1+0.4*np.sin(2*np.pi*dv/365.25))[:,None])]:
            # shift varies by season for S8-like; Xs/Ys add shift
            shin=(shift_true*(1+0.4*np.sin(2*np.pi*din/365.25))[...,None]) if shift_name!="none" else 0
            shout=(shift_true*(1+0.4*np.sin(2*np.pi*(dv[:,None]+np.arange(1,6))/365.25))[...,None]) if shift_name!="none" else 0
            Xs,Ys=Xv+shin,Yv+shout
            for k in [None,0,1,2]:
                if k is None: X2,Y2=Xs,Ys; back=lambda p:p
                else:
                    b=offset_fit(Xs,din,k)
                    o_in=np.einsum("ndk,kv->ndv",harmonics(din,k),b) if False else (harmonics(din.reshape(-1),k)@b).reshape(len(Xs),10,4)
                    dout=dv[:,None]+np.arange(1,6)
                    o_out=(harmonics(dout.reshape(-1),k)@b).reshape(len(Xs),5,4)
                    X2,Y2=Xs-o_in,Ys-o_out; back=None
                p=linear.fit_predict("ridge_seas",tr,(X2,dv))
                err=np.abs(p-Y2).mean()  # error unchanged by adding offset back
                res.setdefault((shift_name,k),[]).append(err)
for key,v in res.items(): print(key, round(np.mean(v),4))
