import numpy as np, linear
from common import *
raw,doy=load_train(); S,T,_=raw.shape
for k in [2,3,4,6,8]:
    sc=[]
    for lo,hi in cv_folds(T,5):
        mask=np.ones((S,T),bool); mask[:,lo:hi]=False
        A=to_anom(raw,doy,fit_climatology(raw,doy,mask,k))
        spans=[(a,b) for a,b in [(0,lo),(hi,T)] if b-a>15]
        P=[];Yt=[]
        for s in range(S):
            tr=linear.station_windows(A[s],doy[s],spans); Xv,Yv,dv=make_windows(A[s],doy[s],lo,hi)
            P.append(linear.fit_predict("ridge_seas",tr,(Xv,dv))); Yt.append(Yv)
        # score in original units: scaled anomaly error is relative to the k-dependent clim, so compare raw errors
        sc.append(np.abs(np.concatenate(P)-np.concatenate(Yt)).mean())
    print("k_clim",k,"ridge_seas CV MAE (scaled raw units):",round(np.mean(sc),4),flush=True)
