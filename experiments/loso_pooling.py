"""Leave-one-station-out check: pooled station-normalised linear model vs. own-station model (last 20% held out)."""
import numpy as np, linear
from common import *
raw,doy=load_train(); S,T,_=raw.shape
cut=int(T*0.8)
mask=np.ones((S,T),bool); mask[:,cut:]=False
A=to_anom(raw,doy,fit_climatology(raw,doy,mask,4))
sig=np.abs(A[:,:cut]).mean(1)          # (S,4) train mean|anom|
normal=[1,2,3,4,5,7,9]
print("held-out station | wind MAE own vs pooled-z (others, normal only) | all-var MAE own vs pooled")
for s in normal:
    own=linear.station_windows(A[s],doy[s],[(0,cut-15)])
    Xv,Yv,dv=make_windows(A[s],doy[s],cut,T)
    po=[]; 
    for o in normal:
        if o==s: continue
        X,Y,d=linear.station_windows(A[o]/sig[o],doy[o],[(0,cut-15)]); po.append((X,Y,d))
    pool=[np.concatenate(z) for z in zip(*po)]
    p_own=linear.fit_predict("lad_seas",own,(Xv,dv))
    p_pool=linear.fit_predict("lad_seas",pool,(Xv/sig[s],dv))*sig[s]
    e=lambda p:np.abs(p-Yv).mean((0,1))
    print(f"S{s}: wind {e(p_own)[3]:.3f} vs {e(p_pool)[3]:.3f} | all {e(p_own).mean():.3f} vs {e(p_pool).mean():.3f}")
