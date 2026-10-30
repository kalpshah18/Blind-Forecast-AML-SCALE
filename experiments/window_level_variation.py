import numpy as np, adapt
from common import *
raw,doy=load_train(); S,T,_=raw.shape
coefs=fit_climatology(raw,doy,None,4); A=to_anom(raw,doy,coefs)
ca=adapt.adapted_coefs(coefs); wids,st,X,d,_=load_test(ca)
r=adapt.dispersion_ratio(A,doy,X,d,st)
print("Extra between-window level variation after adaptation: std(10-day window mean) test/train, divided by dispersion ratio (1.0 = none)")
print("        tmax  tmin    rh  wind")
for s in range(S):
    Xs,_,_=make_windows(A[s],doy[s])
    tr_wm=Xs.mean(1).std(0); te_wm=X[st==s].mean(1).std(0)
    print(f"S{s}: ", " ".join(f"{x:5.2f}" for x in te_wm/tr_wm/r[s]))
