import numpy as np
from common import *
import adapt
raw,doy=load_train(); S,T,_=raw.shape
coefs=fit_climatology(raw,doy,None,4); A=to_anom(raw,doy,coefs)
ca=adapt.adapted_coefs(coefs); wids,st,X,d,_=load_test(ca)
din=d[:,None]+np.arange(-9,1)
def prof(s,v,center,w=10):
    dd=np.abs(((doy[s]-center+182.6)%365.25)-182.6); return np.abs(A[s][dd<=w,v]).mean()
def ac(x,l): # pooled lag-l autocorr within windows (x: (n,10))
    a,b=x[:,:-l].ravel(),x[:,l:].ravel(); return np.corrcoef(a,b)[0,1]
print("ratio = test mean|resid| / season-matched train mean|anom|;  lag-1 autocorr test vs train")
for v,name in enumerate(VARS):
    print(name)
    for s in range(S):
        m=st==s; x=X[m][:,:,v]
        trp=np.array([prof(s,v,c) for c in np.unique(din[m])]); cnt=np.array([(din[m]==c).sum() for c in np.unique(din[m])])
        tr_abs=(trp*cnt).sum()/cnt.sum()
        a=A[s][:,v]; tr_ac=np.corrcoef(a[:-1],a[1:])[0,1]
        print(f"  S{s} ratio {np.abs(x).mean()/tr_abs:5.2f}  ac1 test {ac(x,1):5.2f} train {tr_ac:5.2f}")
