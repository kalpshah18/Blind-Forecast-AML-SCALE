"""NN capacity test (folds 0 and 4, 1 seed, augmentation 0.3/0.2): wider+longer net (h=512, 24 epochs) and station-normalised inputs.
Result: wide+long overfits (clean 0.6477); normalised is neutral (0.6305) vs. reference ~0.6298. Run: python -m experiments.nn_capacity_wide_and_normalised"""
import numpy as np, gbm, nn
from common import *
raw,doy=load_train(); S,T,_=raw.shape
VARIANTS={"wide_long":dict(h=512,epochs=24),"sig_norm":dict(norm=True)}
PERT={"clean":(0,0),"pert_mild":(0.3,0.2)}
rng=np.random.default_rng(0)
res={(v,p):[] for v in VARIANTS for p in PERT}
for lo,hi in [cv_folds(T,5)[i] for i in (0,4)]:
    mask=np.ones((S,T),bool); mask[:,lo:hi]=False
    A=to_anom(raw,doy,fit_climatology(raw,doy,mask,4))
    tr=gbm.build(A,doy,[(0,lo),(hi,T)],range(S),1)
    parts=[make_windows(A[s],doy[s],lo,hi) for s in range(S)]
    Xv,Yv,dv=[np.concatenate(z) for z in zip(*parts)]
    sv=np.concatenate([np.full(len(p[0]),s) for s,p in enumerate(parts)])
    sets={}
    for p,(so,ss) in PERT.items():
        o=rng.normal(0,so,(len(Xv),1,4)) if so else 0; c=np.exp(rng.normal(0,ss,(len(Xv),1,4))) if ss else 1
        sets[p]=(Xv*c+o, Yv*c+o)
    Xall=np.concatenate([sets[p][0] for p in PERT]); n=len(Xv)
    for name,kw in VARIANTS.items():
        pred=nn.fit_predict(tr,(Xall,np.tile(dv,len(PERT)),np.tile(sv,len(PERT))),seeds=1,aug=(0.3,0.2),**kw)
        for i,p in enumerate(PERT):
            res[(name,p)].append(np.abs(pred[i*n:(i+1)*n]-sets[p][1]).mean())
        print("fold",lo,name,[round(res[(name,p)][-1],4) for p in PERT],flush=True)
print("\nmean over folds")
print(f"{'variant':10s}"+"".join(f"{p:>13s}" for p in PERT))
for v in VARIANTS: print(f"{v:10s}"+"".join(f"{np.mean(res[(v,p)]):13.4f}" for p in PERT))
