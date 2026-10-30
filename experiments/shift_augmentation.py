"""Shift/scale augmentation of NN training windows (folds 0/2/4, 1 seed), scored on clean and perturbed validation windows.
Perturbation = per-window offset N(0,so) and lognormal scale applied to val inputs AND targets, mimicking imperfect test-time adaptation.
Run: python -m experiments.shift_augmentation"""
import numpy as np, gbm, nn
from common import *
raw,doy=load_train(); S,T,_=raw.shape
VARIANTS={"base":(0,0),"aug_small":(0.3,0.2),"aug_big":(0.7,0.4)}
PERT={"clean":(0,0),"pert_mild":(0.3,0.2),"pert_strong":(0.7,0.4)}
rng=np.random.default_rng(0)
res={(v,p):[] for v in VARIANTS for p in PERT}
for lo,hi in [cv_folds(T,5)[i] for i in (0,2,4)]:
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
    for name,aug in VARIANTS.items():
        pred=nn.fit_predict(tr,(Xall,np.tile(dv,len(PERT)),np.tile(sv,len(PERT))),seeds=1,aug=aug)
        for i,p in enumerate(PERT):
            res[(name,p)].append(np.abs(pred[i*n:(i+1)*n]-sets[p][1]).mean())
        print("fold",lo,name,[round(res[(name,p)][-1],4) for p in PERT],flush=True)
print("\nmean over folds")
print(f"{'variant':10s}"+"".join(f"{p:>13s}" for p in PERT))
for v in VARIANTS: print(f"{v:10s}"+"".join(f"{np.mean(res[(v,p)]):13.4f}" for p in PERT))
