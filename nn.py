"""Idea 2: pooled residual MLP (linear skip + MLP) on scaled anomalies, L1 loss, station embedding."""
import os
import numpy as np
import torch
import torch.nn as nn
from common import *

torch.set_num_threads(min(8, os.cpu_count() or 1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRIDE = 1
EPOCHS, BATCH, LR, SEEDS = 14, 1024, 2e-3, 2


class Net(nn.Module):
    def __init__(self, n_in, n_st=10, emb=8, h=256):
        super().__init__()
        self.emb = nn.Embedding(n_st, emb)
        d = n_in + emb
        self.skip = nn.Linear(d, OUT * 4)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, OUT * 4))

    def forward(self, x, s):
        z = torch.cat([x, self.emb(s)], 1)
        return self.skip(z) + self.mlp(z)


class GRUNet(nn.Module):
    """GRU over the 10 days (4 vars/step) + season feats + station embedding, with a linear skip."""
    def __init__(self, n_in, n_st=10, emb=8, h=128):
        super().__init__()
        self.emb = nn.Embedding(n_st, emb)
        self.gru = nn.GRU(4, h, num_layers=2, batch_first=True)
        d = h + (n_in - IN * 4) + emb
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, OUT * 4))
        self.skip = nn.Linear(n_in + emb, OUT * 4)

    def forward(self, x, s):
        e = self.emb(s)
        seq = x[:, :IN * 4].view(-1, IN, 4)
        _, hN = self.gru(seq)
        z = torch.cat([hN[-1], x[:, IN * 4:], e], 1)
        return self.head(z) + self.skip(torch.cat([x, e], 1))


def inputs(X, d):
    return np.hstack([X.reshape(len(X), -1), season_feats(d, 2)[:, 1:]]).astype(np.float32)


def fit_predict(train, test, seeds=SEEDS, aug=(0.0, 0.0), aug_p=0.5, arch="mlp", h=256, epochs=None, norm=False):
    X, Y, d, s = train
    Xt, dt, st = test
    epochs = epochs or EPOCHS
    if norm:  # station-normalised dynamics: divide anomalies by each station's own mean |anomaly|
        sig = np.stack([np.abs(X[s == k]).mean((0, 1)) for k in range(10)])
        X, Y, Xt = X / sig[s][:, None], Y / sig[s][:, None], Xt / sig[st][:, None]
    F = torch.tensor(inputs(X, d), device=DEVICE)
    T = torch.tensor(Y.reshape(len(Y), -1), dtype=torch.float32, device=DEVICE)
    S = torch.tensor(s, dtype=torch.long, device=DEVICE)
    Ft, St = torch.tensor(inputs(Xt, dt), device=DEVICE), torch.tensor(st, dtype=torch.long, device=DEVICE)
    preds = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = (GRUNet(F.shape[1]) if arch == "gru" else Net(F.shape[1], h=h)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
        steps = epochs * ((len(F) + BATCH - 1) // BATCH)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
        for _ in range(epochs):
            perm = torch.randperm(len(F), device=DEVICE)
            net.train()
            for i in range(0, len(F), BATCH):
                b = perm[i:i + BATCH]
                fb, tb = F[b], T[b]
                if aug[0] > 0 or aug[1] > 0:
                    # shift/scale augmentation: x' = c*x + o on inputs and targets (regime-shift equivariance)
                    on = (torch.rand(len(b), 1, device=DEVICE) < aug_p).float()
                    o = torch.randn(len(b), 4, device=DEVICE) * aug[0] * on
                    c = torch.exp(torch.randn(len(b), 4, device=DEVICE) * aug[1] * on)
                    fb = fb.clone()
                    fb[:, :IN * 4] = (fb[:, :IN * 4].view(-1, IN, 4) * c[:, None] + o[:, None]).view(len(b), -1)
                    tb = (tb.view(-1, OUT, 4) * c[:, None] + o[:, None]).view(len(b), -1)
                loss = (net(fb, S[b]) - tb).abs().mean()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            preds.append(net(Ft, St).cpu().numpy())
    out = np.mean(preds, 0).reshape(-1, OUT, 4)
    return out * sig[st][:, None] if norm else out


def fit_predict6(train, test):
    return fit_predict(train, test, seeds=6)


def fit_predict6a(train, test):
    return fit_predict(train, test, seeds=6, aug=(0.3, 0.2))


def fit_predict6g(train, test):
    return fit_predict(train, test, seeds=6, aug=(0.3, 0.2), arch="gru")


def fit_predict12b(train, test):
    """12 seeds, larger shift/scale augmentation (offset sd 0.7, log-scale sd 0.4)."""
    return fit_predict(train, test, seeds=12, aug=(0.7, 0.4))
