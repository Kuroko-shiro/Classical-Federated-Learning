#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix, precision_score
from torch.utils.data import Dataset, DataLoader


class WindowDataset(Dataset):
    def __init__(self, data_dir: Path, manifest: pd.DataFrame, condition: str):
        self.x = np.load(data_dir / "windows_ecg_art_qc_f16.npy", mmap_mode="r")
        self.idx = manifest["p2_tensor_index"].to_numpy(dtype=np.int64, copy=True)
        self.y = manifest["label"].to_numpy(dtype=np.float32, copy=True)
        self.condition = condition
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        x = np.asarray(self.x[int(self.idx[i])], dtype=np.float32)
        if self.condition == "ecg_only": x = x[0:1]
        elif self.condition == "art_only": x = x[1:2]
        elif self.condition != "ecg_art": raise ValueError(self.condition)
        return torch.from_numpy(x.copy()), torch.tensor(self.y[i], dtype=torch.float32)


class Encoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(1,16,15,stride=4,padding=7),nn.GroupNorm(4,16),nn.ReLU(),
            nn.Conv1d(16,32,9,stride=4,padding=4),nn.GroupNorm(8,32),nn.ReLU(),
            nn.Conv1d(32,64,7,stride=2,padding=3),nn.GroupNorm(8,64),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(64,latent),nn.ReLU())
    def forward(self,x): return self.net(x)


class UniModel(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=Encoder(); self.head=nn.Linear(64,1)
    def forward(self,x): return self.head(self.enc(x)).squeeze(-1)


class MultiModel(nn.Module):
    def __init__(self):
        super().__init__(); self.a=Encoder(); self.b=Encoder(); self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,x):
        a=F.normalize(self.a(x[:,0:1]),dim=1); b=F.normalize(self.b(x[:,1:2]),dim=1)
        return self.head(torch.cat([a,b],dim=1)).squeeze(-1)


def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def predict(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            p=torch.sigmoid(model(x.to(device))).cpu().numpy(); ps.append(p); ys.append(y.numpy())
    return np.concatenate(ys),np.concatenate(ps)

def val_threshold(y,p):
    cand=np.unique(np.quantile(p,np.linspace(.01,.99,199))); best=(.5,-1.)
    for t in cand:
        f=f1_score(y,p>=t,zero_division=0)
        if f>best[1]: best=(float(t),float(f))
    return best[0]

def metrics(y,p,t):
    pred=p>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"auprc":float(average_precision_score(y,p)),"auroc":float(roc_auc_score(y,p)),
            "f1":float(f1_score(y,pred,zero_division=0)),"sensitivity":float(tp/(tp+fn)) if tp+fn else 0.,
            "specificity":float(tn/(tn+fp)) if tn+fp else 0.,"ppv":float(precision_score(y,pred,zero_division=0)),
            "prevalence":float(np.mean(y)),"n_windows":int(len(y))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--condition',choices=['ecg_only','art_only','ecg_art'],required=True); ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--patience',type=int,default=3)
    args=ap.parse_args(); seed_all(args.seed); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    data=Path(args.data_dir); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    m=pd.read_parquet(data/'vitaldb_p2_window_manifest_qc.parquet')
    trm=m[m.split.eq('train')].copy(); vam=m[m.split.eq('val')].copy(); tem=m[m.split.eq('test')].copy()
    trds=WindowDataset(data,trm,args.condition); vads=WindowDataset(data,vam,args.condition); teds=WindowDataset(data,tem,args.condition)
    g=torch.Generator().manual_seed(args.seed)
    tr=DataLoader(trds,batch_size=args.batch_size,shuffle=True,num_workers=2,generator=g,persistent_workers=True)
    va=DataLoader(vads,batch_size=args.batch_size,shuffle=False,num_workers=2,persistent_workers=True)
    te=DataLoader(teds,batch_size=args.batch_size,shuffle=False,num_workers=2,persistent_workers=True)
    model=(MultiModel() if args.condition=='ecg_art' else UniModel()); device=torch.device('cpu'); model=model.to(device)
    nparams=sum(p.numel() for p in model.parameters())
    pos=float(trm.label.sum()); neg=float(len(trm)-pos); pw=torch.tensor([neg/max(pos,1.)],dtype=torch.float32)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pw); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=-1.; best_epoch=-1; stale=0; ckpt=out/'best.pt'; hist=[]; st=time.time()
    for ep in range(args.epochs):
        model.train(); losses=[]
        for x,y in tr:
            opt.zero_grad(set_to_none=True); z=model(x); loss=loss_fn(z,y); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        vy,vp=predict(model,va,device); vap=float(average_precision_score(vy,vp)); vau=float(roc_auc_score(vy,vp))
        row={"epoch":ep,"train_loss":float(np.mean(losses)),"val_auprc":vap,"val_auroc":vau}; hist.append(row); print(json.dumps(row),flush=True)
        if vap>best+1e-5: best=vap; best_epoch=ep; stale=0; torch.save(model.state_dict(),ckpt)
        else:
            stale+=1
            if stale>=args.patience: break
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True)); vy,vp=predict(model,va,device); thr=val_threshold(vy,vp); ty,tp=predict(model,te,device)
    res={"condition":args.condition,"seed":args.seed,"diagnostic_only":True,"best_epoch":best_epoch,"selection_metric":"validation AUPRC",
         "threshold":thr,"threshold_selection":"validation max F1","n_params":nparams,"pos_weight":float(pw.item()),
         "val":metrics(vy,vp,thr),"test":metrics(ty,tp,thr),"wall_seconds":float(time.time()-st)}
    (out/'metrics.json').write_text(json.dumps(res,indent=2),encoding='utf-8'); pd.DataFrame(hist).to_csv(out/'history.csv',index=False)
    pd.DataFrame({"y":ty,"p":tp}).to_csv(out/'test_predictions.csv',index=False); print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__': main()
