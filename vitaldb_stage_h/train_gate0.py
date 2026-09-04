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
    def __init__(self, tensor_path, manifest, condition):
        self.x = np.load(tensor_path, mmap_mode="r")
        self.indices = manifest["tensor_index"].to_numpy(dtype=np.int64, copy=True)
        self.labels = manifest["label"].to_numpy(dtype=np.float32, copy=True)
        self.condition = condition
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.asarray(self.x[idx], dtype=np.float32)
        if self.condition == "ecg_only": x = x[0:1]
        elif self.condition == "pleth_only": x = x[1:2]
        elif self.condition == "ecg_pleth": pass
        else: raise ValueError(self.condition)
        return torch.from_numpy(x.copy()), torch.tensor(self.labels[i], dtype=torch.float32)


class Encoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=4, padding=7), nn.GroupNorm(4,16), nn.ReLU(),
            nn.Conv1d(16, 32, 9, stride=4, padding=4), nn.GroupNorm(8,32), nn.ReLU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.GroupNorm(8,64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, latent), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class UniModel(nn.Module):
    def __init__(self, latent=64):
        super().__init__(); self.enc=Encoder(latent); self.head=nn.Sequential(nn.Linear(latent,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,1))
    def forward(self,x): return self.head(self.enc(x)).squeeze(-1)


class MultiModel(nn.Module):
    def __init__(self, latent=64):
        super().__init__(); self.ecg=Encoder(latent); self.pleth=Encoder(latent); self.head=nn.Sequential(nn.Linear(2*latent,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,x):
        a=F.normalize(self.ecg(x[:,0:1]), dim=1); b=F.normalize(self.pleth(x[:,1:2]), dim=1)
        return self.head(torch.cat([a,b],dim=1)).squeeze(-1)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def predict(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); p=torch.sigmoid(model(x)).cpu().numpy(); ps.append(p); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def threshold_from_val(y,p):
    qs=np.linspace(0.01,0.99,199)
    cand=np.unique(np.quantile(p,qs))
    best=(0.5,-1.0)
    for t in cand:
        f=f1_score(y,p>=t,zero_division=0)
        if f>best[1]: best=(float(t),float(f))
    return best


def metrics(y,p,thr):
    pred=p>=thr
    auprc=float(average_precision_score(y,p))
    auroc=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float("nan")
    f1=float(f1_score(y,pred,zero_division=0))
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=float(tp/(tp+fn)) if tp+fn else 0.0
    spec=float(tn/(tn+fp)) if tn+fp else 0.0
    ppv=float(precision_score(y,pred,zero_division=0))
    return dict(auprc=auprc,auroc=auroc,f1=f1,sensitivity=sens,specificity=spec,ppv=ppv,prevalence=float(np.mean(y)),n_windows=int(len(y)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--condition",choices=["ecg_only","pleth_only","ecg_pleth"],required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--epochs",type=int,default=12)
    ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--patience",type=int,default=3)
    args=ap.parse_args()
    seed_all(args.seed); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    data=Path(args.data_dir); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_parquet(data/"vitaldb_window_manifest_qc.parquet")
    tensor=data/"windows_qc_f16.npy"
    train=manifest[manifest.split=="train"].copy(); val=manifest[manifest.split=="val"].copy(); test=manifest[manifest.split=="test"].copy()
    trds=WindowDataset(tensor,train,args.condition); vds=WindowDataset(tensor,val,args.condition); tds=WindowDataset(tensor,test,args.condition)
    g=torch.Generator(); g.manual_seed(args.seed)
    tr=DataLoader(trds,batch_size=args.batch_size,shuffle=True,num_workers=2,generator=g,persistent_workers=True)
    va=DataLoader(vds,batch_size=args.batch_size,shuffle=False,num_workers=2,persistent_workers=True)
    te=DataLoader(tds,batch_size=args.batch_size,shuffle=False,num_workers=2,persistent_workers=True)
    device=torch.device("cpu")
    model=(MultiModel() if args.condition=="ecg_pleth" else UniModel()).to(device)
    nparams=sum(p.numel() for p in model.parameters())
    pos=float(train.label.sum()); neg=float(len(train)-pos); pos_weight=torch.tensor([neg/max(pos,1.0)],dtype=torch.float32,device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best_ap=-1.0; best_epoch=-1; stale=0; ckpt=out/"best.pt"; history=[]; start=time.time()
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for x,y in tr:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True); z=model(x); loss=loss_fn(z,y); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        vy,vp=predict(model,va,device); vap=float(average_precision_score(vy,vp)); vau=float(roc_auc_score(vy,vp))
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_auprc":vap,"val_auroc":vau})
        print(json.dumps(history[-1]),flush=True)
        if vap>best_ap+1e-5:
            best_ap=vap; best_epoch=epoch; stale=0; torch.save(model.state_dict(),ckpt)
        else:
            stale+=1
            if stale>=args.patience: break
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
    vy,vp=predict(model,va,device); thr,_=threshold_from_val(vy,vp); ty,tp=predict(model,te,device)
    vm=metrics(vy,vp,thr); tm=metrics(ty,tp,thr)
    result={"condition":args.condition,"seed":args.seed,"best_epoch":best_epoch,"selection_metric":"validation AUPRC","threshold":thr,"threshold_selection":"validation max F1","n_params":nparams,"pos_weight":float(pos_weight.item()),"val":vm,"test":tm,"wall_seconds":float(time.time()-start)}
    (out/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    pd.DataFrame(history).to_csv(out/"history.csv",index=False)
    pd.DataFrame({"y":ty,"p":tp}).to_csv(out/"test_predictions.csv",index=False)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
