#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader


class WindowDataset(Dataset):
    def __init__(self, tensor_path: Path, manifest: pd.DataFrame):
        self.x = np.load(tensor_path, mmap_mode="r")
        self.indices = manifest["tensor_index"].to_numpy(dtype=np.int64, copy=True)
        self.labels = manifest["label"].to_numpy(dtype=np.float32, copy=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.asarray(self.x[idx], dtype=np.float32)
        return torch.from_numpy(x.copy()), torch.tensor(self.labels[i], dtype=torch.float32)


class Encoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=4, padding=7), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv1d(16, 32, 9, stride=4, padding=4), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, latent), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.ecg = Encoder(latent)
        self.pleth = Encoder(latent)
        self.head = nn.Sequential(
            nn.Linear(2 * latent, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, x):
        a = F.normalize(self.ecg(x[:, 0:1]), dim=1)
        b = F.normalize(self.pleth(x[:, 1:2]), dim=1)
        return self.head(torch.cat([a, b], dim=1)).squeeze(-1)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            ps.append(torch.sigmoid(model(x)).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def eval_metrics(model, loader, device):
    y,p = predict(model, loader, device)
    return {
        "auprc": float(average_precision_score(y,p)),
        "auroc": float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
        "prevalence": float(np.mean(y)),
        "n_windows": int(len(y)),
    }


def make_clients(train: pd.DataFrame, k: int, seed: int):
    """Subject-disjoint, approximately IID client partition.

    Subjects are stratified by whether they have any positive training window. Within each
    stratum, large subjects are greedily assigned to the currently least-loaded client.
    This stage intentionally avoids non-IID stress: it is only a homogeneous FedAvg plumbing test.
    """
    subj = train.groupby("subjectid").agg(
        has_positive=("label","max"), n_windows=("label","size"), n_positive=("label","sum")
    ).reset_index()
    rng=np.random.default_rng(seed)
    assignments={i:[] for i in range(k)}
    loads=np.zeros(k,dtype=np.int64)
    for cls in [1,0]:
        s=subj[subj.has_positive==cls].copy()
        # randomized tie-breaking, then descending size for balanced greedy packing
        s["tie"]=rng.random(len(s))
        s=s.sort_values(["n_windows","tie"],ascending=[False,True])
        for r in s.itertuples(index=False):
            cid=int(np.argmin(loads))
            assignments[cid].append(int(r.subjectid))
            loads[cid]+=int(r.n_windows)
    out=[]
    for cid in range(k):
        m=train[train.subjectid.isin(assignments[cid])].copy()
        out.append(m)
    return out


def local_train(global_model, ds, seed, batch_size, lr, local_epochs, device):
    model=copy.deepcopy(global_model).to(device)
    g=torch.Generator(); g.manual_seed(seed)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=2,generator=g,persistent_workers=True)
    # Local optimizer is intentionally reset each round; only model parameters are federated.
    pos=float(np.sum(ds.labels)); neg=float(len(ds)-pos)
    pos_weight=torch.tensor([neg/max(pos,1.0)],dtype=torch.float32,device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    model.train(); losses=[]
    for _ in range(local_epochs):
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y)
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return state,float(np.mean(losses)) if losses else None


def fedavg(states, weights):
    total=float(sum(weights))
    out={}
    for key in states[0]:
        acc=None
        for st,w in zip(states,weights):
            term=st[key].to(torch.float64)*(float(w)/total)
            acc=term if acc is None else acc+term
        out[key]=acc.to(states[0][key].dtype)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--seed",type=int,default=202)
    ap.add_argument("--clients",type=int,default=8)
    ap.add_argument("--rounds",type=int,default=10)
    ap.add_argument("--local-epochs",type=int,default=1)
    ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--lr",type=float,default=1e-3)
    args=ap.parse_args()

    seed_all(args.seed); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    data=Path(args.data_dir); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_parquet(data/"vitaldb_window_manifest_qc.parquet")
    tensor=data/"windows_qc_f16.npy"
    train=manifest[manifest.split=="train"].copy(); val=manifest[manifest.split=="val"].copy(); test=manifest[manifest.split=="test"].copy()
    clients=make_clients(train,args.clients,args.seed)

    # Hard leakage audit.
    sets=[set(m.subjectid.astype(int).unique()) for m in clients]
    for i in range(args.clients):
        for j in range(i+1,args.clients):
            if sets[i]&sets[j]: raise RuntimeError(f"client subject overlap {i},{j}")
    if set(train.subjectid.astype(int).unique()) != set().union(*sets):
        raise RuntimeError("client partition does not cover train subjects exactly")
    if set(train.subjectid)&set(val.subjectid) or set(train.subjectid)&set(test.subjectid) or set(val.subjectid)&set(test.subjectid):
        raise RuntimeError("train/val/test subject leakage")

    client_summary=[]
    client_ds=[]
    for cid,m in enumerate(clients):
        client_summary.append({
            "client":cid,"n_subjects":int(m.subjectid.nunique()),"n_cases":int(m.caseid.nunique()),
            "n_windows":int(len(m)),"n_positive":int(m.label.sum()),"prevalence":float(m.label.mean())
        })
        client_ds.append(WindowDataset(tensor,m))
    pd.DataFrame(client_summary).to_csv(out/"client_summary.csv",index=False)

    device=torch.device("cpu")
    model=FusionModel().to(device)
    nparams=sum(p.numel() for p in model.parameters())
    if nparams!=55681: raise RuntimeError(f"unexpected parameter count {nparams}")
    va=DataLoader(WindowDataset(tensor,val),batch_size=args.batch_size,shuffle=False,num_workers=2,persistent_workers=True)
    te=DataLoader(WindowDataset(tensor,test),batch_size=args.batch_size,shuffle=False,num_workers=2,persistent_workers=True)

    best_ap=-1.0; best_round=-1; best_state=None; history=[]; start=time.time()
    for rnd in range(1,args.rounds+1):
        states=[]; weights=[]; losses=[]
        for cid,ds in enumerate(client_ds):
            st,loss=local_train(model,ds,args.seed+rnd*1009+cid,args.batch_size,args.lr,args.local_epochs,device)
            states.append(st); weights.append(len(ds)); losses.append(loss)
        model.load_state_dict(fedavg(states,weights),strict=True)
        vm=eval_metrics(model,va,device)
        row={"round":rnd,"mean_local_loss":float(np.mean(losses)),"val_auprc":vm["auprc"],"val_auroc":vm["auroc"]}
        history.append(row); print(json.dumps(row),flush=True)
        if vm["auprc"]>best_ap:
            best_ap=vm["auprc"]; best_round=rnd
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is None: raise RuntimeError("no global checkpoint")
    model.load_state_dict(best_state)
    vm=eval_metrics(model,va,device); tm=eval_metrics(model,te,device)

    bytes_model=sum(p.numel()*p.element_size() for p in model.parameters())
    # Full participation: one model download + one upload for each client per round.
    comm_per_round_bytes=2*args.clients*bytes_model
    result={
        "stage":"minimal_homogeneous_fl",
        "method":"full-parameter FedAvg",
        "modalities":"ECG_II+PLETH",
        "seed":args.seed,"clients":args.clients,"rounds":args.rounds,"local_epochs":args.local_epochs,
        "client_partition":"subject-disjoint approximately IID; stratified by subject has_positive and greedily balanced by train-window count",
        "best_round":best_round,"selection_metric":"global validation AUPRC",
        "n_params":nparams,"val":vm,"test":tm,
        "communication":{
            "model_bytes":int(bytes_model),"upload_plus_download_bytes_per_round":int(comm_per_round_bytes),
            "total_bytes":int(comm_per_round_bytes*args.rounds)
        },
        "integrity":{
            "client_subject_disjoint":True,"train_val_test_subject_disjoint":True,"all_8_clients_participate_each_round":True
        },
        "wall_seconds":float(time.time()-start),
    }
    (out/"fedavg_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    pd.DataFrame(history).to_csv(out/"round_history.csv",index=False)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
