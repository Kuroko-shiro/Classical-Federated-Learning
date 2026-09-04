#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metrics-root',required=True); ap.add_argument('--qc-summary',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    root=Path(args.metrics_root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for p in root.rglob('metrics.json'):
        d=json.loads(p.read_text())
        t=d['test']; v=d['val']
        rows.append({'condition':d['condition'],'seed':d['seed'],'best_epoch':d['best_epoch'],'threshold':d['threshold'],'n_params':d['n_params'],
                     'auprc':t['auprc'],'auroc':t['auroc'],'f1':t['f1'],'sensitivity':t['sensitivity'],'specificity':t['specificity'],'ppv':t['ppv'],
                     'prevalence':t['prevalence'],'n_windows':t['n_windows'],'val_auprc':v['auprc'],'wall_seconds':d['wall_seconds']})
    df=pd.DataFrame(rows).sort_values(['condition','seed'])
    if len(df)!=9: raise RuntimeError(f'expected 9 metrics files, found {len(df)}')
    df.to_csv(out/'vitaldb_gate0_metrics.csv',index=False)
    stats=df.groupby('condition').agg(
        n_seeds=('seed','size'), mean_auprc=('auprc','mean'), sd_auprc=('auprc','std'),
        mean_auroc=('auroc','mean'), sd_auroc=('auroc','std'), mean_f1=('f1','mean'), sd_f1=('f1','std'),
        mean_sensitivity=('sensitivity','mean'), mean_specificity=('specificity','mean'), mean_ppv=('ppv','mean'),
        prevalence=('prevalence','mean'), n_windows=('n_windows','first'), n_params=('n_params','first')
    ).reset_index()
    stats.to_csv(out/'vitaldb_gate0_summary_by_condition.csv',index=False)

    piv=df.pivot(index='seed',columns='condition',values='auprc')
    piv['best_unimodal']=piv[['ecg_only','pleth_only']].max(axis=1)
    piv['fusion_gain']=piv['ecg_pleth']-piv['best_unimodal']
    piv.reset_index().to_csv(out/'vitaldb_gate0_fusion_gain_by_seed.csv',index=False)
    gain=float(piv.fusion_gain.mean()); gain_sd=float(piv.fusion_gain.std(ddof=1)); positive=int((piv.fusion_gain>0).sum())

    piv_auc=df.pivot(index='seed',columns='condition',values='auroc')
    piv_auc['best_unimodal']=piv_auc[['ecg_only','pleth_only']].max(axis=1)
    piv_auc['fusion_gain']=piv_auc['ecg_pleth']-piv_auc['best_unimodal']
    gain_auc=float(piv_auc.fusion_gain.mean())

    qc=json.loads(Path(args.qc_summary).read_text())
    means={r.condition:r for r in stats.itertuples(index=False)}
    prevalence=float(df.prevalence.mean())
    cohort_pass=qc.get('qc_pass_cases',0)>=2500
    unimodal_pass=(means['ecg_only'].mean_auprc>prevalence and means['pleth_only'].mean_auprc>prevalence)
    fusion_target=gain>=0.02
    sign_pass=positive>=2
    if cohort_pass and unimodal_pass and fusion_target and sign_pass:
        decision='STRONG_GO'
    elif gain>0 and cohort_pass:
        decision='YELLOW'
    else:
        decision='NO_GO'
    decision_note=('Automated project heuristic follows the execution specification: cohort >=2500, both unimodal mean AUPRC above test prevalence, '
                   'mean fusion AUPRC gain >=0.02, and positive gain in >=2/3 seeds. Scientific review of leakage/capacity remains required before FL.')
    summary={'pair':'P1 ECG_II+PLETH','decision':decision,'decision_note':decision_note,'qc_pass_cases':int(qc.get('qc_pass_cases',0)),
             'qc_pass_subjects':int(qc.get('qc_pass_subjects',0)),'test_prevalence_mean':prevalence,
             'fusion_gain_auprc_mean':gain,'fusion_gain_auprc_sd':gain_sd,'fusion_gain_positive_seeds':positive,
             'fusion_gain_auroc_mean':gain_auc,'cohort_pass':cohort_pass,'unimodal_signal_pass':unimodal_pass,'fusion_target_pass':fusion_target,'sign_stability_pass':sign_pass,
             'conditions':{r.condition:{'mean_auprc':r.mean_auprc,'sd_auprc':r.sd_auprc,'mean_auroc':r.mean_auroc,'sd_auroc':r.sd_auroc,'mean_f1':r.mean_f1,'n_params':int(r.n_params)} for r in stats.itertuples(index=False)}}
    (out/'vitaldb_gate0_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=['# VitalDB centralized complementarity Gate 0','',f"Decision: **{decision}**",'',f"QC-passed cases: **{summary['qc_pass_cases']:,}**; subjects: **{summary['qc_pass_subjects']:,}**",f"Mean test prevalence: **{prevalence:.4f}**",'',
        '| condition | AUPRC mean±sd | AUROC mean±sd | F1 mean | params |','|---|---:|---:|---:|---:|']
    for r in stats.itertuples(index=False): md.append(f"| {r.condition} | {r.mean_auprc:.4f} ± {r.sd_auprc:.4f} | {r.mean_auroc:.4f} ± {r.sd_auroc:.4f} | {r.mean_f1:.4f} | {int(r.n_params):,} |")
    md += ['',f"Mean fusion gain (AUPRC): **{gain:+.4f} ± {gain_sd:.4f}**",f"Positive fusion gain seeds: **{positive}/3**",f"Mean fusion gain (AUROC): **{gain_auc:+.4f}**",'',decision_note]
    (out/'vitaldb_gate0_report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
