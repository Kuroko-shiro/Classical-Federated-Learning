#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

SEEDS=[101,202,303]; CONDITIONS=['ecg_only','art_only','ecg_art']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metrics-root',required=True); ap.add_argument('--qc-summary',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); root=Path(a.metrics_root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); qc=json.load(open(a.qc_summary))
    rows=[]
    for s in SEEDS:
        for c in CONDITIONS:
            m=json.load(open(root/f'{c}_seed{s}'/'metrics.json'))
            rows.append({'condition':c,'seed':s,'auprc':m['test']['auprc'],'auroc':m['test']['auroc'],'f1':m['test']['f1'],'n_params':m['n_params'],'best_epoch':m['best_epoch']})
    df=pd.DataFrame(rows); df.to_csv(out/'vitaldb_p2_metrics.csv',index=False)
    sm=df.groupby('condition').agg(mean_auprc=('auprc','mean'),sd_auprc=('auprc','std'),mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),mean_f1=('f1','mean'),n_params=('n_params','first')).reset_index(); sm.to_csv(out/'vitaldb_p2_summary_by_condition.csv',index=False)
    piv=df.pivot(index='seed',columns='condition',values='auprc')
    piv['gain_ecg_art_vs_best_single']=piv['ecg_art']-piv[['ecg_only','art_only']].max(axis=1)
    piv['art_gain_vs_ecg']=piv['art_only']-piv['ecg_only']; piv.reset_index().to_csv(out/'vitaldb_p2_gain_by_seed.csv',index=False)
    means=sm.set_index('condition')['mean_auprc']; p1_fusion=0.11924836632667496; p1_pleth=0.10591363400687867; p1_gain=0.01333473231979629
    p2_gain=float(piv['gain_ecg_art_vs_best_single'].mean()); art_vs_ecg=float(piv['art_gain_vs_ecg'].mean()); pos=int((piv['gain_ecg_art_vs_best_single']>0).sum())
    art_dominates=bool(float(means['art_only'])>p1_fusion and float(means['art_only'])>float(means['ecg_only'])+0.02)
    if art_dominates:
        verdict='VITALDB_P1_MULTIMODAL_CLAIM_WEAK_ART_DOMINATES'
    elif p2_gain>p1_gain:
        verdict='ART_ECG_FUSION_STRONGER_THAN_P1_SENSOR_FUSION'
    else:
        verdict='P1_REMAINS_COMPARABLE_TO_ART_DIAGNOSTIC'
    summary={'diagnostic_only':True,'canonical_p1_gate0_decision':'YELLOW','p2_verdict':verdict,'p2_qc_cases':qc['p2_qc_cases'],'p2_qc_subjects':qc['p2_qc_subjects'],'p2_qc_windows':qc['p2_qc_windows'],
             'mean_auprc':{c:float(means[c]) for c in CONDITIONS},'mean_p2_fusion_gain_vs_best_single':p2_gain,'positive_p2_fusion_gain_seeds':pos,'mean_art_only_gain_vs_ecg':art_vs_ecg,
             'reference_p1':{'ecg_pleth_mean_auprc':p1_fusion,'pleth_only_mean_auprc':p1_pleth,'fusion_gain':p1_gain},
             'art_dominates_p1_fusion':art_dominates,
             'interpretation':'ART uses the same arterial pressure process that defines the future MAP label, so strong ART performance is evidence that the task can collapse toward blood-pressure forecasting; it is diagnostic evidence, not a candidate primary P1 model.'}
    (out/'vitaldb_p2_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# VitalDB P2 ART diagnostic','',f"P1 canonical decision remains: **YELLOW**",f"P2 verdict: **{verdict}**",'',f"Common P2 cohort after ART QC: **{qc['p2_qc_cases']:,} cases / {qc['p2_qc_subjects']:,} subjects / {qc['p2_qc_windows']:,} windows**",'', '| condition | AUPRC mean±sd | AUROC mean±sd | params |','|---|---:|---:|---:|']
    for r in sm.itertuples(index=False): lines.append(f'| {r.condition} | {r.mean_auprc:.4f} ± {r.sd_auprc:.4f} | {r.mean_auroc:.4f} ± {r.sd_auroc:.4f} | {int(r.n_params):,} |')
    lines += ['',f'P2 ECG+ART gain over best P2 single sensor: **{p2_gain:+.4f} AUPRC** ({pos}/3 positive seeds)',f'ART-only − ECG-only: **{art_vs_ecg:+.4f} AUPRC**','',f'Reference P1 ECG+PLETH: AUPRC **{p1_fusion:.4f}**, fusion gain **{p1_gain:+.4f}**.','',summary['interpretation']]
    (out/'vitaldb_p2_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
