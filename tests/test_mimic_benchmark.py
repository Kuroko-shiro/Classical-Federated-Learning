import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qflbench.mimic import audit_access, audit_cohort, build_fl_partition, evaluate_complementarity_gate


def _manifest(n=400):
    rows=[]
    for i in range(n):
        rows.append({"subject_id":i,"stay_id":10000+i,"study_id":20000+i,"dicom_id":f"d{i}","split":"train" if i<320 else ("val" if i<360 else "test"),"mortality":1 if i%7==0 else 0,"has_ehr":True,"has_cxr":True,"cxr_time":"2026-01-01 00:00:00","prediction_time":"2026-01-02 00:00:00"})
    return pd.DataFrame(rows)


def test_access_audit_stops_cleanly_when_data_absent(tmp_path):
    a=audit_access(tmp_path/"missing_iv",tmp_path/"missing_cxr",tmp_path/"out")
    assert not a["ready_for_cohort_reproduction"]


def test_cohort_audit_patient_split_and_timing(tmp_path):
    s=audit_cohort(_manifest(),tmp_path)
    assert s["patient_level_split_separation_pass"] and s["cxr_timing_pass"] and s["leakage_audit_pass"]


def test_gate0_go_requires_positive_all_three_seeds(tmp_path):
    rows=[]
    for seed in range(3):
        rows += [{"condition":"ehr_only","seed":seed,"auprc":.30,"auroc":.70},{"condition":"cxr_only","seed":seed,"auprc":.20,"auroc":.65},{"condition":"multimodal","seed":seed,"auprc":.35+.01*seed,"auroc":.75}]
    r=evaluate_complementarity_gate(pd.DataFrame(rows),tmp_path)
    assert r["decision"]=="GO" and r["fusion_gain_positive_all_seeds"]


def test_partition_is_patient_disjoint_viable_and_modality_controlled(tmp_path):
    r=build_fl_partition(_manifest(),tmp_path,k=10,seed=0,alpha=1.0,min_positive=1,min_negative=1)
    p=r["manifest"]
    assert p.groupby("subject_id")["client_id"].nunique().max()==1
    assert r["client_stats"]["positive"].min()>=1 and r["client_stats"]["negative"].min()>=1
    assert r["summary"]["modality_counts"]["ehr+cxr"]==5
