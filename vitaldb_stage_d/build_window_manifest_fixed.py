#!/usr/bin/env python3
import numpy as np
import build_window_manifest as b


def stratified_subject_split_fixed(manifest, seed):
    subj = manifest.groupby("subjectid").agg(
        has_positive=("label", "max"),
        n_windows=("label", "size"),
        n_cases=("caseid", "nunique"),
    ).reset_index()
    rng = np.random.default_rng(seed)
    split_map = {}
    for cls in [0, 1]:
        # pandas 3 may expose a read-only NumPy view; copy before shuffling.
        ids = subj.loc[subj["has_positive"] == cls, "subjectid"].to_numpy(copy=True)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        for sid in ids[:n_train]:
            split_map[int(sid)] = "train"
        for sid in ids[n_train:n_train + n_val]:
            split_map[int(sid)] = "val"
        for sid in ids[n_train + n_val:]:
            split_map[int(sid)] = "test"
    subj["split"] = subj["subjectid"].map(split_map)
    return split_map, subj


b.stratified_subject_split = stratified_subject_split_fixed
b.main()
