"""
Pilot study on NSL-KDD (official new-attack split, one seed). Six questions, six tables -> results/quick_study.md

Q1 raw vs SHAP input for the second-stage detector      (Barnard et al.)
Q2 feature reduction: top-10 SHAP features              (Nugraha et al.)
Q3 SHAP vs LIME agreement                               (Nugraha et al., E-XAI)
Q4 explaining the alarm: fidelity / stability / sparsity (E-XAI, X-IDS surveys)
Q5 robustness to SHAP-guided evasion                    (Khan survey / Alani et al., E-XAI)
Q6 cost per flow                                        (Nugraha, E-XAI)

    python quick_study.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from lime.lime_tabular import LimeTabularExplainer

from data import make_split
from detectors import InputTransform, make_detector
from explain import fit_xgb, first_stage, make_explainer, shap_values
from metrics import threshold_at_fpr
from run_experiments import evaluate_scores

warnings.filterwarnings('ignore')
SEED = 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'quick_study.md')
DETECTORS = ['ae_small', 'iforest']
FPR = 0.02          # alert budget: 2% of benign validation flows
CONTROLLABLE = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate']   # attacker-controllable traffic stats
report = []


def emit(title, df, note=''):
    text = f"\n### {title}\n\n" + df.to_markdown(index=False, floatfmt='.3f') + "\n" + (f"\n{note}\n" if note else "")
    print(text, flush=True)
    report.append(text)


def row_for(policy_rows):
    return next(r for r in policy_rows if r['policy'] == 'val_fpr2')


def fit_eval(split, fs, p_test, inp, det_name, train_set, shap_key=''):
    """Fit one detector and evaluate; returns (result row, fitted pieces)."""
    sh = {k: fs[f'shap_{k}{shap_key}'] for k in ('train', 'val', 'test')}
    fit_mask = split.y_train == 0 if train_set == 'benign' else np.ones(len(split.y_train), bool)
    es_mask = split.y_val == 0 if train_set == 'benign' else np.ones(len(split.y_val), bool)
    T = InputTransform(inp).fit(split.X_train[fit_mask], sh['train'][fit_mask])
    Z = {k: T.transform(getattr(split, f'X_{k}'), sh[k]) for k in ('train', 'val', 'test')}
    t0 = time.time()
    det = make_detector(det_name, SEED).fit(Z['train'][fit_mask], Z['val'][es_mask])
    t_fit = time.time() - t0
    s_fit, s_val = det.score(Z['train'][fit_mask]), det.score(Z['val'])
    t0 = time.time()
    s_test = det.score(Z['test'])
    t_score = (time.time() - t0) / len(s_test)
    row = row_for(evaluate_scores({}, split, p_test, s_fit, s_val, s_test))
    return row, dict(row=row, det=det, T=T, Z=Z, s_val=s_val, s_test=s_test, t_fit=t_fit, t_score=t_score,
                     thr=threshold_at_fpr(s_val[split.y_val == 0], FPR))


def main():
    split = make_split('nsl_kdd', 'official', seed=SEED)
    zd, normal = split.zd_test, split.y_test == 0
    print(f"NSL-KDD official split: {len(split.y_train)} train, {zd.sum()} zero-day test flows, {normal.sum()} normal test flows")
    fs = first_stage(split, SEED)
    p_test = fs['p_test']
    model = fit_xgb(split.X_train, split.y_train, SEED)
    xgb_pred = (p_test >= 0.5)
    print(f"XGBoost alone: acc={(xgb_pred == split.y_test).mean():.3f}, zero-day recall={xgb_pred[zd].mean():.3f}, FPR={xgb_pred[normal].mean():.3f}")

    # ---------------------------------------------------------------- Q1
    q1, keep = [], {}
    for train_set in ('benign', 'all'):
        for inp in ('raw', 'shap', 'concat'):
            for d in DETECTORS:
                r, k = fit_eval(split, fs, p_test, inp, d, train_set)
                keep[(train_set, inp, d)] = k
                q1.append({'train data': train_set, 'input': inp, 'detector': d, 'zero-day AUROC': r['zd_auroc'],
                           'AUROC among XGB-missed': r['res_auroc'], 'zero-day recall (2nd stage)': r['ad_zd_recall'],
                           'fused zero-day recall': r['fused_zd_recall'], 'fused FPR': r['fused_fpr'], 'fused F1': r['fused_f1'],
                           'fused MCC': r['fused_mcc']})
    base = pd.DataFrame([{'train data': '-', 'input': '-', 'detector': 'XGBoost only', 'zero-day AUROC': np.nan,
                          'AUROC among XGB-missed': np.nan, 'zero-day recall (2nd stage)': np.nan,
                          'fused zero-day recall': xgb_pred[zd].mean(), 'fused FPR': xgb_pred[normal].mean(),
                          'fused F1': 2 * (xgb_pred[split.y_test == 1] .mean() * (split.y_test[xgb_pred] == 1).mean()) /
                                      (xgb_pred[split.y_test == 1].mean() + (split.y_test[xgb_pred] == 1).mean()),
                          'fused MCC': np.corrcoef(xgb_pred, split.y_test)[0, 1]}])
    emit(f"Q1. Raw vs SHAP input to the second stage (threshold = {FPR:.0%} FPR on validation, OR-fused with XGBoost)",
         pd.concat([base, pd.DataFrame(q1)], ignore_index=True),
         "Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.")

    # ---------------------------------------------------------------- Q2
    imp = np.abs(fs['shap_train']).mean(0)
    top = np.argsort(imp)[::-1][:10]
    names = np.array(split.feature_names)
    m10 = fit_xgb(split.X_train[:, top], split.y_train, SEED)
    ex10 = make_explainer(m10, split.X_train[:, top], SEED)
    t0 = time.time()
    fs10 = {'p_test': m10.predict_proba(split.X_test[:, top])[:, 1]}
    for k in ('train', 'val', 'test'):
        fs10[f'shap_{k}'] = shap_values(ex10, getattr(split, f'X_{k}')[:, top])
    t_shap10 = (time.time() - t0) / (len(split.X_train) + len(split.X_val) + len(split.X_test))
    sp10 = type('S', (), {})()
    sp10.__dict__.update(split.__dict__)
    for k in ('train', 'val', 'test'):
        setattr(sp10, f'X_{k}', getattr(split, f'X_{k}')[:, top])
    p10 = fs10['p_test']
    pred10 = p10 >= 0.5
    q2 = [{'features': 'all 41', 'XGB acc': (xgb_pred == split.y_test).mean(), 'XGB zd recall': xgb_pred[zd].mean(),
           'SHAP ms/flow': float(fs['t_shap_per_flow']) * 1e3, 'AE(SHAP) zd AUROC': keep[('benign', 'shap', 'ae_small')]['row']['zd_auroc']}]
    r10, _ = fit_eval(sp10, fs10, p10, 'shap', 'ae_small', 'benign')
    q2.append({'features': 'top 10 by mean|SHAP|', 'XGB acc': (pred10 == split.y_test).mean(), 'XGB zd recall': pred10[zd].mean(),
               'SHAP ms/flow': t_shap10 * 1e3, 'AE(SHAP) zd AUROC': r10['zd_auroc']})
    emit("Q2. Feature reduction (Nugraha et al.)", pd.DataFrame(q2), "Top-10 features: " + ', '.join(names[top]))

    # ---------------------------------------------------------------- Q3
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(split.X_test), 200, replace=False)
    lime = LimeTabularExplainer(split.X_train[rng.choice(len(split.X_train), 20000, replace=False)], feature_names=list(names),
                                discretize_continuous=False, mode='classification', random_state=SEED)
    t0 = time.time()
    overlaps, rank_agree_top1 = [], []
    for i in idx:
        e = lime.explain_instance(split.X_test[i], model.predict_proba, num_features=len(names), num_samples=1000, labels=(1,))
        w = np.zeros(len(names))
        for f, v in e.as_map()[1]:
            w[f] = v
        s_top = set(np.argsort(-np.abs(fs['shap_test'][i]))[:5])
        l_top = set(np.argsort(-np.abs(w))[:5])
        overlaps.append(len(s_top & l_top) / 5)
        rank_agree_top1.append(np.argmax(np.abs(fs['shap_test'][i])) == np.argmax(np.abs(w)))
    t_lime = (time.time() - t0) / len(idx)
    ov = np.array(overlaps)
    emit("Q3. SHAP vs LIME agreement on 200 random test flows",
         pd.DataFrame([{'mean top-5 overlap': ov.mean(), 'flows with overlap >= 0.6': (ov >= 0.6).mean(),
                        'top-1 feature agrees': np.mean(rank_agree_top1), 'LIME s/flow': t_lime,
                        'SHAP ms/flow': float(fs['t_shap_per_flow']) * 1e3}]),
         "Overlap < 0.5 is flagged for analyst review in Nugraha et al.; here we report the raw overlap.")

    # ---------------------------------------------------------------- Q4
    k = keep[('benign', 'shap', 'ae_small')]
    det, Z = k['det'], k['Z']
    flagged = np.where(k['s_test'] > k['thr'])[0]
    flagged_zd = flagged[zd[flagged]]
    sub = flagged_zd[rng.permutation(len(flagged_zd))[:500]]

    @torch.no_grad()
    def per_feature_err(model, X):
        t = torch.from_numpy(X)
        return (model(t) - t).abs().numpy()

    err = per_feature_err(det.model, Z['test'][sub])
    top5 = np.argsort(-err, axis=1)[:, :5]
    # sparsity: share of total reconstruction error carried by the top-5 features
    sparsity = np.take_along_axis(err, top5, 1).sum(1) / err.sum(1)
    # fidelity (deletion): set the top-5 abnormal explanation features to the benign mean; compare score drop to random-5
    mu = Z['train'][split.y_train == 0].mean(0)
    Zs = Z['test'][sub].copy()
    Zd, Zr = Zs.copy(), Zs.copy()
    for j in range(len(sub)):
        Zd[j, top5[j]] = mu[top5[j]]
        r5 = rng.choice(Zs.shape[1], 5, replace=False)
        Zr[j, r5] = mu[r5]
    s0, sd, sr = det.score(Zs), det.score(Zd), det.score(Zr)
    # stability: top-5 abnormal features from a second AE trained with a different seed
    det2 = make_detector('ae_small', SEED + 1).fit(Z['train'][split.y_train == 0], Z['val'][split.y_val == 0])
    top5b = np.argsort(-per_feature_err(det2.model, Z['test'][sub]), axis=1)[:, :5]
    stab = np.array([len(set(a) & set(b)) / 5 for a, b in zip(top5, top5b)])
    emit("Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)",
         pd.DataFrame([{'flows analysed': len(sub), 'score drop, top-5 abnormal feats': ((s0 - sd) / s0).mean(),
                        'score drop, random 5 feats': ((s0 - sr) / s0).mean(),
                        'stability (top-5 overlap, 2 seeds)': stab.mean(), 'sparsity (error share in top-5 of 41)': sparsity.mean()}]),
         "Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than replacing random ones.")
    common = pd.Series(names[top5.ravel()]).value_counts().head(6)
    emit("Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows",
         pd.DataFrame({'feature': common.index, 'times in top-5': common.values}))

    # ---------------------------------------------------------------- Q5
    attack_hit = np.where((split.y_test == 1) & (p_test >= 0.5))[0]
    sel = rng.choice(attack_hit, 300, replace=False)
    ctrl = np.array([split.feature_names.index(c) for c in CONTROLLABLE])
    benign_med = np.median(split.X_train[split.y_train == 0], axis=0)
    ex = make_explainer(model, split.X_train, SEED)
    Xa = split.X_test[sel].copy()
    changed = np.zeros(len(sel), int)
    for step in range(5):
        pa = model.predict_proba(Xa)[:, 1]
        act = np.where(pa >= 0.5)[0]
        if len(act) == 0:
            break
        sv = shap_values(ex, Xa[act])
        for n, j in enumerate(act):
            c = ctrl[np.argsort(-sv[n, ctrl])]
            for f in c:                       # most attack-pushing controllable feature not yet at the benign median
                if Xa[j, f] != benign_med[f]:
                    Xa[j, f] = benign_med[f]
                    changed[j] += 1
                    break
    pa = model.predict_proba(Xa)[:, 1]
    evaded = pa < 0.5
    q5 = [{'XGBoost evasion success': evaded.mean(), 'median features changed (evaded)': np.median(changed[evaded]) if evaded.any() else np.nan}]
    emit("Q5a. SHAP-guided evasion of XGBoost (300 attacks XGBoost caught; up to 5 controllable features moved to benign median)", pd.DataFrame(q5))
    if evaded.sum() > 0:
        sh_ev = shap_values(ex, Xa[evaded])
        q5b = []
        for inp in ('raw', 'shap', 'concat'):
            for d in DETECTORS:
                kk = keep[('benign', inp, d)]
                Zev = kk['T'].transform(Xa[evaded], sh_ev)
                Zor = kk['T'].transform(split.X_test[sel][evaded], fs['shap_test'][sel][evaded])
                q5b.append({'input': inp, 'detector': d, 'evaded attacks caught by 2nd stage': (kk['det'].score(Zev) > kk['thr']).mean(),
                            'same attacks before evasion': (kk['det'].score(Zor) > kk['thr']).mean(),
                            'benign flows flagged': (kk['s_test'][normal] > kk['thr']).mean()})
        emit("Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)", pd.DataFrame(q5b))

    # ---------------------------------------------------------------- Q6
    q6 = [{'component': 'XGBoost fit (s)', 'cost': float(fs['t_xgb_fit'])},
          {'component': 'TreeSHAP, 41 feat (ms/flow, 8 cores)', 'cost': float(fs['t_shap_per_flow']) * 1e3},
          {'component': 'TreeSHAP, top-10 feat (ms/flow, 8 cores)', 'cost': t_shap10 * 1e3},
          {'component': 'LIME (ms/flow, 1 core)', 'cost': t_lime * 1e3}]
    for d in DETECTORS:
        kk = keep[('benign', 'shap', d)]
        q6 += [{'component': f'{d} fit (s)', 'cost': kk['t_fit']}, {'component': f'{d} score (ms/flow)', 'cost': kk['t_score'] * 1e3}]
    emit("Q6. Cost", pd.DataFrame(q6))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write("# Pilot study: NSL-KDD official split, seed 0\n" + ''.join(report))
    print(f"\nWrote {OUT}")


if __name__ == '__main__':
    main()
