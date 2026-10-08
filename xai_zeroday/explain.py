"""
First stage: XGBoost classifier + TreeSHAP explanations (as in Barnard et al. 2022), with on-disk caching.
"""

import multiprocessing as mp
import os
import time

import numpy as np
import shap
import xgboost

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SHAP_BACKGROUND = 500


def fit_xgb(X, y, seed):
    """XGBoost with the defaults Barnard et al. used (xgboost 1.x defaults written out explicitly)."""
    model = xgboost.XGBClassifier(objective="binary:logistic", n_estimators=100, max_depth=6, learning_rate=0.3,
                                  random_state=seed, n_jobs=os.cpu_count())
    model.fit(X, y)
    return model


def make_explainer(model, X_train, seed, background_n=SHAP_BACKGROUND):
    """Interventional TreeSHAP decomposing P(attack|x), as in the original code, but with a fixed-size background sample."""
    rng = np.random.default_rng(seed)
    background = X_train[rng.choice(len(X_train), size=min(background_n, len(X_train)), replace=False)]
    return shap.TreeExplainer(model, background, feature_perturbation="interventional", model_output="probability")


_EXPLAINER = None  # set before forking so worker processes inherit it


def _shap_chunk(X_chunk):
    return np.asarray(_EXPLAINER.shap_values(X_chunk), dtype=np.float32)


def shap_values(explainer, X, n_jobs=None):
    """Interventional TreeSHAP is single-threaded in the shap package, so split rows across forked worker processes."""
    global _EXPLAINER
    n_jobs = n_jobs or os.cpu_count()
    if n_jobs == 1 or len(X) < 2000:
        return np.asarray(explainer.shap_values(X), dtype=np.float32)
    _EXPLAINER = explainer
    chunks = np.array_split(X, n_jobs * 4)
    with mp.get_context('fork').Pool(n_jobs) as pool:
        out = pool.map(_shap_chunk, chunks)
    _EXPLAINER = None
    return np.concatenate(out, axis=0)


def first_stage(split, seed, cache=True):
    """
    Fit XGBoost on split.X_train, then return a dict with predicted attack probabilities and SHAP vectors
    for train/val/test, plus timing. Results are cached per (split, seed).
    """
    key = split.name.replace('/', '__') + f'__seed{seed}__n{len(split.X_train)}'
    path = os.path.join(CACHE_DIR, key + '.npz')
    if cache and os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}

    t0 = time.time()
    model = fit_xgb(split.X_train, split.y_train, seed)
    t_fit = time.time() - t0

    res = {'t_xgb_fit': np.array(t_fit)}
    for part in ['train', 'val', 'test']:
        res[f'p_{part}'] = model.predict_proba(getattr(split, f'X_{part}'))[:, 1].astype(np.float32)

    explainer = make_explainer(model, split.X_train, seed)
    t0 = time.time()
    for part in ['train', 'val', 'test']:
        res[f'shap_{part}'] = shap_values(explainer, getattr(split, f'X_{part}'))
    n_explained = len(split.X_train) + len(split.X_val) + len(split.X_test)
    res['t_shap_per_flow'] = np.array((time.time() - t0) / n_explained)

    # local accuracy check: base value + sum of SHAP values should reproduce P(attack|x)
    err = np.abs(explainer.expected_value + res['shap_test'].sum(1) - res['p_test']).max()
    assert err < 1e-3, f"SHAP additivity violated (max err {err})"

    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, **res)
    return res
