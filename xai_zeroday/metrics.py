"""
Evaluation metrics and the thresholding / fusion rules.
"""

import numpy as np
from sklearn import metrics


def binary_stats(y_true, y_pred, prefix=''):
    """Same confusion-matrix metrics as compute_performance_stats() in the original code, plus MCC."""
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a) / b if b else float('nan')
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {
        prefix + 'acc': div(tp + tn, tp + tn + fp + fn),
        prefix + 'precision': prec,
        prefix + 'recall': rec,
        prefix + 'f1': div(2 * prec * rec, prec + rec),
        prefix + 'fpr': div(fp, fp + tn),
        prefix + 'mcc': metrics.matthews_corrcoef(y_true, y_pred),
        prefix + 'tp': int(tp), prefix + 'fp': int(fp), prefix + 'tn': int(tn), prefix + 'fn': int(fn),
    }


def score_auc(scores, positive_mask, negative_mask=None, prefix=''):
    """AUROC / AUPR of a continuous anomaly score, positives vs negatives (default: everything else)."""
    if negative_mask is None:
        negative_mask = ~positive_mask
    keep = positive_mask | negative_mask
    y, s = positive_mask[keep].astype(int), scores[keep]
    if y.min() == y.max():
        return {prefix + 'auroc': float('nan'), prefix + 'aupr': float('nan')}
    return {prefix + 'auroc': metrics.roc_auc_score(y, s), prefix + 'aupr': metrics.average_precision_score(y, s)}


# ------------------------------------------------------------------ thresholds (never computed from test data)

def threshold_at_fpr(val_benign_scores, fpr):
    """Threshold so that a fraction `fpr` of benign VALIDATION flows would be flagged."""
    return float(np.quantile(val_benign_scores, 1.0 - fpr))


def threshold_percentile(fit_scores, pct=95):
    """Barnard et al.'s rule: the pct-th percentile of the detector's scores on its own training data."""
    return float(np.percentile(fit_scores, pct))


def or_fusion(p_attack, anomaly_flag):
    """Barnard et al.'s rule: alert if XGBoost says attack OR the anomaly detector says anomalous."""
    return ((p_attack >= 0.5) | anomaly_flag).astype(int)
