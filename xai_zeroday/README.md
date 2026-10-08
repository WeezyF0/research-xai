# xai_zeroday: does the explanation domain help zero-day detection?

Controlled re-evaluation of the XGBoost -> TreeSHAP -> autoencoder pipeline (Barnard et al. 2022),
plus a pilot study covering feature reduction, SHAP-vs-LIME agreement, explaining alarms, evasion robustness and cost.

## Setup
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python numpy pandas scikit-learn xgboost shap scipy matplotlib torch lime tabulate

## Data (committed under data/)
- NSL-KDD: KDDTrain+.txt, KDDTest+.txt -> data/nsl_kdd/  (github.com/defcom17/NSL_KDD)
- UNSW-NB15 official train/test CSVs with attack_cat -> data/unsw_nb15/{train,test}.csv
  (huggingface.co/datasets/Mireu-Lab/UNSW-NB15; note the file names are swapped vs the official partitions, handled in data.py)

## Run
    python quick_study.py            # pilot study, NSL-KDD official split -> results/quick_study.md
    python run_experiments.py --quick   # smoke test of the full grid runner
    python run_experiments.py           # full grid (NSL-KDD + UNSW-NB15 leave-one-family-out, 5 seeds); resumable

## Files
data.py (loaders, zero-day splits) | explain.py (XGBoost + parallel TreeSHAP, cached) | detectors.py (AE, IForest, kNN, PCA)
| metrics.py (metrics, thresholds from validation only, OR fusion) | run_experiments.py | quick_study.py
