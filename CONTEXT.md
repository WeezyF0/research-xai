# Project context: explanation-based zero-day intrusion detection

Handover document for anyone, human or LLM, picking up this project. It covers the goal, what was read, what was built, every result, the decisions made and why, known problems, and what is left to do. Last updated 2026-10-08.

---

## 1. The team and the goal

- **Team:** B.Tech undergraduates writing a research paper. Keep proposals simple and feasible; avoid speculative ideas.
- **Core idea they started from** (Barnard et al., 2022):
  1. A tree model (XGBoost) detects intrusions.
  2. SHAP's `TreeExplainer` turns each prediction into an explanation vector.
  3. An autoencoder trained on those explanation vectors reconstructs them.
  4. A flow whose explanation reconstructs badly (high reconstruction error) is flagged as a possible new (zero-day) attack.
- **Motivation:** trees are good at known attacks but weak at novel ones; deep models are good at novelty but weak at explainability. The combination aims to get both.
- **Paper format:** IEEE conference template (`IEEE Template.doc` in the repo root: Word, A4, two-column). We write in LaTeX with the `IEEEtran` conference class, which is IEEE's LaTeX version of the same template.
- **Constraint from the user:** only a few hours were available, so experiments were scoped to run in roughly 1–3 hours of CPU time.

---

## 2. Repository layout

```
research-xai/
├── CONTEXT.md                       <- this file
├── README.md                        <- user-edited (currently a placeholder)
├── IEEE Template.doc                <- target paper format (IEEE conference, Word)
├── *.pdf, *.docx                    <- papers we read + the team's own notes (see §3)
├── Robust Network Intrusion Detection through Explainable Artificial Intelligence (XAI)/   <- Barnard et al. 2022 original code [P1]
├── Don’t Just Explain, Enhance! .../                                                      <- Barnard et al. follow-up code [P2]
├── xai_zeroday/                     <- OUR experiment harness (all new code is here)
│   ├── data.py                      loaders + zero-day splits (NSL-KDD, UNSW-NB15)
│   ├── explain.py                   XGBoost + parallel interventional TreeSHAP, cached
│   ├── detectors.py                 input transforms + AE / IsolationForest / kNN / PCA detectors
│   ├── metrics.py                   metrics, validation-only thresholds, OR fusion
│   ├── run_experiments.py           full-grid runner (C1 ablation); resumable CSV output
│   ├── quick_study.py               six-question study (Q1–Q6); used for pilot + extended runs
│   ├── run_extended.sh              batch script for the extended pilot (12 runs)
│   ├── aggregate_extended.py        mean±std + Wilcoxon over extended runs
│   ├── data/nsl_kdd/, data/unsw_nb15/   datasets (committed at the user's request)
│   ├── results/quick_study.md       pilot results (NSL-KDD, seed 0)
│   ├── results/FINDINGS.md          pilot findings summary
│   └── results/extended/            extended results: summary.md + per-run report.md and q*.csv
└── paper/
    ├── refs.bib                     shared bibliography (all entries verified, see §9)
    ├── pilot/main.tex               paper from the pilot run
    ├── extended/main.tex            paper from the extended run (more reliable)
    └── README.md                    how to compile on Overleaf
```

Git-ignored (not in the repo): `xai_zeroday/.venv/` (Python env), `xai_zeroday/cache/` (SHAP caches), `*.log`.

---

## 3. What was read

The four journal PDFs, the README and the two `.docx` notes were read in full. The three long surveys (Rjoub, Khan, Neupane) were read by helper agents, which extracted the relevant sections. Key takeaways:

| File | Paper | What we took from it |
|---|---|---|
| `XAI_intrusion_detection_final.pdf` | **Barnard, Marchetti, DaSilva**, "Robust Network Intrusion Detection Through Explainable AI (XAI)", *IEEE Networking Letters* 4(3):167–171, 2022 | The base pipeline. NSL-KDD only. XGBoost acc 79.2% → full pipeline 93.3%. The AE is trained on SHAP of *all* training data, with a threshold at the 95th percentile of training error and OR fusion. |
| `A_versatile_XAI-based_framework_for_efficient_and_.pdf` | **Nugraha et al.**, *Annals of Telecommunications* 80(11–12):1095–1120, 2025 | ANOVA + SHAP feature reduction to the top 10 features; a SHAP–LIME consistency flag (top-5 overlap κ < 0.5 → analyst review); explanation latency (LIME 36 s → 4.9 s, an 87% speed-up). Basis for our Q2, Q3 and Q6. |
| `E-XAI_Evaluating_Black-Box_Explainable_AI_Framewor.pdf` | **Arreche et al.**, "E-XAI", *IEEE Access* 12:23954–23988, 2024 | Six explanation metrics: descriptive accuracy, sparsity, stability, efficiency, robustness, completeness. Uses Wilcoxon tests. Its robustness test is based on Slack et al., "Fooling LIME and SHAP". Basis for our Q4 metrics and statistics. |
| `Explainable_Intrusion_Detection_Systems_X-IDS_A_Su.pdf` | **Neupane et al.**, X-IDS survey, *IEEE Access* 10:112392–112415, 2022 | Open challenges VI-A to VI-F: definitions, stakeholders, **evaluation metrics**, **adversarial AI**, **misleading explanations**, **scalability**. Cites Islam et al. for leave-one-attack-out testing, which justifies our protocol. Does not cite Barnard. |
| `Explainable_AI-based_Intrusion_Detection_System_fo.pdf` | **Khan et al.**, X-IDS for Industry 5.0 survey, arXiv:2408.03335, 2024 | No surveyed work feeds SHAP vectors into a second detector (supports novelty). Table 5 lists attacks that use SHAP to *evade* XGBoost detectors (Alani et al.), which motivates Q5. |
| `Survey_on_Explainable_Artificial_Intelligence__XAI__TNNLS.pdf` | **Rjoub et al.**, "A Survey on XAI for Cybersecurity", *IEEE TNSM* 20(4):5115–5140, 2023 | The file name says TNNLS but the paper is in TNSM. Provides a citable gap statement: evaluation of XAI in security is lacking. Thin on SHAP specifics. |
| `A Comprehensive Analysis of Explainable AI.docx`, `Explainable Intrusion Detection Systems.docx` | **The team's own reading notes** | The team's stated interests are evaluation metrics, adversarial robustness of explanations, and misleading explanations. |

Related work found by web search: Antwarg et al., *Expert Systems with Applications* 186:115736 (2021), explains autoencoder anomalies *with* SHAP. This is the opposite direction to Barnard and must be cited as related work.

---

## 4. Problems found in the original Barnard code and paper

These motivate our work and are written into the papers:

1. **No controlled ablation.** The same AE and threshold were never run on *raw features*, so it is unknown whether the gain comes from SHAP or from adding any anomaly detector with OR fusion.
2. **The false-alarm rate rose about 4×, and this was unreported.** Recomputed from Barnard's Table I (KDDTest+ has 9,711 normal and 12,833 attack flows): XGBoost alone FPR ≈ 2.9%; full pipeline FPR ≈ 12.7%.
3. **The AUC is not a real ROC.** `second_stage()` in the follow-up `utility_funcs.py` sets the probability to 1.0 for flagged flows instead of using the AE score as a continuous score.
4. **Inconsistent AE training data.** P1 trains on all training data; the P2 Exp1 script trains on benign data only. The two are never compared.
5. **Tuning on test data.**
   - `CICIDS_17_18_Generalisation.py` sweeps the threshold percentile on the test set.
   - `NIDS_NSL_normal_PRETRAIN.py` uses test-set normal flows as grid-search validation.
   - P1's `get_hyper_Autoencoder` picks the "best" model by *training* loss.
6. **One seed and one fixed zero-day split.**
7. **Oversized AE** (41→1456→724→14→632→1644→41) with no comparison against a small one.

---

## 5. Decisions made with the user (in order)

1. **First plan:** a controlled re-evaluation only ("C1": raw vs SHAP, several detectors, leave-one-family-out splits, 5 seeds, NSL-KDD + UNSW-NB15). The full harness for this exists in `run_experiments.py`, but the **full grid was never run**.
2. **The user then asked for "the simplest thing that encompasses everything"**, so the scope became six questions, each tied to a paper:
   - **Q1** Raw vs SHAP input (Barnard)
   - **Q2** Top-10 SHAP feature reduction (Nugraha)
   - **Q3** SHAP vs LIME agreement (Nugraha, E-XAI)
   - **Q4** Explaining the second-stage alarm: fidelity, stability, sparsity (E-XAI, surveys)
   - **Q5** Robustness to SHAP-guided evasion (Khan/Alani, E-XAI, team notes)
   - **Q6** Cost per flow (Nugraha, E-XAI, Neupane)
3. **Only a few hours were available**, so the scope became a **pilot**: NSL-KDD official split, one seed, two detectors (small AE and Isolation Forest). This is `quick_study.py`.
4. **Extended pilot** (requested after the pilot): NSL-KDD official plus UNSW-NB15 with DoS, Exploits and Reconnaissance held out, 3 seeds each, 12 runs in total. To fit the time budget, SHAP and the second-stage detectors use a **50k training / 20k validation subsample**. XGBoost still trains on the full training set, and the test set is always used in full.
5. **Paper format:** LaTeX with IEEEtran. Two papers were written, one per run; the user will decide which to use.
6. **The user asked to commit the datasets too**, so they are in git. The largest file is 32 MB, under GitHub's limit.

---

## 6. Method (the harness)

- **Environment:** `uv venv --python 3.11 xai_zeroday/.venv`, then install `numpy pandas scikit-learn xgboost shap scipy matplotlib torch lime tabulate`.
  - Tested versions: xgboost 3.2, shap 0.51, scikit-learn 1.9, torch 2.14 (CPU).
  - The machine has 8 threads and no GPU.
- **First stage:** `XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3)`, i.e. Barnard's defaults written out explicitly.
- **SHAP:** `shap.TreeExplainer(model, background_500_rows, feature_perturbation="interventional", model_output="probability")`.
  - The shap package runs this single-threaded, so `explain.py` forks 8 worker processes; about 0.8–1 ms per flow.
  - Additivity is asserted (max error < 1e-3).
- **Second-stage inputs:**
  - Raw: signed `log1p`, then MinMax to (−1, 1).
  - SHAP: MinMax to (−1, 1).
  - Concat: both side by side.
  - Scalers are fit on detector-training rows only.
- **Detectors:**
  - `ae_small`: d→64→16→64→d, ReLU, tanh output, MAE loss, Adam 1e-3, batch 512, early stopping on validation with patience 10.
  - `iforest`: 200 trees.
  - Also implemented but not used in the pilot runs: `ae_p1` (Barnard's large AE), `knn`, `pca`.
- **Threshold:** the score at which 2% of **benign validation** flows are flagged. Test data are never used for fitting or tuning.
- **Fusion:** alert if `p_xgb ≥ 0.5` OR the detector flags the flow (Barnard's rule).
- **Detector training data:** benign-only or all training flows.
- **Zero-day definitions:**
  - NSL-KDD official: the 3,750 KDDTest+ flows whose attack type is absent from KDDTrain+ (matches Barnard's code exactly).
  - UNSW-NB15 leave-one-family-out: remove the family from train/val; its test rows are the zero-days.
- **Data notes:**
  - UNSW-NB15 comes from the Hugging Face mirror `Mireu-Lab/UNSW-NB15`.
  - The CSV names there are swapped: `test.csv` (175,341 rows) is the official *training* partition and `train.csv` (82,332 rows) is the official *testing* partition. `data.py` handles this.
  - An earlier GitHub mirror (InitRoot) was dropped because it has no `attack_cat` column.
- **Q-specific details** (all in `quick_study.py`):
  - **Q2:** retrain XGBoost on the 10 features with the highest mean |SHAP|.
  - **Q3:** LIME with 1000 samples and no discretisation, on 200 test flows; compare the top-5 sets.
  - **Q4:**
    - Alarm explanation = the top-5 features by per-feature AE reconstruction error on the SHAP vector.
    - Deletion test: set those 5 to the benign mean, compare the score drop with 5 random features.
    - Stability = top-5 overlap with an AE trained from another seed.
    - Sparsity = share of the error carried by the top 5.
  - **Q5:**
    - Take 300 attacks that XGBoost detects.
    - For up to 5 rounds, set the attacker-controllable feature with the largest positive SHAP value to its benign median, stopping once XGBoost says benign.
    - The 14 controllable features per dataset are listed in `CONTROLLABLE` in `quick_study.py`.
    - Then measure how many evading flows each second stage still flags.

---

## 7. Results

### 7.1 Pilot (NSL-KDD official, seed 0, full training set): `xai_zeroday/results/quick_study.md`

- **Q1:**
  - XGBoost alone: zero-day recall 0.379, FPR 0.027, F1 0.786.
  - Best: SHAP AE trained on all data → fused zero-day recall 0.785, F1 0.888 (raw AE: 0.600, 0.849).
  - Among flows XGBoost misses, AUROC is 0.94–0.98 for SHAP vs 0.90–0.94 for raw.
  - SHAP fused FPR is 3.9–4.8% vs 3.1–3.4% for raw.
  - Whole-test zero-day AUROC is not better for SHAP (benign AE: 0.671 SHAP vs 0.747 raw).
- **Q2:** top 10 features → XGBoost accuracy 0.795→0.786, zero-day recall 0.379→0.335; SHAP time did not fall (0.84→0.95 ms).
- **Q3:** SHAP–LIME top-5 overlap 0.50; top-1 agreement 81.5%.
- **Q4:** deletion drop 0.459 vs 0.052 for random features; stability 0.73; sparsity 0.53.
- **Q5:** evasion beats XGBoost on 89% of attacks (median 2 features changed); the SHAP AE still catches 94.8% (raw AE 82.4%), but the SHAP IForest catches only 69.3% (raw 78.3%).
- **Q6:** SHAP 0.84 ms/flow (8 workers), LIME 37 ms/flow (1 core), AE fit 69 s, IForest fit 1.5 s.

### 7.2 Extended pilot (NSL-KDD + UNSW-NB15 DoS/Exploits/Reconnaissance, seeds 0–2, 50k subsample): `xai_zeroday/results/extended/summary.md`

These results are more reliable than the pilot and change its message:

- **Paired Wilcoxon test, SHAP vs raw (48 pairs):**

  | Metric | Mean (SHAP − raw) | SHAP better in | p |
  |---|---|---|---|
  | AUROC on XGB-missed flows | +0.222 | 98% of pairs | < 0.001 |
  | Fused zero-day recall | +0.042 | 94% | < 0.001 |
  | Fused FPR (worse) | +0.012 | — | < 0.001 |
  | Fused F1 | — | — | 0.92, not significant |
  | Fused MCC | — | — | 0.84, not significant |
  | Whole-test zero-day AUROC | — | — | 0.53, not significant |

- **NSL-KDD:** XGBoost zero-day recall 0.348; benign-trained SHAP AE gives fused recall 0.770 and F1 0.901 (raw AE: 0.654, 0.852).
- **UNSW-NB15:** XGBoost alone already catches 97.3–99.8% of each held-out family, because the binary "attack" class generalises, and it has an FPR of about 25% on the test partition. The second stage mostly adds false alarms here.
- **Q2:** feature reduction hurts zero-day detection on NSL-KDD (XGBoost recall 0.348→0.274; AE AUROC 0.645→0.603); mixed on UNSW-NB15.
- **Q3:** top-5 overlap 0.44–0.55 on every split; top-1 agreement 50% on NSL-KDD (large variance across seeds) and 14–29% on UNSW-NB15.
- **Q4:** the deletion test passes on every split (0.27–0.44 vs about 0); stability 0.59–0.76. UNSW DoS/Reconnaissance have few flagged zero-days (43 and 15).
- **Q5:** evasion works on NSL-KDD (89.6%; SHAP AE recovers 96.4% vs raw AE 76.4%). On UNSW-NB15 it only succeeds for 3–5% of attacks (about 8–14 flows), too few to conclude anything.
- **Q6:** SHAP about 1 ms/flow, LIME 38–39 ms/flow, AE fit 23–34 s on 50k flows, IForest 1.2–1.8 s.

**One-line message of the extended paper:** explanations help exactly where the classifier fails to generalise; they significantly improve separation of *missed* zero-days and make alarms explainable, but they raise false alarms and give no net F1 gain when the classifier already catches the held-out families.

---

## 8. Papers

- `paper/pilot/main.tex`: "Does Explaining Help Detecting? A Pilot Evaluation of SHAP-Based Zero-Day Network Intrusion Detection". Single seed, NSL-KDD.
- `paper/extended/main.tex`: "Where Do Explanations Help? A Multi-Criteria Evaluation of SHAP-Based Zero-Day Network Intrusion Detection". Recommended.
- Both papers:
  - IEEEtran conference class, sections I–VII, tables with captions above.
  - Author blocks are placeholders.
  - Never compiled locally (no LaTeX on the machine): compile on Overleaf by uploading the whole `paper/` folder and setting the main file.
- Every number in both papers was checked against the results files.
- Two speculative claims in the extended paper were softened: the reason evasion fails on UNSW, and the UNSW train/test shift.

---

## 9. References: verification status

Every entry in `paper/refs.bib` was checked against **Crossref** DOI metadata, **arXiv** (Khan et al.) or the **NeurIPS** proceedings page (Lundberg & Lee). All matched. Fixes applied:
- Added issue 11–12 to Nugraha et al.
- Added volume 30 to Lundberg & Lee, and dropped its page numbers because the official proceedings page lists none.

---

## 10. Git, credentials, and the machine

- Remote: `https://github.com/WeezyF0/research-xai`, branch `main`.
- **Commit attribution:** the user's git identity (Aditya) with a `Co-Authored-By: Claude …` trailer.
- **Pushing:**
  - The assistant cannot push. There are no stored credentials, and an attempt to use a pasted token was blocked by Claude Code's credential-leak safety check.
  - The user pushes manually with `git push origin main` and a GitHub fine-grained personal access token as the password.
  - **A token was pasted into the chat once and should be considered compromised and revoked.**
- **Remote edits:** the user sometimes edits files on GitHub (e.g. `README.md`). Run `git fetch` and `git rebase origin/main` before pushing.
- **Machine:** remote Linux over SSH, 8 threads, about 11 GB RAM, no GPU, no LaTeX, no `gh` CLI, no `unzip` (Python `zipfile` was used instead).

---

## 11. How to reproduce

```bash
cd xai_zeroday
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy pandas scikit-learn xgboost shap scipy matplotlib torch lime tabulate
.venv/bin/python quick_study.py                 # pilot -> results/quick_study.md (~20 min)
./run_extended.sh                               # extended pilot, 12 runs (~2.5 h); skips finished runs
.venv/bin/python aggregate_extended.py          # -> results/extended/summary.md
.venv/bin/python run_experiments.py --quick     # smoke test of the C1 grid runner
.venv/bin/python run_experiments.py             # full C1 grid (never run; ~4-6 h)
```

---

## 12. Known limitations and open items

- **Not done yet:**
  - The full 5-seed C1 grid (`run_experiments.py`).
  - Reproducing Barnard's numbers with their large AE (`--repro`).
  - CIC-IDS2017.
  - More UNSW families.
  - kNN/PCA detectors in the Q1–Q6 study.
- **The evasion attack is naive:** greedy, with a hand-picked controllable-feature list, and it does not know about the second stage. An adaptive attacker remains future work.
- **Statistical caveats:**
  - Both datasets show a train/test shift that inflates the FPR (UNSW about 25% even for XGBoost alone).
  - On UNSW-NB15, the "missed zero-day" subsets and the evaded-flow subsets are small.
- **The extended run used 50k subsamples** for SHAP and the detectors, so its NSL-KDD numbers differ slightly from the pilot's.
- **Future-work ideas discussed but not implemented:**
  - Class-conditional second stage (one detector per XGBoost-predicted class).
  - Score-level fusion instead of OR.
  - Adaptive attacks that target the explanation (Slack et al.).
- **Before submitting:** fill in author blocks, compile on Overleaf, check the page limit of the target venue, and remove any template guidance text. Candidate venues discussed: IEEE CSNet, NetSoft workshops, IEEE Access, Annals of Telecommunications.
