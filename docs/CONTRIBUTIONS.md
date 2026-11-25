# Who did what

The report asks for a short appendix on each member's contribution. Fill this in as the
work happens and keep it honest, because the peer assessment asks about real contributions
and the commit history is visible.

| Member | Student ID | Contributions |
|---|---|---|
| Kushik Siddapura Kumar | H00549226 | Pipeline orchestration, classical machine learning baseline models (`src/baselines.py`), report compilation, and presentation deck. |
| Rachana Ramesh | H00549153 | Dataset auditing, quality control, group-level stratified partitioning (`src/data.py`), and leave-one-sample-out generalization (`src/cross_sample.py`). |
| Shaik Shah Ali Safwan | H00561860 | Exploratory data analysis, illumination shortcut probe (`src/eda.py`), unsupervised k-means and GMM clustering (`src/clustering.py`). |
| Rohit Nadan Raju | H00534323 | Feature extraction (GLCM, LBP, Fourier, color), mutual information feature selection (`src/features.py`), multi-view hypothesis testing (`src/multiview.py`). |
| Abhishek Gowda Jayalakshmi | H00536176 | Deep learning models: MLP and CNN fine-tuning with Grad-CAM (`src/neural.py`), generative VAE reconstruction anomaly detection (`src/generative.py`). |

A split that works and keeps everyone defensible in the viva: each person owns one module
completely, meaning they wrote it, ran it, and can explain any line of it.

| Work | Module | Owner |
|---|---|---|
| Data audit, QC, splitting | `src/data.py` | Rachana Ramesh |
| EDA and the lighting shortcut test | `src/eda.py` | Shaik Shah Ali Safwan |
| Features and feature selection | `src/features.py` | Rohit Nadan Raju |
| Clustering | `src/clustering.py` | Shaik Shah Ali Safwan |
| Classical learners | `src/baselines.py` | Kushik Siddapura Kumar |
| MLP and CNN | `src/neural.py` | Abhishek Gowda Jayalakshmi |
| VAE | `src/generative.py` | Abhishek Gowda Jayalakshmi |
| View analysis | `src/multiview.py` | Rohit Nadan Raju |
| Cross-sample generalization | `src/cross_sample.py` | Rachana Ramesh |
| Report and slides | - | Kushik Siddapura Kumar |

Practical notes. Commit from your own GitHub account, not one shared login, and commit
regularly rather than in one dump at the end. Everyone should be able to run
`python -m src.run_all --stages report` and read the tables, even for the parts they did not
write, because the Q&A can ask anyone anything.

## Generative AI Declaration (§8)
In accordance with Heriot-Watt University Coursework §8 (Framework Statement 6):
Generative AI was used by team members as a developer copilot for syntax assistance, code refactoring, and test debugging. All architecture choices, feature designs, machine learning algorithms, and experimental evaluations were directed, executed, and validated by the student authors. All report text and conclusions are original work written in our own words.
