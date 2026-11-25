so whwere # Multi-view embroidery defect detection

F21DL group coursework, group PG18, Heriot-Watt University Dubai. We are looking at
whether photographing embroidered fabric from more than one angle makes it easier for a
model to tell good embroidery from defective embroidery.

Dataset: Vina-Embroidery (Mendeley Data, V2). 1,080 RGB images of four pieces of fabric,
shot at 90, 45 and 60 degrees.

## The question

Manual inspection is slow and people get tired. Defects are also easy to miss depending on
how the light hits them: a bit of shine shows up at 45 degrees and disappears straight on.
So we ask:

> Does giving a model more than one viewing angle improve defect detection compared with
> the straight-on 90 degree view alone?

Two hypotheses come out of that:

- H1: multiple viewing angles improve detection compared with a single angle.
- H2: the angles are not equally useful. Some are harder than others.

The plan is the one we pitched in week 4. The data turned out to be messier than the
dataset README suggests, so the splitting needed more care than we expected. That is
summarised in the results tables.

## The data

| What | How many |
| --- | --- |
| Images | 1,080 |
| Normal | 492 |
| Abnormal | 588 |
| 90 degrees (straight) | 313 |
| 45 degrees | 377 |
| 60 degrees | 390 |
| Physical samples | 4 |
| Groups (sample + defect type) | 35 |

A group is one physical sample with one defect type, and every photo of the same defect
shares a group id. That matters for splitting: photos of the same defect taken a second
apart are almost identical, so if some land in training and some in test, the model looks
much better than it actually is. All splits are therefore made on groups, not on images.

![Normal and defective embroidery at each camera angle](results/figures/eda_examples.png)

*Two frames per view and class. The patches differ between classes: the normal 90 degree
rows come from the normal-only groups, which are separate designs from the defect groups.
That overlap between class and design is one of the things we control for later and one of
the caveats in the discussion.*

Three things about this dataset shape the whole project. All three shape the experimental design.

1. There are only four physical samples. We are testing generalisation to another defect on
   the same four garments, not to new garments.
2. 16 of the 35 groups contain both normal and abnormal images, so a group is not a class.
3. The split files shipped with the dataset change their group-to-split assignment between
   view cases (10 of 35 groups move between `straight_only` and `straight_60`). Comparing
   "straight only" with "straight plus 45" on those files would compare different test sets
   while claiming to compare views. We build one partition and derive all three view
   conditions from it.

There is an auxiliary YOLO folder with 954 annotated images and 1,883 polygons across four
defect classes. Those are annotated copies of abnormal images from the main set, not extra
images, so they do not go into the 1,080. They are there for localisation later.

## What we did

- 88 features per image: colour and HSV statistics, brightness and contrast, GLCM texture,
  local binary patterns, edge strength, a gradient direction histogram, FFT band energies
  and quadrant brightness.
- k-means and a Gaussian mixture fitted with EM, to see whether defects separate on their
  own. Silhouette and Davies-Bouldin for the internal view, ARI and NMI against the label
  only to interpret the clusters.
- Seven classical learners: decision tree, Gaussian naive Bayes, logistic regression,
  perceptron, k-NN, random forest and gradient boosting. The brief asks for three.
- An MLP on the same features, and a small CNN on the images with augmentation, early
  stopping and Grad-CAM.
- A VAE trained on normal images only, scored by reconstruction error. That covers the
  generative side of the brief and gives an unsupervised check on the same question.
- H1 tested on the same 90 degree test images in all three view conditions, so the only
  thing that changes is the training data, with McNemar's exact test on the paired
  predictions. H2 is a train-angle by test-angle transfer matrix.

## What we found

Short version, with the numbers the pipeline produced:

- **Single-view detection works** on held-out defect groups: view-specific models reach F1
  1.00 at 90 degrees, 0.93 at 45 and 0.84 at 60.
- **Extra angles do not help.** On the same 90 degree test images, logistic regression drops
  from F1 1.00 to 0.66 once 45 degree images enter training (McNemar p < 0.001), and 90+60
  changes nothing. The CNN shows no statistically significant change when adding the 45 degree view (p = 0.58) or 60 degree view (p = 0.22). H1 is rejected.
- **The angles are not interchangeable.** Training on 45 and testing on 90 gives AUC 0.000,
  and 90 to 45 gives 0.107, while own-view AUC is about 0.99. H2 is supported.
- **Generalisation is partial and unstable.** Holding out one whole garment gives a mean F1
  of 0.75 (group split: 0.79), but the range across the four garments is 0.45 to 0.97.
- **The VAE is a strong detector** (AUC 0.89, F1 0.82), showing that unsupervised reconstruction
  error reliably detects embroidery defects without requiring defect labels during training.
- **The label is partly predictable from lighting alone** (AUC 0.60, accuracy 0.63), and the normal-only
  groups are separate patch designs, which is the most likely reason the numbers look as good
  as they do.

## Results

Everything the report quotes is generated by the pipeline and collected in
`results/report_tables.md`. After a run, `results/tables/headline_numbers.json` has the
short version.

The tables worth opening first:

- `baselines_metrics.csv`: seven learners, three view conditions, group bootstrap CIs.
- `mlp_metrics.csv` and `cnn_metrics.csv`: the neural networks on the same splits.
- `vae_anomaly.csv`: unsupervised anomaly detection.
- `h1_paired_common.csv` and `h1_mcnemar.csv`: the actual answer to H1.
- `h2_cross_view.csv`: which angle transfers worst.
- `shortcut_probe.csv`: how much of the label is predictable from lighting alone.

## Running it

Python 3.12, CPU is fine.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python -m src.run_all
```

By default the dataset is read from `Vina-Embroidery/` in the repo root (git-ignored).
Elsewhere, pass `--dataset <folder that contains raw/vai_theu/manifest.csv>`. A full run
takes about an hour on a CPU laptop, mostly the CNN and the VAE. Stages can be run on their
own:

```bash
python -m src.run_all --stages eda,clustering
python -m src.run_all --stages baselines
```


## Repo layout

```
src/        pipeline modules, roughly one per stage
docs/       team contributions
results/    generated tables and figures, not hand edited
```

## Team

Kushik Siddapura Kumar (H00549226), Rachana Ramesh (H00549153), Shaik Shah Ali Safwan
(H00561860), Rohit Nadan Raju (H00534323), Abhishek Gowda Jayalakshmi (H00536176).

Who did what is in `docs/CONTRIBUTIONS.md` and in the report appendix.

## Data credit

Vina-Embroidery dataset, Mendeley Data V2. If you reuse this code, cite the dataset record
rather than this repo.
