# Generated report tables

All tables below are produced by `python -m src.run_all`; do not edit by hand.


## Dataset composition (R2)

| view | images | normal | abnormal | samples | groups |
|---|---|---|---|---|---|
| straight | 313 | 118 | 195 | 4 | 33 |
| 45 | 377 | 186 | 191 | 4 | 24 |
| 60 | 390 | 188 | 202 | 4 | 28 |
| all | 1080 | 492 | 588 | 4 | 35 |


## Canonical group-level partition

| split | images_45 | images_60 | images_straight | normal_45 | normal_60 | normal_straight |
|---|---|---|---|---|---|---|
| train | 213 | 241 | 175 | 93 | 112 | 44 |
| val | 63 | 83 | 71 | 24 | 42 | 40 |
| test | 101 | 66 | 67 | 69 | 34 | 34 |


## Quality control (R2)

| images | missing_files | unreadable | exact_duplicates | distinct_resolutions | min_size | max_size |
|---|---|---|---|---|---|---|
| 1080 | 0 | 0 | 0 | 2 | 1920x1200 | 2448x2048 |


## Group structure: 35 sample x defect-type groups, 4 physical samples

- groups: **35**
- groups containing both labels: **16**
- images in the largest group: **90**


## Shortcut probe: is the label predictable from lighting alone? (R2 caveat)

| features | n_features | accuracy | f1 | roc_auc |
|---|---|---|---|---|
| lighting_stats_only | 7 | 0.632 | 0.673 | 0.603 |
| full_descriptor | 88 | 0.909 | 0.922 | 0.977 |


## Top-15 selected features (R2)

| feature | mutual_info | rf_importance |
|---|---|---|
| glcm_contrast_d4 | 0.325 | 0.060 |
| hsv_V_hist0 | 0.537 | 0.055 |
| lbp_5 | 0.292 | 0.053 |
| rgb_R_std | 0.347 | 0.053 |
| hsv_S_hist4 | 0.365 | 0.048 |
| lbp_7 | 0.213 | 0.039 |
| glcm_dissimilarity_d4 | 0.255 | 0.038 |
| rgb_R_mean | 0.503 | 0.034 |
| hsv_H_hist4 | 0.409 | 0.031 |
| hsv_H_hist6 | 0.272 | 0.029 |
| hsv_S_mean | 0.344 | 0.026 |
| glcm_contrast_d2 | 0.206 | 0.026 |
| hsv_V_hist4 | 0.118 | 0.026 |
| hsv_V_hist7 | 0.163 | 0.026 |
| hsv_S_hist7 | 0.328 | 0.026 |


## Clustering (R2): best k-means (silhouette) / best GMM (BIC)

| algorithm | k | silhouette | davies_bouldin | ari_vs_label | nmi_vs_label | bic |
|---|---|---|---|---|---|---|
| kmeans | 4 | 0.433 | 0.949 | 0.003 | 0.006 |  |
| kmeans | 6 | 0.425 | 0.954 | 0.124 | 0.185 |  |
| gmm_em | 3 | 0.400 | 1.079 | 0.004 | 0.006 | -256704.032 |
| gmm_em | 5 | 0.402 | 1.105 | 0.158 | 0.196 | -254472.876 |


## What the k-means clusters contain

| cluster | abnormal | normal | n | normal_frac |
|---|---|---|---|---|
| 0 | 127 | 120 | 247 | 0.486 |
| 1 | 162 | 90 | 252 | 0.357 |
| 2 | 266 | 254 | 520 | 0.488 |
| 3 | 33 | 28 | 61 | 0.459 |


## Classical learners (R3) - straight_only

straight

| case | model | cv_f1 | accuracy | precision | recall | f1 | balanced_accuracy | roc_auc | f1_lo | f1_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| straight_only | logistic_regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | knn | 0.882 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | perceptron | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | random_forest | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | decision_tree | 0.878 | 0.493 | 0.493 | 1.000 | 0.660 | 0.500 | 0.500 | 0.239 | 1.000 |
| straight_only | naive_bayes | 0.882 | 0.493 | 0.493 | 1.000 | 0.660 | 0.500 | 0.500 | 0.239 | 1.000 |
| straight_only | gradient_boosting | 0.876 | 0.493 | 0.493 | 1.000 | 0.660 | 0.500 | 0.500 | 0.239 | 1.000 |


## Classical learners (R3) - straight_45

straight+45

| case | model | cv_f1 | accuracy | precision | recall | f1 | balanced_accuracy | roc_auc | f1_lo | f1_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| straight_45 | gradient_boosting | 0.870 | 0.976 | 0.942 | 1.000 | 0.970 | 0.981 | 0.999 | 0.918 | 1.000 |
| straight_45 | random_forest | 0.827 | 0.976 | 0.942 | 1.000 | 0.970 | 0.981 | 1.000 | 0.923 | 1.000 |
| straight_45 | decision_tree | 0.817 | 0.976 | 0.984 | 0.954 | 0.969 | 0.972 | 0.990 | 0.906 | 1.000 |
| straight_45 | knn | 0.827 | 0.792 | 0.650 | 1.000 | 0.788 | 0.830 | 0.830 | 0.407 | 1.000 |
| straight_45 | logistic_regression | 0.823 | 0.792 | 0.650 | 1.000 | 0.788 | 0.830 | 0.762 | 0.407 | 1.000 |
| straight_45 | perceptron | 0.814 | 0.792 | 0.650 | 1.000 | 0.788 | 0.830 | 0.754 | 0.407 | 1.000 |
| straight_45 | naive_bayes | 0.791 | 0.732 | 0.598 | 0.938 | 0.731 | 0.770 | 0.833 | 0.376 | 0.987 |


## Classical learners (R3) - straight_60

straight+60

| case | model | cv_f1 | accuracy | precision | recall | f1 | balanced_accuracy | roc_auc | f1_lo | f1_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| straight_60 | logistic_regression | 0.936 | 0.992 | 0.985 | 1.000 | 0.992 | 0.993 | 1.000 | 0.976 | 1.000 |
| straight_60 | gradient_boosting | 0.918 | 0.985 | 0.970 | 1.000 | 0.985 | 0.985 | 0.998 | 0.957 | 1.000 |
| straight_60 | random_forest | 0.893 | 0.977 | 0.956 | 1.000 | 0.977 | 0.978 | 1.000 | 0.937 | 1.000 |
| straight_60 | perceptron | 0.916 | 0.970 | 0.942 | 1.000 | 0.970 | 0.971 | 0.998 | 0.938 | 1.000 |
| straight_60 | knn | 0.884 | 0.917 | 0.855 | 1.000 | 0.922 | 0.919 | 0.985 | 0.828 | 1.000 |
| straight_60 | decision_tree | 0.956 | 0.895 | 0.892 | 0.892 | 0.892 | 0.895 | 0.939 | 0.826 | 0.962 |
| straight_60 | naive_bayes | 0.835 | 0.617 | 0.573 | 0.846 | 0.683 | 0.622 | 0.839 | 0.335 | 0.930 |


## Neural networks (R4)

| case | model | accuracy | precision | recall | f1 | balanced_accuracy | roc_auc | f1_lo | f1_hi | best_params |
|---|---|---|---|---|---|---|---|---|---|---|
| straight_only | mlp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | {"hidden": [64], "alpha": 0.001} |
| straight_45 | mlp | 0.958 | 0.983 | 0.908 | 0.944 | 0.949 | 0.976 | 0.850 | 1.000 | {"hidden": [256, 128], "alpha": 0.001} |
| straight_60 | mlp | 0.925 | 1.000 | 0.846 | 0.917 | 0.923 | 0.996 | 0.818 | 0.987 | {"hidden": [256, 128], "alpha": 0.001} |
| straight_only | cnn | 0.925 | 0.868 | 1.000 | 0.930 | 0.926 | 1.000 | 0.680 | 1.000 | {"width": 32, "size": 128, "batch": 32, "lr": 0.001, "dropout": 0.3, "val_auc_per_setting": {"lr=0.001,dropout=0.3": 1.0, "lr=0.0003,dropout=0.3": 1.0}} |
| straight_45 | cnn | 0.613 | 0.500 | 0.908 | 0.645 | 0.667 | 0.864 | 0.509 | 0.889 | {"width": 32, "size": 128, "batch": 32, "lr": 0.001, "dropout": 0.3, "val_auc_per_setting": {"lr=0.001,dropout=0.3": 0.9344, "lr=0.0003,dropout=0.3": 0.7525}} |
| straight_60 | cnn | 0.827 | 0.750 | 0.969 | 0.846 | 0.830 | 0.929 | 0.730 | 0.992 | {"width": 32, "size": 128, "batch": 32, "lr": 0.001, "dropout": 0.3, "val_auc_per_setting": {"lr=0.001,dropout=0.3": 0.9773, "lr=0.0003,dropout=0.3": 0.9162}} |


## Generative anomaly detection: VAE (trained on normals only)

| model | n | accuracy | precision | recall | f1 | roc_auc | threshold |
|---|---|---|---|---|---|---|---|
| vae_anomaly | 234 | 0.838 | 0.757 | 0.897 | 0.821 | 0.891 | 0.029 |


## Generalisation check: leave-one-sample-out vs the group split

| case | scheme | n_train | n_test | accuracy | f1 | roc_auc |
|---|---|---|---|---|---|---|
| all_views | leave_out_sample_1 | 863 | 217 | 0.682 | 0.670 | 0.757 |
| all_views | leave_out_sample_2 | 800 | 280 | 0.979 | 0.974 | 1.000 |
| all_views | leave_out_sample_3 | 805 | 275 | 0.825 | 0.897 | 0.718 |
| all_views | leave_out_sample_4 | 772 | 308 | 0.581 | 0.478 | 0.771 |
| all_views | group_split (same garments) | 629 | 234 | 0.778 | 0.789 | 0.932 |
| straight_only | loso_sample_1 | 239 | 74 | 1.000 | 1.000 | 1.000 |
| straight_only | loso_sample_2 | 233 | 80 | 1.000 | 1.000 | 1.000 |
| straight_only | loso_sample_3 | 215 | 98 | 1.000 | 1.000 | 1.000 |
| straight_only | loso_sample_4 | 252 | 61 | 0.459 | 0.000 | 1.000 |
| straight_45 | loso_sample_1 | 539 | 151 | 0.828 | 0.780 | 0.975 |
| straight_45 | loso_sample_2 | 521 | 169 | 0.598 | 0.534 | 0.729 |
| straight_45 | loso_sample_3 | 504 | 186 | 0.871 | 0.924 | 0.511 |
| straight_45 | loso_sample_4 | 506 | 184 | 0.690 | 0.759 | 0.739 |
| straight_60 | loso_sample_1 | 563 | 140 | 0.771 | 0.820 | 1.000 |
| straight_60 | loso_sample_2 | 512 | 191 | 0.812 | 0.818 | 0.999 |
| straight_60 | loso_sample_3 | 516 | 187 | 0.856 | 0.907 | 0.901 |
| straight_60 | loso_sample_4 | 518 | 185 | 0.476 | 0.000 | 0.764 |


## Mean pixel statistics per view and class (shortcut check)

| view | label | brightness | contrast | sharpness |
|---|---|---|---|---|
| 45 | abnormal | 0.137 | 0.196 | 0.025 |
| 45 | normal | 0.175 | 0.198 | 0.026 |
| 60 | abnormal | 0.148 | 0.208 | 0.027 |
| 60 | normal | 0.165 | 0.187 | 0.025 |
| straight | abnormal | 0.353 | 0.217 | 0.031 |
| straight | normal | 0.281 | 0.180 | 0.028 |


## H1 - paired comparison on identical 90° test images

| case | model | n_common | accuracy | f1 | roc_auc | acc_lo | acc_hi |
|---|---|---|---|---|---|---|---|
| straight_45 | cnn | 67 | 0.881 | 0.871 | 0.818 | 0.533 | 0.975 |
| straight_60 | cnn | 67 | 0.985 | 0.985 | 1.000 | 0.909 | 1.000 |
| straight_only | cnn | 67 | 0.925 | 0.930 | 1.000 | 0.873 | 1.000 |
| straight_45 | decision_tree | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_60 | decision_tree | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | decision_tree | 67 | 0.493 | 0.660 | 0.500 | 0.135 | 1.000 |
| straight_45 | gradient_boosting | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_60 | gradient_boosting | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | gradient_boosting | 67 | 0.493 | 0.660 | 0.500 | 0.135 | 1.000 |
| straight_45 | knn | 67 | 0.493 | 0.660 | 0.500 | 0.135 | 1.000 |
| straight_60 | knn | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | knn | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_45 | logistic_regression | 67 | 0.493 | 0.660 | 0.399 | 0.135 | 1.000 |
| straight_60 | logistic_regression | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | logistic_regression | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_45 | mlp | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_60 | mlp | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | mlp | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_45 | naive_bayes | 67 | 0.507 | 0.653 | 0.601 | 0.184 | 1.000 |
| straight_60 | naive_bayes | 67 | 0.493 | 0.660 | 0.958 | 0.135 | 1.000 |
| straight_only | naive_bayes | 67 | 0.493 | 0.660 | 0.500 | 0.135 | 1.000 |
| straight_45 | perceptron | 67 | 0.493 | 0.660 | 0.241 | 0.135 | 1.000 |
| straight_60 | perceptron | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | perceptron | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_45 | random_forest | 67 | 0.985 | 0.985 | 1.000 | 0.975 | 1.000 |
| straight_60 | random_forest | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| straight_only | random_forest | 67 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |


## H1 - McNemar exact test (straight-only vs multi-view)

| model | comparison | n | n_discordant | p_value | a_only_correct | b_only_correct |
|---|---|---|---|---|---|---|
| cnn | straight_only vs straight_45 | 67 | 13 | 0.581 | 8 | 5 |
| cnn | straight_only vs straight_60 | 67 | 6 | 0.219 | 1 | 5 |
| decision_tree | straight_only vs straight_45 | 67 | 34 | 0.000 | 0 | 34 |
| decision_tree | straight_only vs straight_60 | 67 | 34 | 0.000 | 0 | 34 |
| gradient_boosting | straight_only vs straight_45 | 67 | 34 | 0.000 | 0 | 34 |
| gradient_boosting | straight_only vs straight_60 | 67 | 34 | 0.000 | 0 | 34 |
| knn | straight_only vs straight_45 | 67 | 34 | 0.000 | 34 | 0 |
| knn | straight_only vs straight_60 | 67 | 0 | 1.000 | 0 | 0 |
| logistic_regression | straight_only vs straight_45 | 67 | 34 | 0.000 | 34 | 0 |
| logistic_regression | straight_only vs straight_60 | 67 | 0 | 1.000 | 0 | 0 |
| mlp | straight_only vs straight_45 | 67 | 0 | 1.000 | 0 | 0 |
| mlp | straight_only vs straight_60 | 67 | 0 | 1.000 | 0 | 0 |
| naive_bayes | straight_only vs straight_45 | 67 | 5 | 1.000 | 2 | 3 |
| naive_bayes | straight_only vs straight_60 | 67 | 0 | 1.000 | 0 | 0 |
| perceptron | straight_only vs straight_45 | 67 | 34 | 0.000 | 34 | 0 |
| perceptron | straight_only vs straight_60 | 67 | 0 | 1.000 | 0 | 0 |
| random_forest | straight_only vs straight_45 | 67 | 1 | 1.000 | 1 | 0 |
| random_forest | straight_only vs straight_60 | 67 | 0 | 1.000 | 0 | 0 |


## H2 - single-view models

| view | n_train | n_test | cv_f1 | accuracy | f1 | roc_auc |
|---|---|---|---|---|---|---|
| straight | 175 | 67 | 1.000 | 1.000 | 1.000 | 1.000 |
| 45 | 213 | 101 | 0.961 | 0.950 | 0.928 | 0.986 |
| 60 | 241 | 66 | 0.946 | 0.818 | 0.842 | 0.985 |


## H2 - cross-view transfer (ROC-AUC)

| train_view | test_45 | test_60 | test_straight |
|---|---|---|---|
| 45 | 0.986 | 0.748 | 0.000 |
| 60 | 0.882 | 0.985 | 0.799 |
| straight | 0.107 | 0.361 | 1.000 |


## Figures

- `figures/cluster_selection.png`
- `figures/cnn_confusion.png`
- `figures/cnn_gradcam_straight_45.png`
- `figures/cnn_training.png`
- `figures/eda_composition.png`
- `figures/eda_examples.png`
- `figures/eda_feature_correlation.png`
- `figures/eda_pca.png`
- `figures/eda_pixel_stats.png`
- `figures/h1_multi_view.png`
- `figures/h2_cross_view.png`
- `figures/h2_per_view.png`
- `figures/mlp_training.png`
- `figures/vae_anomaly.png`
- `figures/vae_reconstructions.png`
