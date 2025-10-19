Code that belongs to the project in SF2935 at KTH

CNN4.py – trains a tiny convolutional network, extracts embeddings, evaluates VeRi, and saves learning curve and calibration plots.

SVM4.py – builds PCA features from raw pixels and compares three classical baselines: PCA-only retrieval, linear margin (SGD hinge), and RBF via Random Fourier Features. Produces calibration and CI summaries.

Both pipelines:

Expect the VeRi folder with protocol lists.

Compute query-level bootstrap confidence intervals.

Output a single JSON file with metrics for easy logging or dashboards.

Link to the required Dataset: https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset
