# Evaluation and analysis reports

Evaluation creates `evaluation_results.json` and `confusion_matrix.png` under the selected output directory. Benchmarking writes a category-wise JSON summary. These files are generated locally and ignored by Git because they may expose dataset details. Publish metrics only with the model version, dataset provenance/version, split policy, threshold, sample counts and limitations.
