# Dataset preparation

The training/evaluation tools expect owner-provided data in this structure:

```text
dataset/
  dataset_version.txt
  train/
    real/
    ai/
  validation/
    real/
    ai/
  test/
    real/
    ai/
```

Use `real` for camera photographs and `ai` for generated images. Keep manipulated real images in separate benchmark categories until a multi-class model and clear labeling policy are implemented; the current baseline is binary and does not claim manipulation classification.

## Splits and duplicate prevention

Assign sources, prompt families, photographers, cameras, and generation systems to splits before feature extraction. Prefer 70/15/15 train/validation/test, with no original, crop, resize, recompression, or close perceptual match crossing splits. The CLI checks exact SHA-256 and 64-bit pHash distance at most four across split boundaries, with a mean-color guard to reduce grayscale-hash collisions, and refuses training/evaluation when matches are found. This is a useful check, not a substitute for source-level grouping or manual review.

The trainer fits only `train`, fits optional probability calibration only on `validation`, and never reads `test` for fitting. Evaluation requires `test` and reports test metrics. Keep the test split locked until model and threshold choices are final.

## Provenance, licensing and ethics

Record source URL or collection, acquisition date, image author/source, generator and model version, prompt/license terms where known, transformations, labels, and consent/privacy restrictions. Verify dataset and per-image licenses before using, redistributing, or publishing results. The project does not download datasets automatically. Avoid private or sensitive imagery unless you have a lawful basis and permission. Strip or restrict location and identifying metadata in research exports.

## Robustness categories and unseen generators

For `scripts/benchmark.py`, organize the held-out set as `test/<category>/{real,ai}/`, where categories may include `clean`, `edited_real`, `ai_edited`, `social_compressed`, `screenshots`, `resized`, and `metadata_stripped`. For optional AI-only unseen generator checks use `test/unseen_generators/<generator>/`; these report sensitivity only and should be paired with real controls. Keep all variants derived from one source within a single split.
