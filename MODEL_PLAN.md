# Model Implementation Plan: strawberry good vs damaged

**Goal:** a small CNN that labels a photo of one strawberry as **good** or **damaged**, small and fast enough to run on a phone, with honest results we can show in the pitch.

**Approach in one line:** fine-tune small networks that were pre-trained on ImageNet (transfer learning), compare them on accuracy *and* size the way Sustika 2018 did, and test them on a dataset they weren't trained on.

## 1. Data

Full details: [`plans/01_data_and_splits.md`](plans/01_data_and_splits.md) (notebook 01, done).

- Two sources: Mahendra 2018 (webcam, single berries, GPL) and the Mendeley Fresh and Rotten Fruits set (phone photos, CC BY 4.0).
- After cleaning: **1,123 images (574 good, 549 damaged)**. Mendeley 201–737 were augmented copies and are dropped. Re-shots of the same berry are grouped (Mahendra bursts, hand-marked Mendeley photo series, near-duplicates), and 9 val/test near-copies of training photos are purged. Photos showing piles of berries are tagged `multi_berry`.
- Everything is in `data/manifest.csv`, which every later notebook reads.

## 2. Splits and experiments

Full details: [`plans/01_data_and_splits.md`](plans/01_data_and_splits.md) §4 and §6.

- **Split:** 70 / 15 / 15 by group, stratified by source × label, seed 42. Test set: 161 images (126 single-berry).
- **E1**, the main result: train and test on both sources mixed.
- **E2a / E2b**, the cross-dataset test: train on one source, test on all of the other.
- **Shortcut check:** single-berry scores and Grad-CAM, for "pile = rotten" and juice stains.

## 3. Models

Detailed design for training and evaluation (§3–5, plus E1/E2): [`plans/02_train_eval.md`](plans/02_train_eval.md).

| Model | Why | Approx. size |
| --- | --- | --- |
| Small CNN from scratch (3–4 conv blocks) | baseline, in the spirit of Sustika; shows what pre-training adds | < 2 MB |
| **MobileNetV3-Small** (ImageNet) | made for phones; our main candidate | ~4–6 MB |
| EfficientNet-B0 (ImageNet) | usually more accurate, still phone-sized | ~16–20 MB |

The output head for every model is global average pooling → dropout 0.2 → Dense(1, sigmoid).

## 4. Training setup

- **Framework:** TensorFlow / Keras on Google Colab (free GPU). Data comes from Google Drive: upload `seventx_prototype/` to `MyDrive` with `dataset.zip`; each notebook unzips the images to Colab's local disk (`/content/seventx`) and writes `data/`, `models/` and `results/` back to Drive. Keras exports directly to TFLite, which the native Flutter app runs on the device.
- **Input:** 224×224 RGB, with the aspect ratio kept by padding. Keras MobileNetV3 and EfficientNet already rescale their inputs, so feed them raw 0–255 pixels and don't normalise twice.
- **Augmentation (training only, applied on the fly):** random flip, rotation (±20°), zoom (±15%), brightness and contrast (±20%).
- **Two training stages:**
  1. Freeze the backbone and train only the head: Adam, learning rate 1e-3, up to 10 epochs.
  2. Unfreeze the top ~30% of the layers (BatchNorm layers stay frozen): learning rate 1e-4, up to 20 epochs. (Planned as 1e-5 / 15; changed after validation runs, see the tuning log in `plans/02_train_eval.md`.)
  - Both stages: batch size 32, early stopping on validation loss (patience 4), keep the best checkpoint.
- **Baseline CNN:** trained from scratch, learning rate 1e-3, up to 40 epochs (matching Sustika's epoch count), same early stopping.
- **Seed:** fixed. If time allows, repeat E1 with 3 seeds and report mean ± sd. Sustika reported one split only.

## 5. Evaluation (what goes in the results table)

- Accuracy, plus precision, recall and F1 for each class, and a confusion matrix.
- **Recall on "damaged"** is the key number: a rotten berry passed as good reaches a buyer and costs the farmer's reputation.
- Model size (MB) of the `.tflite` file, and inference time per image (on a laptop CPU, and on a phone if the demo app is ready).
- Re-check accuracy **after** TFLite quantization (float16, and int8 if needed), so the number we report is the number of the model we ship.
- **Reference points, not head-to-head comparisons:** Sustika 2018 got 96.49% on 2 classes (a different, non-public dataset); Mahendra 2018 got 90.73% with SVM + SURF on the same `color/` images but with 5-fold cross-validation. Say in the pitch that the datasets and methods differ.

Results template (`results/results.md`):

| Model | E1 acc | E1 damaged recall | E2a acc | E2b acc | .tflite MB | ms / image |
| --- | --- | --- | --- | --- | --- | --- |

## 6. Choosing and exporting the model

Detailed design for the export: [`plans/03_export.md`](plans/03_export.md). The app is a native Flutter app that runs the exported `.tflite` on the device (see §7b).

- **Rule:** compare models on **validation** scores measured **after** TFLite quantization: E1 val accuracy and E1 val damaged recall (mean over 3 seeds). Pick the smallest model that is within ~1 point of the best on both. Switch to a larger model (e.g. EfficientNet-B0) only if it is clearly better on val damaged recall, not on accuracy alone.
- **Test scores are not used to choose.** E1 test and E2 are computed once, after the choice, and reported as they are. E2 scores are test scores too (on the other dataset), so they don't decide the model either. If we ever let them, say so in the pitch. Speed counts too: EfficientNet-B0 needs ~6–7× more computation per image than MobileNetV3-Small, which matters on low-end phones and in the TF.js demo.
- **Shipped (notebook 03, 2026-11-27):** EfficientNet-B0, run E1 seed 42, **float16** TFLite, 8.1 MB, about 30 ms/photo on a Colab CPU, **threshold 0.261**.
  - Val: 92.7% accuracy, 95.0% damaged recall.
  - Test: 96.9% accuracy, 96.4% damaged recall (98.1% accuracy at 0.5).
  - Full int8 was rejected (80.6% val). Results: `results/crop_lr1e-4/export.md`.
- **Chosen: EfficientNet-B0** (on validation scores, float32): 92.1 ± 0.0% val accuracy against 89.7 ± 0.6% for MobileNetV3-Small, and 92.5% against 90.1% when tuned for 95% damaged recall. Notebook 03 confirms it after quantization and measures ms/image; if quantization costs more than ~1 point, ship float16, or fall back to MobileNetV3-Small if it is too slow on the phone.
- **Export:**
  - `strawberry_2class.tflite`: float16, plus int8 if it keeps accuracy. For a native app.
  - TF.js conversion was dropped (it failed on Colab's Python 3.13). The app runs the `.tflite` natively. `preprocess.js` stays as a verified, Pillow-exact reference port, useful when porting the crop to Dart.
  - `labels.txt`: `good`, `damaged`.
  - The **crop-to-berry function** (`berry_box` in notebook 02). The model is trained on photos cropped to the berry, so the app must crop the same way before calling it, or tell the user to fill the frame with one berry.
- **Decision threshold:** chosen on the shipped (quantized) model's **validation** predictions as the largest threshold giving ≥95% damaged recall, stored in `metadata.json` (never chosen on test). 0.5 is reported alongside it.

## 7. Files (in `seventx_prototype/`)

```
seventx_prototype/
  MODEL_PLAN.md                   this file (overview)
  plans/
    01_data_and_splits.md         data, splits and experiments (notebook 01)
    02_train_eval.md              detailed design for notebook 02
    03_export.md                  detailed design for notebook 03
    04_app.md                     design for the grading app (Flutter, native Android)
  dataset/                        (exists)
  dataset.zip                     (exists) the file to upload to Google Drive
  papers/                         (exists)
  notebooks/
    01_manifest_and_split.ipynb   (done) removing copies and duplicates, groups, split, manifest.csv
    02_train_eval.ipynb           (done) E1, E2a, E2b for the 3 models; Grad-CAM; results in results/crop_lr1e-4/
    03_export.ipynb               (done) TFLite export (both demos), re-check after quantization, test vectors, metadata
    lib/preprocess.py, .js        the crop + resize used by training, export and both apps (source of truth)
  data/manifest.csv, excluded.csv (done)
  models/                         .keras, export/ (.tflite, labels.txt, metadata.json, preprocess.py/.js, test_vectors/)
  results/
    preds/                        test (and val) predictions per run
    runs.csv                      one row of metrics per run
    results.md                    table + confusion matrices + Grad-CAM images
```

## 7b. App

Design: [`plans/04_app.md`](plans/04_app.md).
- **Stack:** a **native Flutter app** (Android APK; iOS possible later), **fully offline**.
- **Model:** `strawberry_2class.tflite` runs on the device with `tflite_flutter`.
- **Crop:** ported to Dart and verified against `models/export/test_vectors/`.
- **Language:** Bahasa Indonesia, with an English toggle.
- **Submission:** the **mobile** category (`.apk`).
- **Changes along the way:**
  - A web app (React PWA, then Capacitor) was planned first.
  - Switched to native because the rubric scores a fitting architecture and an APK ready to install, and offline use is the main goal.

## 8. Order of work (rough hours, multi-day hackathon)

1. Manifest, duplicate check and split: ~1 h **(done)**
2. Training loop with MobileNetV3-Small on E1, end to end: ~2 h **(done)**
3. Add EfficientNet-B0 and the baseline, then run E2a and E2b: ~2 h **(done)**
4. Grad-CAM check and filling in the results table: ~1 h **(done)**
5. Export and re-check after quantization: ~1 h **(done)**
6. Hand over to the demo apps: `models/export/` (download it from Drive)

## 9. Risks and limits (say these in the pitch)

- **Single berry only:** the model grades one strawberry per image. Tray grading needs a detection or cropping step first; that is the stretch goal. Most Mendeley rotten photos are piles, so "pile = rotten" is a possible shortcut: check the single-berry test scores and Grad-CAM.
- **Surface only:** like any camera-based tool, it can't see spoilage inside the fruit.
- **Two classes only:** neither dataset separates "second grade" (misshapen but edible) from "reject" (rotten). Finer grades are future work.
- **Data from elsewhere:** none of the images are from Rancabali. The cross-dataset test (E2) is our evidence on how well the model carries over to new photos.
- **Licences:** the Mendeley set is CC BY 4.0, so credit it with its title and DOI (10.17632/bdd69gyhv8.1). Mahendra's data is GPL: credit it, and don't redistribute the raw images inside the app.

## Sources
- Sustika et al. 2018 and Mahendra et al. 2018: Markdown versions in `papers/` (kept as Markdown on purpose, so they are easy to read and search).
- Fresh and Rotten Fruits Dataset: https://data.mendeley.com/datasets/bdd69gyhv8/1
- Mahendra dataset: https://github.com/okamahendra123/dataset_strawberry

