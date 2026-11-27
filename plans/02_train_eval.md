# Plan: notebook 02, training and evaluation

**Notebook:** `notebooks/02_train_eval.ipynb`
**Covers:** `MODEL_PLAN.md` §3 (models), §4 (training), §5 (evaluation), and runs the E1/E2 experiments defined in [`01_data_and_splits.md`](01_data_and_splits.md) §6.
**Depends on:** `data/manifest.csv` from notebook 01. Notebook 02 never re-splits; it only reads the `split`, `source`, `label` and `multi_berry` columns.

## 1. Inputs and outputs

**Input**
- `PROJECT/data/manifest.csv`: 1,123 images, 574 good and 549 damaged.
- Images under `ROOT`, from the same Colab/Drive setup cell as notebook 01. That cell mounts Drive and unzips `dataset.zip` to `/content/seventx`.

**Outputs** (all written under `PROJECT`, so they stay on Drive)

| File | What it holds |
| --- | --- |
| `models/{exp}_{model}_s{seed}.keras` | best checkpoint of each run |
| `results/preds/{exp}_{model}_s{seed}.csv` | one row per test image: `path, source, label, multi_berry, prob_damaged`. Val predictions go in a matching `_val.csv` |
| `results/history/{exp}_{model}_s{seed}.json` | training history, best epoch, params, `.keras` size, training time |
| `results/runs.csv` | one row of metrics per run × val/test × subset × threshold |
| `results/results.md` | the results table, generated from `runs.csv` |
| `results/confusion_{model}_{val\|test}.png` | confusion matrices (E1, seed 42) |
| `results/gradcam_{model}_{val\|test}.png` | Grad-CAM grids |
| `results/gradcam_notes.md` | written by hand after looking at the grids; `results.md` includes it |

Notebook 03 fills in the `.tflite` size and ms/image columns, because they depend on the exported, compressed model.

## 2. Notebook sections

### 0. Setup
- The setup cell copied from notebook 01: it detects Colab, mounts Drive, sets `PROJECT` and `ROOT`, and unzips the images.
- A GPU check that warns if no GPU is attached. Print the TensorFlow and Keras versions.
- Set the global seeds with `keras.utils.set_random_seed`.
- `SMOKE = False` flag (see §5).

### 1. Image cache
- Decode every manifest image once. Use PIL `draft()` so the 3000–4000 px Mendeley JPEGs decode at reduced size, which is much faster.
- **Crop to the berry first** (`CROP_TO_BERRY = True`, added after the first run).
  - **How:** take the pixels that differ from the median border colour **and** are coloured (HSV saturation > 70). Saturation excludes shadows and the blue tint at the Mahendra webcam edge. Take their 1–99 percentile box, add a 15% margin, and make it square.
  - **Why:** in the first run (no crop), a Mahendra berry was only about 60 px wide at 224×224. Mahendra val accuracy was 80.4% against 91.4% for Mendeley, and Grad-CAM showed heat off the berry.
  - Piles and crates fill their photo, so they barely change.
  - **The app must apply the same crop** before calling the model. Notebook 03 exports the crop function with the model.
  - Outputs go to `models/crop/` and `results/crop/`. The first no-crop run is the `fitpad` variant; its files are in `results/` and `models/` directly.
- **Fit inside 224×224, keeping the aspect ratio. Pad with the image's median border colour, not black.** Mahendra photos are always 4:3, so black bars would only ever appear on one source and give the model a cue that has nothing to do with the berry.
- The result is one `uint8` array of shape (1123, 224, 224, 3), about 170 MB, kept in memory and saved in `/content`. It is not stored on Drive: rebuilding it takes 1–2 minutes per session.
- Sanity check: show a grid of 16 cached images with their source, label and `multi_berry`, plus a few augmented versions (from section 3).

### 2. Experiment index sets
Built from manifest columns only.

| Exp | Train | Early stopping (val) | Test |
| --- | --- | --- | --- |
| **E1** | `split == train`, both sources | `split == val`, both sources | `split == test`, both sources (161) |
| **E2a** | Mendeley `train` | Mendeley `val` | all Mahendra (724) |
| **E2b** | Mahendra `train` | Mahendra `val` | all Mendeley (399) |

Validation always comes from the **training** source. The target dataset in E2 must stay completely unseen, including for early stopping and threshold choice.

### 3. Augmentation
- Applied as a `tf.data` map, **training batches only**, using Keras preprocessing layers:
  - flip, horizontal and vertical
  - rotation ±20°, with `fill_mode='reflect'`
  - zoom ±15%
  - brightness ±20% and contrast ±20%, with `value_range=(0, 255)`
- Kept **out of the model**, so the exported `.tflite` and TF.js models contain no augmentation layers.
- Pixels stay in 0–255 throughout.

### 4. Models
`build(name)` returns: input (224, 224, 3) → backbone → GlobalAveragePooling → Dropout 0.2 → Dense(1, sigmoid). The output is **P(damaged)**: label `damaged` = 1, `good` = 0.

| Name | Backbone | Input scaling |
| --- | --- | --- |
| `baseline` | `Rescaling(1/255)`, then 4 blocks of Conv3×3-BN-ReLU-MaxPool with 32, 64, 128, 128 filters | inside the model |
| `mnv3s` | `keras.applications.MobileNetV3Small`, ImageNet weights, `include_top=False` | built into the backbone. Feed raw 0–255 |
| `effb0` | `keras.applications.EfficientNetB0`, ImageNet weights, `include_top=False` | built into the backbone. Feed raw 0–255 |

- Pretrained backbones are called as `backbone(x, training=False)`. This keeps every BatchNorm layer in inference mode in both training stages, which is how MODEL_PLAN's "BatchNorm layers stay frozen" is implemented.
- Never add an extra `/255` or `preprocess_input` in front of `mnv3s` or `effb0`.

### 5. Training

| | Pretrained (`mnv3s`, `effb0`) | `baseline` |
| --- | --- | --- |
| Stage 1 | backbone frozen, Adam 1e-3, ≤ 10 epochs | Adam 1e-3, ≤ 40 epochs |
| Stage 2 | unfreeze the top 30% of backbone layers (BatchNorm layers stay frozen), recompile, Adam 1e-4, ≤ 20 epochs (planned 1e-5 / 15; changed in tuning run 3) | – |

- **All runs:** batch 32, binary cross-entropy, `EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)`.
- The classes are close to balanced (574 / 549), so no class weights.
- **Resume-safe:** before training, a run is skipped if its `.keras` file and preds CSV already exist on Drive. Colab sessions disconnect; rerunning the notebook continues where it stopped.
- Log each run's training history (loss and accuracy per epoch) and plot the curves for the E1 seed-42 runs.

### 6. Run matrix
A config cell lists the runs to do:

```
MODELS = ['mnv3s', 'effb0', 'baseline']
RUNS = [('E1', s) for s in (42, 43, 44)] + [('E2a', 42), ('E2b', 42)]
```

- **Order:** run `mnv3s` E1 seed 42 end to end first, through **validation** scores and the results table. That proves the pipeline works. Then run the rest. Test scores come last (§7).
- **Full matrix:** 3 models × 5 runs = 15 runs. On ~790 training images at 224 px, this should take well under an hour on a Colab T4.
- **Three seeds for E1 are the default, not optional.** The E1 test set has only 161 images, and only 10 of them are single-berry rotten Mendeley photos. With one seed, a difference of a few points between models is noise.

### 7. Evaluation
Computed from the saved prediction CSVs only, so it can be re-run or extended without retraining.

**Test-set discipline.** The split (notebook 01) keeps the same photos and berries out of test. It can't stop test scores from steering our decisions; these rules do:
- **Development uses val only.** Fix bugs, try settings and compare models on val scores. A config flag `SHOW_TEST = False` keeps test metrics hidden until the settings are frozen. Then set it to `True` and compute test scores once.
- **The model is chosen on val scores** (E1 val accuracy and val damaged recall), not test scores. See `MODEL_PLAN.md` §6.
- If a test result leads to a change (a setting, more augmentation, cropping), the old test scores no longer count as unseen. Say so in `results.md`.

- **Metrics:** accuracy; precision, recall and F1 for each class; confusion matrix. **Damaged recall** is the key number.
- **Subsets**, each reported separately:
  - all test images
  - single-berry images only (`multi_berry == 0`), because the demo grades one berry
  - per source (Mahendra, Mendeley)
- **Threshold:**
  - The main results use 0.5.
  - Secondary line: pick the threshold on the **val** predictions that gives damaged recall ≥ 0.95, then report test scores at that threshold, with the drop in good-class precision.
  - Never choose a threshold on test.
- **E1:** report the mean ± sd over seeds 42/43/44. **E2a/E2b:** report seed 42.
- `runs.csv` holds both val and test metrics for every run. The val columns are the ones used to choose a model.

### 8. Grad-CAM
- For `mnv3s` and `effb0` (E1, seed 42), using the last convolutional layer of the backbone.
- **Diagnose on val, show on test.** While developing, look at Grad-CAM on **val** images to check for shortcuts. Any change made because of a heatmap must come from val. Test-image grids are made once, with the final test scores, for the results and the pitch.
- **12 images** (same mix for the val and the test grid):
  - Mendeley rotten: 2 single-berry and 2 piles
  - Mendeley fresh: 2, including 1 pile if there is one in test
  - Mahendra: 2 good and 2 damaged
  - up to 2 misclassified images
- Save as a grid PNG: the image, the heatmap overlay, the label and the predicted probability.
- **By hand**, write a short note in `results.md`: does the heat sit on the berry, on juice stains, or on the pile layout / background?

### 9. `results/results.md`
Generated from `runs.csv`:

| Model | E1 val acc | E1 val damaged recall | E1 acc | E1 damaged recall | E1 single-berry acc | E2a acc | E2b acc | params | .keras MB | .tflite MB | ms / image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

- E1 columns show mean ± sd. The val columns decide the model; the test columns are reported. The last two columns are filled by notebook 03.
- Below the table:
  - the per-source and single-berry breakdown
  - the secondary-threshold line
  - the confusion matrices and Grad-CAM grids
  - the Grad-CAM note
- End with the reference-points note: Sustika 2018 (96.49%) and Mahendra 2018 (90.73%) used different data and methods. They are not head-to-head comparisons.

## Tuning log (validation scores only)

MobileNetV3-Small, E1, seed 42. One change per run; a gain counts only if it is a few points (one val image = 0.6 points) or shows in validation loss.

| Run | Variant (folder) | Change | Val acc | Val damaged recall | Mahendra / Mendeley acc | Best val loss | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `fitpad` (`results/`) | none: fit + pad, stage 2 lr 1e-5 / 15 epochs | 84.2 | 86.2 | 80.4 / 91.4 | 0.391 | Mahendra berries too small |
| 2 | `crop` | crop to berry, 15% margin | 85.5 | 88.8 | 82.2 / 91.4 | 0.350 | **keep**: loss −10%, 86.7% acc at 95% damaged recall (was 77.6%) |
| 3 | `crop_lr1e-4` | stage 2 lr 1e-4, up to 20 epochs | **89.7** | 88.8 | **86.9 / 94.8** | **0.279** | **keep**: loss −20%, +7 val images; 90.3% acc at 95% damaged recall. Early stopping ended stage 2 at epoch 9 (best: 5) |

**Full matrix, `crop_lr1e-4` (E1 val, mean ± sd over 3 seeds):**

| Model | Val acc | Val damaged recall | Mahendra / Mendeley acc | Acc at 95% damaged recall | Params |
| --- | --- | --- | --- | --- | --- |
| mnv3s | 89.7 ± 0.6 | 89.6 ± 1.4 | 87.2 / 94.3 | 90.1 | 0.94 M |
| effb0 | 92.1 ± 0.0 | 92.5 ± 3.3 | 88.5 / 98.9 | 92.5 | 4.05 M |
| baseline (fixed rerun) | 72.7 ± 6.4 | 68.3 ± 7.3 | 67.9 / 81.6 | 71.9 | 0.24 M |

- **First baseline run failed** (48.5%): it predicted "damaged" for every val image from epoch 1.
  - Cause: BatchNorm momentum 0.99 (the Keras default) with about 25 steps per epoch, so the running averages used at evaluation never settled. Patience 4 also stopped it at epoch 5.
  - Fix: BatchNorm momentum 0.9 and baseline patience 8. The 5 baseline runs are retrained in place (`REPLACE_CHANGED`); the pretrained runs are unaffected.
- **EfficientNet-B0 vs MobileNetV3-Small:**
  - EfficientNet-B0 is about 2.4 points better on val accuracy, consistently across seeds. Its damaged-recall gain (+2.9) is within its own spread (±3.3).
  - Grad-CAM for effb0 sits on the berries, including the crate photo, which it gets right. There's no sign of a shortcut.
- **Pre-training adds about 17–19 points** of val accuracy over the from-scratch baseline (72.7 → 89.7 / 92.1).
- **Decision: EfficientNet-B0**, chosen on val scores (float32). Reasons:
  - best val accuracy on every seed;
  - still ahead when the threshold is set for 95% damaged recall (92.5 vs 90.1);
  - Grad-CAM on the berries.

  Condition: notebook 03 must show that the quantized model keeps its val scores (within ~1 point), with measured ms/image. Otherwise use float16, or fall back to MobileNetV3-Small.

**Final test scores** (computed once with `SHOW_TEST = True`, after the model choice; not used for any decision):

| Model | E1 test acc | E1 test damaged recall | E1 test single-berry acc | E2a acc (Mendeley → Mahendra) | E2b acc (Mahendra → Mendeley) |
| --- | --- | --- | --- | --- | --- |
| **effb0** | **97.1 ± 0.9** | **96.8 ± 1.8** | **96.3 ± 1.2** | 62.7 | 74.9 |
| mnv3s | 92.8 ± 1.6 | 88.1 ± 4.3 | 90.7 ± 2.0 | 57.6 | 74.9 |
| baseline | 67.7 ± 6.0 | 59.5 ± 3.6 | 59.3 ± 8.0 | 55.9 | 46.4 |

- **E1 test is about 5 points above val** (97.1 vs 92.1).
  - It's a small set: 161 images, ±2.6 points at 95% confidence.
  - Its Mendeley part was easy: every pretrained model scored 100% on it.
  - Report it as "92–97% on held-out photos from the same two datasets".
- **E2 drops to 63–75%.** E2 can't leak, so it's the most trustworthy measure of a new camera or setting. The model as trained is **not** ready for Rancabali photos; collecting and fine-tuning on local photos is the next step.
- **Test Grad-CAM (effb0):**
  - Mendeley heat sits on the berries.
  - The errors are Mahendra damaged berries with small or subtle defects, called "good", with heat on the edge or background. This is the costly error type.

**Settings frozen after run 3** (crop, 15% margin, stage 2 lr 1e-4 / ≤20 epochs). More tuning on 165 val images would start fitting the val set itself. The full matrix runs with these settings in `crop_lr1e-4`.

## 3. Decisions that refine `MODEL_PLAN.md`
1. **E1 uses 3 seeds by default.** MODEL_PLAN had this as "if time allows". The small test set makes it necessary.
2. **Padding uses the image's border colour,** not black, to avoid a source cue.
3. **Augmentation lives in `tf.data`,** not inside the model, so the export stays clean.
4. **E2 early stopping uses the training source's val split.** The target dataset stays unseen.
5. **Predictions are saved per run,** and all metrics are computed from them. Subsets and thresholds can be re-analysed without retraining.
6. **Test is looked at once.** Development, model choice and Grad-CAM diagnosis use val. MODEL_PLAN's model-choice rule now uses val scores instead of test scores.
7. **Crop to the berry before resizing.** Decided on validation scores after the first run; see §2.1. The model then expects close-up berries, so the app crops the same way.

## 4. Risks to watch
- **Shortcuts:**
  - "pile = rotten": most rotten Mendeley photos are piles
  - juice stains under rotten berries

  If single-berry scores are clearly lower than all-image scores, or Grad-CAM shows heat off the berry, report it.
- **E2 drops:** the two datasets differ a lot (webcam vs phone photos). A big drop is a finding, not a failure.
- **Colab disconnects:** covered by the resume logic and by writing to Drive after every run.
- **Keras 3 API differences:** the `keras.applications` arguments and layer names used by Grad-CAM can differ between versions. Print versions in the setup cell, and look up the last conv layer by type, not by a hard-coded name.

## 5. Verification
- **Smoke run:** `SMOKE = True` uses 64 training images, 1 epoch per stage, and all three models on E1 only. It writes to `results/smoke/` and `models/smoke/`. It should finish in a few minutes, and is the first thing to run on Colab, or locally on CPU if TensorFlow can be installed.
- **Checks built into the notebook** (assertions). Overlap between splits is already checked in notebook 01, so it isn't repeated here.
  - **The manifest is the fixed one:** 1,123 rows. Otherwise stop with "upload the new data/manifest.csv". The old, leaky manifest had 1,132 rows. Update the number if notebook 01 changes.
  - Each preds CSV has exactly as many rows as its test set (161 / 724 / 399).
- **Plausibility flags:** print a warning if `mnv3s` E1 accuracy is below ~85% (something is broken), or exactly 100% (check for leakage before celebrating).
- **Done when:**
  - all 15 runs are in `runs.csv`
  - `results.md` is generated
  - the Grad-CAM note is written
