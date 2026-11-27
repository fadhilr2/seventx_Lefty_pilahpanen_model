# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Hackathon prototype: a small image classifier that labels a photo of **one strawberry** as `good` or `damaged`, small enough to run on a phone. Code so far: `notebooks/01_manifest_and_split.ipynb` (done) and `notebooks/02_train_eval.ipynb` (done on Colab; results in `results/crop_lr1e-4/` on Drive, summarised in the tuning log of `plans/02_train_eval.md`). `notebooks/03_export.ipynb` is done: EfficientNet-B0 float16 shipped as `models/export/strawberry_2class.tflite` (8.1 MB, threshold 0.261) on Drive. Open items: test on the team's own phone photos, and check the model in the browser once. The crop code shared by training, export and both apps lives in `notebooks/lib/preprocess.py` (reference) and `preprocess.js` (web port, Pillow-exact resampling). Notebook 03 embeds both, so after editing them, regenerate it: `python notebooks/build/make_nb03.py notebooks/03_export.ipynb`. `notebooks/build/` holds the generators of all three notebooks. Notebooks 01 and 02 have since been patched in place, so for those the `.ipynb` files are the source of truth; regenerating them would also wipe the run outputs saved in them. No build, tests or dependency files, and the folder is not a git repo. `MODEL_PLAN.md` is the source of truth for everything to be built. Read it before starting work, and update it if a decision changes. Detailed per-notebook designs live in `plans/` (`plans/01_data_and_splits.md`, `plans/02_train_eval.md`, `plans/03_export.md`); read the matching one before building a notebook.

## Data (`dataset/`)

Two sources with different folder layouts. Each folder name maps to a label:

| Folder | Label | Source (`manifest.csv` value) | Notes |
| --- | --- | --- | --- |
| `color/good` (382), `color/damaged` (350) | good / damaged | `mahendra` | webcam 640×480, GPL licence: don't bundle raw images in the app |
| `FreshStrawberry/` (737), `RottenStrawberry/` (737) | good / damaged | `mendeley` | high-res photos on a white background, CC BY 4.0 (DOI 10.17632/bdd69gyhv8.1) |

Facts found by inspecting the images (details in `plans/01_data_and_splits.md`):
- Mendeley files numbered 201–737 are augmented copies of 1–200 and are excluded. Never add them back; that would leak copies of training photos into test.
- Many Mendeley photos show piles of berries (`multi_berry = 1`, hand-tagged in notebook 01): ~1/3 of fresh, ~2/3 of rotten. Report test scores on all images and on `multi_berry == 0`.
- Rotten Mendeley berries often sit in juice stains. Together with piles, this is a possible shortcut; Grad-CAM checks exist for this reason.

## Running notebooks: Colab (main) and local

The user runs the notebooks **from VS Code with the Colab extension**: the `.ipynb` files stay in this local project and only the kernel (and GPU) is on Colab.
- Never tell the user to upload notebooks. Edits to the local notebook files take effect directly; the user must reload a notebook that is open in VS Code after it is edited on disk.
- Cell outputs are saved into the local `.ipynb`, so read them there to see run results.
- The kernel can't see local files. Data (`dataset.zip`, `data/manifest.csv`) must be on Google Drive, and outputs (`models/`, `results/`) are written to Drive. Exported files must be downloaded from Drive to reach this folder.

Training runs on **Google Colab, reading from Google Drive**. Every notebook starts with the same setup cell (copy it from notebook 01):
- It detects Colab, mounts Drive, and sets two paths:
  - `PROJECT = /content/drive/MyDrive/seventx_prototype`: outputs (`data/`, `models/`, `results/`) are written here so they persist.
  - `ROOT = /content/seventx`: a local copy of the images, unzipped from `PROJECT/dataset.zip` (fallback: copies `PROJECT/dataset/`). Reading ~520 MB of photos from Drive directly is too slow.
- Paths in `manifest.csv` are relative to `ROOT`. Always read images via `ROOT / path` and write outputs under `PROJECT`, never the other way round.
- Locally, `PROJECT = ROOT =` the project folder.

`dataset.zip` (project root, ~607 MB, all 2,206 images with paths `dataset/...`) is the file to upload to Drive. Rebuild it if `dataset/` changes.

Run notebook 01 headless locally (needs only Pillow and NumPy) from `notebooks/`:

```
python -m nbconvert --to notebook --execute --inplace 01_manifest_and_split.ipynb
```

(The `jupyter` launcher on this machine exits silently; use `python -m nbconvert`.)

## Planned pipeline (from `MODEL_PLAN.md`)

- **Stack:** TensorFlow/Keras on Google Colab. One TFLite model serves both demos: the Android app, and the web page via the LiteRT / TFLite web runtime. TF.js was dropped.
- **`notebooks/01_manifest_and_split.ipynb` (done):** writes `data/manifest.csv` (1,123 rows) and `data/excluded.csv`. The split is 70/15/15 **by group** (Mahendra bursts, hand-marked Mendeley photo series and near-duplicates stay together), stratified by source × label, seed 42, followed by a purge of val/test images within 8 hash bits of an earlier split. Every later step reads splits from this manifest. Never re-split anywhere else, and never split image by image.
- **`notebooks/02_train_eval.ipynb`:** three models (a from-scratch CNN baseline, **MobileNetV3-Small** as the main candidate, and EfficientNet-B0), each with the head GAP → Dropout 0.2 → Dense(1, sigmoid).
  - Experiments: **E1** trains and tests on both sources mixed. **E2a/E2b** train on one source and test on the other.
  - Results go to `results/results.md`.
- **`notebooks/03_export.ipynb`** (design: `plans/03_export.md`):
  - Converts effb0 (and mnv3s as fallback) to four TFLite variants: float32, float16, dynamic int8, full int8.
  - Picks the **fastest** variant within ~1 point of float32 on **val** (float16; "smallest" picked a 4× slower dynamic int8), measures ms/image, and sets the val-tuned threshold.
  - Web check: Node runs `preprocess.js` on the test vectors, and its output goes through the shipped `.tflite`.
  - Writes everything to `models/export/`. The numbers reported must come from the shipped model.

## Conventions that are easy to get wrong

- Keras MobileNetV3 and EfficientNet rescale their inputs internally, so feed them raw 0–255 pixels and don't normalise twice. Input is 224×224 with the aspect ratio kept by padding.
- Images are **cropped to the berry** before resizing (`berry_box` in notebook 02: coloured pixels that differ from the background, 15% margin, square). Training, evaluation, export and the app must all use the same crop.
- Fine-tuning has two stages: first train the head with the backbone frozen (lr 1e-3), then unfreeze the top ~30% (lr 1e-4, ≤20 epochs; 1e-5 was too slow) while BatchNorm layers stay frozen. Settings are frozen after tuning run 3 (`RUN_TAG = 'lr1e-4'`); see the tuning log in `plans/02_train_eval.md`. Early stopping on validation loss uses patience 4 (the from-scratch baseline uses patience 8 and BatchNorm momentum 0.9; with the Keras default 0.99 it predicted one class for everything).
- The key metric is **recall on `damaged`**. Tune the decision threshold only on the validation set, never on the test set.
- Model choice rule: compare models after quantization on **validation** scores (E1 val accuracy and val damaged recall). Pick the smallest model within ~1 point of the best. Move to a bigger model only for a clear gain on val damaged recall. Test scores (E1 test, E2) are computed once after the choice and never used to choose.
- **Chosen model: EfficientNet-B0** (`effb0`, variant `crop_lr1e-4`), on val scores. Notebook 03 must confirm it after quantization.
- The apps get three things, all in `models/export/`: the model, the **crop** (`preprocess.py` / `preprocess.js`, checked against `test_vectors/`) and the **threshold** (`metadata.json`, val-tuned for ≥95% damaged recall). Input is float32 RGB 0–255 with no normalisation; dividing by 255 in an app is a bug.
- Test-set discipline: develop, tune and diagnose (including Grad-CAM) on val only. Notebook 02 hides test metrics behind `SHOW_TEST = False` until settings are frozen.
- The paper accuracies (Sustika 96.49%; Mahendra 90.73% with SVM+SURF and 5-fold cross-validation) are reference points on different setups. Never present them as head-to-head comparisons.

## Reference material

`papers/` holds Markdown notes on Sustika et al. 2018 (a comparison of CNN models) and Mahendra et al. 2018 (SVM with hand-picked features on the `color/` dataset). They are kept as Markdown on purpose, so they are easy to read and search. Don't swap them for PDFs.
