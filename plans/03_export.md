# Plan: notebook 03, export

**Notebook:** `notebooks/03_export.ipynb`
**Covers:** `MODEL_PLAN.md` §6. It confirms the chosen model after quantization, measures its speed, sets the shipped threshold, and packages everything both demos need.
**Depends on:** notebook 02 (`models/crop_lr1e-4/`, `results/crop_lr1e-4/`) and `data/manifest.csv`.
**Demos:** an **Android app** and a **web page**, both running the same `strawberry_2class.tflite`. The web page uses the LiteRT / TFLite web runtime; see §6.

## 1. Inputs and outputs

**Inputs** (Drive, under `PROJECT`)
- `data/manifest.csv`, and the images via `dataset.zip`, with the same setup cell as notebooks 01–02.
- `models/crop_lr1e-4/E1_{effb0,mnv3s}_s{42,43,44}.keras`
- `results/crop_lr1e-4/history/*.json` and `results/crop_lr1e-4/preds/*.csv`

**Outputs**

| File | What it holds |
| --- | --- |
| `models/export/strawberry_2class.tflite` | the shipped model (the chosen variant, §4) |
| `models/export/tflite_variants/{model}_{variant}.tflite` | every converted variant, for reference |
| `models/export/labels.txt` | `good`, `damaged` (index 0, 1) |
| `models/export/metadata.json` | input/output contract, crop parameters, threshold, scores, attribution (§8) |
| `models/export/preprocess.py` | reference crop + resize; notebook 03 imports it |
| `models/export/preprocess.js` | the same for the web demo |
| `models/export/test_vectors/` | 10 Mendeley images with expected crop boxes and probabilities (§7) |
| `results/crop_lr1e-4/export.md`, `export_runs.csv` | size, speed and accuracy per model × variant |

**Getting the export to the apps.** The notebook runs from VS Code with the Colab extension, so its kernel writes to Google Drive, not to this project folder. After the run, download `models/export/` from Drive into the project's `models/export/`. That local copy is what the Android and web builders use. `results/crop_lr1e-4/export.md` can be downloaded the same way, to keep it next to the plans.

## 2. Which trained model to export

- **Model:** EfficientNet-B0 (`effb0`) only, chosen in notebook 02 on val scores. MobileNetV3-Small isn't converted by default; add `'mnv3s'` to `MODELS` to convert it too, as a fallback or for a size/speed comparison.
- **Seed:** for each model, the E1 seed with the best **val** accuracy, tie-broken by the lowest val loss (from the history files). This is a val-only choice.
- **No retraining on train + val.** Retraining would leave no clean val set for choosing the quantization variant and the threshold.

## 3. Preprocessing: one source of truth

The model was trained on photos **cropped to the berry**. Every app must preprocess exactly the same way, or accuracy drops.

- **Source:** `notebooks/lib/preprocess.py` and `notebooks/lib/preprocess.js` in the project. The notebook generator embeds them in notebook 03, and the kernel writes them to `models/export/`. Edit the `lib/` files, then regenerate the notebook.
- **Checked locally:**
  - `preprocess.fit_pad` is byte-identical to notebook 02's on all 1,123 images.
  - `preprocess.js` gives identical crop boxes on 17 mixed photos, with pixels within 2 levels.
- `preprocess.py` holds `berry_box` + `fit_pad`, copied unchanged from notebook 02:
  - colour difference from the median border colour > 60 **and** HSV saturation > 70;
  - full-width lines removed;
  - 1–99 percentile box, 15% margin, made square;
  - EXIF rotation applied;
  - fitted inside 224×224, padded with the median border colour.

  Notebook 03 imports it from `models/export/`, so the notebook and the apps use the same file.
- **Parity check:**
  - Predict val and test with the float32 Keras model on images preprocessed by `preprocess.py`.
  - Compare with notebook 02's saved predictions (`results/crop_lr1e-4/preds/E1_{model}_s{seed}[_val].csv`).
  - Assert max |Δp| < 1e-3. This proves the export uses the same preprocessing as training.
- **Model input/output contract** (written into `metadata.json`):

| | |
| --- | --- |
| Input | 1 × 224 × 224 × 3, float32, RGB, values **0–255**, after `preprocess` |
| Normalisation | **none**: EfficientNet-B0 and MobileNetV3 rescale inside the model |
| Output | 1 × 1, float32, **P(damaged)** |
| Decision | `damaged` if P ≥ `threshold` (from `metadata.json`), else `good` |

## 4. TFLite conversion and choice of variant

1. Export each Keras model as a SavedModel (`model.export(dir)`), then convert with `tf.lite.TFLiteConverter.from_saved_model` into four variants:

| Variant | How | Expected size (effb0) |
| --- | --- | --- |
| `float32` | no optimisation; the reference | ~16 MB |
| `float16` | `target_spec.supported_types = [tf.float16]` | ~8 MB |
| `dynamic_int8` | `optimizations = [DEFAULT]`, weights in int8 | ~4–5 MB |
| `full_int8` | `DEFAULT` + a representative dataset of 200 **train** images; input and output stay float32 so the contract doesn't change | ~4–5 MB |

2. **Evaluate every variant** with `tf.lite.Interpreter`:
   - on E1 **val**, which is used for the choice;
   - on E1 **test**, reported once, never used.
3. **Speed:**
   - Interpreter with `num_threads=4` on the Colab CPU: 5 warm-up runs, then the median of 50 single-image runs, in ms/image.
   - This is a CPU reference, not a phone number. Phone timing comes from the Android app (an app log, or the TFLite benchmark tool) and from the web demo (`performance.now()`).
4. **Choose the variant:** among the variants whose val accuracy **and** val damaged recall are both within ~1 point of float32, the **fastest**; ties go to the smaller file.
   - **Changed after the first run.** The rule was "smallest", and it picked dynamic int8: 4.6 MB, but 92 ms/image against 21 ms for float16 (it de-quantizes weights while running). Its predictions also shifted: val damaged recall went from 88.8% to 93.8%, so it wasn't the model evaluated in notebook 02.
   - Float16 is 8.1 MB and gives the same val scores as float32.
   - This choice uses speed and val scores only; test was 98.1% for both.
   - **Changed again after the second run.** Timing on free Colab's shared CPU is too noisy to rank variants that run at the same speed. Float16 and float32 swapped places between runs (21 vs 34 ms, then 35 vs 23 ms), and "fastest" shipped float32 at 16 MB.
   - Since 2026-11-27, `SHIP_VARIANT = 'float16'` pins the decision. The notebook still checks that float16 is within 1 point of float32 on val; if not, the fastest-passing rule decides. EfficientNet often loses accuracy in full int8 (its swish activations and squeeze-excite blocks quantize poorly), so float16 is the likely pick.
5. **If something goes wrong:**
   - **No effb0 variant passes:** ship effb0 float16 anyway.
   - **effb0 is slow** (over ~150 ms/image on the Colab CPU, a rough stand-in for a mid-range phone): the notebook warns and still ships it; measure on the phone. With `'mnv3s'` added to `MODELS`, it would ship MobileNet instead.

## 5. Threshold (val-tuned for 95% damaged recall)

- **How it's chosen:** on the **chosen variant's** val predictions, take the largest threshold that still gives damaged recall ≥ 95%.
  - This is the same rule as `recall_threshold` in notebook 02, but not capped at 0.5.
  - If it lands above 0.5, that means 0.5 already exceeds 95% recall on val; record it as it is.
- **Reported once on test:** accuracy, damaged recall and good precision at that threshold, next to the scores at 0.5.
- **Stored in `metadata.json`.** The apps read it from there and never hard-code it.
- **Why this threshold:** a rotten berry passed as good is the costly error (`MODEL_PLAN.md` §5). In notebook 02, effb0 at a val-tuned threshold kept 92.5% val accuracy at 95% damaged recall.

## 6. Web demo: the same `.tflite` in the browser

**Changed after the first run.** The planned TF.js conversion failed on Colab: Python 3.13 couldn't create the separate converter environment, and `tensorflowjs` has further dependency problems on 3.13. Decision: the web demo runs **the same `strawberry_2class.tflite`** in the browser with Google's LiteRT / TFLite web runtime (WebAssembly).
- **One model file** for both demos, so their scores are identical and nothing has to be kept in sync.
- **No converter** to install.

**Check in the notebook** (`RUN_WEB_CHECK`):
1. Node runs `preprocess.js` on the test-vector photos. Python does the full-size decode first, so only `preprocess.js` is under test.
2. Its 224×224 output goes through the shipped `.tflite`.
3. Compare with the Python reference: boxes within 1% of the longer side, P(damaged) within 0.02.

Tested locally end to end: identical boxes, |Δ| 0.0004.

**Check once in the browser:** load the `.tflite` with the web runtime and run the test vectors. The runtime itself can't be tested on Colab.

## 7. The crop in the apps

- **`preprocess.js`** ports `preprocess.py` to the browser. It works on canvas `ImageData`: median border colour, per-pixel HSV saturation, the same thresholds, line filter, percentiles, margin and padding. The browser applies EXIF rotation when it decodes images (`image-orientation: from-image`), and the port relies on that.
- **Android:** no Kotlin port in this notebook. Whoever builds the app ports `preprocess.py` and checks the port against the test vectors.
- **`test_vectors/`:**
  - **10 Mendeley images only.** Mendeley is CC BY 4.0. Mahendra's images are GPL and must not be redistributed with the apps.
  - **The mix:** fresh single berries, piles, a dark mouldy rotten berry, a mashed rotten berry, and a crate.
  - **For each image** (in `test_vectors.json`): file name, expected crop box, mean and std of the 224×224 output, and the expected P(damaged) from the float32 model.
- **Pass criteria for any port:** boxes within **1% of the photo's longer side**, and P within 0.02. (The box is found on a 256×256 copy, so one step there is about 16 px on a 4000-px photo; an absolute pixel tolerance makes no sense.)
- **Decode at full size.** Training decoded big JPEGs at reduced size (PIL `draft`), while phones and browsers decode at full size. The test vectors record P for both decodes, so the effect is visible; it's small.

## 8. `metadata.json`

- **Model:** name and version, source run (`effb0`, seed, variant `crop_lr1e-4`), shipped TFLite variant, TF/Keras versions, date.
- **Input/output contract** from §3.
- **Crop parameters:** thresholds 60 / 70, margin 0.15, line filter 0.8, percentiles 1–99, size 224.
- **Threshold,** with a note that it was chosen on val for ≥95% damaged recall.
- **Labels:** `["good", "damaged"]`.
- **Scores for the shipped variant:**
  - val and test at both thresholds (0.5 and the shipped one);
  - **E2a / E2b** (63–75%), as an explicit warning that new cameras and settings need local data.
- **Attribution:** "Fresh and Rotten Fruits Dataset, Mendeley Data, DOI 10.17632/bdd69gyhv8.1, CC BY 4.0" and "Mahendra et al. 2018 strawberry dataset (GPL), github.com/okamahendra123/dataset_strawberry". Show both in the apps' About screen.

## 9. `results/crop_lr1e-4/export.md`

- **One row per model × variant:** size (MB), ms/image (Colab CPU), val acc, val damaged recall, Δ vs float32, then test acc and test damaged recall.
- **The shipped choice:** variant, threshold, and the test scores at that threshold.
- **The checks:** preprocessing parity, float32 TFLite vs Keras, and the web check (`preprocess.js` + the shipped `.tflite`).
- Notebook 02's `results.md` keeps "(nb 03)" in its `.tflite MB` and `ms / image` columns. `export.md` fills them in.

## 10. Rules

- **Val only for decisions:** seed, variant and threshold are all chosen on val. Test is computed once and reported.
- **The crop is part of the model:** shipping the `.tflite` without the crop is a bug.
- **Licences:** no Mahendra images in `models/export/` or in either app; dataset credits from `metadata.json` go in the About screen.

## 11. Verification

**Assertions in the notebook:**
- preprocessing parity: max |Δp| < 1e-3 against notebook 02's predictions;
- float32 TFLite vs Keras: max |Δp| < 1e-3 (GPU Keras vs CPU TFLite);
- every variant predicts exactly as many images as val (165) and test (161);
- web check: `preprocess.js` boxes match `preprocess.py` within 1% of the longer side, and the shipped `.tflite` on its output matches the Python path within 0.02;
- `metadata.json` loads and has every field in §8.

**Before Colab, locally:** `preprocess.py` gives the same 224×224 arrays as notebook 02's `fit_pad` on all 1,123 images. It needs only Pillow and NumPy.

**Done when:**
- `strawberry_2class.tflite`, `labels.txt`, `metadata.json`, both preprocess files and `test_vectors/` exist in `models/export/`;
- `export.md` is written;
- `MODEL_PLAN.md` §6 records the shipped model, variant, size, speed and threshold.

## 12. Risks

- **int8 accuracy loss on EfficientNet:** expected; float16 is the planned fallback within effb0.
- **Converter breaking Colab's TensorFlow:** avoided with the separate virtualenv; the laptop is the backup.
- **App preprocessing drifting from training** (a different resize filter, a missing EXIF rotation, normalising to 0–1): caught by the test vectors. The most common bug is dividing by 255; the model expects 0–255.
- **Cross-dataset weakness (E2 63–75%):** not fixed by exporting. The apps should say the model was trained on two public datasets, and that local photos are the next step.
