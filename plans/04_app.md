# Plan: the grading app (Flutter, native Android)

**Folder:** `app/` (not built yet)
**Depends on:** `models/export/` from notebook 03 (download it from Drive into the project), and the crop reference `notebooks/lib/preprocess.py`.
**Decisions:**
- **A native app built with Flutter.** The model runs on the phone's native TFLite runtime, and everything works **offline**. Offline is the main goal.
- **Submission: the mobile category,** with an **`.apk`**. The `.ipa` (iOS) is optional; Flutter can build it later on a Mac.
- **UI in Bahasa Indonesia, with an English toggle.**
- **Why native instead of a web app.** The judging rubric scores three things:
  - whether the main features work, and the code is clean;
  - whether the technology architecture fits;
  - "APK siap pasang", an APK ready to install.

  On-device TFLite in a native app is the standard architecture for this. It ships the model inside the APK and needs no network or caching step.

## 1. Goal and scope

**MVP**
- A farmer takes or picks a photo of **one strawberry**. The app says **good** or **damaged**, and shows the crop it graded.
- Over a session of several photos, it tallies the counts and gives advice:
  - good → **sell fresh**;
  - damaged → **sort again** (processing such as jam or juice, or compost).
- It estimates kg as count × an average berry weight, which the farmer can edit.
- It keeps a harvest log on the phone.
- **Fully offline:** no network permission is needed for grading.

**Out of scope for the MVP**
- **Grading a whole tray from one photo:** optional (§8), because it needs berry detection.
- **More than 2 grades:** no dataset separates second grade from reject.
- **Anything below the surface:** a camera can't see it.

## 2. Stack, and why

| Part | Choice | Why |
| --- | --- | --- |
| App | **Flutter (Dart)**, Android first | Native APK; one codebase for iOS later |
| Model runtime | **`tflite_flutter`** (TensorFlow Lite / LiteRT), CPU with 4 threads | Runs `strawberry_2class.tflite` natively, offline. GPU delegate optional if speed needs it |
| Crop | **Dart port of `preprocess.py`**, in `lib/core/preprocess/` | Training cropped every photo, so the app must crop the same way. Port it from `notebooks/lib/preprocess.js`, which already reproduces Pillow exactly, and **prove it against `test_vectors/`** (§9) |
| Heavy work | A background **isolate** (`Isolate.run`) for decoding and the crop; `IsolateInterpreter` for inference | The UI never freezes on 12 MP phone photos |
| Photos | `image_picker` (camera + gallery) | Simplest reliable path to the full-size photo. A live preview (`camera` plugin) is optional |
| Storage | `sqflite` | Harvest log on the device |
| Translations | `flutter_localizations` + `intl` ARB files: `app_id.arb` (default), `app_en.arb` | Farmers read Indonesian; judges may read English |
| State | `provider` with `ChangeNotifier` | Small and readable, which supports the "clean code" criterion |

**Technical unknowns, all settled in Phase 0 (§8):**
1. **`tflite_flutter` loads the model** and gives the same P as the Python path.
2. **The Dart crop matches the test vectors.**
3. **Decoding and EXIF rotation.** Does Flutter's image decoding apply EXIF rotation? If not, apply it explicitly, for example with the `image` package's `bakeOrientation`. And how fast is decoding a 12 MP photo?

## 3. The contract with the model

The app reads `metadata.json` and `labels.txt` from its assets at start-up. Nothing below is hard-coded.

| | |
| --- | --- |
| Input | An upright photo → the crop (port of `preprocess.py`) → Float32 `[1, 224, 224, 3]`, RGB, values **0–255**. **No /255**: the model rescales internally |
| Output | P(damaged), a single number |
| Decision | **damaged if P ≥ `threshold`** (0.261, chosen on validation for ≥95% damaged recall), else good |
| Labels | `labels.txt`: `good`, `damaged` |

**Decoding size.** Training decoded big JPEGs at reduced size (Pillow `draft`, at least 896 px), and on the test vectors P was the same for full-size and reduced decodes. So the app may decode with a long side of about 1,024–1,600 px for speed, **if** the test vectors still pass (§9). Phase 0 measures both.

**Uncertain results** (show these instead of a verdict):
- **No berry found** (crop box is null): "Retake on a plain, light background, one berry, filling the frame."
- **P close to the threshold:** within a band, e.g. 0.15–0.40, tuned on the demo photos. Show "Unsure: take another photo."

**Limits stated in the app** (About screen):
- one berry per photo;
- surface only;
- trained on two public datasets, so accuracy on new phones and settings is unproven. Cross-dataset tests gave 63–75%. See `MODEL_PLAN.md`.

## 4. Architecture

```
image_picker (camera / gallery) → file path
  → isolate: decode (+ EXIF upright, optional downscale) → RGB bytes
             → berryBox → crop → thumbnail (Lanczos) → pad → Float32List [1,224,224,3]
  → IsolateInterpreter (tflite_flutter, model loaded once at start-up) → P(damaged)
  → Decision (threshold, uncertain band) → Result screen (+ crop preview)
```

Code layout, in layers, for clean code:
```
app/lib/
  main.dart
  core/
    preprocess/berry_crop.dart     # port of preprocess.py: berryBox, fitPad, Pillow-exact resample
    model/grader.dart              # loads model + metadata once; grade(File) -> GradeResult
    decision.dart                  # threshold, uncertain band, kg estimate (pure Dart, unit-tested)
  data/
    metadata.dart                  # parses metadata.json
    session_repository.dart        # sqflite: sessions + counts
  ui/
    screens/ grade, result, summary, history, settings, about, self_test
    widgets/
  l10n/ app_id.arb  app_en.arb
```

## 5. Screens

All text lives in the ARB files, with a language toggle in Settings and in the app bar.

1. **Grade (home):** a big **Take photo** button, **Choose from gallery**, and the current session's running count.
2. **Result:**
   - **Verdict:** good or damaged, in large text with a colour and an icon.
   - **Confidence:** in words ("very likely", "likely") rather than a raw percentage.
   - **The crop preview.**
   - **Buttons:** **Next berry** and **Retake**.
3. **Session summary:** counts per class, estimated kg per class, and the advice per class. **Save session** writes it to History.
4. **History:** saved sessions (date, counts, kg), newest first.
5. **Settings:** language, average berry weight (grams), and clearing history.
6. **About:**
   - how it works and its limits (§3);
   - the model version and threshold, from `metadata.json`;
   - **dataset credits** from `metadata.json`: Mendeley CC BY 4.0 with its DOI, and Mahendra et al. 2018 (GPL).
7. **Self-test** (hidden: long-press the version number on About):
   - The user picks the 10 test-vector photos and `test_vectors.json` from phone storage (copied there from `models/export/test_vectors/`). They are **not** bundled, which keeps the APK small.
   - Runs them through the real pipeline.
   - Checks: box within 1% of the photo's longer side, P within 0.02 of `p_damaged_tflite`.
   - Shows pass/fail and ms per photo.

## 6. Assets and source of truth

```
app/
  pubspec.yaml
  tool/sync_model.dart         # copies models/export/{strawberry_2class.tflite, metadata.json, labels.txt}
                               # into app/assets/model/; prints the model version. Run before building.
  assets/model/                # synced copies (git-ignored)
  test/
    decision_test.dart
    berry_crop_test.dart       # runs the crop on models/export/test_vectors/*.jpg, compares boxes/stats
  android/                     # signing config reads key.properties (git-ignored)
```
- **The model and its metadata come only from `models/export/`.** To change the model, re-export in notebook 03, download, then run `tool/sync_model.dart`.
- **The crop reference is `notebooks/lib/preprocess.py`.** The Dart port must keep passing `berry_crop_test.dart`. If the reference changes, change the port and rerun the test.

## 7. Licences

- **No Mahendra (GPL) images** in the app or its assets. The test vectors are Mendeley only (CC BY 4.0) and aren't bundled.
- **The About screen credits both datasets,** using the strings in `metadata.json`.

## 8. Phases (multi-day hackathon)

Each phase ends with a check. Don't start the next phase until it passes.

**Phase 0: prove the model and the crop run natively (first day, the gate)**
- A bare Flutter app that bundles the model, loads it with `tflite_flutter`, runs the Dart crop, and has the self-test screen.
- **Check:**
  - `berry_crop_test.dart` passes on the computer;
  - the self-test passes on a **mid-range Android phone**, in about 2 s per photo or less (decode + crop + inference);
  - a **release APK** installs on a phone that never had the app.
- **Also measure:** full-size vs reduced-size decoding (§3), and whether EXIF rotation is applied. Test with a photo taken in portrait mode.
- **If `tflite_flutter` fails:** call the Android TFLite Interpreter through a small Kotlin platform channel. The crop and UI don't change.

**Phase 1: one photo → result**
- Grade and Result screens, isolate pipeline, crop preview, Indonesian/English.
- **Check:** the 10 test photos give the expected verdicts in the UI, and an empty or non-berry photo gives "no berry found" or "unsure".

**Phase 2: session**
- Tally, advice per class, kg estimate, Summary screen.
- **Check:** a session of mixed photos gives the right counts and kg.

**Phase 3: history, settings, about, release build**
- History (sqflite), Settings, About.
- App name, icon and Android permissions (camera only; **no internet permission**).
- A **signed release APK** (`flutter build apk --release`). Keep the keystore and `key.properties` out of git.
- **Check:** with airplane mode on, a fresh install grades photos and saves a session.

**Phase 4: demo-ready**
- Polish, and a demo script for the video.
- **Test on the team's own phone photos:** 50–100 single berries on a plain sheet, labelled by the team. Report the accuracy honestly, and record it in `MODEL_PLAN.md` and the notes file.
- **Check:** a fresh install of the exact APK file to be submitted runs the whole demo script.

**Optional, if time allows**
- **Tray grading:** split a tray photo into single-berry crops (e.g. extend `berryBox` to connected components on a plain sheet), grade each one, and show the counts. Berries that touch will merge; say so.
- **iOS `.ipa`:** optional in the mobile category, and needs a Mac and an Apple developer account. The same Flutter code builds it.
- **GPU delegate** in `tflite_flutter`, if CPU speed is short of the budget on low-end phones.

## 9. Verification

- **Crop port, on the computer:** `berry_crop_test.dart` on all 10 test vectors. Box within 1% of the longer side; 224×224 mean and std close to `test_vectors.json`.
- **Whole pipeline, on the phone:** the self-test screen. Box within 1%, and P within 0.02 of `p_damaged_tflite`.
- **Unit tests:** `decision_test.dart` covers the threshold, the uncertain band and the kg estimate.
- **Offline:** airplane mode on a fresh install; grading and history work.
- **Speed budget:** about 2 s per photo or less on a mid-range Android phone, measured on the self-test screen and recorded in `MODEL_PLAN.md`.
- **Demo check:** the expected verdicts on the test photos, and "no berry found" or "unsure" on a non-strawberry photo, not a confident verdict.

## 10. Submission (mobile category)

What the competition asks for:
- **`.apk`:** a signed release build. `.ipa` is optional.
- **GitHub repository link.**
- **Proposal document.**
- **Originality statement** (surat pernyataan orisinalitas).
- **Demo video.**

Checklist:
- **Install the exact `.apk` file you submit** on a phone that never had the app ("install from unknown sources"), and run the demo script offline.
- **The repository** builds after `tool/sync_model.dart`, and contains no dataset, no Mahendra photos, no keystore and no `key.properties`.
- **The demo video** follows the Phase 4 demo script, recorded on a real phone in airplane mode, so the offline claim is visible.
