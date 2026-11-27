# Plan: data, splits and experiments

**Notebook:** `notebooks/01_manifest_and_split.ipynb` (**done**)
**Outputs:** `data/manifest.csv` (1,123 images) and `data/excluded.csv` (1,083 images, with the reason for each)
**Used by:** notebook 02 (`plans/02_train_eval.md`), which runs the experiments defined in §6 below.

## 1. Sources

Location: `seventx_prototype/dataset/`, uploaded to Colab as `dataset.zip`.

| Source | Folder | good | damaged | Camera, setting | Licence |
| --- | --- | --- | --- | --- | --- |
| Mahendra 2018, [GitHub](https://github.com/okamahendra123/dataset_strawberry) | `color/good`, `color/damaged` | 382 | 350 | webcam 640×480, one berry, plain light background | GPL: credit it, don't bundle raw images in the app |
| Fresh and Rotten Fruits Dataset, [Mendeley bdd69gyhv8](https://data.mendeley.com/datasets/bdd69gyhv8/1) | `FreshStrawberry/` → good, `RottenStrawberry/` → damaged | 737 | 737 | phone photos, white background; **many show piles of berries** (§2) | CC BY 4.0: credit title and DOI 10.17632/bdd69gyhv8.1 |

On disk: 1,119 good and 1,087 damaged. **After cleaning: 574 good and 549 damaged (1,123 images).**

## 2. What we found in the images, and what we do about it

| Finding | What notebook 01 does |
| --- | --- |
| **Mendeley 201–737 are augmented copies.** In both folders, 1–200 are the original photos (camera sizes). 201–737 are rotated 1020×1020 versions of them, with black corners. | Drop 201–737 (`excluded.csv`). This leaves 200 fresh and 200 rotten. We do our own augmentation in training. |
| **No exact duplicates (MD5), but many re-shots of the same berry.** Mahendra photographed each berry several times, seconds apart, and sometimes again much later (one berry at 21:23 and 21:48). Mendeley photographed each berry or pile in a series of consecutive file numbers, and fresh 37–47 and 189–200 repeat earlier berries. | Split **by group**, never image by image, then purge near-duplicates that cross splits (§4). |
| **Many Mendeley photos show more than one berry:** 68 of 200 fresh (88–150 are small piles, 173–177 are crates) and 131 of 200 rotten. All Mahendra photos are single berries. | Keep them, and tag `multi_berry = 1` by hand (ranges below). Report test scores on all images **and** on single-berry images only, since the demo grades one berry. |
| **Rotten Mendeley berries often sit in juice stains.** | Nothing in the data. Notebook 02 checks for it with Grad-CAM (§6). |

**Hand tags** (from looking at every Mendeley original; edit them in notebook 01 if one is wrong):
- Fresh with `multi_berry = 1`: 88–150, 173–177.
- Rotten with `multi_berry = 0` (single berry): 1, 5, 10–13, 21, 50–52, 84–87, 90, 92, 93, 95–101, 151–166, 169–197. Every other rotten original is a pile.

## 3. The manifest

`data/manifest.csv`, one row per image used. Every later notebook reads it, and **nothing re-splits the data**.

| Column | Meaning |
| --- | --- |
| `path` | relative to `ROOT`, e.g. `dataset/color/good/image_….jpg` |
| `source` | `mahendra` or `mendeley` |
| `label` | `good` or `damaged` |
| `split` | `train`, `val` or `test` |
| `group` | photos that must stay together (same berry, burst or near-duplicate) |
| `multi_berry` | 1 if the photo shows more than one berry |
| `width`, `height` | original size in pixels |
| `md5`, `phash` | exact hash, and a 64-bit perceptual hash of a crop around the berry |

## 4. Groups and split

**Groups.** A group never spans two splits. Images go into the same group when:
- **Mahendra:** photos with the same label taken ≤ 30 s apart (one shooting burst).
- **Mendeley:** photos in the same hand-marked **photo series**, a run of file numbers showing the same berry or pile. Where it was unclear whether two neighbouring runs were the same berries, they were merged.
  - Fresh: 1–8, 9–33, 34–41, 42–69, 70–87, 88–113, 114–127, 128–150, 151–160, 161–172, 173–177, 178–194, 195–200.
  - Rotten: 1, 2–9, 10–13, 14–22, 23–29, 30–49, 50–52, 53–83, 84–91, 92–101, 102–110, 111–124, 125–130, 131–150, 151–168, 169–197, 198–200.
  - The first version used fixed blocks of 10 file numbers. A leakage check showed series crossing block edges, e.g. rotten 10 (train) and 11–13 (test) are the same berry, and so are rotten 169–170 (test) and 171–172 (train).
- **Both sources:** the perceptual hash of the berry crop is within 6 bits, for the same source and label. This links far-apart re-shots, e.g. fresh 37–47 with 189–200. The crop matters: on the whole photo, Mahendra's small berry on a big white background matched unrelated photos. Pairs this close across labels or sources would be printed for a manual look. There are none.

The rules are deliberately cautious. They may merge different berries into one group, but they shouldn't put the same berry in both train and test. The cost is big, uneven groups: fresh Mendeley has only 9 groups of 5–43 photos.

**Split:** 70 / 15 / 15 for train / val / test, by group, stratified by **source × label**, seed 42.
- With groups this lumpy, one greedy pass can miss the target badly (it first gave fresh test only 6%).
- So within each stratum, the notebook tries 5,000 seeded group orders. Each order is filled greedily: a group goes to the split furthest below its target.
- It keeps the order closest to 70/15/15 in both image count and single-berry count, so each split gets a fair mix of single berries and piles.

**Purge.** After the split, a val image within **8 bits** of a train image, or a test image within 8 bits of a train or val image (same source and label), is removed and listed in `excluded.csv`.
- This catches re-shots that the grouping missed, like the Mahendra berry photographed 25 minutes apart.
- Merging at 8 bits instead would chain unrelated photos into huge groups, so these images are dropped, not merged. It removed 3 val and 6 test images (8 Mahendra, 1 fresh pile).
- At 9–10 bits, 23 test pairs remain. In an image sheet, pairs at that distance were mostly different berries, so they stay.

Result (single-berry / multi-berry):

| Split | Mahendra good | Mahendra damaged | Mendeley good | Mendeley damaged | Total |
| --- | --- | --- | --- | --- | --- |
| train | 268 / 0 | 245 / 0 | 95 / 49 | 49 / 91 | 797 |
| val | 55 / 0 | 52 / 0 | 25 / 5 | 10 / 18 | 165 |
| test | 52 / 0 | 52 / 0 | 12 / 13 | 10 / 22 | **161** (126 single-berry) |

## 5. Running it

- **Colab (main):**
  - Put `seventx_prototype/` in `MyDrive`, with `dataset.zip` inside.
  - The first cell mounts Drive and unzips the images to `ROOT = /content/seventx` (reading from Drive directly is slow).
  - It writes outputs to `PROJECT = /content/drive/MyDrive/seventx_prototype`.
  - You don't need to rerun notebook 01 on Colab: `data/manifest.csv` is already built.
- **Locally:** `PROJECT = ROOT =` the project folder. Needs only Pillow and NumPy. Run it from `notebooks/`:
  ```
  python -m nbconvert --to notebook --execute --inplace 01_manifest_and_split.ipynb
  ```
- **Checks built into the notebook** (assertions):
  - every Mendeley file number 1–200 is in exactly one series
  - no group spans two splits
  - no exact copy (MD5) is in two splits
  - no val/test image is within 8 bits of an image in an earlier split (same source and label)
  - every image file is in either `manifest.csv` or `excluded.csv`

## 6. Experiments (run in notebook 02)

| Exp | Question | Train | Early stopping (val) | Test |
| --- | --- | --- | --- | --- |
| **E1**, main result | How good is it on our data? | `train`, both sources (797) | `val`, both sources (165) | `test`, both sources (161) |
| **E2a**, cross-dataset | Does it work on photos it hasn't seen? | Mendeley `train` (284) | Mendeley `val` (58) | all Mahendra in the manifest (724) |
| **E2b**, cross-dataset | Does it work on photos it hasn't seen? | Mahendra `train` (513) | Mahendra `val` (107) | all Mendeley in the manifest (399) |

- **E1:** choose settings on val, report once on test. Run it with 3 seeds and report mean ± sd (see §7).
- **E2:**
  - Validation and threshold choice always use the **training** source, so the test dataset stays completely unseen.
  - E2 can't leak at all: the two datasets show different berries. That makes it our cleanest number.
  - A big drop is a finding to report honestly, not a failure to hide. The two datasets differ a lot (webcam vs phone, single berries vs piles).
- **Shortcut check:**
  - Within each source, both classes have similar backgrounds, so the source alone can't give away the label.
  - Two shortcuts remain possible in Mendeley:
    - **"pile = rotten":** two thirds of rotten photos are piles, against one third of fresh.
    - **juice stains** under rotten berries.
  - Notebook 02 compares single-berry scores with all-image scores, and looks at Grad-CAM heatmaps for 12 test images. They should show the model looking at the berry.

## 7. Limits
- **The test set is small:** 161 images, and only 10 are single-berry rotten Mendeley photos (the leaky split had 18, but some had near-copies in train). A difference of a few points between models can be noise, so E1 uses 3 seeds.
- **Fewer images than "2,206" suggests:** after removing copies and near-duplicates, 1,123 photos remain, and many show the same berry. Say so in the pitch.
- **Leakage is reduced, not proven absent.** The hash can't recognise a berry that was turned over, and the series boundaries are a judgement from contact sheets. E2 is the leakage-free check.
- **Cautious grouping** makes some groups large (up to 87 images, one whole Mahendra shooting session; fresh Mendeley has only 9 groups). The split search still lands within 2 points of 70/15/15 in every stratum.
