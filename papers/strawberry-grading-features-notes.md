# Notes: Comparison of Features for Strawberry Grading Classification with Novel Dataset

**Authors:** Oka Mahendra, Hilman F. Pardede, Rika Sustika, R. Budiarianto Suryo Kusumo  
**Affiliation:** Research Center for Informatics, Indonesian Institute of Sciences, Bandung  
**PDF date:** 31 August 2018 (file metadata)

> These are notes written in new wording, not a transcription of the paper.

## At a glance

| Item | Detail |
|---|---|
| Task | Binary classification: good vs. damaged strawberries |
| Classifier | Support Vector Machine (SVM) |
| Features compared | RGB, HSV, RGB histogram, SIFT, SURF, ORB, HOG |
| Dataset | 732 webcam images (382 good, 350 damaged) |
| Validation | 5-fold cross-validation |
| Best feature | SURF, 90.73% accuracy |
| Dataset link | <https://github.com/okamahendra123/dataset_strawberry> (stated as GNU GPL) |

## Motivation

- Manual grading is slow and inconsistent, which motivates computer vision with machine learning.
- SVM was chosen because it performs well in many food-grading tasks, trains more cheaply than ANNs or CNNs, and usually gives small models suited to mobile use.
- SVM depends heavily on how separable its input features are, and features that work for one fruit may not work for another.
- The aim is to find the best of seven features for strawberries and to judge whether it is good enough.

## Features

| Feature | What it captures | Data per sample or keypoint |
|---|---|---|
| RGB | Raw colour values of each pixel | Per pixel, 3 channels |
| HSV | Hue, saturation and value of each pixel | Per pixel, 3 channels |
| RGB histogram | 3D colour distribution, 8 bins per channel | 512 per sample |
| SIFT | Scale-invariant keypoints found with Difference of Gaussians | 128 per keypoint |
| SURF | Blob-like keypoints based on the Hessian matrix and Haar responses | 64 per keypoint |
| ORB | FAST keypoints, Harris ranking and rotated BRIEF descriptors | 32 per keypoint |
| HOG | Histograms of gradient orientation (contour and texture) | 8100 per sample |

## Method

### Image capture
Photos were taken with a Logitech C170 webcam.

### Pre-processing
The goal is to remove the sepals, which say little about damage, and to reduce processing time.

1. Apply a Gaussian blur.
2. Convert BGR to HSV.
3. Filter for reddish pixels: H 0–30 and 150–180, S 50–255, V 50–255.
4. Remove noise by expanding and shrinking the white regions of the mask.
5. Find the largest contour.
6. Crop the original image to it.
7. Resize to 128×128, keeping the aspect ratio with white padding.

### Feature settings
- **SIFT, SURF, ORB:** 100 randomly chosen keypoints per image.
- **HOG:** OpenCV settings with 16×16 blocks, 8×8 cells, 8-pixel stride and 9 bins.

### Training
- **Kernels:** linear, RBF, and polynomial of degree 2, 3 and 4.
- **C values:** 0.1, 0.5 and 1.0.
- **Tools:** Python, scikit-learn and OpenCV 3.4.1.

## Results

Accuracy (%) by SVM setting. "Hist." is the RGB histogram (labelled "HIS" in the paper).

| Kernel, C | RGB | HSV | Hist. | SIFT | SURF | ORB | HOG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear, 1.0 | 63.96 | 69.70 | 52.00 | 67.16 | 90.73 | 83.45 | 59.31 |
| Linear, 0.5 | 63.96 | 69.70 | 52.50 | 67.16 | 90.73 | 83.45 | 59.31 |
| Linear, 0.1 | 63.96 | 69.70 | 50.00 | 67.16 | 90.73 | 83.45 | 59.31 |
| RBF, 1.0 | 58.06 | 64.22 | 50.00 | 66.78 | 90.73 | 83.45 | 52.19 |
| RBF, 0.5 | 52.32 | 56.96 | 50.00 | 66.78 | 90.73 | 83.45 | 52.19 |
| RBF, 0.1 | 52.19 | 52.19 | 50.00 | 66.78 | 90.73 | 83.45 | 52.19 |
| Poly 2, 1.0 | 53.14 | 62.30 | 59.50 | 70.65 | 90.73 | 83.45 | 52.19 |
| Poly 2, 0.5 | 52.19 | 52.59 | 59.50 | 70.65 | 90.73 | 83.45 | 52.19 |
| Poly 2, 0.1 | 52.19 | 52.19 | 59.50 | 70.65 | 90.73 | 83.45 | 52.19 |
| Poly 3, 1.0 | 53.00 | 62.71 | 60.00 | 67.15 | 90.73 | 83.45 | 52.19 |
| Poly 3, 0.5 | 52.19 | 52.46 | 60.00 | 67.15 | 90.73 | 83.45 | 52.19 |
| Poly 3, 0.1 | 52.19 | 52.18 | 60.00 | 67.15 | 90.73 | 83.45 | 52.19 |
| Poly 4, 1.0 | 52.32 | 60.38 | 61.50 | 65.38 | 90.73 | 83.45 | 52.19 |
| Poly 4, 0.5 | 52.19 | 52.19 | 61.50 | 65.38 | 90.73 | 83.45 | 52.19 |
| Poly 4, 0.1 | 52.19 | 52.19 | 61.50 | 65.38 | 90.73 | 83.45 | 52.19 |
| **Average** | **55.07** | **58.77** | **56.50** | **67.42** | **90.73** | **83.45** | **53.61** |

### Authors' interpretation
- **Ranking by average:** SURF, ORB, SIFT, HSV, RGB histogram, RGB, HOG.
- Colour alone struggles to separate damaged fruit from good fruit.
- HOG mostly responds to the fruit's edge rather than the damaged area.
- SURF does well because its blob detection picks up spots, such as decay or fungal growth.

### Effect of the minimum keypoint count (SURF)
Raising the minimum number of keypoints raises accuracy at the higher end, but fewer images qualify. Values below are **approximate readings from Figure 9**.

| Minimum keypoints | Valid samples (%) | Accuracy (%) |
|---:|---:|---:|
| 50 | ~95 | ~52 |
| 60 | ~85 | ~57 |
| 70 | ~62 | ~55 |
| 80 | ~36 | ~54 |
| 90 | ~18 | ~77 |
| 100 | ~9 | ~90 |
| 110 | ~4 | ~94 |

## Conclusions and future work
- SURF is the strongest of the seven features; ORB and SIFT follow, colour features are inadequate, and HOG ranks last on average.
- The main limitation is that keypoint counts vary between images, so images or information get dropped.
- The authors plan a blob detector that gives a descriptor of the same length for every image.

## Figures in the original (not included)

| Figure | Shows |
|---:|---|
| 1 | Good and damaged sample strawberries |
| 2 | SIFT keypoints |
| 3 | SURF keypoints |
| 4 | ORB keypoints |
| 5 | HOG visualisation |
| 6 | SVM maximum-margin hyperplane |
| 7 | Pre-processing flowchart |
| 8 | Image before and after pre-processing |
| 9 | SURF accuracy and valid samples vs. minimum keypoints |

## Points to check before citing
*These are observations added during note-taking, not claims from the paper.*

1. **Headline accuracy:** with 100 keypoints selected, Figure 9 suggests the 90.73% SURF result covers only about 9% of the images (roughly 66 of 732, an estimate).
2. **Chance-level scores:** 52.19% equals 382/732, the score from labelling every image "good".
3. **Identical rows:** SURF and ORB do not change at all across 15 kernel and C settings, which is worth verifying.
4. **Pixel feature length:** 128×128×3 is 49,152, not the 4192 printed.
5. **HOG formula:** 8100 is correct, but the block count should be (128/8 − 1) = 15 per axis, not 128/(8 − 1).
