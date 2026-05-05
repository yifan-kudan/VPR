# VPR

## Setup

The environment requirement is provided in environment.yaml

## Retrieval

### Data Processing

The image data is located under `VPR/retrieval/data/`.

Currently provide a method to convert `.HEIC` photos into `.jpg`.

To make the labelling easier, the photos are collected according to the following rules: 
- For each photo, it has image,direction,light,weather,indoor and construction
- The camera directions sequence are: forward, up, down, left, right, left-up, right-up, left-down, right-down, and zoom. 10 directions intotal.
- To avoid complex labelling, for light, weather, indoor and construction, a single scenario will not contain multiple changes. For exmaple, this experiment only involves light changes.
- For the light change in one scenario, each change will contains 10 directions. For example, if one scenario include day and night, this scenario will have two set of photos that include 10 directions in day and night seperately.

The initial label example is as follows:

```csv
image,direction,light,weather,indoor,construction
3785-3794,forward;up;down;left;right;left-up;right-up;left-down;right-down;zoom,day,sun,indoor,none
3795-3804;3805-3814,forward;up;down;left;right;left-up;right-up;left-down;right-down;zoom,day;light,sun,indoor,none
```
Each line is one scenario, `3785-3794` is the image number range. If there's a change, for example, light change of day and night, the images are represented as `3795-3804;3805-3814`.

There's no place label at first, but will be labeled according to scenario by `label_process.py`, which reorganizes the initial label, put each photo and its attributes on a separate line. Example of the refined label:

```csv
image,direction,light,weather,indoor,construction,place
VPR/retrieval/data/images/converted_jpeg/IMG_3735.jpg,forward,day,sun,indoor,none,0
```
### ORB Matching

### Evaluation

The output will be saved into `VPR/retrieval/results/`

Currently, the image pair of false matches will be saved into the `false_matches`

## TODO:

- More detailed output of the evaluation
  - Confusion matrix
  - Accuracy, Precision, Recall, F1-Score, CPU usage, Memory usage, Time consuming
- Implement the SIFT + DBoW2
- Implement the SuperPoint + NetVlad