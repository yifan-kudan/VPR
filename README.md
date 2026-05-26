# Visual Place Recognition(VPR)

## Setup

The environment requirement environment.yaml is created by conda. To setup the environment, run

```bash
conda env create -f environment.yaml
conda activate img_matching
```

This project also uses Hierarchical-Localization (`hloc`) for SIFT,
SuperPoint, SuperGlue, and NetVLAD components. Install it separately after
activating the environment. In `external_repositores`:

```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git
cd Hierarchical-Localization
pip install -e .
```

## Feature Extractor

ORB, SIFT, SuperPoint are selected to extract features and descriptors

### ORB

ORB matching method is realized based on opencv

### SIFT

CPU version is based on the DoG extractor repository<a href="https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/extractors/dog.py">Hierarchical-Localization</a>

GPU version is depends on Kornia.

### SuperPoint

It is based on the repository<a href="https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/extractors/superpoint.py">Hierarchical-Localization</a> The SuperPoint uses a pre-trained model from <a herf="https://github.com/magicleap/SuperGluePretrainedNetwork/tree/ddcf11f42e7e0732a0c4607648f9448ea8d73590">SuperGluePretrainedNetwork</a>

## Retrieval

## Data Processing

The image data is located under `VPR/image_matching/data/`.

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
VPR/image_matching/data/images/converted_jpeg/IMG_3735.jpg,forward,day,sun,indoor,none,0
```

### NetVLAD

Based on the repository<a href="https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/extractors/superpoint.py">Hierarchical-Localization</a> It is pretrained.

### DBoW2

DBoW2 is realised by combining C++ library. The based repository is located at `VPR/external_repositories/DBoW2`. Clone DBoW2 repository and place it under `VPR/external_repositories/`

```bash
cd external_repositories
git clone https://github.com/dorian3d/DBoW2.git
```

`VPR/image_matching/bindings/dbow2_pybind.cpp` is the binding to allow pyhton call the C++ functions.

## Feature Matching

KNN based brut fore method to match features between top K predicted images, then use RANSAC to verify the features, to select the best matching.

## Evaluation

The output will be saved into `VPR/image_matching/results/`

Currently, the image pair of false matches will be saved into the `false_matches`

And a jupyter notebook is provided for more convenient result analysis.

