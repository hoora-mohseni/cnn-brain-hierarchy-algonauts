# Hierarchical similarity in CNNs and human vision

**Neuromatch Academy — Deep Learning 2025 · Group project**
Team *Hierarchical Visionaries* (Megapod Ontario, Pod Isis)

Do the layers of a convolutional neural network (CNN) line up with the stages of the human visual system? We extracted features from pretrained CNNs (AlexNet and ResNeXt) for short naturalistic videos, then trained encoding models (ridge regression and small multilayer perceptrons) to predict fMRI responses in visual regions of interest (ROIs) from those features.

## Key findings

- CNN features predict regional fMRI activity with considerable accuracy. AlexNet features alone were enough to predict mean regional BOLD activity.
- Earlier visual regions (V1–V3) are best predicted by earlier CNN layers, and higher-level regions by deeper layers. That is the expected hierarchical correspondence.
- The trend holds for both AlexNet and ResNeXt features, and for both linear (ridge) and non-linear (MLP) voxel-wise encoding models.
- How strongly each voxel follows this layer–region relationship varies a lot from voxel to voxel.

The full results are in [`slides/`](slides/).

## Repository contents

```
.
├── notebooks/
│   ├── 01_video_cnn_sandbox.ipynb          # Video loading, frame sampling, hemodynamic shift,
│   │                                       # and a draft AlexNet + LSTM video→voxel architecture
│   └── 02_roi_encoding_alexnet_mlp.ipynb   # Main pipeline: AlexNet feature extraction + PCA,
│                                           # MLP encoding of ROI-mean responses, layer × ROI
│                                           # R² heatmaps, leave-one-subject-out evaluation
├── features/
│   └── resnext_features_pca50/             # ResNeXt activations (conv1, layer1–4), PCA → 50 dims
│                                           # train_*.npy: (1000, 50) · test_*.npy: (102, 50), float32
├── slides/
│   └── Hierarchical_Visionaries_Algonauts_presentation.pptx
├── requirements.txt
└── LICENSE
```

## Data

The analyses use the [Algonauts Project 2021](http://algonauts.csail.mit.edu/2021/index.html) dataset: fMRI responses from 10 subjects watching 1,000+ three-second videos, with ROIs V1, V2, V3, V4, LOC, EBA, FFA, STS and PPA.

**The dataset is not included in this repository.** Its terms of use do not allow redistribution. To get it, request access through the Algonauts 2021 form linked in the [official development kit](https://github.com/Neural-Dynamics-of-Visual-Cognition-FUB/Algonauts2021_devkit). The notebooks then download and unpack it into the Colab workspace (`/content/participants_data_v2021`, `/content/AlgonautsVideos268_All_30fpsmax`).

The files in `features/` are low-dimensional PCA projections of CNN activations, not the original videos or fMRI data.

## How to run

The notebooks were written for **Google Colab** with a GPU runtime:

1. Open a notebook in Colab (*File → Upload notebook*, or open it from GitHub).
2. Set *Runtime → Change runtime type → GPU*.
3. Run the installation and data-download cells at the top.
4. Run the rest of the notebook in order.

To run locally, install the dependencies with `pip install -r requirements.txt` and change the `/content/...` paths to local folders.

## Team

Sophia G., Hoora M., Tanner M., Atharva P., Supraja V.
Teaching assistants: Arash A. (pod TA), Morteza M. (project TA)

## Acknowledgments

- Parts of `02_roi_encoding_alexnet_mlp.ipynb` (data loading, the AlexNet definition, feature extraction, PCA and submission helpers) are adapted from the [Algonauts 2021 development kit](https://github.com/Neural-Dynamics-of-Visual-Cognition-FUB/Algonauts2021_devkit).
- Training utilities such as `set_seed` come from the [Neuromatch Academy Deep Learning course](https://github.com/NeuromatchAcademy/course-content-dl) materials.

## Reference

Cichy, R. M., Dwivedi, K., Lahner, B., Lascelles, A., Iamshchinina, P., Graumann, M., Andonian, A., Murty, N. A. R., Kay, K., Roig, G., & Oliva, A. (2021). *The Algonauts Project 2021 Challenge: How the human brain makes sense of a world in motion.* arXiv:2104.13714.
