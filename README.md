# CVAM-Pose: Conditional Variational Autoencoder for Multi-Object Monocular Pose Estimation <br> <span style="float: right"><sub><sup>BMVC 2024 (Oral Presentation)</sub></sup></span>
![pipeline_2](https://github.com/user-attachments/assets/5cbf3634-5c4c-461f-bb49-7259b09a9178)

Authors: [Jianyu Zhao](https://scholar.google.com/citations?user=b6qSMLwAAAAJ&hl=en), [Wei Quan](https://www.uclan.ac.uk/academics/wei-quan), [Bogdan J. Matuszewski](https://scholar.google.co.uk/citations?user=QlUO_oAAAAAJ&hl=en)

Paper link:

If you find this work useful to your research, please consider citing:
```
@inproceedings{zhao2024cvam,
  title={CVAM-Pose: Conditional Variational Autoencoder for Multi-Object Monocular Pose Estimation},
  author={Zhao, Jianyu and Quan, Wei and Matuszewski, Bogdan J},
  booktitle={The 35th British Machine Vision Conference, 25th-28th November 2024, Glasgow, UK},
}
```

This method is developed based on our single-object method [CVML-Pose](https://github.com/JZhao12/CVML-Pose), you may also consider citing:
```
@article{zhao2023cvml,
  title={CVML-Pose: Convolutional VAE Based Multi-Level Network for Object 3D Pose Estimation},
  author={Zhao, Jianyu and Sanderson, Edward and Matuszewski, Bogdan J},
  journal={IEEE Access},
  volume={11},
  pages={13830--13845},
  year={2023},
  publisher={IEEE}
}
```

## 1. Usage
### 1.1 Download data
Download and navigate to the CVAM-Pose repository, then download the following data from [BOP datasets](https://bop.felk.cvut.cz/datasets/).
```
cd CVAM-Pose
mkdir original\ data                   # Make the "original data" folder
cd original\ data

export SRC=https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main
wget $SRC/lmo/lmo_base.zip             # Linemod-Occluded base archive
wget $SRC/lmo/lmo_models.zip           # Linemod-Occluded 3D models
wget $SRC/lm/lm_train_pbr.zip          # Linemod PBR images
wget $SRC/lmo/lmo_test_bop19.zip       # BOP Linemod-Occluded test images
```

Unzip the datasets.
```
unzip lmo_base.zip                     # Contains folder "lmo"
unzip lmo_models.zip -d lmo            # Unpacks to "lmo"
unzip lm_train_pbr.zip -d lmo          # Unpacks to "lmo"
unzip lmo_test_bop19.zip -d lmo        # Unpacks to "lmo"
```

We use the default detection results of [BOP Challenge 2022](https://bop.felk.cvut.cz/challenges/bop-challenge-2022/). The results are obtained from the Mask-R-CNN detector pretrained by [CosyPose](https://github.com/ylabbe/cosypose). You can download and unzip them through:
```
wget https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop22_default_detections_and_segmentations.zip
unzip bop22_default_detections_and_segmentations.zip
```

### 1.2 Conda environment
To set up the environment, run:
```
conda create -n CVAM-Pose python=3.9.7
conda activate CVAM-Pose

pip install matplotlib pandas scikit-learn pyrender pypng opencv-python torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

### 1.3 Data preprocessing
Images of objects are preprocessed using the crop-and-resize strategy as described in paper supplementary materials. A detailed illustration can also be found in [CVML-Pose](https://ieeexplore.ieee.org/document/10040668).
```
cd CVAM-Pose
conda activate CVAM-Pose

python scripts_data/pbr.py                       # Extract pbr training images
python scripts_data/recon.py --object 1          # Generate ground truth reconstruction images
python scripts_data/detection_bop.py             # Extract test images with BOP detection bounding box
```

### 1.4 Model training
```
cd CVAM-Pose
conda activate CVAM-Pose

python scripts_method/cvae.py                     # Train the CVAE model
python scripts_method/latent.py                   # Process latent representations
python scripts_method/mlp_r.py                    # Train MLP for rotation
python scripts_method/mlp_c.py                    # Train MLP for centre
python scripts_method/mlp_tz.py                   # Train MLP for distance
python scripts_method/cal_t.py                    # Calculate translation
```

### 1.5 Evaluation
For evaluation, we use the BOP metrics (VSD, MSSD, and MSPD) as implemented in the [BOP Toolkit](https://github.com/thodan/bop_toolkit).
```
cd CVAM-Pose
git clone https://github.com/thodan/bop_toolkit.git
```

To use the BOP Toolkit, change the code ```sys.path.append('/home/jianyu/CVAM-Pose/bop_toolkit/')``` in the script ```scripts_method/eva.py``` to your BOP Toolkit path, then evaluate the estimated poses:
```
python scripts_method/eva.py
```

## 2. License
This repository is released under the Apache 2.0 license as described in the [LICENSE](https://github.com/JZhao12/CVAM-Pose/blob/main/LICENSE).

## 3. Commercial use
We allow commercial use of this work, as permitted by the [LICENSE](https://github.com/JZhao12/CVAM-Pose/blob/main/LICENSE). However, where possible, please inform us of this use for the facilitation of our impact case studies.

## 4. Acknowledgements
This work makes use of existing datasets which are openly available at: https://bop.felk.cvut.cz/datasets/

This work makes use of multiple existing code which are openly available at:
+ [BOP Toolkit](https://github.com/thodan/bop_toolkit)
+ [CosyPose](https://github.com/ylabbe/cosypose)
+ [Pyrender](https://github.com/mmatl/pyrender)
+ [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
+ [PyTorch VAE](https://github.com/AntixK/PyTorch-VAE)
+ [Dive into Deep Learning](https://d2l.ai/index.html)

## 5. Additional information

[Student Profile](https://www.uclan.ac.uk/articles/research/jianyu-zhao)

[UCLan Computer Vision and Machine Learning (CVML) Group](https://www.uclan.ac.uk/research/activity/cvml)

Contact: jzhao12@uclan.ac.uk
