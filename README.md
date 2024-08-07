# Structure-Aware 3D VR Sketch to 3D Shape Retrieval

This repository contains the Pytorch implementation of the paper: Structure-Aware 3D VR Sketch to 3D Shape Retrieval, accepted by 3DV 2022. ([paper](https://arxiv.org/abs/2209.09043), [supplemental](https://drive.google.com/file/d/11rt_fVuqumWUy_jVMAis4di4KW0bRHJr/view?usp=sharing), [video](https://www.youtube.com/watch?v=osskcgV2lLk&list=PLDqmL95Gm8yIOzgysJgj2riAPNyZna61w&index=7&t=6s&ab_channel=3DV2022))

![retrieval](images/retrieval.png)

## Introduction

![teaser](images/teaser.png)

We study the practical task of fine-grained 3D-VR-sketch-based 3D shape retrieval. This task is of particular interest as 2D sketches were shown to be effective queries for 2D images.
However, due to the domain gap, it remains hard to achieve strong performance in 3D shape retrieval from 2D sketches. 
Recent work demonstrated the advantage of 3D VR sketching on this task. 
In our work, we focus on the challenge caused by inherent inaccuracies in 3D VR sketches.
We observe that retrieval results obtained with a triplet loss with a fixed margin value, commonly used for retrieval tasks, contain many irrelevant shapes and often just one or few with a similar __structure__ to the query.
To mitigate this problem, we for the first time draw a connection between adaptive margin values and shape similarities.

In particular, we propose to use a triplet loss with an adaptive margin value driven by a "fitting gap", which is the similarity of two shapes under structure-preserving deformations.
We also conduct a user study which confirms that this fitting gap is indeed a suitable criterion to evaluate the structural similarity of shapes. 

Furthermore, we introduce a dataset of 202 VR sketches for 202 3D shapes drawn from memory rather than from observation.

## Dataset

- [Fine-Grained VR Sketching: Dataset and Insights](https://cvssp.org/data/VRChairSketch/) for training, validation and testing. 
- [Sketch from memory dataset](https://drive.google.com/file/d/1lwofDAX-_z4rHebsmHJiYUd4Tui_jFIT/view?usp=sharing) (for testing only): 202 VR sketches for 202 3D shapes drawn from memory rather than from observation. 
- ShapeNetCore obj files: only needed for training deformer.

If you want to retrain the model using the original dataset, after downloading the `3dv_2021_vr_sketches_full.tar.gz` from the dataset website, please unzip the file to your path as follows:

```
— your_path
    — data
        — list # the splits for train/val/test
            — train.txt
            — val.txt
            — ...
        — pointcloud # contains all samples for retrieval.
            — shape
                — XXX.npy
                — ...
            — aligned_sketch
                — XXX.npy
                — ...
            — FVRS-M # Sketch from memory dataset, optional
                — XXX.npy
                — ...
    — mesh (original obj files from ShapeNetCore, not included in the 3DV dataset, only required to train deformer)
        — 03001627
            — XXX.obj
            — ...
```
 
The data factory is located in `src/datasets/SketchyVRLoader.py`. The `Shapes` class is the base class for all dataloader classes, setting the list file and data loading method. Specific paths can be customized in `configs/datamodule/SketchyVR_dataModule.yaml`: 
- The data path `data_dir` should be updated to `your_path/data/pointcloud`
- The sketch data path, `sketch_dir`, is set to `aligned_sketch` by default. 
- The `test_data` is the testing data dirname which can be chosen from:
    - `aligned_sketch`
    - `FVRS-M`: sketch from memory dataset.
- __Stage1 training deformer__ (optional): Requires ShapeNet mesh obj files. The `mesh_dir` should be updated to `your_path/mesh`. 
- __Multifold training__ (optional): In the paper, we use multifold training. Therefore, the training config file `configs/experiments/retrieval_multifold.yaml` uses a multifold variant of `SketchyVRDataModule`: `SketchyVRDataModule_multifold`.


## Environments
This project is based on Pytorch Lightning.

```
conda env create -f environment.yaml
conda activate structure
```

## Usage

### For easy inference

All pretrained models (deformer and retrieval) for chair(03001627) can be dowloaded [here](https://drive.google.com/file/d/18R59QDMkdsVBk40ojdaFYpfm_GvOqm6Q/view?usp=sharing). Please download and unzip the `logs.zip` as:

```
- images
- project
    - configs: Hydra configs
        - experiment: Train model with chosen experiment config
    - logs
        - experiments
            - deformer
            - retrieval
    - ...
```

Then run inference by:

```shell
python project/src/run.py inference=True
            experiment=retrieval // the config file to be loaded
            name=retrieval // the trained model to be loaded
            +test_ckpt=last // use the last checkpoint
```

If you want to train from scratch, please follow the intructions below:

### Step 1: Train Deformer

You can train deformer for any specif category like chair(03001627), lamp(03636649), and ariplane category(02691156) of ShapeNet by claiming `++datamodule.category==CATEGORY_ID`.

```shell
category = 03636649
name = template_$(category)
JobBatchName = $(config)_$(name)

python project/run.py name=$(JobBatchName) experiment=deformer_cage_sh2sh_shapenet 
++datamodule.category=$(category) 
resume_training=True

```

### Step 2: Compute Fitting Gap

```shell
category = 03001627

python project/val_sh2sh.py --category $(category)
```

### Step 3: Train

```shell
category = 03001627

python project/run.py 
++datamodule.category=$(category) 
resume_training=True

```

## Contact

If you have any questions about this project please feel free to open an issue or contact Ling Luo at `ling.rowling.luo@gmail.com`.

## Cite

If you find this work useful, please consider citing our work:

```
@inproceedings{luo2022structure,
  title={Structure-aware 3D VR sketch to 3D shape retrieval},
  author={Luo, Ling and Gryaditskaya, Yulia and Xiang, Tao and Song, Yi-Zhe},
  booktitle={2022 International Conference on 3D Vision (3DV)},
  pages={383--392},
  year={2022},
  organization={IEEE}
}
```
## Acknowledgement

Our project is built upon the following work:

- [KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control](https://github.com/tomasjakab/keypoint_deformer)