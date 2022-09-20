# Structure-Aware 3D VR Sketch to 3D Shape Retrieval

This repository contains the Pytorch implementation of the paper: Structure-Aware 3D VR Sketch to 3D Shape Retrieval, 3DV 2022. ([Arxiv link](https://arxiv.org/abs/2209.09043))


Code and dataset will be available soon!

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

- [Fine-Grained VR Sketching: Dataset and Insights](https://cvssp.org/data/VRChairSketch/)
- Sketch from memory dataset: 202 VR sketches for 202 3D shapes drawn from memory rather than from observation. This one will be available soon!

## Environments

## Usage

## Contact

If you have any questions about this project please feel free to open an issue or contact Ling Luo at ling.rowling.luo@gmail.com.

## Cite
If you find this work useful, please consider citing our work:

## Acknowledgement

Our project is built upon the following work:

- [KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control](https://github.com/tomasjakab/keypoint_deformer)