# OCT-MAE
Cervical cancer seriously endangers the health of the female reproductive system and even risks women's life in severe cases. Optical coherence tomography (OCT) is a non-invasive, real-time, high-resolution imaging technology for cervical tissues. However, since the interpretation of cervical OCT images is a knowledge-intensive, time-consuming task, it is tough to acquire a large number of high-quality labeled images quickly, which is a big challenge for supervised learning. In this study, we introduce the vision Transformer (ViT) architecture, which has recently achieved impressive results in natural image analysis, into the classification task of cervical OCT images. Our work aims to develop a computer-aided diagnosis (CADx) approach based on a self-supervised ViT-based model to classify cervical OCT images effectively. We leverage masked autoencoders (MAE) to perform self-supervised pre-training on cervical OCT images, so the proposed classification model has a better transfer learning ability. In the fine-tuning process, the ViT-based classification model extracts multi-scale features from OCT images of different resolutions and fuses them with the cross-attention module.

This repository contains all our materials except the whole dataset of our paper. Note that the .gitkeep file is **only for** keeping the integrity of our directory structure. 

## Overview
The MAE pretrained weight for imageNet can be downloaded here: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth,
https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large_full.pth, and then put them in the folder mae_imagenet_weight.

## Module envirment
The `requirement.txt` file records all dependencies and versions our work needs.  Run `pip install -r requirements.txt`.  

## Runing
The train commands are in the train_command.txt,  The test commands are in the test_command.txt

## Citation
If you think that our method and code are useful for your work, please help cite the following paper.

Qingbin Wang, Kaiyi Chen, Wanrong Dou, and Yutao Ma. [Cross-Attention Based Multi-Resolution Feature Fusion Model for Self-Supervised Cervical OCT Image Classification](https://doi.org/10.1109/TCBB.2023.3246979). _IEEE/ACM Transactions on Computational Biology and Bioinformatics_, DOI: 10.1109/TCBB.2023.3246979, 2023.
