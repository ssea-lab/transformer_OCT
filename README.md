# OCT-MAE
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
