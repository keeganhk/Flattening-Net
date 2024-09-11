# Flattening-Net: Deep Regular 2D Representation for 3D Point Cloud Analysis

This is the official implementation of **[[Flattening-Net](https://arxiv.org/pdf/2212.08892.pdf)] (TPAMI 2023)**, an unsupervised deep neural architecture to represent irregular 3D point clouds of arbitrary geometry and topology as a completely regular 2D point geometry image (PGI) structure, in which the coordinates of spatial points are captured in colors of image pixels. This code has been tested with Python 3.9, PyTorch 1.10.1, CUDA 11.1 and cuDNN 8.0.5 on Ubuntu 20.04.

<p align="center"> <img src="https://github.com/keeganhk/Flattening-Net/blob/master/imgs/toy_example.png" width="85%"> </p>

<p align="center"> <img src="https://github.com/keeganhk/Flattening-Net/blob/master/imgs/pgi_gallery.png" width="85%"> </p>


### Instruction
- The pre-processed datasets can be downloaded from [here](https://drive.google.com/drive/folders/1UYMwVram1uePkyINRaIPn7tX_FSpeeaC?usp=sharing), which should be put in the ```data``` folder.

- The scripts for training Flattening-Net and a demo script for PGI creation are provided in ```scripts/para_scripts```.

- The experiments of downstream task evaluation are conducted in ```scripts/task_evaluations```.

- The pre-trained model parameters (for Flattening-Net and different task networks) are stored in ```ckpt```.

### News
- We extended the regular representation paradigm to dynamic point cloud sequences in **[SPCV](https://github.com/ZENGYIMING-EAMON/SPCV) (TPAMI 2024)**.
- We further investigated real-sense surface parameterization via the unsupervised neural architecture of **[Flatten Anything Model (FAM)](https://github.com/keeganhk/FlattenAnything)**.

### Citation
If you find our work useful in your research, please consider citing:

	@article{zhang2023flattening,
	  title={Flattening-Net: Deep Regular 2D Representation for 3D Point Cloud Analysis},
	  author={Zhang, Qijian and Hou, Junhui and Qian, Yue and Zeng, Yiming and Zhang, Juyong and He, Ying},
	  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
	  year={2023}
	}

 
