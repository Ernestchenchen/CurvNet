# <h1 align="center"> LRANet: Towards Accurate and Efficient Scene Text Detection with Low-Rank Approximation Network </h1> 

<div align="center">
    <img src="figs/framework.pdf" width="80%">
</div>

## Introduction

This is the official implementation of Paper: [SG-LRA: Self-Generating Automatic Scoliosis Cobb Angle Measurement with Low-Rank Approximation]. The implementation is based on [LRANet: Towards Accurate and Efficient Scene Text Detection with Low-Rank Approximation Network](https://arxiv.org/abs/2306.15142.pdf) (AAAI 2024 Oral), which is built upon mmocr-0.2.1. SG-Lra is a network framework with self-generative capabilities for automatic measurement of spinal scoliosis Cobb angles.

## Function

Reproducing the code will give you the following functions: 
Function 1. Ask SG-LRA to generate an X-ray image for you (based on the distribution characteristics of our private dataset Spinal2023), and you can continue to complete Function 3, or generate an entire dataset and continue to complete Function 4.
Function 2. Provide a private spine X-ray dataset, train the generation module, and SG-LRA to generate X-ray images based on your own data features, continuing to complete Function 3 or Function 4.
Function 3. Provide a private or SG-LRA generated spine X-ray image, and SG-LRA will automatically detect spinal landmarks, Cobb angles, and visualize the results. 
Function 4. Provide your private or SG-LRA generated unlabeled spine X-ray dataset, and SG-LRA will use the Data Engine proposed in the paper to generate relatively usable data annotations for the dataset.
 
## Environment

Regardless of your specific needs, you will need to complete the following dependency installations. It is recommended to use a virtual [Anaconda](https://www.anaconda.com/) environment to manage your environment setup. Run the following commands to install dependencies.


```
conda create -n SG-LRA python=3.7 -y
conda activate SG-LRA
 conda install pytorch=1.8 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -c conda-forge
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install mmdet==2.14.0
git clone 
cd LSG-LRA
pip install -r requirements.txt
python setup.py build develop
```

## Dataset

You have the following options for your spinal X-ray dataset:

Prepare your own private spinal X-ray dataset.
Download our open-source [Spinal-AI2024](https://anonymous.4open.science/r/Spinal-AI2024) dataset.
Additionally, our code provides sample data that can be used for testing purposes.


Please download and extract the above datasets into the `data` folder following the file structure below.
```
data
├─Spinal-AI2024
│  │ Spinal-AI2024_train.json
│  └─train
│  │ Spinal-AI2024_test.json
│  └─test
├─Your_Custom_Dataset
│  │ Your_Custom_train.json
│  └─train
│  │ Your_Custom_test.json
│  └─test
├─sample_data
│  │ sample_data_train.json
│  └─train
```

## Function1
```
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/lranet/chenchen_spinal_det.py work_dirs/chenchen_spinal_1 1
CUDA_VISIBLE_DEVICES=0 ./tools/train.py configs/lranet/chenchen_spinal_det.py work_dirs/chenchen_spinal_1 1
```

## Function2
```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/lranet/chenchen_spinal_det.py work_dirs/chenchen_spinal/latest.pth --eval hmean-e2e
```

## Trained Model
Total-Text : [One Drive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvYy81YWE2OWZiZTU4NDY0MDYxL0VZdmxkOXBEWUFGSnM2SERNNWFscWFjQlRpejVtWG5WZmxoQ1JiUFlmX0x1SXc%5FZT1rY3RBa3k&cid=5AA69FBE58464061&id=5AA69FBE58464061%21sda77e58b60434901b3a1c33396a5a9a7&parId=root&o=OneUp)


## Acknowledgement
We sincerely thank [MMOCR](https://github.com/open-mmlab/mmocr) for their excellent works.