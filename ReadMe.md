# Solution of Team kxyang for AortaSeg24 Challenge
Our methods are built upon [nnU-Net V1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

## Team Details
### Groop Leader
    Zhiwei Wang; zwwang@hust.edu.cn
    Wuhan National Laboratory for Optoelectronics, Huazhong University of Science and Technology; 
### Team Members
    Kaixiang Yang; kxyang@hust.edu.cn
    Wuhan National Laboratory for Optoelectronics, Huazhong University of Science and Technology; 

## Environments and Weights
We provide docker version here, if you want to run in server, please set as follow. For more details like path setting, please refer to [nnU-Netv1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

```bash
download nnU-NetV1
cd nnUNet
pip install -e .
```
The weight should be put in .../Aorta_UNet_Baseline/resources/fold_0, the dir tree is like:

    Aorta_UNet_Baseline
    ├── resources
    │   └── fold_0
    │       ├── model_final_checkpoint.model
    │       └── model_final_checkpoint.model.pkl
    ...

### 1. docker
```bash
sh .../Aorta_UNet_Baseline/build.sh
sh .../Aorta_UNet_Baseline/save.sh
```

### 2. server
Change the `input` and `output` path in [inference.py](inference.py/#L22)
```bash
cd .../Aorta_UNet_Baseline
python inference.py
```