# RXP-YOLO
## Quick Links
+ [Overview](https://github.com/hys-pp/RXP-YOLO/edit/main/README.md#overview)  

+ [Requirements](https://github.com/hys-pp/RXP-YOLO/edit/main/README.md#requirements)  

+ [Code Usage](https://github.com/hys-pp/RXP-YOLO/edit/main/README.md#code-usage)
## Overview
In this paper, we propose a new model called RXP-YOLO, which can achieve drone target detection more efficiently and lightweight.
![RXP-YOLO](https://github.com/hys-pp/RXP-YOLO/blob/main/data/RXP-YOLO4.jpg)

## Requirements
This project uses PyTorch for its implementation. Please ensure your system includes the following package versions:  

+ Ubuntu 22.04
  
+ Python 3.12
  
+ PyTorch 2.3.1
  
+ CUDA 12.1
  
Additional required packages can be installed via pip:

```
pip install -r requirements.txt
```

```
pip install timm
```
## Code Usage

### Training data set
```
python train.py
```
### Validation of the dataset
```
yolo detect val data=data\data.yaml model=weights\best.pt batch=16 imgsz=640 split=val device=0 workers=8
```
### Test data set
```
yolo detect val data=data\data.yaml model=weights\best.pt batch=16 imgsz=640 split=test device=0 workers=8
```
### Pruning (the best weight file obtained through training is used as a pre-trained model for pruning operations)
```
python prune_ok.py
```
### Customizing Hyperparameters
Hyperparameter settings are flexible and can be adjusted within either `train.py` or `ultralytics\cfg\default.yaml`.
