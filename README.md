# MCDet: Multi-content Collaboration Detector for Multiscale Remote Sensing Object
* This is a pytorch implementation of MCDet

## 1 Requirements：
* Python 3.6
* Pytorch 1.8
* pycocotools(Linux: ```pip install pycocotools```;  Windows: ```pip install pycocotools-windows```)
* opencv_python 4.3.0.36
* tensorboard 2.1.0

## 2 File structure：
```
  ├── cfg: configuration file directory
  │    ├── hyp.yaml: hyperparameters of the training network
  │    └── MCDet.cfg: MCDet network structure configuration  
  │ 
  ├── data: 
  │    └── pascal_voc_classes.json: pascal voc dataset tags
  │ 
  ├── build_utils: Tools used in building the training network
  │     ├── datasets.py: Data reading and pre-processing methods
  │     ├── img_utils.py:
  │     ├── layers.py:
  │     ├── parse_config.py: 
  │     ├── torch_utils.py:
  │     └── utils.py: 
  │
  ├── train_utils: 
  ├── weights: Storage of pre-training weights
  ├── model.py: Model building files
  ├── train.py: Train the model
  ├── calculate_dataset.py: 1)Count the data from the training and validation sets and generate the corresponding .txt files
  │                         2)Create data.data file
  └── predict_test.py: Prediction test using trained weights
```

## 3 Data format
* Data set in yolo format
* The labeled datasets should be arranged according to the following directory structure:
```
├── my_yolo_dataset 
│         ├── train  Training Set
│         │     ├── images  
│         │     └── labels  
│         └── val    Validation Set
│               ├── images  
│               └── labels      
```

## 4 Data Prepare：
* use ```calculate_dataset.py``` to generate ```my_train_data.txt```, ```my_val_data.txt```, and ```my_data.data```文件
* Before executing the script, you need to modify the following parameters according to your path
```python
# The path of the training set labels
train_annotation_dir = "/home/wz/my_project/my_yolo_dataset/train/labels"
# The path of the val set labels
val_annotation_dir = "/home/wz/my_project/my_yolo_dataset/val/labels"
# 
classes_label = "./data/my_data_label.names"
```

## 5 Pre-training weights download (download and place in weights folder):
* ```yolov3-spp-ultralytics-416.pt```: link: https://pan.baidu.com/s/1cK3USHKxDx-d5dONij52lA  code: r3vm
 

## 6 Train
* Run ```train.py```
* The following parameters can be adjusted
```python
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--cfg', type=str, default='cfg/MCDet.cfg', help="*.cfg path")
parser.add_argument('--img-size', type=int, default=416, help='test size')
```
* If you want to modify the setting of SCR and SSP in MCDet,
please adjust the followings in ```MCDet.cfg```.
```python
[Shallow_Clue_Refinement] # SCR module
kernel_size=3
stride=2
padding=1
in_channels=512
out_channels=512

[self_dilating_Pooling] # SSP module
in_channels=1024
out_channels=1024
```

## Reference
* This project is based on YOLOv3
* [WZMIAOMIAO/yolov3](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/yolov3_spp)