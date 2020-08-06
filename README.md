# Automatic Number Plate Recognition :india:

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Directory Tree](#directory-tree)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)
  * [Resources](#resources)
  
  
## Demo
Link: [https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/demo%20video.mp4](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/demo%20video.mp4)


## Overview
This is a four phased Object Detection project mainly focussing on detecting Vehicle's license plates and thereby reading the license number and saving them in a text file for use by the concerned authority.This Deep Learning Project uses YOLOv4(You Only Look Once) as its Neural Network Architecture which is made above a framework called Darknet.It is then made deployment ready with Tensorflow Lite making it compatible for using it in various edge devices such as android, iOS, raspberry Pi etc. 


## Motivation

- Automatic License Plate Recognition (ALPR) has been a frequent topic of research due to many practical applications, such as automatic toll collection, traffic law
enforcement, private spaces access control and road traffic monitoring.

- ALPR systems typically have three stages: License Plate (LP) detection, character segmentation and character recognition. The earlier stages require higher accuracy or almost
perfection, since failing to detect the LP would probably lead to a failure in the next stages either. Many approaches search first for the vehicle and then its LP in order to reduce processing time and eliminate false positives.

- Although ALPR has been frequently addressed in the literature, many studies and solutions are still not robust enough on real-world scenarios. These solutions commonly depend on certain constraints, such as specific cameras or viewing angles, simple backgrounds, good lighting conditions, search in a fixed region, and certain types of vehicles (they would not detect LPs from vehicles such as motorcycles, trucks or buses).

- Many computer vision tasks have recently achieved a great increase in performance mainly due to the availability of large-scale annotated datasets (i.e., ImageNet [4]) and the hardware (GPUs) capable of handling a large amount of data. In this scenario, Deep Learning (DL) techniques arise. However, despite the remarkable progress of DL approaches in
ALPR, there is still a great demand for ALPR datasets with vehicles and LPs annotations. The amount of training data is determinant for the performance of DL techniques.

- Higher amounts of data allow the use of more robust network architectures with more parameters and layers. Hence, we propose a larger benchmark dataset, called UFPR-ALPR, focused on different real-world scenarios.

- As great advances in object detection were achieved through YOLO-inspired models, we decided to fine-tune i for ALPR. YOLO is a state-of-the-art real-time object detection that uses a model with 19 convolutional layers and 5 maxpooling layers. On the other hand, Fast-YOLO is a model focused on a speed/accuracy trade-off that uses fewer convolutional layers (9 instead of 19) and fewer filters in those layers. Therefore, Fast-YOLO is much faster but less accurate
than YOLOv2.

- In this work, we propose a new robust real-time ALPR system based on the YOLO object detection Convolutional Neural Networks (CNNs). Since we are processing video frames, we also employ temporal redundancy such that we process each frame independently and then combine the results to create a more robust prediction for each vehicle.

![](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/images/WHO.png)


Link: [https://twitter.com/i/status/1268986094042992640](https://twitter.com/i/status/1268986094042992640)



## Technical Aspect

In order to have an efficient object detection (here, the vehicle number plates), I have used the most accurate

![](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/images/face_mask_detection_phases.png)\


### Dataset Resource:

Link: https://drive.google.com/drive/folders/1FHPJRCab-cyLq8IVz83LkU71gOc7gTS8?usp=sharing

![](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/images/face_mask_detection_dataset.jpg)


### Project structure

```
├── dataset
│   ├── with_mask [690 entries]
│   └── without_mask [686 entries]
├── examples
│   ├── example_01.png
│   ├── example_02.png
│   └── example_03.png
├── face_detector
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── detect_mask_image.py
├── detect_mask_video.py
├── mask_detector.model
├── evaluation.png
└── Data Augmentation and Model Training.ipynb
├── requirements.txt
└── mask-detector-model.model
```


### Important Python Scripts:

1. Data Augmentation and Preprocessing.ipynb: In this notebook Accepts our dataset is taken as input and fine-tuning is donw with MobileNetV2 DNN architecture upon it to create our mask-detector-model.model. A training history evaluation.png containing accuracy/loss curves is also produced for better visualization of Model Evaluation through a plot.Some important processes which we performed here:

a. Data augmentation
b. Loading the MobilNetV2 classifier (we will fine-tune this model with pre-trained ImageNet weights)
c. Building a new fully-connected (FC) head
d. Pre-processing
e. Loading image data

Libraries Significance:

scikit-learn: for binarizing class labels, segmenting our dataset, and printing a classification report.
imutils: To find and list images in our dataset. 
matplotlib: To plot our training curves.

2. detect_mask_from_webcam.py: Using your webcam, this script applies face mask detection to every frame in the stream using webcom to read the real-time video.

Some command line arguments in this script include:
```
--image: The path to the input image containing faces for inference
--face: The path to the face detector model directory (we need to localize faces prior to classifying them)
--model: The path to the face mask detector model that we trained earlier in this tutorial
--confidence: An optional probability threshold can be set to override 50% to filter weak face detections
```


## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

```bash
## Run
> STEP 1
After unzipping the forked zip file of this project into your local machine, type the follwing command from the directory where you saved the project files in the command prompt: 
pip install -r requirements.txt

This will install thw following libraries:

absl-py==0.9.0
astunparse==1.6.3
cachetools==4.1.1
certifi==2020.6.20
chardet==3.0.4
cycler==0.10.0
gast==0.3.3
google-auth==1.19.2
google-auth-oauthlib==0.4.1
google-pasta==0.2.0
grpcio==1.30.0
h5py==2.10.0
idna==2.10
importlib-metadata==1.7.0
imutils==0.5.3
joblib==0.16.0
Keras-Preprocessing==1.1.2
kiwisolver==1.2.0
Markdown==3.2.2
matplotlib==3.3.0
numpy==1.19.1
oauthlib==3.1.0
opencv-python==4.3.0.36
opt-einsum==3.3.0
Pillow==7.2.0
protobuf==3.12.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
pyparsing==2.4.7
python-dateutil==2.8.1
requests==2.24.0
requests-oauthlib==1.3.0
rsa==4.6
scikit-learn==0.23.1
scipy==1.4.1
six==1.15.0
sklearn==0.0
tensorboard==2.2.2
tensorboard-plugin-wit==1.7.0
tensorflow==2.2.0
tensorflow-estimator==2.2.0
termcolor==1.1.0
threadpoolctl==2.1.0
urllib3==1.25.10


> STEP 2
Open Jupyter Notebook and run Data Augmentation and Preprocessing.ipynb in order to train your custom dataset within your loacl machine and preprocess the images meanwhile.

> STEP 3
Run detect_mask_from_webcam.py from the same directory of your project folder in the command prompt in order to test the detector in real- time using the webcam.
```

## Results/Classification Report

```
              precision    recall  f1-score   support

   with_mask       0.97      1.00      0.99       138
without_mask       1.00      0.97      0.99       138

    accuracy                           0.99       276
   macro avg       0.99      0.99      0.99       276
weighted avg       0.99      0.99      0.99       276

```

## Accuracy/Loss Plot

![](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/evaluation.png)\

...
## To Do
1. This approach reduces our computer vision pipeline to a single step — rather than applying face detection and then our face mask detector model, all we need to do is apply the object detector to give us bounding boxes for people both with_mask and without_mask in a single forward pass of the network.

2. An integration of this project to a web app/android app.


## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/issues/new). Please include sample queries and their corresponding results.


## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://keras.io/img/logo.png" width=200>](https://keras.io/) 

[<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=170>](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)

[<img target="_blank" src="https://www.gstatic.com/devrel-devsite/prod/vbf66214f2f7feed2e5d8db155bab9ace53c57c494418a1473b23972413e0f3ac/tensorflow/images/lockup.svg" width=280>](https://www.gstatic.com/devrel-devsite/prod/vbf66214f2f7feed2e5d8db155bab9ace53c57c494418a1473b23972413e0f3ac/tensorflow/images/lockup.svg)

[<img target="_blank" src="http://image-net.org/index_files/logo.jpg" width=200>](http://image-net.org/index_files/logo.jpg) 

[<img target="_blank" src="https://jupyter.org/assets/nav_logo.svg" width=200>](https://jupyter.org/assets/nav_logo.svg) 


## Team
Ashish Agarwal

LinkedIn Profile: [https://www.linkedin.com/in/ashish-agarwal-502203113/](https://www.linkedin.com/in/ashish-agarwal-502203113/)


## License
[![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)](http://www.apache.org/licenses/LICENSE-2.0e)

Copyright 2020 Ashish Agarwal

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Resources

1. https://www.who.int/publications/i/item/advice-on-the-use-of-masks-in-the-community-during-home-care-and-in-healthcare-settings-in-the-context-of-the-novel-coronavirus-(2019-ncov)-outbreak
2. https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
3. http://www.image-net.org/
4. https://arxiv.org/abs/1801.04381
