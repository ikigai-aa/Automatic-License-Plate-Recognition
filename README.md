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

- Many computer vision tasks have recently achieved a great increase in performance mainly due to the availability of large-scale annotated datasets (i.e., ImageNet) and the hardware (GPUs) capable of handling a large amount of data. In this scenario, Deep Learning (DL) techniques arise. However, despite the remarkable progress of DL approaches in
ALPR, there is still a great demand for ALPR datasets with vehicles and LPs annotations. The amount of training data is determinant for the performance of DL techniques.

- Higher amounts of data allow the use of more robust network architectures with more parameters and layers. Hence, we propose a larger benchmark dataset, called UFPR-ALPR, focused on different real-world scenarios.

- In this work, we propose a new robust real-time ALPR system based on the YOLO object detection Convolutional Neural Networks (CNNs). Since we are processing video frames, we also employ temporal redundancy such that we process each frame independently and then combine the results to create a more robust prediction for each vehicle.



## Technical Aspect

In order to have an efficient object detection (here, the vehicle number plates), I have used the most recently developed algorithm of YOLO series i.e. YOLOv4 backed up over a framework of Darknet.

Yolov4: Yolo abbreviates to You Only Look Once depicting its ability to detect objects and entities by using CNN (Convolutional Neural Network).Neural Network in YOLO uses weights trained by the user through annotated training data by using bounding boxes. Hence YOLO takes an image as input puts it through a Neural Network and gives the output in the image through bounding boxes.The input image is divided into SXS grid of cells.Each cell contributes to the object detection. Each cell predicts Bounding Boxes as well as Class probabilities. The prediction consists of 5 components (x,y,w,h,confidence).(x,y) represents the centre of the bounding box and (w,h) are the width and the height of the boxes.Confidence represents the Estimated Prediction Accuracy of the object.YOLO is extremely fast and accurate as compared to other algorithms and hence was our primary choice for this project.

![](data:image/png;base64,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)


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
