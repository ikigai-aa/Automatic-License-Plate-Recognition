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
![](https://github.com/ikigai-aa/Automatic-License-Plate-Recognition/blob/master/images/demo.gif)


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

- In this work, we propose a new robust real-time ALPR system based on the YOLO object detection Convolutional Neural Networks (CNNs). Since we are processing video frames, we also employ temporal redundancy such that we process each frame independently and then combine the results to create a more robust prediction for each vehicle.



## Technical Aspect

In order to have an efficient object detection (here, the vehicle number plates), I have used the most recently developed algorithm of YOLO series i.e. YOLOv4 backed up over a framework of Darknet.

Yolov4: Yolo abbreviates to You Only Look Once depicting its ability to detect objects and entities by using CNN (Convolutional Neural Network).Neural Network in YOLO uses weights trained by the user through annotated training data by using bounding boxes. Hence YOLO takes an image as input puts it through a Neural Network and gives the output in the image through bounding boxes.The input image is divided into SXS grid of cells.Each cell contributes to the object detection. Each cell predicts Bounding Boxes as well as Class probabilities. The prediction consists of 5 components (x,y,w,h,confidence).(x,y) represents the centre of the bounding box and (w,h) are the width and the height of the boxes.Confidence represents the Estimated Prediction Accuracy of the object.YOLO is extremely fast and accurate as compared to other algorithms and hence was our primary choice for this project.

![](https://github.com/ikigai-aa/Automatic-License-Plate-Recognition/blob/master/images/architecture.png)


### Dataset Source:

As, this project needed a lot of images to perform transfer learning with the weights of pre-trained model of YOLOv4 architecture which is trained on large scale dataset called COCO dataset(having more than 80 classes for object detection), so I used the Google Open Image Dataset for my custom retraining with that model.For extracting images for both training and testing, I used an toolkit called OIDToolKit4 which extracts the images from the Open Images Dataset by Google and create annotations for each files as YOLO architecture takes the image annotations present in a text file as an input. A complete directory for that toolkit is present above in this repository.Kindly follow the readme.md file of it to get into the specifications.




### Important Python Scripts:




## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

```bash


```


## Accuracy/Loss Plot

![](https://github.com/ikigai-aa/Automatic-License-Plate-Recognition/blob/master/images/mAP.png)

...
## To Do



## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here]https://github.com/ikigai-aa/Automatic-License-Plate-Recognition/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/ikigai-aa/Automatic-License-Plate-Recognition/issues/new). Please include sample queries and their corresponding results.


## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://keras.io/img/logo.png" width=200>](https://keras.io/) 

[<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=170>](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)

[<img target="_blank" src="https://www.gstatic.com/devrel-devsite/prod/vbf66214f2f7feed2e5d8db155bab9ace53c57c494418a1473b23972413e0f3ac/tensorflow/images/lockup.svg" width=280>](https://www.gstatic.com/devrel-devsite/prod/vbf66214f2f7feed2e5d8db155bab9ace53c57c494418a1473b23972413e0f3ac/tensorflow/images/lockup.svg)

[<img target="_blank" src="http://image-net.org/index_files/logo.jpg" width=200>](http://image-net.org/index_files/logo.jpg) 

[<img target="_blank" src="https://jupyter.org/assets/nav_logo.svg" width=200>](https://jupyter.org/assets/nav_logo.svg) 

[<img target="_blank" src="https://pjreddie.com/media/image/yologo_2.png" width=200>](https://pjreddie.com/media/image/yologo_2.png)

[<img target="_blank" src="https://pjreddie.com/static/img/darknet.png" width=200>](https://pjreddie.com/static/img/darknet.png)

[<img target="_blank" src="https://opencv.org/wp-content/uploads/2020/07/cropped-OpenCV_logo_white_600x.png" width=200>](https://opencv.org/wp-content/uploads/2020/07/cropped-OpenCV_logo_white_600x.png)



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

