# Lego motion detection and image classification

This repository contains two main folders of scripts. Firstly, the Computer_files folder contains scripts to create and train an image recognition CNN model for lego technic pieces using Tensorflow (we trained it on Google colab). Secondly, the RPI_files folder contains motion_detection_and_image_classification.py, that when run on RaspberryPi, will detect motion of lego pieces using picamera, extract its bounding box as an image, and pass it through a tensorflowlite model to output its most probable class. The relevant files are found in the respective folders.

The database in this repository contains a small sample size of images. The models in the models folder were trained with 6000 images spanning across 7 possible categories of lego technic. It achieved 93% testing accuracy from an 80/20 split between training and testing data. Graphs of the accuracy and loss across epochs are shown below. A confusion matrix was also plotted to visualize the performance of the classification algorithm. It depicts the count value of true versus false predictions across each category.

![Unknown-5](https://user-images.githubusercontent.com/91732309/190358182-58fa5671-263d-490b-8f54-616cb2daf764.png)

More images can be taken by editing the motion_detection_and_image_classification.py script.

The motion detection portion of the RaspberryPi script was adapted from pyimagesearch's project 'Building a Raspberry Pi security camera with OpenCV' and can be found at
https://pyimagesearch.com/2019/03/25/building-a-raspberry-pi-security-camera-with-opencv/

Depicted below are 2 examples of lego pieces being classified by the model, through the RaspberryPi's Picamera livestream. When motion counter exceeds 8 consequtive frames, bounding box is extracted and run through model, almost instantaneously outputting the class of lego. Image counter increases by 1 while motion counter is reset back to zero.


![](https://github.com/racketmaestro/Lego-motion-detection-and-image-recognition-ak-ag/blob/main/misc/lego_classification_1.gif)

![](https://github.com/racketmaestro/Lego-motion-detection-and-image-recognition-ak-ag/blob/main/misc/lego_classification_2.gif)
