# This program combines motion detection and object classification. It will ouput the most probable category of lego pieces
# after the picamera detects it in realtime.
# The motion detection portion of the script was adapted from pyimagesearch's project
# 'Building a Raspberry Pi security camera with OpenCV' and can be found at
# https://pyimagesearch.com/2019/03/25/building-a-raspberry-pi-security-camera-with-opencv/

# To run, open the terminal in RPI and navigate to folder containing the python script.
# Run python3 'path_to_script' --conf conf.json

# This script, when run, will activate the picamera to detect motion of objects (preferably against a white background)
# and enclose it in a green boundary box.
# If successive frames of motion is detected by the picamera, the boundary box will be extracted and image saved to a
# pre-specified folder in the RPI. The image contrast will be increased, and resized before being converted into an input tensor.
# The input tensor will be passed into the interpretor (a tensorflow lite model) which will output a probability vector.
# The vector index of the highest probability will be extracted to output the most likely class of the lego piece.

# This script can be modified to take the images required for the database. The motionCounter can be decreased to take more images.

from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import os

#imports and initialisations for image recognition
from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageOps
import numpy as np

# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path="lego_tflite_model/detect.tflite") # insert path to the tflite model
interpreter.allocate_tensors()
path = r'/home/nullspacepi/Desktop/opencv-test/lego-pieces' # create variable for path to where camera pictures will be saved to

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# define a function that will convert the image captured into an array
def img_to_array(img, data_format='channels_last', dtype='float32'):
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)

    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x

# define a function that will increase the contrast of the image by manipulating its array. This will increase the likelihood
# of its features to be detected by the image classification tensorflow model
def increase_contrast_more(s):
    minval = np.percentile(s, 2)
    maxval = np.percentile(s, 98)
    npImage = np.clip(s, minval, maxval)

    npImage = npImage.astype(int)

    min=np.min(npImage)        # result=144
    max=np.max(npImage)        # result=216

    # Make a LUT (Look-Up Table) to translate image values
    LUT=np.zeros(256,dtype=np.float32)
    LUT[min:max+1]=np.linspace(start=0,stop=255,num=(max-min)+1,endpoint=True,dtype=np.float32)
    s_clipped = LUT[npImage]
    return s_clipped

# Read the labels from the text file as a Python list.
def load_labels(path): 
    with open(path, 'r') as f:
        return [line.strip() for i, line in enumerate(f.readlines())]
    
# Read class labels and create a vector. 
labels = load_labels("lego_tflite_model/labelmap.txt")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
motionCounter = 0
image_number = 0

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    frame = f.array
    text = "No piece"

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        continue


    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
        cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        piece_image = frame[y:y+h,x:x+w]
        text = "Piece found"
        # cv2.imshow("Image", image)
        
    
    if text == "Piece found":
           # to save images of bounding boxes
        
        
        motionCounter += 1
        print("motionCounter= ", motionCounter)
        print("image_number= ", image_number)

#       # Save image if motion is detected for 8 or more successive frames
        if motionCounter >= 8:
            image_number +=1
            image_name = str(image_number)+"image.jpg"
            cv2.imwrite(os.path.join(path, image_name), piece_image)
            motionCounter = 0 #reset the motion counter
            
            # Open the image, resize it and increase its contrast
            input_image = Image.open('lego-pieces/'+ image_name)
            input_image = ImageOps.grayscale(input_image)
            input_image = input_image.resize((128,128))
            input_data = img_to_array(input_image)
            input_data = increase_contrast_more(input_data)
            input_data.resize(1,128,128,1)
            
            # Pass the np.array of the image through the tflite model. This will output a probablity vector
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Get the index of the highest value in the probability vector.
            # This index value will correspond to the labels vector created above (i.e index value 1 will mean the object is most likely labels[1])
            category_number = np.argmax(output_data[0])
            

            # Return the classification label of the image    
            classification_label = labels[category_number]                
            print("Image Label for " + image_name + " is :", classification_label)
            
            
            
    else:
        motionCounter = 0


        
# check to see if the frames should be displayed to screen
    if conf["show_video"]:
        # display the feed
        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
