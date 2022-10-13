# to test tflite model on individual images
# run on your own computer as raspberry pi can't install tensorflow, and we need the img_to_array function

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="OGmodel.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_image = Image.open('lego-testing/testing/12image.jpg')
input_image = ImageOps.grayscale(input_image)
input_image = input_image.resize((28,28))

input_data = img_to_array(input_image)
input_data.resize(1,28,28,1)
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(np.argmax(output_data[0]))