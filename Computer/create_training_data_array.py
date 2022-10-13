# script to create training data npy file from the database of images
# the npy file can then be uploaded to google drive and read in the jupyter notebook
# can then create training_data for model training

import os
import cv2
import numpy as np

# initialize target image size for the training and testing data
img_height = 128
img_width = 128

categories = ["straight-liftarm", 'pins', 'bent-liftarm', 'gears-and-disc', 'special-connector', 'axles', 'axle-connectors-stoppers']

training_data = []
def get_category_images(list,path,label):
    #print("old:", str(len(training_data)))
    current = len(training_data)
    for i in range(len(list)):
        try:
            image = cv2.imread(os.path.join(path,list[i]),
                            cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128,128))
            training_data.append([image, label])
        except Exception:
            pass
    new = len(training_data)  
    print(new - current)


for cat in categories:
    cat_path = "RPI3_project/lego-test-data/database/" + cat
    cat_list = os.listdir(cat_path)
    cat_label = categories.index(cat)
    get_category_images(cat_list, cat_path, cat_label)
    
print(len(training_data))
td_array = np.array(training_data)
len(td_array)
np.save('td_array_7cat', td_array)
