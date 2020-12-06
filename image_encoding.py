import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import numpy as np

import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image

import pickle

from tqdm import tqdm


def image_preprocessing(img_path, target_size=(299,299)):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.inception_v3.preprocess_input(img)
    return img


#Pretrained InceptionNet
inception_v3 = keras.applications.InceptionV3()
model = keras.Model(inputs=inception_v3.input, outputs=inception_v3.layers[-2].output)

encoded_img_dict = {}

img_files = glob.glob(os.path.join("flickr8k/images", "*"))
for img_file in tqdm(img_files):
    img = image_preprocessing(img_file)
    output = model(img)
    encoded = np.reshape(output, output.shape[1])
    encoded_img_dict[img_file.split("/")[2]] = encoded

pickled_files = open("img_encoded/img_inceptionv3_encoding.pkl", "wb")
pickle.dump(encoded_img_dict, pickled_files)
pickled_files.close()



