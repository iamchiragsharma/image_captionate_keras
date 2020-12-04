import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import pandas as pd
import string

from collections import Counter
import random

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image

def text_cleaning(text):
    punct_table = str.maketrans("", "", string.punctuation)
    text = text.lower()
    words = text.split()
    words = [word.translate(punct_table) for word in words]
    words = [word for word in words if len(word)>1]
    words = [word for word in words if word.isalpha()]

    return " ".join(words)


def captions_2_dict(file_path = "flickr8k/captions.txt", subset=None):
    df = pd.read_csv(file_path)
    df['image'] = df['image'].apply(lambda x : x[:-4])
    df['caption'] = df['caption'].apply(text_cleaning)
    df = df.groupby(['image']).apply(lambda sub_df : sub_df["caption"].to_list())

    if subset is None:
        return df.to_dict()

    if subset == 'train':
        df = df.iloc[:6000]
    else:
        df = df.iloc[6000:]
    
    return df.to_dict()

def generate_dataset(image_dir = "flickr8k/images", file_path = "flickr8k/captions.txt"):
    df = pd.read_csv(file_path)
    img_files = glob.glob(os.path.join(*[image_dir, "*"]))
    img_files = dict(zip(map(lambda x : x.split("/")[2], img_files), img_files))
    df['file_path'] = df['image'].map(img_files)

    # train_df = df.groupby(['image']).apply(lambda sub_df : sub_df.sort_values("file_path").iloc[:3,:]).reset_index(drop=True)
    # test_df = df.groupby(['image']).apply(lambda sub_df : sub_df.sort_values("file_path").iloc[3:,:]).reset_index(drop=True)
    
    train_df = df.iloc[:6000*5]
    test_df = df.iloc[6000*5:]
    return train_df,test_df



