import pickle

import numpy as np
import pandas as pd

from collections import Counter
import random

import tensorflow.keras as keras

from utils import captions_2_dict,generate_dataset


CAPTION_TXT_FILE = "flickr8k/captions.txt"
IMAGES_DIR = "flick8k/images"

descriptions = captions_2_dict()

vocab = Counter()

for desc_list in descriptions.values():
    for desc in desc_list:
        vocab.update(desc.split())

print("Original Vocab Size : ", len(vocab))

threshold = 10
vocab = dict(list(filter(lambda x : x[1] >= threshold, vocab.items())))
print(f"Vocab Size with >= {threshold} is {len(vocab)}")


#Loading Image Encodings
read_file = open("img_encoded/img_inceptionv3_encoding.pkl", "rb")
img_encodings = pickle.load(read_file)
read_file.close()


print("Total Images : ", len(img_encodings))

# print(img_encodings[list(img_encodings.keys())[0]]) : Prints first img numpy array


#Caption Dataset
train_df,test_df = generate_dataset()

############# Data Specifications ########################
print("Unique Train Images :", len(set(train_df['image'])))
print("Unqiue Test Images : ", len(set(test_df['image'])))

print("Total Train Instances :", train_df.shape[0])
print("Total Test Instances : ", test_df.shape[0])
###########################################################

train_df['caption'] = train_df['caption'].apply(lambda text: "<SOS> " + text + " <EOS>")
test_df['caption'] = test_df['caption'].apply(lambda text: "<SOS> " + text + " <EOS>")

train_descriptions = train_df.groupby(['image']).apply(lambda sub_df : sub_df["caption"].to_list()).to_dict()
test_descriptions = test_df.groupby(['image']).apply(lambda sub_df : sub_df["caption"].to_list()).to_dict()


word_2_idx = {word : idx for idx,word in enumerate(vocab.keys(),1)}
idx_2_word = {word : idx for idx,word in word_2_idx.items()}


sent_length = train_df['caption'].apply(lambda x : len(x.split()))
max_sent_length =  max(sent_length)

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key]
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [np.array(X1), np.array(X2)], np.array(y)
                X1, X2, y = list(), list(), list()
                n=0


# Load Glove vectors
glove_path = 'glove_embeddings/glove.6B.200d.txt'
embeddings_index = {} # empty dictionary
f = open(glove_path, encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#Loading Embeddings
embedding_dim = 200
vocab_size = len(vocab) + 1
# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_2_idx.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

# image feature extractor model
inputs1 = keras.layers.Input(shape=(2048,))
fe1 = keras.layers.Dropout(0.5)(inputs1)
fe2 = keras.layers.Dense(256, activation='relu')(fe1)

# partial caption sequence model
inputs2 = keras.layers.Input(shape=(max_sent_length,))
se1 = keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = keras.layers.Dropout(0.5)(se1)
se3 = keras.layers.LSTM(256)(se2)

# decoder (feed forward) model
decoder1 = keras.layers.add([fe2, se3])
decoder2 = keras.layers.Dense(256, activation='relu')(decoder1)
outputs = keras.layers.Dense(vocab_size, activation='softmax')(decoder2)

# merge the two input models
model = keras.Model(inputs=[inputs1, inputs2], outputs=outputs)

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False
model.compile(loss='categorical_crossentropy', optimizer='adam')

epochs = 10
number_pics_per_batch = 3
steps = len(train_descriptions)//number_pics_per_batch



for i in range(epochs):
    generator = data_generator(train_descriptions, img_encodings, word_2_idx, max_sent_length, number_pics_per_batch)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('./model_weights/model_' + str(i) + '.h5')
    break