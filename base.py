import pickle

import numpy as np
from collections import Counter

from utils import captions_2_dict,generate_dataset
from data_gen import data_generator
from model import captionate_model


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


model = captionate_model(max_sent_length, vocab_size, embedding_dim, embedding_matrix)

epochs = 10
number_pics_per_batch = 3
steps = len(train_descriptions)//number_pics_per_batch



for i in range(epochs):
    generator = data_generator(train_descriptions, img_encodings, word_2_idx, max_sent_length, number_pics_per_batch, vocab_size)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('./model_weights/model_' + str(i) + '.h5')