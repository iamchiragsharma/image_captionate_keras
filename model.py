import tensorflow.keras as keras


def captionate_model(max_sent_length, vocab_size, embedding_dim, embedding_matrix):
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

    return model