# https://confusedcoders.com/data-science/deep-learning/how-to-build-deep-neural-network-for-custom-ner-with-keras
# https://appliedmachinelearning.blog/2019/04/01/training-deep-learning-based-named-entity-recognition-from-scratch-disease-extraction-hackathon/
#
# with elmo embedding (! not running in keras 2 !):
# https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede
# https://github.com/nxs5899/Named-Entity-Recognition_DeepLearning-keras/blob/master/nre.py

import json
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import LSTM, Input, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam


DATA_FILE_PATH = "../../data/ner_datasets/entity-annotated-corpus/ner_dataset.csv"
MODEL_DIR = "models/0_1"
NUM_TRAIN = 10000
NUM_TEST = 10
PERC_TEST = 0.3  # if train_test_split

# *****************************
# get tagged data
# https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus#ner_dataset.csv
# *****************************
def get_data():
    '''
    1)
        FROM:
        Sentence #      Word	        POS	    Tag
        Sentence: 1	    Thousands	    NNS	    O
                        of	            IN	    O
        TO:
            Sentence #           Word Tag
        0  Sentence: 1      Thousands   O
        1  Sentence: 1             of   O
    2)
        [
            [('Thousands', 'O'), ('of', 'O'), ... , ('that', 'O'), ('country', 'O'), ('.', 'O')]
        ]
    '''
    print("Load data...")
    data = pd.read_csv(DATA_FILE_PATH, encoding="latin1").drop(['POS'], axis=1).fillna(method="ffill")
    print("Data loaded")

    tags = list(set(data["Tag"].values))
    words = list(set(data["Word"].values))
    
    zipped_data = lambda s: [
        (w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())
    ]
    grouped_data = data.groupby("Sentence #").apply(zipped_data)
    sentences = [sentence for sentence in grouped_data]

    return tags, sentences, words

def prepare_data(tags, sentences, words):
    print("process incoming sequences ...")

    words.append("__EOS__")

    vocab_size = len(words)
    num_tags = len(tags)
    max_seq_len = max([len(s) for s in sentences])

    word2id = {w: i for i, w in enumerate(words)}
    tag2id = {t: i for i, t in enumerate(tags)}
    id2tag = {v: k for k, v in tag2id.items()}

    X, y = text_encoder(sentences, word2id, tag2id, max_seq_len, vocab_size, num_tags)

    # **********
    # save pseudo tokenizer for later use in prediction
    # **********
    with open(f"{MODEL_DIR}/word2id.json", "w") as f:
        f.write(json.dumps(word2id, indent=2, ensure_ascii=False))
    with open(f"{MODEL_DIR}/id2tag.json", "w") as f:
        f.write(json.dumps(id2tag, indent=2, ensure_ascii=False))

    return X, y, vocab_size, num_tags, max_seq_len


# *****************************
# 
# *****************************
def text_encoder(sentences, word2id, tag2id, max_seq_len, vocab_size, num_tags):
    print("encode X ...")
    X = [[word2id[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_seq_len, sequences=X, padding="post", value=word2id.get("__EOS__"))

    print("encode y ...")
    y = [[tag2id[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_seq_len, sequences=y, padding="post", value=int(tag2id["O"]))
    y = to_categorical(y, num_classes=num_tags)

    return X, y

def text_decoder(encoded_sequences, id2tag):
    '''
    decode one hot encoded labels
    '''
    one_hot_decoded = list()
    for encoded_sequence in encoded_sequences:  # each 'sentence'
        one_hot_decoded.append([np.argmax(vector) for vector in encoded_sequence])  # append list of indices to list
    decoded_sentences = []
    for seq in one_hot_decoded:
        decoded_sentences.append([id2tag[str(idx)] for idx in seq])
    
    return decoded_sentences

# *****************************
# build model
# *****************************
def build_model(hidden_units, vocab_size, max_seq_len, num_tags):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,output_dim=10,input_length=max_seq_len))  # embed words into vectors of 10 dimensions
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, recurrent_dropout=0.1)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_tags, activation="softmax")))

    return model

# *****************************
# start training
# *****************************
def train(model_name="test"):
    # *****************************
    # get data and process sequences and labels
    # *****************************
    tags, sentences, words = get_data()
    X, y, vocab_size, num_tags, max_seq_len = prepare_data(tags, sentences, words)

    # *****************************
    # Split train and test data -> get TRAIN data
    # *****************************
    X_train, y_train = (X[:NUM_TRAIN], y[:NUM_TRAIN])
    # X_train, X_test, _, _ = train_test_split(X, y, test_size=PERC_TEST, random_state=2018)

    # *****************************
    #
    # *****************************
    hidden_units = 128
    model = build_model(hidden_units, vocab_size, max_seq_len, num_tags)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    model.fit(
        np.array(X_train), 
        np.array(y_train), 
        batch_size=32, 
        epochs=20
    )

    model.save(f"{MODEL_DIR}/{model_name}.h5")

# *****************************
# predict and print samples
# *****************************
def predict(model_name="test"):
    # *****************************
    # print gt / prediction of sample TEST sentences
    # *****************************
    def print_sample(X_test, gt_tags, predictions, num_samples=1):
        for i, prediction in enumerate(predictions):
            if i == num_samples:
                break

            print(f"---------------- {i} ---------------")
            print("{:15} {:8} {}".format("Word", "GT", "Pred"))
            print(f"----------------------------------")
            for w, gt_tag, p_tag in zip(X_test[i], gt_tags[i], prediction):
                if words[w] == "__EOS__":
                    break
                print("{:15} {:8} {}".format(words[w], gt_tag, p_tag))
            print()

    # *****************************
    # get model, data and process sequences and labels (according to model shape)
    # *****************************
    model = load_model(f"{MODEL_DIR}/{model_name}.h5")
    model.summary()
    
    with open(f"{MODEL_DIR}/word2id.json") as f:
        word2id = json.load(f)
    with open(f"{MODEL_DIR}/id2tag.json") as f:
        id2tag = json.load(f)
    
    _, sentences, _ = get_data()
    
    max_seq_len = int(model.input.shape[1])

    words = list(word2id.keys())
    tag2id = {v: k for k, v in id2tag.items()}

    num_tags = len(id2tag.keys())
    vocab_size = len(words)

    # *****************************
    # encode (X: index / y: one hot)
    # *****************************
    X, y = text_encoder(sentences, word2id, tag2id, max_seq_len, vocab_size, num_tags)

    # *****************************
    # Split train and test data -> get TEST data
    # *****************************
    X_test, y_test = (X[NUM_TRAIN:NUM_TRAIN + NUM_TEST], y[NUM_TRAIN:NUM_TRAIN + NUM_TEST])
    # _, _, X_test, y_test = train_test_split(X, y, test_size=PERC_TEST, random_state=2018)

    gt_tags = text_decoder(y_test, id2tag)

    # *****************************
    # predict
    # *****************************
    predictions = model.predict(np.array(X_test))
    predictions = text_decoder(predictions, id2tag)

    # *****************************
    # eval
    # *****************************
    print_sample(X_test, gt_tags, predictions, num_samples=1)
    


# train(model_name="model")
predict(model_name="model")
