import os
import numpy as np
import pandas as pd 
import keras.preprocessing.text as kpt
from keras.models import model_from_json
from keras.utils.data_utils import pad_sequences
from keras.preprocessing.text import Tokenizer,tokenizer_from_json

max_len = 300
num_words = 50000
padding = 'post'
sheet = "dataset"
name_org_data = 'all'
filter_characters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\''

path_data = os.path.join("Data", name_org_data + ".xlsx")
path_model_dir = os.path.join("Model")

def get_path_model_json(labels):
    path_model_json = ""
    if len(labels) == 0:
        path_model_json = os.path.join(path_model_dir, name_org_data +".json")
    else:
        path_model_json = os.path.join(path_model_dir,"_".join(labels) + ".json")
    return path_model_json

def get_path_model_h5(labels):
    path_model_h5 = ""
    if len(labels) == 0:
        path_model_h5 = os.path.join(path_model_dir, name_org_data +".h5")
    else:
        path_model_h5 = os.path.join(path_model_dir,"_".join(labels) + ".h5")
    return path_model_h5

def load_model(labels):
    json_file = open(get_path_model_json(labels), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(get_path_model_h5(labels))
    return model

def load_data(labels):
    data = pd.read_excel(path_data, sheet).to_numpy()
    for i in range(len(labels)):
        data = [row for row in data if row[i+1] == labels[i]]
    return data

def shorten_str(str, tokenizer):
    words = kpt.text_to_word_sequence(str, filters = filter_characters, lower = True)
    shortened_str = ""
    for word in words:
        if word in tokenizer.word_index:
            shortened_str = shortened_str + word + " "
    return shortened_str

def get_data_and_list_labels(labels):
    data = load_data(labels)
    list_labels = []
    texts = []
    for row in data:
        texts.append(row[0]) 
        if row[len(labels)+1] not in list_labels:
            list_labels.append(row[len(labels)+1])
    return texts, list_labels

def process(str, labels):
    texts, list_labels = get_data_and_list_labels(labels)
    tokenizer = Tokenizer(num_words = num_words, filters = filter_characters, lower=True)
    tokenizer.fit_on_texts(texts)
    shortened_str = shorten_str(str, tokenizer)
    return shortened_str, tokenizer, list_labels

def predict(str, labels, tok_sam, list_labels):
    model = load_model(labels)
    text = tok_sam.texts_to_sequences(np.expand_dims(str, axis = 0)) 
    seq = pad_sequences(text, maxlen = max_len, padding = padding)
    pred = model.predict(seq)
    return list_labels[np.argmax(pred)]

def predict_final(str, num_class):
    labels = []
    for i in range(0,num_class):
        str, tokenizer, list_labels = process(str, labels)
        label = predict(str, labels, tokenizer, list_labels)
        labels.append(label) 
    return str, labels[num_class-1]

