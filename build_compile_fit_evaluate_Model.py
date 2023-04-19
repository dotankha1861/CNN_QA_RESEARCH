import datetime
import numpy as np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
from keras.callbacks import ModelCheckpoint

# thiet lap chung
from preprocessing_data import load_data_temp
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json
import keras.preprocessing.text as kpt
print(datetime.datetime.now())
t1 = datetime.datetime.now()
from keras_preprocessing.sequence import pad_sequences
model_json_full = "D:\\NCKH_HTHD\\Model\\all1.json"

EMBEDDING_DIM = 300
filter_sizes = [3, 4, 5]
num_filters = 298
drop = 0.2
epoch = 100
batch_size = 16
train_len = 0
L2 = 0.0001
num_labels = 5
np.random.seed(0)

#load data
X_train, y_train, X_test, y_test, X_val, y_val,  embedding_layer = load_data_temp()

#build model
sequence_length = X_train.shape[1]
inputs = Input(shape=(sequence_length,))
embedding = embedding_layer(inputs)
reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_3 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_4 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)

maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)
maxpool_3 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_3)
maxpool_4 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_4)

merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4], axis=1)
flatten = Flatten()(merged_tensor)
reshape = Reshape((5*num_filters,))(flatten)
dropout = Dropout(drop)(flatten)
output = Dense(units = num_labels, activation='softmax',kernel_regularizer=regularizers.l2(L2))(dropout)
model = Model(inputs, output)
adam = Adam(learning_rate=1e-3)

model.summary()
from keras import metrics
#compile_model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy', metrics.Precision(), metrics.Recall()])
callbacks = [EarlyStopping(monitor='val_loss')]

# fit_model
checkpoint_filepath = 'D:\\NCKH_HTHD\\Model\\all1.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=False)  

model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1,
          validation_data=(X_val, y_val),
          callbacks=[model_checkpoint_callback])  # starts training

#save model
model_json = model.to_json()
with open(model_json_full, 'w') as json_file:
    json_file.write(model_json)

#evaluate model
scores = model.evaluate(X_test, y_test)
scores = model.evaluate(X_test, y_test)
print("Loss:", (scores[0]))
print("Accuracy:", (scores[1]*100))
print("Precision:", (scores[2]*100))
print("Recall:", (scores[3]*100))
