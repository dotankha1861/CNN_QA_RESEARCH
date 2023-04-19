from predict_label import load_model, load_data, process
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.metrics as metrics

labels = ['chuong1']
model = load_model(labels)
data = load_data(labels)
str, tok, list_label = process("abcxyz",labels)

X_test = [x[0] for x in data]
y_test = [x[2] for x in data]
X_test = X_test[:100]
y_test = y_test[:100]
sequences_test = tok.texts_to_sequences(X_test)
X_test = pad_sequences(sequences_test, maxlen=300, padding='post')

dict_label = { key:value for key, value in zip(list_label, range(len(list_label)))}

y_test = to_categorical([dict_label[x] for x in y_test] ,num_classes=2)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy', metrics.Precision(), metrics.Recall()])
scores = model.evaluate(X_test, y_test)
print("Loss:", (scores[0]))
print("Accuracy:", (scores[1]*100))
print("Precision:", (scores[2]))
print("Recall:", (scores[3]*100))