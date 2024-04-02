import pandas as pd
import numpy as np
import pickle
import string
import re

#Data Visualization
import matplotlib.pyplot as plt

#Text Color
from termcolor import colored

#Train Test Split
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

#Model Evaluation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from mlxtend.plotting import plot_confusion_matrix

#Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

# file_path = "/content/drive/MyDrive/kuliah/nlp/news-classification/detik_news.csv"
file_path = "detik_news.csv"
df = pd.read_csv(file_path)

def preprocess(text):
  # convert to lower case
  text = text.lower()

  # remove punctuation
  text = text.translate(str.maketrans("","",string.punctuation))

  # remove number
  text = re.sub(r"\d+", "", text)

  # remove whitespace
  text = re.sub('\s+',' ',text.strip())

  # remove single char
  text = re.sub(r"\b[a-zA-Z]\b", "", text)
  return text

# preprocessing dataset
df['title'] = df['title'].apply(lambda x:preprocess(x))
df.sample(10)

# representasi label menjadi angka
le = preprocessing.LabelEncoder()
le.fit(df.category)
df['label'] = le.transform(df.category)

X = df['title']
y = df['label']
df.sample(20)

# Data splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Menghitung jumlah data per kategori pada set training dan validasi
train_counts = y_train.value_counts().sort_index()
val_counts = y_val.value_counts().sort_index()

# Plotting data pelatihan
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.bar(train_counts.index, train_counts.values, color = ['blue', 'green', 'red', 'orange', 'purple', 'yellow', 'cyan'])
plt.title('Distribusi Data Pelatihan')
plt.xlabel('Kategori')
plt.ylabel('Jumlah Data')

# Plotting data validasi
plt.subplot(1, 2, 2)
plt.bar(val_counts.index, val_counts.values, color = ['blue', 'green', 'red', 'orange', 'purple', 'yellow', 'cyan'])
plt.title('Distribusi Data Validasi')
plt.xlabel('Kategori')
plt.ylabel('Jumlah Data')

plt.tight_layout()
plt.show()

maxlen = X_train.map(lambda x: len(x.split())).max()

vocab_size = 10000 # arbitrarily chosen
embed_size = 32 # arbitrarily chosen

# Create and Fit tokenizer
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

# Tokenize data
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)

# Pad data
X_train = pad_sequences(X_train, maxlen=maxlen)
X_val = pad_sequences(X_val, maxlen=maxlen)

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = Sequential()
model.add(Embedding(vocab_size, embed_size, input_length=maxlen))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(GlobalMaxPooling1D()) #Pooling Layer decreases sensitivity to features, thereby creating more generalised data for better test results.
model.add(Dense(1024))
model.add(Dropout(0.25)) #Dropout layer nullifies certain random input values to generate a more general dataset and prevent the problem of overfitting.
model.add(Dense(512))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax')) #softmax is used as the activation function for multi-class classification problems where class membership is required on more than two class labels.
model.summary()

callbacks = [
    EarlyStopping(     #EarlyStopping is used to stop at the epoch where val_accuracy does not improve significantly
        monitor='val_accuracy',
        min_delta=1e-4,
        patience=4,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='weights.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
]

#Compile and Fit Model
model.compile(loss='sparse_categorical_crossentropy', #Sparse Categorical Crossentropy Loss because data is not one-hot encoded
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    batch_size=8,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    callbacks=callbacks)

plt.figure(figsize=(9,7))
plt.title('Accuracy score')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.show()

plt.figure(figsize=(9,7))
plt.title('Loss value')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

model.load_weights('weights.h5')
model.save('model.hdf5')

labels = le.classes_

preds = [np.argmax(i) for i in model.predict(X_val)]
cm  = confusion_matrix(y_val, preds)
plt.figure()
plot_confusion_matrix(cm, figsize=(16,12), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(7), labels, fontsize=12)
plt.yticks(range(7), labels, fontsize=12)
plt.show()

print("Recall of the model is {:.3f}".format(recall_score(y_val, preds, average='micro')))
print("Precision of the model is {:f}".format(precision_score(y_val, preds, average='micro')))
print("Accuracy of the model is {:f}".format(accuracy_score(y_val, preds)))
print("F1 score of the modell is {:f}".format(f1_score(y_val, preds, average='micro')))