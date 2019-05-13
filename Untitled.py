
# coding: utf-8

# In[22]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


data1 = pd.read_csv("pagination.csv")
data2 = pd.read_csv("pagination2.csv")
data3 = pd.read_csv("pagination3.csv")

frame = [data1,data2,data3]
result = pd.concat(frame)
result.head()
result = result.dropna(subset=['keluhan','tipe'])
result = result[result['keluhan'].map(len) <300]
result.shape


# In[9]:


result.reset_index()
result.head()


# In[10]:


pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


# In[11]:


feature = result['tipe'].value_counts()
feature


# In[ ]:


threshold = 10 
to_remove = feature[feature <= threshold].index
result.replace(to_remove, np.nan, inplace=True)
result.reset_index()


# In[13]:


result['l'] = result['keluhan'].apply(lambda x: len(str(x).split(' ')))
print("mean length of sentence: " + str(result.l.mean()))
print("max length of sentence: " + str(result.l.max()))
print("std dev length of sentence: " + str(result.l.std()))


# In[14]:


sequence_length = 143
max_features = 20000
tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(result['keluhan'].values)
X = tokenizer.texts_to_sequences(result['keluhan'].values)
X = pad_sequences(X, sequence_length)
X


# In[15]:


y = pd.get_dummies(result['tipe']).values
y


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print("test set size " + str(len(X_test)))


# In[17]:


embeddings_index = {}
f = open('glove_wiki_id_50.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[18]:


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[19]:


num_words = min(max_features, len(word_index)) + 1
print(num_words)

embedding_dim = 50

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector
        embedding_matrix[i] = np.random.randn(embedding_dim)


# In[14]:


model = Sequential()
model.add(Embedding(num_words,
                    embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=sequence_length,
                    trainable=True))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(32)))
model.add(Dropout(0.25))
model.add(Dense(units=293, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[22]:


# X_train.shape
batch_size = 128
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
history = model.fit(X_train, y_train, epochs=200, batch_size=batch_size, verbose=1, validation_split=0.1, callbacks=[checkpointer])


# In[23]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[21]:


model = Sequential()
model.add(Embedding(num_words,
                    embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=sequence_length,
                    trainable=True))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100))
model.add(Dense(293, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[24]:


history = model.fit(X_train, y_train, validation_split=0.4, epochs = 3)


# In[25]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[34]:


X_train.shape


# In[26]:


model = Sequential()
masuk = Input(shape=(sequence_length,), dtype='int32')
encoder = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=sequence_length, trainable=True)(masuk)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)
merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1)(merged)
output = Activation('sigmoid')(merged)
model = Model(inputs=[masuk], outputs=[output])
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()


# In[28]:


from keras.callbacks import ModelCheckpoint
filepath="CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit(X_train, y_train, batch_size=32, epochs=5,validation_data=(X_test, y_test), callbacks = [checkpoint])


# In[34]:


X_train[12]

