#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

import collections

import numpy as np

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from tensorflow.keras.layers import Embedding
from keras.optimizers import Adam 
from keras.losses import sparse_categorical_crossentropy
import sklearn
from sklearn.model_selection import train_test_split


# Bringing in the Texx

# In[3]:


en_path = 'small_vocab_en.txt'
fr_path = 'small_vocab_fr.txt'


# In[4]:


with open(en_path, 'r') as file:
    en_sentences = file.read()
    en_sentences = en_sentences.split('\n')
    en_sentences = [sentence.strip() for sentence in en_sentences if sentence.strip()]

with open(fr_path, 'r') as file:
    fr_sentences = file.read()
    fr_sentences = fr_sentences.split('\n')
    fr_sentences = [sentence.strip() for sentence in fr_sentences if sentence.strip()]

print("text uploaded")


# In[5]:


en_sentences


# In[6]:


print(len(en_sentences))


# In[7]:


fr_sentences


# In[8]:


print(len(fr_sentences))


# Tokenize Function

# In[9]:


def tokenize(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    return tokenizer.texts_to_sequences(data), tokenizer


# Padding Sentences Function

# In[10]:


def pad(data, length=None):
    return pad_sequences(data, maxlen = length, padding = 'post')


# Preprocessing Function

# In[11]:


def preprocess(lang1, lang2):
    preprocess_lang1, lang1_tk = tokenize(lang1)
    preprocess_lang2, lang2_tk = tokenize(lang2)

    preprocess_lang1 = pad(preprocess_lang1)
    preprocess_lang2 = pad(preprocess_lang2)

    preprocess_lang2 = preprocess_lang2.reshape(*preprocess_lang2.shape, 1)
    return preprocess_lang1, preprocess_lang2, lang1_tk, lang2_tk


# Preprocessing Data

# In[12]:


preproc_en_sentences, preproc_fr_sentences, en_tokenizer, fr_tokenizer = preprocess(en_sentences, fr_sentences)


# In[30]:


tokenizer_path = 'fr_tokenizer.npy'
np.save(tokenizer_path, fr_tokenizer.word_index)

tokenizer_path = 'en_tokenizer.npy'
np.save(tokenizer_path, en_tokenizer.word_index)


# In[13]:


print(en_sentences[:5], preproc_en_sentences[:5], len(preproc_en_sentences[0]))


# In[14]:


max_en_sequence_length = len(preproc_en_sentences[0])
max_fr_sequence_length = len(preproc_fr_sentences[0])

en_vocab_size = len(en_tokenizer.word_index)
fr_vocab_size = len(fr_tokenizer.word_index)


# In[15]:


print(f"Max en sentences length: {max_en_sequence_length}")
print(f"Max fr sentences length: {max_fr_sequence_length}")


# In[16]:


print(f"En vocab size: {en_vocab_size}")
print(f"Fr vocab size: {fr_vocab_size}")


# Function to Convert From Id's to Text

# In[17]:


def id_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return " ".join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


# Building the Network(Bidirectional RNN)

# In[18]:


def model(input_shape, output_sequence_length, en_vocab_size, fr_vocab_size):
    model = Sequential()
    
    model.add(Bidirectional(GRU(128, return_sequences = True), input_shape = input_shape[1:]))
    model.add(TimeDistributed(Dense(1024, activation = 'relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(fr_vocab_size, activation = 'softmax')))

    model.compile(loss = sparse_categorical_crossentropy, optimizer = Adam(learning_rate = 0.003), metrics = ['accuracy'])

    return model



# In[19]:


tmp_lang1 = pad(preproc_en_sentences, preproc_fr_sentences.shape[1])
tmp_lang1 = tmp_lang1.reshape((-1, preproc_fr_sentences.shape[-2]))
tmp_lang1 = np.expand_dims(tmp_lang1, axis=-1)
print(tmp_lang1.shape, preproc_fr_sentences.shape)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(tmp_lang1, preproc_fr_sentences, test_size=0.1, random_state=42)


# In[21]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[22]:


model = model(X_train.shape, y_train.shape[1], len(en_tokenizer.word_index) + 1, len(fr_tokenizer.word_index) + 1)


# In[23]:


model.fit(X_train, y_train, batch_size= 1024, epochs = 15, validation_split=0.2)
model.summary()


# In[24]:


model.save("translator")


# In[25]:


print(id_to_text(model.predict(tmp_lang1[:1])[0], fr_tokenizer))


# In[26]:


predictions = model.predict(X_test)


# In[27]:


evaluation = model.evaluate(X_test, y_test)


# In[28]:


print("Predictions:", predictions)


# In[29]:


print("Evaluation:", evaluation)

