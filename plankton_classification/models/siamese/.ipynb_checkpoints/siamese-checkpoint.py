import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from keras.models import Sequential  
from keras.layers import * 

DEFAULT_INPUT_SHAPE = (224, 224, 3)
DEFAULT_WEIGHTS = "imagenet"


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    # return the computed contrastive loss to the calling function
    return loss


def build_network():

  network = Sequential()
  network.add(Embedding(name="synopsis_embedd",input_dim =len(t.word_index)+1, 
                       output_dim=len(embeddings_index['no']),weights=[embedding_matrix], 
                       input_length=train_q1_seq.shape[1],trainable=False))
  network.add(LSTM(64,return_sequences=True, activation="relu"))
  network.add(Flatten())
  network.add(Dense(128, activation='relu',
                  kernel_regularizer=l2(1e-3),
                  kernel_initializer='he_uniform'))
  
  network.add(Dense(2, activation=None,
                  kernel_regularizer=l2(1e-3),
                  kernel_initializer='he_uniform'))
  
  #Force the encoding to live on the d-dimentional hypershpere
  network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))

  return network


def create_siamese_model(num_classes, input_shape=DEFAULT_INPUT_SHAPE):

    input_1 = Input(shape=(train_q1_seq.shape[1],))
    input_2 = Input(shape=(train_q2_seq.shape[1],))
    
    network = build_network()
    
    encoded_input_1 = network(input_1)
    encoded_input_2 = network(input_2)
    
    distance = Lambda(euclidean_distance)([encoded_input_1, encoded_input_2])
    
    # Connect the inputs with the outputs
    model = Model([input_1, input_2], distance)

    model.compile(loss=contrastive_loss, optimizer=Adam(0.001))

    return model

def create_pairs(x, digit_indices, num_classes):
  pairs = []
  labels = []
  
  n=min([len(digit_indices[d]) for d in range(num_classes)]) -1
  
  for d in range(num_classes):
    for i in range(n):
      z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
      pairs += [[x[z1], x[z2]]]
      inc = random.randrange(1, num_classes)
      dn = (d + inc) % num_classes
      z1, z2 = digit_indices[d][i], digit_indices[dn][i]
      pairs += [[x[z1], x[z2]]]
      labels += [1,0]
  return np.array(pairs), np.array(labels)
