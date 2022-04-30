# -*- coding: utf-8 -*-
"""bo_cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SRMV3wIDjn-yGXPRHKWPVdbQE3OW3HUO
"""

from bayes_opt import BayesianOptimization
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, callbacks

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
import matplotlib.pyplot as plt


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier

import time


import warnings

warnings.filterwarnings('ignore')
# Make scorer accuracy
score_acc = make_scorer(accuracy_score)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Set hyperparameter space
hyperspace_cnn = {
    'conv1_act': (0, 7),
    'conv2_act': (0, 7),
    'conv3_act': (0, 7),
    'conv1_ker': (2, 10),
    'conv2_ker': (2, 10),
    'conv3_ker': (2, 10),
    'conv1_filter': (20, 32),
    'conv2_filter': (20, 32),
    'conv3_filter': (50, 100),
    'pool1_ker': (2, 3.9),
    'pool2_ker': (2, 3.9),
    'pool3_ker': (2, 3.9),
    'dense1_neuron': (32, 64),
    'dense2_neuron': (32, 64),
    'dense1_act': (0, 7),
    'dense2_act': (0, 7),
    'dropout_rate':(0, 0.5),
    'learning_rate':(0.0005, 0.002),
    'beta_1':(0.9,1),
    'beta_2':(0.95,1),
 }

def bo_tune_cnn(conv1_act,conv2_act,conv3_act,conv1_ker,conv2_ker,conv3_ker,conv1_filter,conv2_filter,conv3_filter,
                pool1_ker,pool2_ker,pool3_ker,dense1_act,dense2_act,dense1_neuron,dense2_neuron,dropout_rate,learning_rate,beta_1,beta_2):

    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu','elu', 'exponential']
    
    conv1_act = activationL[int(conv1_act)]
    conv2_act = activationL[int(conv2_act)]
    conv3_act = activationL[int(conv3_act)]
    dense1_act = activationL[int(dense1_act)]
    dense2_act = activationL[int(dense2_act)]

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=beta_1,beta_2=beta_2)
    
    

    model = models.Sequential()
    model.add(layers.Conv2D(int(conv1_filter), int(conv1_ker), activation=conv1_act, padding="same", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D(int(pool1_ker)))
    model.add(layers.Conv2D(int(conv2_filter), int(conv2_ker), activation=conv2_act, padding="same"))
    model.add(layers.MaxPooling2D(int(pool2_ker)))
    model.add(layers.Conv2D(int(conv3_filter), int(conv3_ker), activation=conv3_act, padding="same"))
    model.add(layers.MaxPooling2D(int(pool3_ker)))
    model.add(layers.Flatten())
    model.add(layers.Dense(int(dense1_neuron), activation=dense1_act))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(int(dense2_neuron), activation=dense2_act))
    model.add(layers.Dense(10))
    
    model.compile(optimizer = opt,
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    
    es = callbacks.EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=5)


    history = model.fit(train_images, train_labels, epochs = 3, validation_split = 0.1, callbacks=[es],verbose = 0 )
    
    return max(history.history['val_accuracy'])

def run_cnn(best_param):
        conv1_ker = best_param['conv1_ker']
        conv2_ker = best_param['conv2_ker']
        conv3_ker = best_param['conv3_ker']
        conv1_filter = best_param['conv1_filter']
        conv2_filter = best_param['conv2_filter']
        conv3_filter = best_param['conv3_filter']
        pool1_ker = best_param['pool1_ker']
        pool2_ker = best_param['pool2_ker']
        pool3_ker = best_param['pool3_ker']
        dense1_neuron = best_param['dense1_neuron']
        dense2_neuron = best_param['dense2_neuron']
        dropout_rate = best_param['dropout_rate']
        learning_rate = best_param['learning_rate']
        beta_1 = best_param['beta_1']
        beta_2 = best_param['beta_2']


        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=beta_1,beta_2=beta_2)

                        
        activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu','elu', 'exponential']  
        conv1_act = activationL[int(best_param['conv1_act'])]
        conv2_act = activationL[int(best_param['conv2_act'])]
        conv3_act = activationL[int(best_param['conv3_act'])]
        dense1_act = activationL[int(best_param['dense1_act'])]
        dense2_act = activationL[int(best_param['dense2_act'])]

        model = models.Sequential()
        model.add(layers.Conv2D(int(conv1_filter), int(conv1_ker), activation=conv1_act, padding="same", input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D(int(pool1_ker)))
        model.add(layers.Conv2D(int(conv2_filter), int(conv2_ker), activation=conv2_act, padding="same"))
        model.add(layers.MaxPooling2D(int(pool2_ker)))
        model.add(layers.Conv2D(int(conv3_filter), int(conv3_ker), activation=conv3_act, padding="same"))
        model.add(layers.MaxPooling2D(int(pool3_ker)))
        model.add(layers.Flatten())
        model.add(layers.Dense(int(dense1_neuron), activation=dense1_act))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(int(dense2_neuron), activation=dense2_act))
        model.add(layers.Dense(10))

        model.compile(optimizer = opt,
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])


        es = callbacks.EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=5)


        history = model.fit(train_images, train_labels, epochs=15, validation_split = 0.1, callbacks=[es])
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        
        return test_loss, test_acc

# rand_run_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
rand_run_list = [100, 200, 300, 400, 500]
for count, random_state in enumerate(rand_run_list):
    cnn_bo = BayesianOptimization(bo_tune_cnn, hyperspace_cnn, random_state=0)
    cnn_bo.maximize(init_points=50, n_iter=100,acq='ucb')
    best_params = cnn_bo.max['params']

    bo_val_loss = []
    for i,res in enumerate(cnn_bo.res):
        bo_val_loss.append(cnn_bo.res[i]['target'])    

    loss = np.array(bo_val_loss)
    loss_minimum_accumulate = np.minimum.accumulate(loss)
    df = pd.DataFrame(loss_minimum_accumulate).transpose()
    df.to_csv("BO_tune_cnn.csv", mode = 'a', index = False, header = False)

    eval_result = run_cnn(best_params)
    df_test = pd.DataFrame(eval_result).transpose()
    df_test.to_csv("BO_test_cnn.csv", mode = 'a', index = False, header = False)