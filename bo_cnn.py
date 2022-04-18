#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:19:35 2022

@author: zixiao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:32:01 2022

@author: zixiao
"""
from bayes_opt import BayesianOptimization
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
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

def bo_tune_cnn(batch_size,conv1_act,conv2_act,conv3_act,conv1_ker,conv2_ker,conv3_ker,conv1_filter,conv2_filter,conv3_filter,
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


    history = model.fit(train_images, train_labels, epochs=2, batch_size = int(batch_size), validation_split = 0.1, callbacks=[es])
    
    return -1.0 * min(history.history['val_loss'])

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
    'pool1_ker': (2, 4),
    'pool2_ker': (2, 4),
    'pool3_ker': (2, 4),
    'dense1_neuron': (32, 64),
    'dense2_neuron': (32, 64),
    'dense1_act': (0, 7),
    'dense2_act': (0, 7),
    'dropout_rate':(0, 0.5),
    'learning_rate':(0.0005, 0.002),
    'batch_size':(20, 100),
    'beta_1':(0.9,1),
    'beta_2':(0.95,1),
 }

# Run Bayesian Optimization
start = time.time()
cnn_bo = BayesianOptimization(bo_tune_cnn, hyperspace_cnn, random_state=123)
cnn_bo.maximize(init_points=2, n_iter=2,acq='ucb')
end = time.time()
print("The execution time of BO is :", end-start)


best_param = cnn_bo.max['params']
print(best_param)


batch_size = best_param['batch_size']
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


history = model.fit(train_images, train_labels, epochs=10, batch_size = int(batch_size), validation_split = 0.1, callbacks=[es])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

bo_val_loss = []
for i,res in enumerate(cnn_bo.res):
    bo_val_loss.append(-1.0 * cnn_bo.res[i]['target'])
    

loss = np.array(bo_val_loss)
np.minimum.accumulate(loss)
plt.figure()
plt.plot(np.minimum.accumulate(loss))
