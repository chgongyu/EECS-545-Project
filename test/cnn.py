from tuning import BayesOpt_KSD
from tensorflow.keras import datasets, layers, models, callbacks
import numpy as np
import pandas as pd
from ax import (
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from gpytorch.kernels import MaternKernel
from botorch.acquisition.analytic import UpperConfidenceBound

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

params_space = {
    'conv1_act': (ParameterType.INT, 0, 7),
    'conv2_act': (ParameterType.INT, 0, 7),
    'conv3_act': (ParameterType.INT, 0, 7),
    'conv1_ker': (ParameterType.INT, 2, 10),
    'conv2_ker': (ParameterType.INT, 2, 10),
    'conv3_ker': (ParameterType.INT, 2, 10),
    'conv1_filter': (ParameterType.INT, 20, 32),
    'conv2_filter': (ParameterType.INT, 20, 32),
    'conv3_filter': (ParameterType.INT, 50, 100),
    'pool1_ker': (ParameterType.INT, 2, 3),
    'pool2_ker': (ParameterType.INT, 2, 3),
    'pool3_ker': (ParameterType.INT, 2, 3),
    'dense1_neuron': (ParameterType.INT, 32, 64),
    'dense2_neuron': (ParameterType.INT, 32, 64),
    'dense1_act': (ParameterType.INT, 0, 7),
    'dense2_act': (ParameterType.INT, 0, 7),
    'dropout_rate': (ParameterType.FLOAT, 0, 0.5),
    'learning_rate': (ParameterType.FLOAT, 0.0005, 0.002),
    'beta_1': (ParameterType.FLOAT, 0.9, 0.999),
    'beta_2': (ParameterType.FLOAT, 0.95, 0.999),
}

search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name=key, parameter_type=val[0], lower=val[1], upper=val[2]
        )
        for key, val in params_space.items()
    ]
)


def bo_tune_cnn(best_param, test=False):
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']

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
    conv1_act = activationL[int(best_param['conv1_act'])]
    conv2_act = activationL[int(best_param['conv2_act'])]
    conv3_act = activationL[int(best_param['conv3_act'])]
    dense1_act = activationL[int(best_param['dense1_act'])]
    dense2_act = activationL[int(best_param['dense2_act'])]

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

    model = models.Sequential()
    model.add(
        layers.Conv2D(int(conv1_filter), int(conv1_ker), activation=conv1_act, padding="same", input_shape=(32, 32, 3)))
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

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    es = callbacks.EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=5)

    if test:
        history = model.fit(train_images, train_labels, epochs=15, validation_split=0.1, callbacks=[es])
        _, test_acc = model.evaluate(test_images, test_labels)
        return test_acc
    history = model.fit(train_images, train_labels, epochs=3, validation_split=0.1, callbacks=[es])
    return -max(history.history['val_accuracy'])

inits = (30, 20)
iters = 100
runs = 20
GLOBAL_Y_COLLECTOR = np.zeros((runs, sum(inits)+iters))
GLOBAL_TEST_ACC = np.zeros((runs, ))

test_case = BayesOpt_KSD(search_space, bo_tune_cnn)

for run in range(runs):
  test_case.optimize(MaternKernel, UpperConfidenceBound, n_init=inits, n_iter=iters, alpha=0.4)
  best_params = test_case.get_best_params()

  test_acc = bo_tune_cnn(best_params, test=True)
  GLOBAL_TEST_ACC[run] = test_acc
  GLOBAL_Y_COLLECTOR[run] = test_case.Y_COLLECTOR

  test_perform = pd.DataFrame({"acc": GLOBAL_TEST_ACC})
  test_perform.to_csv(f'boksd_new_test_perform_{run}.csv', index=False)

  np.savetxt(f"boksd_new_y_tune_{run}.csv", np.minimum.accumulate(GLOBAL_Y_COLLECTOR, axis=1), delimiter=",")
