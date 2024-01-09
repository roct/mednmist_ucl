from medmnist import PathMNIST
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

def load_from_medmnist():
    """Load files from medmnist dataset"""
    dataset_train = PathMNIST(split='train', download=True)
    dataset_test = PathMNIST(split='test', download=True)
    dataset_val = PathMNIST(split='val', download=True)

    dataset_train_val = dataset_train + dataset_val
    train_val_images = np.array([np.array(dataset_train_val[n][0]) for n in range(len(dataset_train_val))])
    train_val_labels = np.array([np.array(dataset_train_val[n][1]) for n in range(len(dataset_train_val))])
    test_images = np.array([np.array(dataset_test[n][0]) for n in range(len(dataset_test))])
    test_labels = np.array([np.array(dataset_test[n][1]) for n in range(len(dataset_test))])
    return train_val_images, train_val_labels, test_images, test_labels


def create_CNN_model(num_layers=1, nodes=64, activation='relu', learning_rate=0.001):
    model = tf.keras.Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(layers.Conv2D(nodes, activation=activation, kernel_size=(3, 3), input_shape=(28, 28, 3)))
        else:
            model.add(layers.Conv2D(nodes, activation=activation, kernel_size=(3, 3)))
    # model.add(layers.MaxPool2D()),
    model.add(layers.Flatten())
    model.add(layers.Dense(9, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

def train_and_save_optimal_model():
    train_val_images, train_val_labels, test_images, test_labels = load_from_medmnist()
    optimal_model = create_CNN_model(num_layers=7, nodes=32, activation='relu', learning_rate=0.001)
    training_optimal = optimal_model.fit(train_val_images, train_val_labels, epochs=9, batch_size=32, validation_split=0.2, verbose=1)
    optimal_model.save('optimal_model.h5')
    
def load_model_and_evaluate():
    train_val_images, train_val_labels, test_images, test_labels = load_from_medmnist()
    optimal_model = tf.keras.models.load_model('optimal_model.h5')
    optimal_model_eval = optimal_model.evaluate(test_images, test_labels, verbose=1)
    print('TaskB model testing evaluation:', optimal_model_eval[1])

# load_model_and_evaluate()
train_and_save_optimal_model()