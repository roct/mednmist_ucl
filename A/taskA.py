from medmnist import PneumoniaMNIST
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

def load_from_medmnist():
    """Load files from medmnist dataset"""
    dataset_train = PneumoniaMNIST(split='train', download=True)
    dataset_test = PneumoniaMNIST(split='test', download=True)
    dataset_val = PneumoniaMNIST(split='val', download=True)

    dataset_train_val = dataset_train + dataset_val

    train_val_images = np.array([np.array(dataset_train_val[n][0]) for n in range(len(dataset_train_val))])
    train_val_labels = np.array([np.array(dataset_train_val[n][1]) for n in range(len(dataset_train_val))])
    test_images = np.array([np.array(dataset_test[n][0]) for n in range(len(dataset_test))])
    test_labels = np.array([np.array(dataset_test[n][1]) for n in range(len(dataset_test))])

    return train_val_images, train_val_labels, test_images, test_labels

def create_CNN_model(num_layers=1, nodes=64, activation='relu', learning_rate=0.001):
    """Create CNN model"""
    model = tf.keras.Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(layers.Conv2D(nodes, activation=activation, kernel_size=(3, 3), input_shape=(28, 28, 1)))
        else:
            model.add(layers.Conv2D(nodes, activation=activation, kernel_size=(3, 3)))
    # model.add(layers.MaxPool2D()),
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

def train_and_save_optimal_model():
    train_val_images, train_val_labels, test_images, test_labels = load_from_medmnist()
    best_model = create_CNN_model(num_layers=1, nodes=32, activation='relu', learning_rate=0.001)
    best_model_CNN_training = best_model.fit(train_val_images, train_val_labels, epochs=9, batch_size=32, verbose=1, validation_split=0.2)
    # save model to file
    best_model.save('best_model.h5')

def load_model_and_evaluate():
    train_val_images, train_val_labels, test_images, test_labels = load_from_medmnist()
    best_model = tf.keras.models.load_model('A/best_model.h5')
    best_model_eval = best_model.evaluate(test_images, test_labels, verbose=1)
    print('TaskA model testing evaluation:', best_model_eval[1])

def predict(file):
    model = tf.keras.models.load_model('A/best_model.h5')
    try:
        img_array = np.loadtxt(file, delimiter=",")
    except FileNotFoundError:
        print("File not found.")
        return False
    predictions = model.predict(img_array.reshape(1, 28, 28, 1))
    score = predictions[0]
    print(score)
    if score < 0.5:
        print('Pneumonia')
    else:
        print('Normal')

def load_dataset_npz(filepath):
    try:
        loaded_dataset = np.load(filepath)
    except FileNotFoundError:
        print("File not found.")
        return False
    # Extract images and labels from dataset
    print(type(loaded_dataset))
    test_images = loaded_dataset['test_images']
    test_labels = loaded_dataset['test_labels']
    # test_images = np.array([np.array(loaded_dataset[n][0]) for n in range(len(loaded_dataset))])
    # test_labels = np.array([np.array(loaded_dataset[n][1]) for n in range(len(loaded_dataset))])
    best_model = tf.keras.models.load_model('A/best_model.h5')
    best_model_eval = best_model.evaluate(test_images, test_labels, verbose=1)
    print('TaskA model testing evaluation:', best_model_eval[1])
