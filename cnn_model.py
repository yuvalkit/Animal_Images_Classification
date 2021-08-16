import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time


dataset_path = '/home/access/yuval_projects/data/Animals-10'
# dataset_path = '/content/PracticalML_FinalProject/dataset'


categories = ['butterfly',
              'cat',
              'chicken',
              'cow',
              'dog',
              'elephant',
              'horse',
              'sheep',
              'spider',
              'squirrel']

image_size = 128
num_channels = 3


def get_x_and_y_from_dataset():
    start = time.time()
    x = []
    y = []
    for category in categories:
        dir_path = os.path.join(dataset_path, category)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(image, (image_size, image_size))
            x.append(resized_img)
            y.append(categories.index(category))
    x = np.array(x).reshape((-1, image_size, image_size, num_channels))
    y = np.array(y)
    end = time.time()
    print(f'load time: {end - start}')
    return x, y


def save_x_and_y_to_file(x, y, file_path):
    np.savez(file_path, x, y)


def load_x_and_y_from_file(file_path):
    npz = np.load(file_path)
    x = npz['arr_0']
    y = npz['arr_1']
    return x, y


def get_model():
    model = Sequential([
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
               input_shape=(image_size, image_size, num_channels)),
        Dropout(0.2),
        BatchNormalization(),

        Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        BatchNormalization(),

        Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        BatchNormalization(),

        Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        BatchNormalization(),

        Flatten(),
        Dropout(0.2),

        Dense(512, activation='relu'),
        Dropout(0.2),
        BatchNormalization(),

        Dense(128, activation='relu'),
        Dropout(0.2),
        BatchNormalization(),

        Dense(len(categories), activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def plot_fit_metric(fit_log, metric):
    plt.plot(fit_log.history[metric])
    plt.plot(fit_log.history[f'val_{metric}'])
    plt.title(f'model {metric}')
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.show()


def plot_fit_log(fit_log):
    print()
    plot_fit_metric(fit_log, 'loss')
    print()
    plot_fit_metric(fit_log, 'accuracy')
    print()


def write_results_to_file(file, results_arr, title):
    file.write(f'{title}:\n')
    for i, elem in enumerate(results_arr):
        file.write(f'{i + 1}: {elem}\n')
    file.write('\n')


def save_results_to_file(file_name, fit_log, test_results):
    train_loss_history = fit_log.history['loss']
    val_loss_history = fit_log.history['val_loss']
    train_accuracy_history = fit_log.history['accuracy']
    val_accuracy_history = fit_log.history['val_accuracy']

    file = open(file_name, 'w')

    write_results_to_file(file, train_loss_history, 'train_loss')
    write_results_to_file(file, val_loss_history, 'val_loss')
    write_results_to_file(file, train_accuracy_history, 'train_accuracy')
    write_results_to_file(file, val_accuracy_history, 'val_accuracy')

    file.write(f'test_loss: {test_results[0]}\n')
    file.write(f'test_accuracy: {test_results[1]}\n')

    file.close()


def main():
    x, y = get_x_and_y_from_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = get_model()

    fit_log = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=64)
    test_results = model.evaluate(x_test, y_test, verbose=1)

    save_results_to_file('results.txt', fit_log, test_results)


if __name__ == '__main__':
    main()
