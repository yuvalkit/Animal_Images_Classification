import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint


dataset_path = '/home/access/yuval_projects/data/Animals-10'

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
    return x, y


def get_model():
    model = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
               input_shape=(image_size, image_size, num_channels)),

        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(64, activation='relu'),

        Dense(len(categories), activation='softmax')
    ])

    model.compile(optimizer=optimizers.SGD(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


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

    checkpoint_path = 'best_model/checkpoint'
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       monitor="val_accuracy",
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='max')

    fit_log = model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=64,
                        callbacks=[model_checkpoint])

    model.evaluate(x_test, y_test, verbose=1)

    model.load_weights(checkpoint_path)

    test_results = model.evaluate(x_test, y_test, verbose=1)

    save_results_to_file('results.txt', fit_log, test_results)


if __name__ == '__main__':
    main()
