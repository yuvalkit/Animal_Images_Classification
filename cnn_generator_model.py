import os
import numpy as np
import cv2
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tf_keras_vis.activation_maximization import ActivationMaximization
from matplotlib import pyplot as plt


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


def get_flows(x_train, x_val, x_test, y_train, y_val, y_test):
    train_generator = ImageDataGenerator(samplewise_center=True,
                                         rotation_range=30,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         rescale=1/255)

    val_test_generator = ImageDataGenerator(samplewise_center=True, rescale=1/255)

    train_flow = train_generator.flow(x_train, y_train, batch_size=64)
    val_flow = val_test_generator.flow(x_val, y_val, batch_size=64)
    test_flow = val_test_generator.flow(x_test, y_test, batch_size=64)

    return train_flow, val_flow, test_flow


def train_and_evaluate_model():
    x, y = get_x_and_y_from_dataset()
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2)

    train_flow, val_flow, test_flow = get_flows(x_train, x_val, x_test, y_train, y_val, y_test)

    model = get_model()

    checkpoint_path = 'best_model/checkpoint'
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       monitor="val_accuracy",
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='max')

    fit_log = model.fit(train_flow, validation_data=val_flow, epochs=5,
                        callbacks=[model_checkpoint])

    model.evaluate(test_flow, verbose=1)

    model.load_weights(checkpoint_path)

    test_results = model.evaluate(test_flow, verbose=1)

    save_results_to_file('results.txt', fit_log, test_results)

    return model


def loss(output):
    return output[0, 0], output[1, 1], output[2, 2], output[3, 3], output[4, 4], output[5, 5], output[6, 6], output[7, 7], output[8, 8], output[9, 9]


def model_modifier(m):
    m.layers[-1].activation = tensorflow.keras.activations.linear


def visualize_model(model):
    visualize_activation = ActivationMaximization(model, model_modifier)
    seed_input = tensorflow.random.uniform((10, image_size, image_size, 3), 0, 255)
    activations = visualize_activation(loss, seed_input=seed_input, steps=512)
    images = [activation.astype(np.float32) for activation in activations]
    for i in range(0, len(images)):
        visualization = images[i]
        plt.imshow((visualization * 255).astype(np.uint8), cmap='gray')
        plt.title(categories[i])
        plt.savefig(f'visualizations/{categories[i]}.png')


def main():
    model = train_and_evaluate_model()
    visualize_model(model)


if __name__ == '__main__':
    main()
