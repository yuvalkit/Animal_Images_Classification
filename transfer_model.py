import os
import numpy as np
import cv2
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet101, ResNet152
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

image_size = 224
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


def get_model(keras_application):
    pre_trained_model = keras_application(input_shape=(image_size, image_size, num_channels),
                                          include_top=False, weights='imagenet')

    pre_trained_model.trainable = False

    x = Flatten()(pre_trained_model.output)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)

    outputs = Dense(len(categories), activation='softmax')(x)

    model = Model(inputs=pre_trained_model.inputs, outputs=outputs)

    model.compile(optimizer=optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def write_results_to_file(file, results_arr, title):
    file.write(f'{title}:\n')
    for i, elem in enumerate(results_arr):
        file.write(f'{i + 1}: {elem}\n')
    file.write('\n')


def save_results_to_file(file_name, fit_log, test_results, keras_application_name):
    train_loss_history = fit_log.history['loss']
    val_loss_history = fit_log.history['val_loss']
    train_accuracy_history = fit_log.history['accuracy']
    val_accuracy_history = fit_log.history['val_accuracy']

    file = open(file_name, 'a')

    file.write('=======================================\n')
    file.write(f'{keras_application_name}\n')
    file.write('=======================================\n\n')

    write_results_to_file(file, train_loss_history, 'train_loss')
    write_results_to_file(file, val_loss_history, 'val_loss')
    write_results_to_file(file, train_accuracy_history, 'train_accuracy')
    write_results_to_file(file, val_accuracy_history, 'val_accuracy')

    file.write(f'test_loss: {test_results[0]}\n')
    file.write(f'test_accuracy: {test_results[1]}\n\n')

    file.close()


def get_flows(x_train, x_val, x_test, y_train, y_val, y_test):
    train_val_generator = ImageDataGenerator(samplewise_center=True,
                                             rotation_range=30,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True,
                                             rescale=1/255)

    test_generator = ImageDataGenerator(samplewise_center=True, rescale=1/255)

    train_flow = train_val_generator.flow(x_train, y_train, batch_size=64)
    val_flow = train_val_generator.flow(x_val, y_val, batch_size=64)
    test_flow = test_generator.flow(x_test, y_test, batch_size=64)

    return train_flow, val_flow, test_flow


def train_and_evaluate_model(keras_application, keras_application_name):
    x, y = get_x_and_y_from_dataset()
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2)

    train_flow, val_flow, test_flow = get_flows(x_train, x_val, x_test, y_train, y_val, y_test)

    model = get_model(keras_application)

    checkpoint_path = 'best_model/checkpoint'
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       monitor="val_accuracy",
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='max')

    print(f'starting {keras_application_name} train')
    fit_log = model.fit(train_flow, validation_data=val_flow, epochs=100,
                        callbacks=[model_checkpoint])

    model.evaluate(test_flow, verbose=1)

    model.load_weights(checkpoint_path)

    test_results = model.evaluate(test_flow, verbose=1)

    save_results_to_file('results.txt', fit_log, test_results, keras_application_name)


def main():
    # train_and_evaluate_model(VGG16, 'VGG16')
    # train_and_evaluate_model(VGG19, 'VGG19')
    train_and_evaluate_model(ResNet50, 'ResNet50')
    # train_and_evaluate_model(ResNet101, 'ResNet101')
    # train_and_evaluate_model(ResNet152, 'ResNet152')


if __name__ == '__main__':
    main()
