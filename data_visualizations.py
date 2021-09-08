import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2


one_per_class_path = 'one_per_class'
dataset_path = 'Animals-10'

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


def plot_count_per_category():
    x, y = get_x_and_y_from_dataset()
    counts = [0] * len(categories)
    for _, category in zip(x, y):
        counts[category] += 1
    plt.xlabel('category', fontsize='x-large')
    plt.ylabel('image count', fontsize='x-large')
    plt.title('Number of images per category')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='x-large')
    plt.bar(categories, counts)
    plt.xticks(rotation=30)
    plt.show()


def plot_x_and_y_as_image(x, y):
    to_image = transforms.ToPILImage()
    plt.title(categories[y])
    plt.imshow(to_image(x))
    plt.show()


def plot_one_image_per_class():
    fig = plt.figure(figsize=(224, 224))
    columns = 5
    rows = 2
    for i, dir_name in enumerate(categories):
        dir_path = os.path.join(one_per_class_path, dir_name)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            image = Image.open(file_path)
            resized_image = image.resize((224, 224))
            image.close()
            fig.add_subplot(rows, columns, i + 1)
            plt.title(dir_name, fontsize='xx-large')
            plt.imshow(resized_image)
    plt.show()


def save_gen_images_to_dir():
    x, y = get_x_and_y_from_dataset()
    generator = ImageDataGenerator(samplewise_center=True,
                                   samplewise_std_normalization=True,
                                   rotation_range=30,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   brightness_range=[0.7, 1.3],
                                   shear_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
    i = 0
    for _ in generator.flow(x, y, batch_size=1, save_to_dir='preview'):
        i += 1
        if i > 100:
            return


def plot_fit_metric(epochs, train_values, val_values, metric):
    plt.plot(epochs, train_values, 'b', label='train')
    plt.plot(epochs, val_values, 'r', label='validation')
    plt.title(f'{metric.title()} per epoch', fontsize='x-large')
    plt.ylabel(metric, fontsize='x-large')
    plt.xlabel('epoch', fontsize='x-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.legend(fontsize='x-large')
    plt.show()


def get_epochs_and_values(element):
    tokens = element.split('\n')[1:]
    splits = [tuple(token.split(': ')) for token in tokens]
    epochs, values = list(zip(*splits))
    epochs = [float(epoch) for epoch in list(epochs)]
    values = [float(value) for value in list(values)]
    return epochs, values


def plot_from_results_file():
    f = open('results.txt', 'r')
    content = f.read()
    f.close()

    parts = content.split('\n\n')
    epochs, train_loss = get_epochs_and_values(parts[0])
    _, val_loss = get_epochs_and_values(parts[1])
    _, train_accuracy = get_epochs_and_values(parts[2])
    _, val_accuracy = get_epochs_and_values(parts[3])

    plot_fit_metric(epochs, train_loss, val_loss, 'loss')
    plot_fit_metric(epochs, train_accuracy, val_accuracy, 'accuracy')


def get_val_epochs_loss_accuracy(parts, model_index):
    model_parts = parts[model_index].split('\n\n')[:-1]
    epochs, loss = get_epochs_and_values(model_parts[1])
    _, accuracy = get_epochs_and_values(model_parts[3])
    return epochs, loss, accuracy


def plot_metric_comparison(metric, names, epochs, values):
    for value in values:
        plt.plot(epochs, value)
    plt.title(f'{metric.title()} on validation per epoch', fontsize='x-large')
    plt.ylabel(metric, fontsize='x-large')
    plt.xlabel('epoch', fontsize='x-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.legend(names, title='Pre-trained model used', fontsize='large', title_fontsize='large')
    plt.locator_params(axis="x", nbins=12)
    plt.show()


def plot_transfer_val_comparison():
    f = open('results.txt', 'r')
    content = f.read()
    f.close()

    parts = content.split('=======================================\n\n')[1:]

    epochs, vgg16_loss, vgg16_accuracy = get_val_epochs_loss_accuracy(parts, 0)
    _, vgg19_loss, vgg19_accuracy = get_val_epochs_loss_accuracy(parts, 1)
    _, mobile_net_loss, mobile_net_accuracy = get_val_epochs_loss_accuracy(parts, 2)
    _, resnet50_loss, resnet50_accuracy = get_val_epochs_loss_accuracy(parts, 3)
    _, resnet101_loss, resnet101_accuracy = get_val_epochs_loss_accuracy(parts, 4)
    _, inception_v3_loss, inception_v3_accuracy = get_val_epochs_loss_accuracy(parts, 5)
    _, inception_resnet_v2_loss, inception_resnet_v2_accuracy = get_val_epochs_loss_accuracy(parts, 6)
    _, xception_loss, xception_accuracy = get_val_epochs_loss_accuracy(parts, 7)

    names = ['VGG16', 'VGG19', 'MobileNet', 'ResNet50', 'ResNet101', 'InceptionV3', 'InceptionResNetV2', 'Xception']

    losses = [vgg16_loss, vgg19_loss, mobile_net_loss, resnet50_loss, resnet101_loss,
              inception_v3_loss, inception_resnet_v2_loss, xception_loss]

    accuracies = [vgg16_accuracy, vgg19_accuracy, mobile_net_accuracy, resnet50_accuracy, resnet101_accuracy,
                  inception_v3_accuracy, inception_resnet_v2_accuracy, xception_accuracy]

    plot_metric_comparison('loss', names, epochs, losses)
    plot_metric_comparison('accuracy', names, epochs, accuracies)


def get_test_loss_accuracy(parts, model_index):
    test_values = parts[model_index].split('\n\n')[:-1][-1].split('\n')
    test_loss = float(test_values[0].split(': ')[1])
    test_accuracy = float(test_values[1].split(': ')[1])
    return test_loss, test_accuracy


def plot_metric_bar_comparison(metric, names, values, y_range, label_factor):
    plt.bar(names, values)
    plt.title(f'Test {metric} per pre-trained model used', fontsize='x-large')
    plt.xlabel('model', fontsize='x-large')
    plt.ylabel(f'{metric}', fontsize='x-large')
    plt.xticks(fontsize='medium', rotation=30)
    plt.yticks(fontsize='x-large')

    axes = plt.gca()
    axes.set_ylim(y_range)
    for i, value in enumerate(values):
        label_value = value * label_factor
        plt.text(i, label_value, "{:.4f}".format(value), ha='center', fontsize='medium')

    plt.show()


def plot_transfer_test_comparison():
    f = open('results.txt', 'r')
    content = f.read()
    f.close()

    parts = content.split('=======================================\n\n')[1:]

    vgg16_loss, vgg16_accuracy = get_test_loss_accuracy(parts, 0)
    vgg19_loss, vgg19_accuracy = get_test_loss_accuracy(parts, 1)
    mobile_net_loss, mobile_net_accuracy = get_test_loss_accuracy(parts, 2)
    resnet50_loss, resnet50_accuracy = get_test_loss_accuracy(parts, 3)
    resnet101_loss, resnet101_accuracy = get_test_loss_accuracy(parts, 4)
    inception_v3_loss, inception_v3_accuracy = get_test_loss_accuracy(parts, 5)
    inception_resnet_v2_loss, inception_resnet_v2_accuracy = get_test_loss_accuracy(parts, 6)
    xception_loss, xception_accuracy = get_test_loss_accuracy(parts, 7)

    names = ['VGG16', 'VGG19', 'MobileNet', 'ResNet50', 'ResNet101', 'InceptionV3', 'InceptionResNetV2', 'Xception']

    losses = [vgg16_loss, vgg19_loss, mobile_net_loss, resnet50_loss, resnet101_loss,
              inception_v3_loss, inception_resnet_v2_loss, xception_loss]

    accuracies = [vgg16_accuracy, vgg19_accuracy, mobile_net_accuracy, resnet50_accuracy, resnet101_accuracy,
                  inception_v3_accuracy, inception_resnet_v2_accuracy, xception_accuracy]

    plot_metric_bar_comparison('loss', names, losses, [0.09, 0.185], 1.008)
    plot_metric_bar_comparison('accuracy', names, accuracies, [0.955, 0.985], 1.0005)


def get_images_details():
    heights = []
    widths = []

    for category in categories:
        dir_path = os.path.join(dataset_path, category)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            heights.append(height)
            widths.append(width)

    heights = np.array(heights)
    widths = np.array(widths)

    print(f'min_height: {min(heights)}, max_height: {max(heights)}, avg_height: {np.average(heights)}, height_median: {np.median(heights)}')
    print(f'min_width: {min(widths)}, max_width: {max(widths)}, avg_width: {np.average(widths)}, width_median: {np.median(widths)}')


def plot_some_images():
    fig = plt.figure(figsize=(image_size, image_size))
    columns = 8
    rows = 1
    path = 'images_to_show'
    for i, file_name in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        image = Image.open(file_path)
        resized_image = image.resize((image_size, image_size))
        image.close()
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(resized_image)
    plt.show()


def plot_summary_by_metric(titles, values, values_range, metric, label_factor):
    plt.bar(titles, values)
    axes = plt.gca()
    axes.set_ylim(values_range)
    for i, value in enumerate(values):
        label_value = value * label_factor
        plt.text(i, label_value, "{:.4f}".format(value), ha='center', fontsize='xx-large')
    plt.title(f'Test {metric} per experiment', fontsize='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.xlabel('experiment', fontsize='xx-large')
    plt.ylabel(metric, fontsize='xx-large')
    plt.show()


def plot_summary():
    loss_titles = ['CNN', 'Improved\nCNN', 'Data\nAugmentation', 'Transfer\nLearning\n(Xception)']
    accuracy_titles = loss_titles[:-1] + ['Transfer\nLearning\n(InceptionResNetV2)']
    losses = [2.1321, 1.3976, 0.5371, 0.0971]
    accuracies = [0.6025, 0.7379, 0.8645, 0.9815]
    plot_summary_by_metric(loss_titles, losses, [0.0, 2.5], 'loss', 1.03)
    plot_summary_by_metric(accuracy_titles, accuracies, [0.58, 1.02], 'accuracy', 1.003)


def plot_image_sizes():
    sizes = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000-3000', '3000-5000', '5000-7000']
    widths = [0] * len(sizes)
    heights = [0] * len(sizes)
    for category in categories:
        dir_path = os.path.join(dataset_path, category)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            for i, size in enumerate(sizes):
                values_range = size.split('-')
                from_value = int(values_range[0])
                to_value = int(values_range[1])
                if from_value <= width < to_value:
                    widths[i] += 1
                if from_value <= height < to_value:
                    heights[i] += 1
    plt.plot(sizes, widths, 'b', label='width')
    plt.plot(sizes, heights, 'r', label='height')
    plt.title('Number of images per width and height', fontsize='x-large')
    plt.ylabel('image count', fontsize='x-large')
    plt.xlabel('size range in pixels', fontsize='x-large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.legend(fontsize='x-large')
    plt.xticks(rotation=30)
    plt.show()
