import datetime

import numpy as np
import sys

from PIL import Image
from math import sqrt
from matplotlib import pyplot as plt

__author__ = 'christopher@levire.com'

learning_rate = 0.8
lamb = 2.8


def natural_sig(x, derived=False):
    if derived:
        return np.multiply(sigmoid(x), (1-sigmoid(x)))
    return 1 / (1 + np.exp(-x))


def tan_sig(x, derived=False):
    if derived:
        return 1-np.power(np.tanh(x), 2)
    return np.tanh(x)


sigmoid = natural_sig


def run(data, result, layer_list):

    zlist, activation_list = forward_propagation(data, layer_list)
    back_propagate(zlist, activation_list, result, layer_list)


def forward_propagation(data, layer_list):

    z_list = []
    activation_list = [data]

    for layer in layer_list:
        activation_list[-1] = np.append(np.ones((activation_list[-1].shape[0], 1)), activation_list[-1], 1)
        z1 = np.dot(activation_list[-1], layer)
        activation1 = sigmoid(z1)
        activation_list.append(activation1)
        z_list.append(z1)

    return z_list, activation_list


def back_propagate(zlist, activation_list, expected_result, layer_list):
    last_delta = activation_list[-1] - expected_result
    delta_list = [last_delta]

    for i in range(len(layer_list)-1, 0, -1):
        delta = np.multiply(np.dot(delta_list[0], layer_list[i].T[:, 1:]), sigmoid(zlist[i-1], derived=True))
        delta_list.insert(0, delta)

    for index, layer in enumerate(layer_list):
        gradient_matrix = np.dot(activation_list[index].T, delta_list[index])
        lambda_factor = lamb * np.append(np.zeros((layer.shape[0], 1)), layer[:, 1:], 1)
        layer -= (learning_rate * gradient_matrix + lambda_factor) / expected_result.shape[0]


def train_network(extension, trainings_data, layers, batch_size=5000, epochs=100):
    error_rates = [1.00]
    for epoch in range(epochs):
        for batch_index in range(0, trainings_data.shape[0], batch_size):

            input_matrix, output_matrix = split_and_normalize(batch_index, batch_size, trainings_data)

            run(input_matrix, output_matrix, layers)

        print("Epoch %d saving file at %s" % (epoch, str(datetime.datetime.utcnow())))
        for k, layer in enumerate(layers):
            np.savetxt("layer"+str(k)+str(extension)+".csv", layer, delimiter=",")

        if epoch % 2 == 0:
            input_matrix, output_matrix = split_and_normalize(0, int(trainings_data.shape[0]*0.3), trainings_data)

            zlist, activations = forward_propagation(input_matrix, layers)

            correct = 0
            for index, prediction in enumerate(activations[-1]):
                predicted_number = np.argmax(prediction)
                exprected_number = np.argmax(output_matrix[index])
                if predicted_number == exprected_number:
                    correct += 1
            error = 1-correct / input_matrix.shape[0]
            print("Current Error: %f"%error)
            error_rates.append(error)
            plot_error_rate(error_rates, extension)
            if error < 0.005:
                print("Reached crossvalidation goal at: %d epochs" % epoch)
                for k, layer in enumerate(layers):
                    np.savetxt("layer" + str(k) + str(extension) + ".csv", layer, delimiter=",")
                break

    print("You are ready!")


def split_and_normalize(batch_index, batch_size, trainings_data):
    data_matrix = trainings_data[batch_index:batch_index + batch_size]
    input_matrix, output_matrix = split_io(data_matrix)
    input_matrix = input_matrix / 256.0
    return input_matrix, output_matrix


def plot_error_rate(error_rates, extension):
    plt.plot(error_rates)
    plt.savefig(extension + "error.png")


def split_io(data):
    input_matrix = data[:, 1:]
    output_col = data[:, 0]
    output_matrix = np.zeros((input_matrix.shape[0],10))
    for index, number in enumerate(output_col):
        output_matrix[index][number] = 1
    return input_matrix, output_matrix


def norm_min_max(x, min, max):
    return (x-min)/(max-min)


if __name__ == "__main__":

    print("Starting")
    action = sys.argv[1]

    if action == "test":
        print("Running checking")
        check_file = sys.argv[2]
        test_lines = int(sys.argv[3])
        layer_list = []
        for i in range(4, len(sys.argv)):
            layer_list.append(np.genfromtxt(sys.argv[i], delimiter=",", dtype=float))

        test_data = np.genfromtxt(check_file, max_rows=test_lines, delimiter=",", skip_header=1, dtype=int)

        input_vec, output_vec = split_io(test_data)

        input_vec = input_vec / 256.0

        zlist, activations = forward_propagation(input_vec, layer_list)
        correct = 0
        for index, prediction in enumerate(activations[-1]):
            predicted_number = np.argmax(prediction)
            exprected_number = np.argmax(output_vec[index])
            if predicted_number == exprected_number:
                correct+=1
            else:
                print("Result: "+str(predicted_number) + " Expected: "+str(exprected_number))
        print("Model results - correct: %d, incorrect: %d, tested: %d, error; %.3f %%" % (correct, test_lines-correct, test_lines, float((1-correct/test_lines)*100)))

    elif action == "train":
        print("Running training")
        trainingfile = sys.argv[2]
        epochs = int(sys.argv[3])
        extension = sys.argv[4]
        trainings_data = np.genfromtxt(trainingfile, skip_header=1, max_rows=29000, delimiter=",", dtype=int)
        layer_1 = 2 * np.random.random((785, 1024)) - 1
        layer_5 = 2 * np.random.random((1025, 10)) - 1
        train_network(extension, trainings_data, [layer_1, layer_5], epochs=epochs)

    elif action == "print":

        print("Printing layer")
        layer = sys.argv[2]

        layer_file = np.genfromtxt(layer, delimiter=",", dtype=float)
        mini_picture_list = []
        tile_size = int(sqrt(layer_file.shape[0] - 1))
        for column in range(0, layer_file.shape[1]):
            mini_picture = layer_file[1:, column].reshape((tile_size,tile_size))
            min = mini_picture.min()
            max = mini_picture.max()
            for i in range(0, tile_size):
                for j in range(0, tile_size):
                    mini_picture[i][j] = int(norm_min_max(mini_picture[i][j], min, max) * 255.0)
            mini_picture_list.append(mini_picture)

        spalten = 48
        zeilen = np.math.ceil(len(mini_picture_list)*1.0 / spalten*1.0)

        picture = np.zeros((zeilen*(tile_size+1), spalten*(tile_size+1)), dtype=np.uint8)

        for index, image in enumerate(mini_picture_list):
            x = (index % spalten) * (tile_size+1)
            y = int(index / spalten) * (tile_size+1)
            picture[y:y+tile_size, x:x+tile_size] = image

        img = Image.fromarray(picture, 'L')
        img.save(layer + "_image.jpg")
        img.show()

    elif action == "generate":
        pass
