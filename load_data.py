import os
import numpy as np
import pandas as pd


def loadVectors(dataset='AlexNet'):
    # paths and filenames
    main_path = './Data/Kaggle Challenge/'
    vector_path = 'vectors/'
    labels_path = 'labels/'
    train_vectors = 'trainVectors.csv'
    train_labels = 'trainLbls.csv'
    validation_vectors = 'valVectors.csv'
    validation_labels = 'valLbls.csv'
    test_vectors = 'testVectors.csv'
    global x_train, y_train, x_validation, y_validation, x_test
    # Load all data from csv file
    x_train = pd.read_csv(os.path.join(main_path, vector_path, dataset, train_vectors), header=None)
    y_train = pd.read_csv(os.path.join(main_path, labels_path, train_labels), header=None)

    x_validation = pd.read_csv(os.path.join(main_path, vector_path, dataset, validation_vectors), header=None)
    y_validation = pd.read_csv(os.path.join(main_path, labels_path, validation_labels), header=None)

    x_test = pd.read_csv(os.path.join(main_path, vector_path, dataset, test_vectors), header=None)

    # Transpose all x
    x_train = x_train.T
    x_validation = x_validation.T
    x_test = x_test.T

    # Ravel all y
    y_train = np.ravel(y_train)
    y_validation = np.ravel(y_validation)

    return x_train, y_train, x_validation, y_validation, x_test
