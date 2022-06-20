import keras
from keras import backend as K
from keras.datasets import mnist, cifar10, fashion_mnist
import pickle
import os
from .types import Dataset
from typing import Optional

def load_data(name=None) -> Optional[Dataset]:
    """Load data
    :param name:
        - mnist_flat
        - mnist_conv
        - cifar10
    :return dataset
    """
    num_classes = 10
    dataset = None

    filepath = os.path.abspath(__file__)
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath + '/saved/', exist_ok=True)
    datapath = dirpath + '/saved/' + name

    if os.path.exists(datapath):
        with open(datapath, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    
    if name == 'mnist_flat':        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # 画像データを一次元化
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)

        # データを正規化（0~255の値なので255で割って0~1に変換）
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
        
        dataset = (Dataset(x_train, y_train), Dataset(x_test, y_test))
        with open(datapath, 'wb+') as f:
            pickle.dump(dataset, f)

        return dataset
    
    elif name == 'mnist_conv':
        """
        x_train.shape: (60,000, 28, 28, 1)
        y_train.shape: (60,000, 10)
        x_test.shape: (10,000, 28, 28, 1)
        y_test.shape: (10,000, 10)
        """
        img_rows, img_cols = 28, 28

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

        dataset = (Dataset(x_train, y_train), Dataset(x_test, y_test))
        with open(datapath, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset
    
    elif name == 'cifar10':
        """
        x_train.shape: (50000, 32, 32, 3)       y_train.shape: (50000, 10)
        x_test.shape:  (10000, 32, 32, 3)       y_test.shape:  (10000, 10)
        """

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # normalize
        x_train = x_train.astype('float32') / 255.0
        x_test  = x_test.astype('float32') / 255.0
    
        # Convert class vectors to binary class matrices.
        y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
        y_test  = keras.utils.np_utils.to_categorical(y_test,  num_classes)

        dataset = (Dataset(x_train, y_train), Dataset(x_test, y_test))
        with open(datapath, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset

    elif name == 'fmnist_flat':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # 画像データを一次元化
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)

        # データを正規化（0~255の値なので255で割って0~1に変換）
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
        
        dataset = (Dataset(x_train, y_train), Dataset(x_test, y_test))
        with open(datapath, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset

    elif name == 'fmnist_conv':
        """
        x_train.shape: (60,000, 28, 28, 1)
        y_train.shape: (60,000, 10)
        x_test.shape: (10,000, 28, 28, 1)
        y_test.shape: (10,000, 10)
        """
        img_rows, img_cols = 28, 28

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

        dataset = (Dataset(x_train, y_train), Dataset(x_test, y_test))
        with open(datapath, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset

    else:
        print('No data loaded')
