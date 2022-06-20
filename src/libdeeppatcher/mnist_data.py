from keras.datasets import mnist
import keras
from keras import backend as K
from .types import Dataset

def load_mnist_data():
    """前処理したmnistのデータを返す
    x_train.shape: (60,000, 784)
    y_train.shape: (60,000, 10)
    x_test.shape: (10,000, 784)
    y_test.shape: (10,000, 10)
    :return x_train, y_train: 学習データ(60,000件)
            x_test, y_test: テストデータ(10,000件)
    """
    num_classes = 10
    img_rows, img_cols = 28, 28

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

    return Dataset(x_train, y_train), Dataset(x_test, y_test)

def load_mnist_data_conv():
    """前処理したmnistのデータを返す
    x_train.shape: (60,000, 28, 28, 1)
    y_train.shape: (60,000, 10)
    x_test.shape: (10,000, 28, 28, 1)
    y_test.shape: (10,000, 10)
    :return x_train, y_train: 学習データ(60,000件)
            x_test, y_test: テストデータ(10,000件)
    """
    num_classes = 10
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

    return Dataset(x_train, y_train), Dataset(x_test, y_test)

if __name__ == '__main__':
    print('=====MAIN=====')
    print('dataset for mnist full-connected-layer model')
    train_dataset, test_dataset = load_mnist_data()
    print('[train dataset]')
    print("data: {}\tlabel: {}".format(train_dataset.data.shape, train_dataset.label.shape))
    print('[test dataset]')
    print("data: {}\tlabel: {}".format(test_dataset.data.shape, test_dataset.label.shape))
    
    print('dataset for mnist convolutional model')
    train_dataset_conv, test_dataset_conv = load_mnist_data_conv()
    print('[trian dataset]')
    print("data: {}\tlabel: {}".format(train_dataset_conv.data.shape, train_dataset_conv.label.shape))
    print('[test dataset]')
    print("data: {}\tlabel: {}".format(test_dataset_conv.data.shape, test_dataset_conv.label.shape))
