"""
CIFAR10のデータセット取得用
"""
from keras.datasets import cifar10
import keras
from .types import Dataset

def load_dataset():
    """
    cifar10のデータを読み込む。正規化とone-hot化済みのデータを返す。
    x_train.shape: (50000, 32, 32, 3)       y_train.shape: (50000, 10)
    x_test.shape:  (10000, 32, 32, 3)       y_test.shape:  (10000, 10)
    """
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.np_utils.to_categorical(y_test,  num_classes)

    return Dataset(x_train, y_train), Dataset(x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data_cifar10()
    print("x_train.shape: {}\ty_train.shape: {}".format(x_train.shape, y_train.shape))
    print("x_test.shape:  {}\ty_test.shape:  {}".format(x_test.shape, y_test.shape))

