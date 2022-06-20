import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras import backend as K

import tensorflow as tf

import numpy as np
import os, sys, h5py

sys.path.append( "../" )
from libdeeppatcher.mnist_data import *

def make_model_mnist_dense_dropout():
    
    num_classes = 10

    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'],
    )
    
    return model

def train_model( x_train, y_train, x_val, y_val, batch_size=1, epochs=1, validation_split=0 ):
    """学習したモデルを返す。
    Args:
        x_train, y_trian: 学習データ
        x_val, y_val: validationデータ
        batch_size (int): バッチサイズ
        epochs (int): エポック数
        validation_split (float):
            学習データのうち何％をvalidationデータに使用するかを指定。
            x_val, y_valを指定した場合は無効。
    Returns:
        model : 学習済みモデル
    """
    model = make_model_mnist_dense_dropout()

    model.fit(
        x_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = 1,
        validation_split = validation_split,
        validation_data = ( x_val, y_val )
    )

    return model

if __name__ == '__main__':
    """
    学習の再現性のために環境を設定
    """
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed( 7 ) # for Reproducibility
    # GPU configulations
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads = 1,
        inter_op_parallelism_threads = 1
    )

    tf.set_random_seed( 7 )
    session = tf.Session( graph = tf.get_default_graph(), config = None )
    K.set_session( session )

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

    """
    学習データ、テストデータの取得
    """
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    model = train_mnist_dense_dropout(
        x_train, y_train,
        x_test, y_test,
        batch_size=128,
        epochs=10,
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    print("test accuracy: {}".format(score[1]))

    model.save("./mnist_dense_dropout.hdf5")
