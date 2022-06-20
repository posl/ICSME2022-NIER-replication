import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras import backend as K

import tensorflow as tf

import sys, os
sys.path.append( "../" )
import numpy as np

from libdeeppatcher.cifar10_data import *

def build_model():
    model = Sequential()
    model.add( Conv2D(6, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', input_shape=(32, 32, 3) ) )
    model.add( MaxPooling2D((2, 2), strides=(2, 2)) )
    model.add( Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal' ) )
    model.add( MaxPooling2D((2, 2), strides=(2, 2)) )
    model.add( Flatten() )
    model.add( Dense(120, activation='relu', kernel_initializer='he_normal' ) )
    model.add( Dense(84, activation='relu', kernel_initializer='he_normal' ) )
    model.add( Dense(10, activation='softmax', kernel_initializer='he_normal' ) )
    sgd = optimizers.SGD( lr=.1, momentum=0.9, nesterov=True )
    model.summary()
    model.compile( loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'] )
    return model

def scheduler( epoch ):
    if epoch < 100:
        return 0.01
    if epoch < 150:
        return 0.005
    return 0.001

def train_model( x_train, y_train, x_val, y_val, batch_size=1, epochs=1 ):
    """モデルを学習して返す
    Args:
        x_train, y_train: 学習データ
        x_val, y_val: validationデータ
        batch_size (int): バッチサイズ
        epochs (int): エポック数
    Returns:
        moel: 訓練後のデータ
    """
    model = build_model()
    
    # set callback
    tb_cb = TensorBoard( log_dir='.lenet', histogram_freq=0 )
    change_lr = LearningRateScheduler( scheduler )
    cbks = [ change_lr, tb_cb ]
    
    model.fit(
        x_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        callbacks = cbks,
        verbose = 1,
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

    ## 訓練開始
    (x_train, y_train), (x_test, y_test) = load_data_cifar10()
    
    model = train_model(
        x_train, y_train,
        x_test, y_test,
        batch_size=128,
        epochs=10,
    )
    
    score = model.evaluate( x_test, y_test, verbose=0 )
    print( "test accuracy: {}".format( score[1] ) )
    model.save( "./data_hdf5/LeNet_keras.hdf5" )