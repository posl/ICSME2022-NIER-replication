import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import BatchNormalization, Dropout, Activation
from keras.optimizers import RMSprop

import numpy as np
import os, sys

sys.path.append( "../" )
from libdeeppatcher.mnist_data import *

def make_model_mnist_batch_norm():
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        input_shape = ( 1, img_rows, img_cols )
    else:
        input_shape = ( img_rows, img_cols, 1 )

    model = Sequential()

    model.add( Conv2D( kernel_size=(3,3), filters=12, use_bias=False, padding='same', input_shape=input_shape) )
    model.add( BatchNormalization( center=True, scale=False ) ),
    model.add( Activation( 'relu' ) )

    model.add( Conv2D( kernel_size=(6,6), filters=24, use_bias=False, padding='same', strides=2 ) )
    model.add( BatchNormalization( center=True, scale=False ) )
    model.add( Activation( 'relu' ) )

    model.add( Conv2D( kernel_size=(6,6), filters=32, use_bias=False, padding='same', strides=2 ) )
    model.add( BatchNormalization( center=True, scale=False ) )
    model.add( Activation( 'relu' ) )

    model.add( Flatten() )

    model.add( Dense( 200, use_bias=False) )
    model.add( BatchNormalization( center=True, scale=False ) )
    model.add( Activation( 'relu' ) )

    model.add( Dropout(0.3) )
    model.add( Dense( 10, activation='softmax') )

    model.compile(
        optimizer=RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model

def train_model( x_train, y_train, x_val, y_val, batch_size=1, epochs=1 ):
    """学習したモデルを返す。
    cnnモデルを作成
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
    model = make_model_mnist_batch_norm()

    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val),
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
    # 再現性のためにconfig=session_confとしていたが、何故か学習できなくなってしまうので
    # とりあえず指定せずにすすめる
    session = tf.Session( graph=tf.get_default_graph(), config=None )
    K.set_session( session )

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

    """
    学習データ、テストデータの取得
    """
    (x_train, y_train), (x_test, y_test) = load_mnist_data_conv()

    model = train_mnist_batch_norm(
        x_train, y_train,
        x_test, y_test,
        batch_size=128,
        epochs=10,
    )

    score = model.evaluate( x_test, y_test, verbose=0 )
    print( "test accuracy: {}".format( score[1] ) )

    model.save( "./mnist_baatch_norm.hdf5" )