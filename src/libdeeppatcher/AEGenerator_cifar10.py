from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import DeepFool
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import CarliniWagnerL2

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import os
import numpy as np

from . import utils

class AEGenerator:
    def __init__(self, model):
        self.clever_model = KerasModelWrapper( model )

    def create_adversarial_examples( self, data, session, AE_type, AE_option ):
        x = tf.placeholder( tf.float32, shape=( None, 784 ) )
        y = tf.placeholder( tf.float32, shape=( None, 10 ) )
        adv_model = AE_type( self.clever_model, sess=session )
        adv_gen = adv_model.generate(x, **AE_option )
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            adv_data = sess.run( adv_gen, feed_dict = { x: data } )
        return adv_data

    def create_adversarial_examples_for_convolution( self, data, session, AE_type, AE_option ):
        x = tf.placeholder( tf.float32, shape=( None, 32, 32, 3 ) )
        y = tf.placeholder( tf.float32, shape=( None, 10  ) )
        adv_model = AE_type( self.clever_model, sess=session )
        adv_gen = adv_model.generate(x, **AE_option )
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            adv_data = sess.run( adv_gen, feed_dict = { x: data } )
        return adv_data
