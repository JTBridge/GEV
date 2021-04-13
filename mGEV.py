'''
Activation functions for highly unbalanced data:
- GEV can be used for binary classification to replace the sigmoid activtion
- mGEV can be used for multiclass classification to replace the softmax activtion

Please cite:
J Bridge, Y Meng, Y Zhao, Y Du, M Zhao, R Sun, Y Zheng,
Introducing the GEV Activation Function for Highly Unbalanced Data to Develop COVID-19 Diagnostic Models,
IEEE Journal of Biomedical and Health Informatics, 
vol. 24, no. 10, pp. 2776-2786, Oct. 2020, 
doi: 10.1109/JBHI.2020.3012383.

https://github.com/JTBridge/GEV
joshua.bridge@liverpool.ac.uk

License: Apache License 2.0
'''

import tensorflow as tf
from tensorflow.keras import layers, models, backend, applications, regularizers, initializers
from tensorflow.keras.layers import Layer
import math
import numpy as np
from tensorflow.python.ops import math_ops

 
class GEV(Layer):
    def __init__(self, **kwargs):
        super(GEV, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(1,),
                                  initializer=tf.constant_initializer(0.),
                                  trainable=True,
                                  dtype='float32')
        self.sigma = self.add_weight(name='sigma',
                                     shape=(1,),
                                     initializer=tf.constant_initializer(1.),
                                     trainable=True,
                                     dtype='float32')
        self.xi = self.add_weight(name='xi',
                                  shape=(1,),
                                  initializer=tf.constant_initializer(0.),
                                  trainable=True,
                                  dtype='float32')
        super(GEV, self).build(input_shape)
 
    def call(self, x):

        sigma = backend.maximum(backend.epsilon(), self.sigma)  # sigma<0 doesn't make sense

        # Type 1: For xi = 0 (Gumbel)
        def t1(x=x, mu=self.mu, sigma=sigma):
            return backend.exp(-backend.exp(-(x-self.mu)/sigma))

        # Type 2: For xi>0 (Frechet) or xi<0 (Reversed Weibull) 
        def t23(x=x, mu=self.mu, sigma=sigma, xi=self.xi):
            y = (x - mu) / sigma
            y = xi*y
            y = tf.maximum(tf.constant(-1.), y)
            y = backend.exp(-tf.pow( tf.constant(1.) + y, -tf.constant(1.)/xi))        
            return y 

        GEV = tf.cond(backend.equal(tf.constant(0.), self.xi), t1, t23) # This chooses the type based on xi
        return GEV 

    def compute_output_shape(self, input_shape):
        return input_shape


class mGEV(Layer):

    def __init__(self, **kwargs):
        super(mGEV, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(input_shape[-1],),
                                  initializer=tf.constant_initializer(0.),
                                  trainable=True,
                                  dtype='float32')
        self.sigma = self.add_weight(name='sigma',
                                     shape=(input_shape[-1],),
                                     initializer=tf.constant_initializer(1.),
                                     trainable=True,
                                     dtype='float32')
        self.xi = self.add_weight(name='xi',
                                  shape=(1,),
                                  initializer=tf.constant_initializer(0.1),
                                  regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
                                  trainable=True,
                                  dtype='float32')
        super(mGEV, self).build(input_shape)
 
    def call(self, x):
        mu = backend.cast(self.mu, 'float32')
        sigma = backend.cast(self.sigma, 'float32')
        xi = backend.cast(self.xi, 'float32')
        x = backend.cast(x, 'float32')
        x = tf.clip_by_value(x, -20, 20) 

        sigma = backend.maximum(backend.epsilon(), sigma)
        

        # Type 1: For xi = 0 (Gumbel)
        def t1(x=x, mu=mu, sigma=sigma, xi=xi):
            return backend.exp(-backend.exp(-(x-mu)/sigma)) 

        # Type 2: For xi>0 (Frechet) or xi<0 (Reversed Weibull) 
        def t23(x=x, mu=mu, sigma=sigma, xi=xi):        
            y = (x - mu) / sigma
            y = xi*y
            tf.debugging.assert_all_finite(y, 'xi*y',name=None)
            y = tf.maximum(tf.constant(-1., dtype='float32'), y)
            y = backend.exp(-tf.pow( tf.constant(1., dtype='float32') + y, -tf.constant(1., dtype='float32')/xi))       
            return y 

        mGEV = tf.cond(backend.equal(tf.constant(0., dtype='float32'), backend.cast(xi, 'float32')), t1, t23) 
        mGEV = mGEV/tf.math.reduce_sum(mGEV)
        return mGEV 

    def compute_output_shape(self, input_shape):
        return input_shape
