from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import re
import numpy as np
import tensorflow as tf
import BisTensorFlowUtils as bisutil

from six.moves import xrange

FLAGS = tf.app.flags.FLAGS


# A Complete model

def fcn_inference(inputs, keep_prob,filter_size=3, num_filters=32,num_connected=512,num_classes=2):
    """Build the segmentation model 
    """

    
    with tf.variable_scope('inference') as scope:


        # Get this from input
        num_frames=1;
        
        print('_____ Input:')
        print('_____ inputs shape: %s' % (inputs.get_shape(),))
        print('_____ filter_size = %d, num_filters= %d, num_connected=%d, num_classes=%d'
              % (filter_size,num_filters,num_connected,num_classes))

        
        with tf.variable_scope('c1'):
            print('_____\n_____ Convolution Layer 1');
            relu1=bisutil.createConvolutionLayerRELU(input=inputs,
                                             name='convolution_1_1',
                                             shape=[ filter_size,filter_size,num_frames,num_filters],
                                             padvalue=-1);
            pool1=bisutil.createConvolutionLayerRELUPool(input=relu1,
                                                         mode='max',
                                                         name='convolution_1_2',
                                                         shape=[ filter_size,filter_size,num_filters,num_filters],
                                                         padvalue=0);
            
        with tf.variable_scope('c2'):
            print('_____\n_____ Convolution Layer 2');
            relu2=bisutil.createConvolutionLayerRELU(input=pool1,
                                             name='convolution_2_1',
                                             shape=[ filter_size,filter_size,num_filters,num_filters*2],
                                             padvalue=-1);
            pool2=bisutil.createConvolutionLayerRELUPool(input=relu2,
                                                         mode='max',
                                                         name='convolution_2_2',
                                                         shape=[ filter_size,filter_size,num_filters*2,num_filters*2],
                                                         padvalue=0);
            
        with tf.variable_scope('c3'):
            print('_____\n_____ Convolution Layer 3');
            relu3=bisutil.createConvolutionLayerRELU(input=pool2,
                                                     name='convolution_3_1',
                                                     shape=[ filter_size,filter_size,num_filters*2,num_filters*4],
                                                     padvalue=-1);
            pool3=bisutil.createConvolutionLayerRELUPool(input=relu3,
                                                         mode='max',
                                                         name='convolution_3_3',
                                                         shape=[ filter_size,filter_size,num_filters*4,num_filters*4],
                                                         padvalue=0);
            
        # Fully connected layers
        with tf.variable_scope('f1'):
            print('_____\n_____ Fully Connected Layer 1');
            pool3shape=pool3.get_shape()
            fc1=bisutil.createFullyConnectedLayerWithDropout(input=pool3,
                                                     name='fullyconnected_1',
                                                     shape=[pool3shape[1].value,
                                                            pool3shape[2].value,
                                                            num_filters*4,
                                                            num_connected],
                                                     keep_prob=keep_prob)

        with tf.variable_scope('f2'):            
            print('_____\n_____ Fully Connected Layer 2');
            fc2=bisutil.createFullyConnectedLayerWithDropout(input=fc1,
                                                     name='fullyconnected_2',
                                                     shape=[1,1,num_connected,num_connected],
                                                     keep_prob=keep_prob)

        with tf.variable_scope('f3'):                        
            print('_____\n_____ Fully Connected Layer 3');
            fc3=bisutil.createFullyConnectedLayer(input=fc2,
                                          name='fullyconnected_3',
                                          shape=[1,1,num_connected,num_connected])
            
        # Deconvolve (upscale the image)
        with tf.variable_scope('d1'):
            print('_____\n_____ Deconv Layer 1');
                
            deconv_shape1 = pool2.get_shape()
            fuse1=bisutil.createDeconvolutionLayerFuse(input=fc3,
                                               fuse_input=pool2,
                                               shape=[4,4, deconv_shape1[3].value,num_connected ],
                                               name="deconv_1")
            

        with tf.variable_scope('d2'):
            deconv_shape2 = pool1.get_shape()
            print('_____\n_____ Deconv Layer 2');
            fuse2=bisutil.createDeconvolutionLayerFuse(input=fuse1,
                                               fuse_input=pool1,
                                               shape=[4,4,deconv_shape2[3].value,deconv_shape1[3].value],
                                               name="deconv_2")


        with tf.variable_scope('d3'):
            
            print("_____\n_____ Deconv layer 3")
            shape = inputs.get_shape()
            out_shape = tf.stack([tf.shape(inputs)[0], shape[1].value, shape[2].value, num_classes])
            deconv3=bisutil.createDeconvolutionLayer(input=fuse2,
                                             name="deconv_3",
                                             input_shape=[4,4,num_classes,deconv_shape2[3].value],
                                             output_shape=out_shape);

        if num_classes>1:
            print("_____\n_____ Annotation")    
            annotation_pred = tf.argmax(deconv3, dimension=3, name='prediction')
            return tf.expand_dims(annotation_pred, dim=3), deconv3

        print("_____\n_____ Regression: No annotation")    
        return None,deconv3
