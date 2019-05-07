from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import re
import time
import math
import numpy as np

import tensorflow as tf
import bis_tf_utils as bisutil
import bis_image_patchutil as putil

from six.moves import xrange
from datetime import datetime
FLAGS = tf.app.flags.FLAGS


# -----------------------------------------------------------------------------------
#
# A Complete model (fcn)
#
# -----------------------------------------------------------------------------------

def fcn_inference(inputs, keep_probability=0.85,filter_size=3, num_filters=32,
                  num_connected=512,num_classes=2,max_outputs=6,do_edge=False):
    """Build the segmentation model 
    """

    # Store Parameters
    with tf.variable_scope('Parameters') as scope:
        imgshape=inputs.get_shape()
        patchsize=imgshape[1].value
        num_frames=imgshape[len(imgshape)-1].value;
        print('_____ Inputs shape: %s' % (imgshape,))
        print('_____ filter_size = %d, num_filters= %d, num_connected=%d, num_classes=%d patchsize=%d num_frames=%d'
              % (filter_size,num_filters,num_connected,num_classes,patchsize,num_frames))
        tf.Variable(filter_size,name='filter_size',trainable=False)
        tf.Variable(num_filters,name='num_filters',trainable=False)
        tf.Variable(num_connected,name='num_connected',trainable=False)
        tf.Variable(num_classes,name='num_classes',trainable=False)
        tf.Variable(patchsize,name='patch_size',trainable=False)
        

    # Create Model
    with tf.variable_scope('Inference') as scope:
        
        with tf.variable_scope('c1'):
            if FLAGS.debug:
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
            if FLAGS.debug:
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
            if FLAGS.debug:
                print('_____\n_____ Convolution Layer 3');
            relu3=bisutil.createConvolutionLayerRELU(input=pool2,
                                                     name='convolution_3_1',
                                                     shape=[ filter_size,filter_size,num_filters*2,num_filters*4],
                                                     padvalue=-1);
            pool3=bisutil.createConvolutionLayerRELUPool(input=relu3,
                                                         mode='max',
                                                         name='convolution_3_2',
                                                         shape=[ filter_size,filter_size,num_filters*4,num_filters*4],
                                                         padvalue=0);
            
        # Fully connected layers
        with tf.variable_scope('f1'):
            if FLAGS.debug:
                print('_____\n_____ Fully Connected Layer 1');
            pool3shape=pool3.get_shape()
            fc1=bisutil.createFullyConnectedLayerWithDropout(input=pool3,
                                                             name='fullyconnected_1',
                                                             shape=[pool3shape[1].value,
                                                                    pool3shape[2].value,
                                                                    num_filters*4,
                                                                    num_connected],
                                                             keep_prob=keep_probability)

        with tf.variable_scope('f2'):
            if FLAGS.debug:
                print('_____\n_____ Fully Connected Layer 2');
            fc2=bisutil.createFullyConnectedLayerWithDropout(input=fc1,
                                                             name='fullyconnected_2',
                                                             shape=[1,1,num_connected,num_connected],
                                                             keep_prob=keep_probability)

        with tf.variable_scope('f3'):
            if FLAGS.debug:
                print('_____\n_____ Fully Connected Layer 3');
            fc3=bisutil.createFullyConnectedLayer(input=fc2,
                                          name='fullyconnected_3',
                                          shape=[1,1,num_connected,num_connected])
            
        # Deconvolve (upscale the image)
        with tf.variable_scope('d1'):
            if FLAGS.debug:
                print('_____\n_____ Deconv Layer 1');
                
            deconv_shape1 = pool2.get_shape()
            fuse1=bisutil.createDeconvolutionLayerFuse(input=fc3,
                                               fuse_input=pool2,
                                               shape=[4,4, deconv_shape1[3].value,num_connected ],
                                               name="deconv_1")
            

        with tf.variable_scope('d2'):
            deconv_shape2 = pool1.get_shape()
            if FLAGS.debug:
                print('_____\n_____ Deconv Layer 2');
            fuse2=bisutil.createDeconvolutionLayerFuse(input=fuse1,
                                               fuse_input=pool1,
                                               shape=[4,4,deconv_shape2[3].value,deconv_shape1[3].value],
                                               name="deconv_2")


        with tf.variable_scope('d3'):
            if FLAGS.debug:
                print("_____\n_____ Deconv layer 3")
            shape = inputs.get_shape()
            out_shape = tf.stack([tf.shape(inputs)[0], shape[1].value, shape[2].value, num_classes])
            deconv3=bisutil.createDeconvolutionLayer(input=fuse2,
                                             name="deconv_3",
                                             input_shape=[4,4,num_classes,deconv_shape2[3].value],
                                             output_shape=out_shape);

        if FLAGS.debug:
            print("_____\n")

            
            
        if num_classes>1:
            print("_____  Annotation (in classification mode)")
            dim=len(deconv3.get_shape())-1
            annotation_pred = tf.argmax(deconv3, dimension=dim, name='prediction')
            output= { "image": tf.expand_dims(annotation_pred, dim=dim),
                      "logits": deconv3,
                      "edgelist": None}
        else:
            print("_____  In Regression mode")
            output = { "image": deconv3, "logits": deconv3, "edgelist" : None };
            
        tf.summary.image('pred_labels', tf.cast(output['image'], tf.float32), max_outputs=max_outputs)
            
        if do_edge:
            outimg=output['image']
            if (num_classes>1):
                outimg=tf.cast(output['image'],tf.float32)
            shapelength=len(outimg.get_shape())
            print('_____  Adding back end smoothness compilation, shapelength=',shapelength,' numclasses=',num_classes)
            if shapelength!=5:
                grad_x = np.zeros([3, 1, 1, 1])
                grad_x[ 0, 0, : , :] = -1
                grad_x[ 1, 0, : , :] =  2
                grad_x[ 2, 0, : , :] = -1
                
                grad_y = np.zeros([1, 3, 1, 1])
                grad_y[ 0, 0, : , : ] = -1
                grad_y[ 0, 1, : , : ] =  2
                grad_y[ 0, 2, : , : ] = -1
                
                edge_conv_x = tf.nn.conv2d(outimg,grad_x,strides=[1,1,1,1],padding='SAME',name='smoothness_final_x');
                edge_conv_y = tf.nn.conv2d(outimg,grad_y,strides=[1,1,1,1],padding='SAME',name='smoothness_final_y');
                output['edgelist']= [ edge_conv_x,edge_conv_y ]
            else:
                grad_x = np.zeros([3, 1, 1, 1, 1])
                grad_x[2, 0, 0, :, : ] = -1
                grad_x[1, 0, 0, :, : ] =  2
                grad_x[0, 0, 0, :, : ] = -1
                
                grad_y = np.zeros([1, 3, 1, 1, 1])
                grad_y[0, 0, 0, :, : ] = -1
                grad_y[0, 1, 0, :, : ] =  2
                grad_y[0, 2, 0, :, : ] = -1
                
                grad_z = np.zeros([1, 1, 3, 1, 1])
                grad_z[0, 0, 0, :, : ] = -1
                grad_z[0, 0, 1, :, : ] =  2
                grad_z[0, 0, 2, :, : ] = -1
                
                conv_x = tf.nn.conv3d(outimg,grad_x,strides=[1,1,1,1,1],padding='SAME',name='smoothness_final_x');
                conv_y = tf.nn.conv3d(outimg,grad_y,strides=[1,1,1,1,1],padding='SAME',name='smoothness_final_y');
                conv_z = tf.nn.conv3d(outimg,grad_z,strides=[1,1,1,1,1],padding='SAME',name='smoothness_final_z');
                output['edgelist']= [ edge_conv_x,edge_conv_y,conv_z ]
        else:
            print('_____  Not adding back end smoothness compilation')
        return output
    
