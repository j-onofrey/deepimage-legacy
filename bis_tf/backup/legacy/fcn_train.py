from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import math

import argparse
import numpy as np
from six.moves import xrange
import tensorflow as tf

# import our code
import bis_tf_utils as bisutil
import bis_tf_legacy as bislegacy


FLAGS = tf.app.flags.FLAGS


def train(training_data):
    print('+++++\n+++++ Training batch size = %d steps=%d patch_size=%d' % (tf.app.flags.FLAGS.batch_size, tf.app.flags.FLAGS.max_steps,FLAGS.patch_size))

    patch_size=FLAGS.patch_size
    use_regression=FLAGS.use_regression
    edge_smoothness=FLAGS.smoothness
    num_classes=2;
    if use_regression:
        num_classes=1

    do_edge=False;
    if edge_smoothness > 0.0:
        do_edge=True;

    
    with tf.Graph().as_default():


        # Get the images and their labels for the images (Fix 3D)
        images_pointer, labels_pointer = bisutil.placeholder_inputs(patch_size)
        
        # Build a Graph that computes the logits predictions from the inference model
        output_dict = bislegacy.fcn_inference(inputs=images_pointer,
                                              keep_probability=0.85,
                                              num_classes=num_classes,
                                              do_edge=do_edge)
        
        # Calculate the loss.
        normalizer_scalar=bisutil.compute_normalizer(images_pointer,FLAGS.batch_size)
        loss_op = bisutil.compute_loss(pred_image=output_dict["image"],
                                       logits=output_dict["logits"],
                                       edgelist=output_dict["edgelist"],
                                       use_l2=use_regression,
                                       labels=labels_pointer,
                                       smoothness=edge_smoothness,
                                       normalizer=normalizer_scalar)

        # Build a Graph that trains the model with one batch of examples and 
        # updates the model parameters.
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads = optimizer.compute_gradients(loss_op, tf.trainable_variables())
        train_op=optimizer.apply_gradients(grads)

        bisutil.run_training(
            training_data=training_data,
            images_pointer = images_pointer,
            labels_pointer = labels_pointer,
            loss_op = loss_op,
            train_op=train_op,
            train_dir=FLAGS.train_dir,
            batch_size=FLAGS.batch_size,
            max_steps=FLAGS.max_steps,
            model_name="fcn.ckpt",
            patch_size=FLAGS.patch_size,
            distorted=True)
        

def main(argv=None):
    # Load the input data into memory
    FLAGS = tf.app.flags.FLAGS
    print('+++++')
    
    training_data = bisutil.load_training_data(FLAGS.input_data, FLAGS.label_data);
    print('+++++')
    print('+++++ Training model output=',os.path.abspath(FLAGS.train_dir))
    train(training_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a fully convolutional NN for 2D images.')
    parser.add_argument('-o','--output', help='Directory for saving the learned model')    
    parser.add_argument('-i','--input', nargs=2, help='txt files containing list of image and label data files')

    parser.add_argument('-g','--usegpu', help='Train using GPU', action='store_true')
    parser.add_argument('-s','--steps', type=int, help='Number of steps (iterations) -- mapped to closest multiple of 10', default=300)
    parser.add_argument('-b','--batch_size', type=int, help='Number of training samples per mini-batch -- mapped to closest power of two', default=16)
    parser.add_argument('-p','--patch_size', type=int, help='patch size -- mapped to closest power of two', default=64)
    parser.add_argument('-r','--regression', help='Train using regression', action='store_true')
    parser.add_argument('-l','--smoothness', help='Smoothness factor (Baysian training with edgemap)', default=0.0)
    parser.add_argument('-d','--debug', help='Extra Debug Output', action='store_true')
        
    args = parser.parse_args()

    tf.app.flags.DEFINE_string('train_dir', args.output,"""Directory where to write event logs and checkpoint.""")
    tf.app.flags.DEFINE_string('input_data', args.input[0],"""File listing training data samples.""")
    tf.app.flags.DEFINE_string('label_data', args.input[1],"""File listing training label data samples.""")
    tf.app.flags.DEFINE_boolean('use_gpu',args.usegpu,"""Train using GPU.""")
    tf.app.flags.DEFINE_boolean('debug',args.debug,"""Enable debug output.""")
    tf.app.flags.DEFINE_integer('max_steps', args.steps,"""Number of batches to run.""")
    tf.app.flags.DEFINE_integer('batch_size', bisutil.getpoweroftwo(args.batch_size,4,1024),"""Number of images to process in a training batch.""")
    tf.app.flags.DEFINE_integer('patch_size', bisutil.getpoweroftwo(args.patch_size,8,128),"""Size of patch.""")
    tf.app.flags.DEFINE_boolean('use_regression', args.regression,"""Use Regression.""")
    tf.app.flags.DEFINE_float('smoothness', args.smoothness,"""Smoothness Factor""")

    tf.app.run()



