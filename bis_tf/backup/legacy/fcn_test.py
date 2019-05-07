from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys

import argparse
import nibabel as nib
import numpy as np
from six.moves import xrange

import tensorflow as tf


import bis_tf_utils as bisutil
import bis_tf_legacy as bislegacy


tf.app.flags.DEFINE_boolean('log_device_placement', False,"""Whether to log device placement.""")

def reconstruct(test_image):
        
    FLAGS = tf.app.flags.FLAGS
    num_classes=2
    if (FLAGS.use_regression):
        num_classes=1

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        cname=ckpt.model_checkpoint_path
        patch_width=int(bisutil.get_tensor_from_checkpoint_file(cname,'Parameters/patch_size',False).item())
        num_classes=int(bisutil.get_tensor_from_checkpoint_file(cname,'Parameters/num_classes',False).item())
        num_connected=int(bisutil.get_tensor_from_checkpoint_file(cname,'Parameters/num_connected',False).item())
        num_filters=int(bisutil.get_tensor_from_checkpoint_file(cname,'Parameters/num_filters',False).item())
        filter_size=int(bisutil.get_tensor_from_checkpoint_file(cname,'Parameters/filter_size',False).item())
    else:
        print('No checkpoint file found in',FLAGS.train_dir)
        sys.exit(0)
        
    print("+++++ Initializing the FCN model...")
    with tf.Graph().as_default():
            
        # Get the images and their labels for the images (3D!!!)
        images_pointer, _ = bisutil.placeholder_inputs(patch_width)

        # Build a Graph that computes the logits predictions from the inference model
        output_dict = bislegacy.fcn_inference(inputs=images_pointer,
                                              keep_probability=1.0,
                                              num_classes=num_classes,
                                              filter_size=filter_size,
                                              num_filters=num_filters,
                                              num_connected=num_connected)

        print('\n\n\n\n')
        print('+++++ Read Parameters filter_size = %d, num_filters= %d, num_connected=%d, num_classes=%d patchsize=%d'
              % (filter_size,num_filters,num_connected,num_classes,patch_width))
            
        return bisutil.reconstruct_image(checkpointname=cname,
                                          test_image=test_image,
                                          images_pointer=images_pointer,
                                          model_output_image=output_dict['image'],
                                          patch_width=patch_width,
                                          batch_size=FLAGS.batch_size);
            
def main(argv=None):
    # Load the input data into memory
    FLAGS = tf.app.flags.FLAGS
    
    print('+++++ Loading data...')
    test_image = bisutil.load_single_image(FLAGS.input_data);
    print('Loaded test image with shape: %s' % (test_image.shape, ))

    print('+++++ Reconstructing...')
    result_image = reconstruct(test_image)
    
    bisutil.save_single_image(result_image, output_path=FLAGS.output_path)
    print('+++++ All done.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a fully convolutional NN for 2D images.')
    parser.add_argument('model_dir', nargs=1, help='Directory where to find the learned model')    
    parser.add_argument('input_data', nargs=1, help='Input image to segment (NIfTI image)')
    parser.add_argument('output_path', nargs=1, help='Output classification result (NIfTI image)')
    parser.add_argument('-g','--usegpu', help='Train using GPU', action='store_true')
    parser.add_argument('-b','--batch_size', type=int, help='Number of training samples per mini-batch -- mapped to closest power of two', default=64)
    parser.add_argument('-r','--regression', help='Train using regression', action='store_true')
    parser.add_argument('-d','--debug', help='Extra Debug Output', action='store_true')
    args = parser.parse_args()


    tf.app.flags.DEFINE_string('train_dir', args.model_dir[0],
                           """Directory where to write event logs and checkpoint.""")
    tf.app.flags.DEFINE_string('input_data', args.input_data[0],
                           """File listing training data samples.""")
    tf.app.flags.DEFINE_string('output_path', args.output_path[0],
                           """NIfTI image output destination.""")
    tf.app.flags.DEFINE_boolean('use_gpu',args.usegpu,"""Train using GPU.""")
    tf.app.flags.DEFINE_integer('batch_size', bisutil.getpoweroftwo(args.batch_size,4,256),"""Number of images to process in a training batch.""")
    tf.app.flags.DEFINE_boolean('use_regression', args.regression,"""Use Regression.""")

    tf.app.flags.DEFINE_boolean('debug',args.usegpu,"""Enable debug output.""")
    tf.app.run()



