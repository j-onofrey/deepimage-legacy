from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import re
import numpy as np
import tensorflow as tf
import nibabel as nib

from six.moves import xrange
import ImagePatchUtil as putil

FLAGS = tf.app.flags.FLAGS

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'
DEVICE_CPU = '/cpu:0'
DEVICE_GPU = '/gpu:0'



def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))



def _variable_on_device(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    DEVICE = DEVICE_CPU
    if FLAGS.use_gpu:
        DEVICE = DEVICE_GPU

    with tf.device(DEVICE):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var



def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
    Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_device(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var




def createConvolutionLayer(input,name,shape,padvalue=-1,stdev=5e-2,wd=0.0,indent=""):
   """Create a Convolution Layer

   Returns:
      convolution layer and biases to add to next layer
   
   padvalue = -1 -- create padding based on filter size
   padvalue =  0 -- no padding
   """

   wname='w_'+name;
   bname='b_'+name;

   shapelength=len(shape)
   threed=False;
   if shapelength==5:
       threed=True
   
   num_layers= shape[shapelength-1]
   width=shape[0]-1
   height=shape[1]-1
   depth=0
   if threed==True:
       depth=shape[2]-1

   padding=input;
   if padvalue>0:
       width=padvalue
       height=padvalue
       depth=padvalue

   if threed==False:
        pad_values=[[0,0],[width,width],[height,height],[0,0]]
   else:
        pad_values=[[0,0],[width,width],[height,height],[depth,depth],[0,0]]
       
   if padvalue !=0 :
       padding = tf.pad(input, paddings=pad_values, mode='SYMMETRIC', name='padding_'+name)
       print('_____',indent,'Adding padding :',pad_values)
   else:
       print('_____',indent,'Not adding padding.')
              

   
   conv_w = _variable_with_weight_decay(wname, shape=shape, stddev=stdev,wd=wd)
   conv_b = _variable_on_device(bname, [num_layers], tf.constant_initializer(0.0))
   if threed:
      conv = tf.nn.conv3d(padding, conv_w, strides=[1, 1, 1, 1, 1], padding='VALID',name="conv3d_"+name)
   else:
      conv = tf.nn.conv2d(padding, conv_w, strides=[1, 1, 1, 1], padding='VALID',name="conv2d_"+name)

   print('_____',indent,'Creating convolution: num_layers = ',num_layers,' shape=',shape,
         ' input_data_shape=',padding.get_shape(),
         '%s shape: %s' % (wname,conv.get_shape()))
   return [ conv,conv_b ]


def createConvolutionLayerRELU(input,name,shape,padvalue=-1,stdev=5e-2,wd=0.0,indent=""):


   print('_____',indent,'Creating convolution+relu ',name)
   conv,conv_b=createConvolutionLayer(input=input,
                                      name=name,
                                      shape=shape,
                                      padvalue=padvalue,
                                      stdev=stdev,
                                      wd=wd,
                                      indent=indent+'    ')
   
   relu = tf.nn.relu( tf.nn.bias_add(conv, conv_b), name='relu_'+name)
   _activation_summary(relu)
   return relu


def createConvolutionLayerRELUPool(input,name,shape,mode='max',
                                   padvalue=-1,in_ksize=2,in_strides=2,
                                   stdev=5e-2,wd=0.0,indent=""):

   print('_____',indent,'Creating convolution+relu+maxpool ',name)
   
   relu=createConvolutionLayerRELU(input=input,
                                   name=name,
                                   shape=shape,
                                   padvalue=padvalue,
                                   stdev=stdev,
                                   wd=wd,
                                   indent=indent+'    ')


   shapelength=len(shape)
   if shapelength==5:
       strides=[1,in_strides,in_strides,in_strides,1]
       ksize=[1,in_ksize,in_ksize,1,1 ]
   else:
       strides=[1,in_strides,in_strides,1]
       ksize=[1,in_ksize,in_ksize,1]
       

   if mode=='max':
        if shapelength==5:
            pool = tf.nn.max_pool3d(relu, ksize=ksize, strides=strides, padding='VALID', name='maxpool_'+name)
        else:
            pool = tf.nn.max_pool(relu, ksize=ksize, strides=strides, padding='VALID', name='maxpool_'+name)
   else:
       mode='avg'
       if shapelength==5:
           pool = tf.nn.avg_pool3d(relu, ksize=ksize, strides=strides, padding='VALID', name='maxpool_'+name)
       else:
           pool = tf.nn.avg_pool(relu, ksize=ksize, strides=strides, padding='VALID', name='maxpool_'+name)

   print('_____',indent,'    Adding maxpool shape: ',pool.get_shape(),' strides=',strides,' ksize=',ksize)
   return pool




def createFullyConnectedLayer(input,name,shape,stdev=5e-2,wd=0.0,indent=""):

    print('_____',indent,'Creating fully connected layer ',name, ' shape=',shape,' input_data shape=',input.get_shape())
    shapelength=len(shape)
    num_layers= shape[shapelength-1]
    threed=False;
    if shapelength==5:
        threed=True

    
    fc1_w = _variable_with_weight_decay('w_'+name, shape=shape, stddev=stdev, wd=wd)
    fc1_b = _variable_on_device('b_'+name, shape=[num_layers], initializer=tf.constant_initializer(0.0))

    if threed==False:
        fc1_conv1 = tf.nn.conv2d(input, fc1_w, strides=[1,1,1,1], padding='SAME',name='conv2d_'+name)
    else:
        fc1_conv1 = tf.nn.conv3d(input, fc1_w, strides=[1,1,1,1,1], padding='SAME',name='conv3d_'+name)
        
    fc1_relu1 = tf.nn.relu(tf.nn.bias_add(fc1_conv1, fc1_b), name='relu1_'+name)
    _activation_summary(fc1_relu1)
    if indent=="":
        print('_____      ',name,':',fc1_relu1.get_shape())
    return fc1_relu1


def createFullyConnectedLayerWithDropout(input,name,shape,keep_prob,stdev=5e-2,wd=0.0,indent=""):


    print('_____',indent,'Creating fully connected + dropout layer ',name)
    fc1_relu1=createFullyConnectedLayer(input=input,
                                        name=name,
                                        shape=shape,
                                        stdev=stdev,
                                        wd=wd,
                                        indent=indent+'    ')

    fc1_dropout = tf.nn.dropout(fc1_relu1, keep_prob=keep_prob,name='dropout_'+name)
    print('_____',indent,' ',name,':',fc1_dropout.get_shape(),' with dropout')
    
    return fc1_dropout


def createDeconvolutionLayer(input,name,output_shape,input_shape,
                             in_strides=2,stdev=5e-2,wd=0.0,indent=""):
    
    wname=name+'_w';
    bname=name+'_b';

    shapelength=len(input_shape)
    threed=False;
    if shapelength==5:
        threed=True
        strides=[1,in_strides,in_strides,in_strides,1]
    else:
       strides=[1,in_strides,in_strides,1]
        
    num_layers= input_shape[shapelength-2]

    deconv1_w = _variable_with_weight_decay('w_'+name,
                                            shape=input_shape,
                                            stddev=5e-2, wd=0.0)

    deconv1_b = _variable_on_device('b_'+name,
                                    shape=[ num_layers ],
                                    initializer=tf.constant_initializer(0.0))


    if threed==False:
        deconv1_conv = tf.nn.conv2d_transpose(input,
                                              deconv1_w,
                                              output_shape=output_shape,
                                              strides=strides, padding='SAME',
                                              name='transpose2d_'+name)
    else:
        deconv1_conv = tf.nn.conv3d_transpose(input,
                                              deconv1_w,
                                              output_shape=output_shape,
                                              strides=strides,
                                              padding='SAME',
                                              name='transpose3d_'+name)
        
    deconv1 = tf.nn.bias_add(deconv1_conv, deconv1_b)
    if indent=="":
        print('_____',indent,'Creating deconvolution ',name, ' shape=',deconv1.get_shape(),' strides=',strides)
    else:
        print('_____ ',indent,name,':',deconv1.get_shape(),' strides=',strides)
    return deconv1

def createDeconvolutionLayerFuse(input,fuse_input,shape,name,
                                 in_strides=2,stdev=5e-2,wd=0.0,indent=""):

  
   deconv1=createDeconvolutionLayer(input=input,
                                    name=name,
                                    input_shape=shape,
                                    output_shape=tf.shape(fuse_input),
                                    in_strides=in_strides,
                                    stdev=stdev,
                                    wd=wd,
                                    indent=indent+"    ");
   
   fuse1 = tf.add(deconv1, fuse_input, name='fuse_'+name)
   _activation_summary(fuse1)
   print('_____',indent,'Creating deconvolution+fuse ',name, ' shape=',fuse1.get_shape())
   return fuse1



def get_batch(TRAINING_DATA, batch_size, distorted=False,image_size=64):
    image_training_data = TRAINING_DATA[0]
    label_training_data = TRAINING_DATA[1]


    if len(image_training_data) != len(label_training_data):
        raise ValueError('Image and label data counts do not match')

    # Get the random sample indexing
    num_samples = len(image_training_data)
    sample_idx = (((num_samples-1)*np.random.rand(batch_size))+0.5).astype(int)
    # print('Sampling index: %s' % (sample_idx,))

    patch_size = (image_size, image_size, 1)

    images = np.zeros((batch_size,image_size, image_size, 1), dtype=np.float32)
    labels = np.zeros((batch_size,image_size, image_size, 1), dtype=np.uint8)
    flip_count = 0
    for i in range(0,batch_size):
        # Only get a single patch at a time
        # [p1, idx] = image_training_data[sample_idx[i]].getPatches(1, dtype=np.float32)
        [p1, idx] = putil.getRandomPatches(image_training_data[sample_idx[i]], patch_size, num_patches=1, dtype=np.float32)
        # [p2, idx] = label_training_data[sample_idx[i]].getPatchesFromIndexes(idx, dtype=np.uint8)
        p2 = putil.getPatchesFromIndexes(label_training_data[sample_idx[i]], idx, patch_size, dtype=np.uint8)
        # print('Get patch index: %s' % (idx,))
        images[i,:,:,0] = np.squeeze(p1, 3)
        labels[i,:,:,0] = np.squeeze(p2, 3)
        if distorted:
            # 50% chance of flipping in a random dimension
            if np.random.random() > 0.5:
                    # flip_dim = np.random.randint(0,2)
                flip_dim = 0
                images[i,:,:,0] = np.flip(images[i,:,:,0],flip_dim)
                labels[i,:,:,0] = np.flip(labels[i,:,:,0],flip_dim)
                flip_count += 1

    # print('%d of %d flipped' % (flip_count, batch_size))

    return [images, labels]



def load_training_data(input_data_path, input_label_path):

    image_training_data = []
    image_input_file = open(input_data_path)
    filenames = image_input_file.readlines()
    pathname = os.path.abspath(os.path.dirname(input_data_path))
    print("+++++ Loading image data (pathname=",pathname,")")
    
    for f in filenames:
        fname = f.rstrip()
        if not os.path.isabs(fname):
            fname=os.path.abspath(os.path.join(pathname,fname))

        if not os.path.isfile(fname):
            raise ValueError('Failed to find file: ' + fname)

        image = nib.load(fname).get_data()
        if len(image.shape) < 3:
            image = np.expand_dims(image, 2)
        image_training_data.append(image)
        print('+++++ \t Loaded image: %s' % fname,image.shape)

        
    print('+++++ Loaded %d image samples' % len(image_training_data))
    print('+++++')
    label_training_data = []
    label_input_file = open(input_label_path)
    filenames = label_input_file.readlines()
    pathname = os.path.abspath(os.path.dirname(input_label_path))

    print("+++++ Loading label data (pathname=",pathname,")")
    for f in filenames:
        fname = f.rstrip()
        if not os.path.isabs(fname):
            fname=os.path.abspath(os.path.join(pathname,fname))

        if not os.path.isfile(fname):
            raise ValueError('Failed to find file: ' + fname)

        image = nib.load(fname).get_data()
        if len(image.shape) < 3:
            image = np.expand_dims(image, 2)
        label_training_data.append(image)
        print('+++++ \t Loaded label: %s ' % fname,image.shape)


    print('+++++ Loaded %d label samples' % len(label_training_data))
    print('+++++')
    return [image_training_data, label_training_data]



def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int32)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(labels, squeeze_dims=[3]), 
                                                       name='entropy'))
    tf.summary.scalar('entropy', loss)
    return loss;


def loss2(logits, labels,normalizer=100000.0,smoothness=0.0):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). [ batch_size,width,height,depth,num_channels ]

    Returns:
    Loss tensor of type float.
    """

    edgeloss=0.0;    
    if smoothness>0.0:
        print('_____ Adding Back End Smoothness lambda=',smoothness)
        grad = np.zeros([3, 3, 1, 1])
    
        grad[1, 1, :, :] =  4
        grad[2, 1, :, :] = -1
        grad[0, 1, :, :] = -1
        grad[1, 0, :, :] = -1
        grad[1, 2, :, :] = -1
        
        conv = tf.nn.conv2d(logits,grad,strides=[1,1,1,1],padding='SAME',name='smoothness_final');
        print('_____ smoothness conv shape: %s normalizer=%f' % (conv.get_shape(),normalizer))
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.float32)
        edgeloss=smoothness*tf.nn.l2_loss(conv,'edgeL2loss')/normalizer


    loss = tf.nn.l2_loss(tf.subtract(logits,labels),'L2loss')/normalizer
    tf.summary.scalar('L2 loss', loss)
    tf.summary.scalar('Edge loss', edgeloss)
    return loss+edgeloss;

