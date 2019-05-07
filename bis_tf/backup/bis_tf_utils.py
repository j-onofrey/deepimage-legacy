from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import re
import math
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from six.moves import xrange
import bis_image_patchutil as putil
from tensorflow.python import pywrap_tensorflow


INTERNAL = {
    'use_gpu' : False,
    'use_fp16' : False,
    'debug' : False,
    'extra_debug' : False,
    'firsttime' : True,
    'device' : '/cpu:0'
}

def set_device(usegpu=True,devicename=''):

    DEVICE_CPU = '/cpu:0'
    DEVICE_GPU = '/gpu:0'

    if (devicename==''):
        if (usegpu):
            INTERNAL['device']=DEVICE_GPU
        else:
            INTERNAL['device']=DEVICE_CPU
    else:
        INTERNAL['device']=devicename

        #    print('+++++ Initialized Device Name as:'+INTERNAL['device'])

def get_device_name():
    return INTERNAL['device']

# -----------------------------------------------------------------------------------
# two levels of debugging statements
# -----------------------------------------------------------------------------------

def debug_print(*arg):

    """Prints arguments (variable length array) if 
    INTERNAL['debug'] == true, else nothing. This is really a "print like" statement.
    
    Args:
    *arg: variable length argument arrray. Stuff to print

    Returns:
    nothing
    """

    if not INTERNAL['debug']:
        return

    for a in arg:
        print(a,end='')
    print('')

def extra_debug_print(*arg):

    """Prints arguments (variable length array) if 
    INTERNAL['extra_debug'] == true, else nothing. This is really a "print like" statement.
    
    Args:
    *arg: variable length argument arrray. Stuff to print

    Returns:
    nothing
    """
    
    if not INTERNAL['extra_debug']:
        return
    for a in arg:
        print(a,end='')
    print('')

def print_dict(values,extra="+++++",header='   '):

    """Pretty-prints a dictionary 
    
    Args:
    values: the dictionary to print
    extra:  an indent string default="+++++"
    header: a short description of the dictionary (print as first line)

    Returns:
    nothing
    """
    
    print(extra+" "+header+" ",end='')
    sum=len(header)+1
    for key in values:
        if sum==0:
            print(extra+"     ",end='')
        a='('+str(key)+":"+str(values[key])+') '
        sum+=len(a)
        print(a,end='')
        if sum>60:
            sum=0
            print('')
                
    if sum!=0:
        print('')


def handle_error_and_exit(e):
    print('!!!!!')
    print('!!!!! E r r o r :',e)
    print('!!!!! E x i t i n g')
    sys.exit(0)
    
# -----------------------------------------------------------------------------------
#
#  Core Utilities from John
#
# -----------------------------------------------------------------------------------

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.


    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % 'PP', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))



def _variable_on_device(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Relies on INTERNAL['use_gpu'] to decide cpu vs gpu

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    
    dtype = tf.float16 if INTERNAL['use_fp16'] else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Relies on INTERNAL['use_gpu'] to decide cpu vs gpu

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
    Variable Tensor
    """
    dtype = tf.float16 if INTERNAL['use_fp16'] else tf.float32
    var = _variable_on_device(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


# -----------------------------------------------------------------------------------
#
#  Set Parameter Ranges
#
# -----------------------------------------------------------------------------------
def getpoweroftwo(val,minv=16,maxv=128):

    """Returns the closest power of two for input
    
    Args:
    val: desired value
    minv: minimum value (should be power of two)
    maxv: maximum value (should be power of two)

    Returns:
    power of two that is closest to val
    """

    
    if val<minv:
        val=minv
    elif val>maxv:
        val=maxv

    a=math.floor(math.log(val,2.0)+0.5)
    val=int(math.pow(2,a))
    return val

def getdivisibleby(val,divider=10):

    if (val<divider):
        val=divider
    else:
        fraction=math.floor(val/divider+0.5)
        val=int(divider*fraction)

    return val

def force_inrange(val,minv=0,maxv=1000):

    if (val<minv):
        val=minv
    elif (val>maxv):
        val=maxv;
        
    return val

def normalize_string(cname):
    try:
        cname=cname.encode('ascii','ignore')
    except:
        a=1
        
    try:
        cname=str(cname,'ascii')
    except:
        a=1

    return cname

def get_step_from_cname(cname):

    cname=normalize_string(cname)
    a=os.path.basename(cname)
    return a.split('-')[-1]

def get_read_checkpoint(initial):

    cname=normalize_string(initial)
    out=''
    
    if os.path.isdir(initial):
        ckpt = tf.train.get_checkpoint_state(initial)
        if not (ckpt and ckpt.model_checkpoint_path):
            raise RuntimeError('No checkpoint file found in '+initial)
        out=ckpt.model_checkpoint_path

    if out=='':
        ext=os.path.splitext(initial)[1]
        if os.path.isfile(initial)==True and ext==".meta":
            out=os.path.splitext(initial)[0]

    if out=='':
        fname=initial+".meta"
        if os.path.isfile(fname):
            out=initial

    if out=='':
        raise ValueError('Bad checkpoint file/directory:'+initial)

#    print('+++++ Mapping (read) '+initial+' -->',out)
    return out

def get_write_checkpoint(initial,model_name="FCN"):

    out=''
    
    path=normalize_string(initial)
    if os.path.isdir(initial):
        out=os.path.join(path, model_name)

    if out=='':
        dirname=os.path.dirname(path)
        basename=os.path.basename(path)
        ext=os.path.splitext(path)[1]

        if (ext==".ckpt"):
            if os.path.isdir(dirname):
                out=path
            else:
                path=dirname
                if (model_name!=''):
                    model_name=basename
                
    if out=='':
        try:
            os.mkdir(path)
            out=os.path.join(path, model_name)
        except OSError as e:
            raise ValueError('Bad checkpoint file/directory:'+initial)

    if model_name=='':
        out=os.path.dirname(out)
#        print('+++++ Mapping (write directory) '+initial+' -->',out)
        return  out
        
#    print('+++++ Mapping (write) '+initial+' ('+model_name+') -->',out)
    return out

# -----------------------------------------------------------------------------------
#
#  Get Tensor From Checkpoint file
#
# -----------------------------------------------------------------------------------

def get_tensor_from_checkpoint_file(file_name, tensor_name, all_tensors):
  """Prints tensors in a checkpoint file.

  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.

  If `tensor_name` is provided, prints the content of the tensor.

  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
  """
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in var_to_shape_map:
        print("***** looking for tensor_name: ", key, ' in ' , file_name)
        print(reader.get_tensor(key))
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
#      print("tensor_name: ", tensor_name)
      return reader.get_tensor(tensor_name)
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")


def get_tensor_list_from_checkpoint_file(file_name, tensor_name_list):
  """Gets tensor list in a checkpoint file.

  Args:
    file_name: Name of the checkpoint file.
    tensor_name_list: List of names of the tensor in the checkpoint file to print.
  """
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    out = {};
    for name in tensor_name_list:
        out[name]=reader.get_tensor('Parameters/'+name)
    return out
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")


# -----------------------------------------------------------------------------------
#
#  Various Utility Functions needed in places
#
# -----------------------------------------------------------------------------------
def compute_normalizer(image_shape,batch_size,regression=True):

    depth=1;
    if (len(image_shape)==5):
        depth=image_shape[3]
    if regression:
        normalizer=1.0/float(100.0*(depth*image_shape[2]*image_shape[1]*batch_size))
    else:
        normalizer=1.0/float(depth*image_shape[2]*image_shape[1]*batch_size)
    return normalizer


# -----------------------------------------------------------------------------------
#
#  Create DNN Elements
#
# -----------------------------------------------------------------------------------
def createConvolutionShape(filter_size=3,num1=1,num2=1,threed=False):
    if (threed):
        return [ filter_size,filter_size,filter_size,num1,num2 ]
    
    return [ filter_size,filter_size,num1,num2 ]



def createConvolutionLayer(input,name,shape,padvalue=-1,stdev=5e-2,wd=0.0,indent="",norelu=False):
    """Create a Convolution Layer

    Returns:
      convolution layer and biases to add to next layer
    
    padvalue = -1 -- create padding based on filter size
    padvalue =  0 -- no padding
    """
    
    wname='w_'+name;
    bname='b_'+name;
    shapelength=len(shape)
    num_layers= shape[shapelength-1]
    
    width=shape[0]-1
    height=shape[1]-1
    threed=False;
    if shapelength==5:
        threed=True
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

    if padvalue !=0:
        padding = tf.pad(input, paddings=pad_values, mode='SYMMETRIC', name='padding_'+name)
        debug_print('*****',indent,'Adding padding :',pad_values)
    else:
        extra_debug_print('*****',indent,'Not adding padding.')

        
    conv_w = _variable_with_weight_decay(wname, shape=shape, stddev=stdev,wd=wd)
    conv_b=None
    if not norelu:
        conv_b = _variable_on_device(bname, [num_layers], tf.constant_initializer(0.0))
        
    if threed:
        conv = tf.nn.conv3d(padding, conv_w, strides=[1, 1, 1, 1, 1], padding='VALID',name="conv3d_"+name)
    else:
        conv = tf.nn.conv2d(padding, conv_w, strides=[1, 1, 1, 1], padding='VALID',name="conv2d_"+name)

    debug_print('*****',indent,'Creating convolution: num_layers = ',num_layers,' shape=',shape,
                    ' in_shape=',padding.get_shape(),
                    ' %s shape: %s' % (wname,conv.get_shape()))
    return [ conv,conv_b ]


def createConvolutionLayerRELU(input,name,shape,padvalue=-1,stdev=5e-2,wd=0.0,indent="",norelu=False):

    if not norelu:
        debug_print('*****',indent,' Creating convolution+relu ',name)
        indent=indent+'    '
        
    conv,conv_b=createConvolutionLayer(input=input,
                                       name=name,
                                       shape=shape,
                                       padvalue=padvalue,
                                       stdev=stdev,
                                       wd=wd,
                                       indent=indent,norelu=norelu)
    if not norelu:
        relu = tf.nn.relu( tf.nn.bias_add(conv, conv_b), name='relu_'+name)
        _activation_summary(relu)
        return relu

    return conv


def createConvolutionLayerRELUPool(input,name,shape,mode='max',
                                   padvalue=-1,in_ksize=2,in_strides=2,
                                   stdev=5e-2,wd=0.0,indent="",norelu=False):

    if (mode!='max'):
        mode='avg'
        
    if norelu:
        debug_print('*****',indent,' Creating convolution+pool ',name,' pool mode=',mode)
    else:
        debug_print('*****',indent,' Creating convolution+relu+pool ',name,' pool mode=',mode)

    if not norelu:
       relu=createConvolutionLayerRELU(input=input,
                                       name=name,
                                       shape=shape,
                                       padvalue=padvalue,
                                       stdev=stdev,
                                       wd=wd,
                                       indent=indent+'   ')
    else:
        relu,_=createConvolutionLayer(input=input,
                                      name=name,
                                      shape=shape,
                                      padvalue=padvalue,
                                      stdev=stdev,
                                      wd=wd,
                                      indent=indent+'    ',norelu=True)
        
    shapelength=len(shape)
    if shapelength==5:
        strides=[1,in_strides,in_strides,in_strides,1]
        ksize=[1,in_ksize,in_ksize,in_ksize,1 ]
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
            
    debug_print('*****',indent,'    Adding ',mode,'pool shape: ',pool.get_shape(),' strides=',strides,' ksize=',ksize)
    return pool



def createFullyConnectedShape(poolshape,num1,num2,threed=False):
    if (threed):
        return [ poolshape[1],
                 poolshape[2],
                 poolshape[3],
                 num1,num2 ]

    
    return [ poolshape[1],
             poolshape[2],
             num1,num2 ]

def createSquareShape(val,num1,num2,threed=False):
    if (threed):
        return [ val,val,val,num1,num2 ]

    return [ val,val, num1,num2 ]


def createFullyConnectedLayer(input,name,shape,stdev=5e-2,wd=0.0,indent=""):

    if indent=="":
        debug_print('*****',indent,' Creating fully connected layer ',name, ' shape=',shape,' input_data shape=',input.get_shape())
    else:
        extra_debug_print('*****',indent,' Creating fully connected layer ',name, ' shape=',shape,' input_data shape=',input.get_shape())
        
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
    if (indent==""):
        debug_print('*****      ',indent,name,':',fc1_relu1.get_shape())
    else:
        extra_debug_print('*****      '+indent+name+':',fc1_relu1.get_shape())
    return fc1_relu1


def createFullyConnectedLayerWithDropout(input,name,shape,keep_prob_tensor,stdev=5e-2,wd=0.0,indent=""):


    debug_print('*****',indent,' Creating fully connected + dropout layer ',name)
    fc1_relu1=createFullyConnectedLayer(input=input,
                                        name=name,
                                        shape=shape,
                                        stdev=stdev,
                                        wd=wd,
                                        indent=indent+'    ')

    fc1_dropout = tf.nn.dropout(fc1_relu1, keep_prob=keep_prob_tensor,name='dropout_'+name)
    debug_print('*****',indent,'    ',name,':',fc1_dropout.get_shape(),' with dropout keep=',keep_prob_tensor)
    
    return fc1_dropout

# --------------------------------------------------------------------------------
def createDeconvolutionDynamicOutputShape(input,
                                          orig_input_data,
                                          nofuse=False,
                                          threed=False,
                                          num_classes=-1):

    if nofuse:                
        shape=input.get_shape()
        if (threed):
            if num_classes<1:
                num_classes=shape[4].value
            out_shape = tf.stack([tf.shape(orig_input_data)[0],
                                  shape[1].value, shape[2].value, shape[3].value,num_classes ])
        else:
            if num_classes<1:
                num_classes=shape[3].value
            out_shape = tf.stack([tf.shape(orig_input_data)[0],
                                  shape[1].value, shape[2].value,
                                  num_classes ])
    else:
        out_shape=tf.shape(input)

    return out_shape
        
def createDeconvolutionShape(val,num1,num2,threed=False):
    if (threed):
        return [ val,val,val,num1,num2 ]

    return [ val,val, num1,num2 ]


def createDeconvolutionLayer(input,name,dynamic_output_shape,input_shape,
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
                                              output_shape=dynamic_output_shape,
                                              strides=strides, padding='SAME',
                                              name='transpose2d_'+name)
    else:
        deconv1_conv = tf.nn.conv3d_transpose(input,
                                              deconv1_w,
                                              output_shape=dynamic_output_shape,
                                              strides=strides,
                                              padding='SAME',
                                              name='transpose3d_'+name)
        
    deconv1 = tf.nn.bias_add(deconv1_conv, deconv1_b)
    if indent=="":
        debug_print('*****',indent,' Creating deconvolution ',name, ' shape=',deconv1.get_shape(),' ',
                    ' strides=',strides)
    else:
        extra_debug_print('*****',indent,name,':',deconv1.get_shape(),' strides=',strides)
    return deconv1

def createDeconvolutionLayerFuse(input,
                                 fuse_input,
                                 shape,
                                 dynamic_output_shape,
                                 name='deconv',
                                 in_strides=2,
                                 stdev=5e-2,
                                 wd=0.0,
                                 nofuse=False,
                                 indent=""):
    
    newindent=indent
    if not nofuse:
        newindent=indent+"    "


#    if not nofuse:
#        output_shape=tf.shape(fuse_input)
#    else:
#        output_shape=dynamic_output_shape


    deconv1=createDeconvolutionLayer(input=input,
                                     name=name,
                                     input_shape=shape,
                                     dynamic_output_shape=dynamic_output_shape,
                                     in_strides=in_strides,
                                     stdev=stdev,

                                     indent=newindent)
    
    if nofuse:
        return deconv1
   
    fuse1 = tf.add(deconv1, fuse_input, name='fuse_'+name)
    _activation_summary(fuse1)
    debug_print('*****',indent,' Creating deconvolution+fuse ',name, ' shape=',fuse1.get_shape())
    return fuse1

# -----------------------------------------------------------------------------------
def createSmoothnessLayer(outimg,
                          num_classes=1,
                          threed=False):

    print('===== Adding back end smoothness compilation, numclasses=',num_classes)
    if num_classes>1:
        outimg=tf.cast(outimg,tf.float32)
        
    with tf.variable_scope('Edge_Outputs'):
                        
        if not threed:
            grad_x = np.zeros([3, 1, 1, 1])
            grad_x[ 0, 0, : , :] = -1
            grad_x[ 1, 0, : , :] =  2
            grad_x[ 2, 0, : , :] = -1
            
            grad_y = np.zeros([1, 3, 1, 1])
            grad_y[ 0, 0, : , : ] = -1
            grad_y[ 0, 1, : , : ] =  2
            grad_y[ 0, 2, : , : ] = -1
            
            edge_conv_x = tf.nn.conv2d(outimg,grad_x,strides=[1,1,1,1],
                                       padding='SAME',name='smoothness_final_x');
            edge_conv_y = tf.nn.conv2d(outimg,grad_y,strides=[1,1,1,1],
                                       padding='SAME',name='smoothness_final_y');
            edgeloss=tf.nn.l2_loss(edge_conv_x)+tf.nn.l2_loss(edge_conv_y)
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
            
            conv_x = tf.nn.conv3d(outimg,grad_x,strides=[1,1,1,1,1],
                                  padding='SAME',name='smoothness_final_x');
            conv_y = tf.nn.conv3d(outimg,grad_y,strides=[1,1,1,1,1],
                                  padding='SAME',name='smoothness_final_y');
            conv_z = tf.nn.conv3d(outimg,grad_z,strides=[1,1,1,1,1],
                                  padding='SAME',name='smoothness_final_z');
            edgeloss=tf.nn.l2_loss(conv_x)+tf.nn.l2_loss(conv_y)+ tf.nn.l2_loss(conv_z)
                
    return tf.identity(edgeloss,name='Regularizer')

                
# -----------------------------------------------------------------------------------
#
#  Standard Loss Functions
#
# -----------------------------------------------------------------------------------
def cross_entropy_loss(logits, labels):
    """Add Classification Loss

    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int32)



    shapelength=len(labels.get_shape())
    threed=False;
    if shapelength==5:
        threed=True
        sq=4
    else:
        sq=3

    print('***** Setting up entropy_loss function: dim=',logits.shape,' labels.shape.length=',shapelength,'sq=',sq)
    entropyloss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(labels, squeeze_dims=[sq]),
                                                       name='entropy'))
    tf.summary.scalar('entropy', entropyloss)
    return entropyloss


def l2_loss(pred_image, labels,normalizer=1.0):
    """Add L2Loss 
    
    Returns:
    Loss tensor of type float.
"""
    print('***** Setting up l2_loss function: dim=',pred_image.shape,' normalizer='+str(normalizer))
    

    l2_loss = tf.identity((tf.nn.l2_loss(tf.subtract(pred_image,labels))*normalizer),name='L2_loss')
    tf.summary.scalar('L2_loss', l2_loss)
    return l2_loss


def compute_loss(pred_image,logits,labels,regularizer,use_l2=True,normalizer=1.0,smoothness=0.0):

    
    if use_l2:
        dataloss=l2_loss(pred_image=pred_image,
                         labels=labels,
                         normalizer=normalizer)

    else:
        dataloss=cross_entropy_loss(logits=logits,
                                    labels=labels)


    if smoothness>0.0:
        print('*****\t and adding regularizer smoothness='+str(smoothness))
        reg=regularizer*normalizer;
        tf.summary.scalar('Regularizer', reg)
        total=tf.identity(dataloss+reg*smoothness,name="Total")
        tf.summary.scalar('Total', total)
    else:
        debug_print('*****\t and not adding regularizer smoothness='+str(smoothness))
        total=tf.identity(dataloss,name='Total')
        
    return total


    
# -----------------------------------------------------------------------------------
#
#  Train and Test Functions
#
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
#
# Generic Training Code 
#
# -----------------------------------------------------------------------------------
def run_training(training_data,
                 images_pointer,
                 labels_pointer,
                 keep_pointer,
                 train_op,
                 loss_op,
                 output_model_path,
                 resume_model_path='',
                 keep_value=1.0,
                 batch_size=32,
                 max_steps=200,
                 model_name='fcn.ckpt',
                 step_fraction=10,
                 patch_size=64,
                 epsilon=0.1,
                 augmentation=False):


    #    max_steps=getdivisibleby(max_steps,step_fraction)
    
    # Create  a Saver
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries
    summary_op = tf.summary.merge_all()

    # Build an init operation to run below
    # init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
    
    # Start running operations on the Graph.
    session = tf.Session()
    session.run(init_op)
    
    # Start the queue runners.
    tf.train.start_queue_runners(sess=session)

    output_model_path=get_write_checkpoint(output_model_path,model_name="")
    print('+++++ Initializing output directory to ',output_model_path,' (from ',output_model_path,')')
    summary_writer = tf.summary.FileWriter(output_model_path, session.graph)
    
    last_loss=1e+30;

    if len(resume_model_path)>2:
        cname=get_read_checkpoint(resume_model_path)
        print("+++++ Read initial tensor values from "+cname+" (from "+resume_model_path+")")
        loader = tf.train.Saver(tf.global_variables())
        loader.restore(session, cname)
        begin_step = int(get_step_from_cname(cname))
    else:
        begin_step=0

    last_step=0
    checkpoint_outpath = get_write_checkpoint(output_model_path, model_name)
    
    if (max_steps>0):
        print("*****")
        print_dict(values={ 'steps' : max_steps,
                            'batch_size' : batch_size,
                            'keep_prob' : keep_value,
                            'augment' :   augmentation,
                            'output' :    checkpoint_outpath+' (from '+output_model_path+')' },
                   extra="*****",
                   header="Beginning Training")
        print("*****")
        
        for step in range(1,max_steps+1):

            truestep=step+begin_step
            start_time = time.time()
            batch = training_data.get_batch(batch_size=batch_size,
                                            augmentation=augmentation,
                                            patch_size=patch_size)
            feed_dict={
                images_pointer: batch[0],
                labels_pointer: batch[1],
                keep_pointer  : keep_value,
            }
            session.run(train_op, feed_dict=feed_dict)
            duration = time.time() - start_time
            
            if (truestep % step_fraction == 0 or truestep == 1 or step == max_steps):

                loss_value, summary_str = session.run([loss_op, summary_op], feed_dict=feed_dict)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                
                dloss=100.0
                if step>0:
                    dloss=(200.0*math.fabs(loss_value-last_loss))/(loss_value+last_loss)
                
                last_loss=loss_value
                num_exmaples_per_step = batch_size
                examples_per_sec = num_exmaples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('***** %s: step %4d, loss = %6.2f (%.1f examples/s; %.3f s/batch)')

                s=datetime.now().strftime('%H:%M:%S')
                print (format_str % (s, truestep, loss_value, examples_per_sec, sec_per_batch))
                
                summary_writer.add_summary(summary_str, truestep)
                
                saver.save(session, checkpoint_outpath, global_step=truestep)
                last_step=truestep
                
                if (dloss<epsilon):
                    step=max_steps+step_fraction


    if (max_steps<0):
        saver.save(session, checkpoint_outpath)
        print('***** Saved initial structure in',checkpoint_outpath,
              ' graph=',checkpoint_outpath+".meta), size=",os.path.getsize(checkpoint_outpath+".meta"))
        return checkpoint_outpath+".json"

    return checkpoint_outpath+"-"+str(last_step)+".json"
        

# -----------------------------------------------------------------------------------
#
# Generic Reconstuct Code 
#
# -----------------------------------------------------------------------------------
# Returns list of reconstructed images
def reconstruct_images(checkpointname,
                       testing_data,
                       images_pointer,
                       keep_pointer,
                       model_output_image,
                       patch_width,
                       keep_value=1.0,
                       threed=False,
                       batch_size=32):
    
    # Create  a Saveer
    saver = tf.train.Saver(tf.global_variables())


    with tf.Session() as sess:

        saver.restore(sess, checkpointname)
        global_step = get_step_from_cname(checkpointname)

        print('***** Restored model %s step=%s' % (checkpointname,global_step))

        imgshape=testing_data.get_image_shape()
        print('***** \t Image Dimensions=',imgshape,' threed=',threed)

        if (threed):
            patch_shape = (patch_width, patch_width, patch_width)
            stride = ( patch_width//2,patch_width//2,patch_width//2)
        else:
            patch_shape = (patch_width, patch_width, 1)
            stride = ( patch_width//2,patch_width//2,1)
            
        print('***** \t Stride: %s' % (stride, ))
        patch_indexes = putil.getOrderedPatchIndexes(testing_data.get_input_data()[0], patch_size=patch_shape, stride=stride, padding='SAME')
        total_patches = patch_indexes.shape[0]
            
        # Evaluate the patch indexes in small batches
        if (threed):
            result_patches = np.zeros((total_patches, patch_width,patch_width, patch_width, 1), dtype=np.float32)
        else:
            result_patches = np.zeros((total_patches, patch_width, patch_width, 1), dtype=np.float32)
            
        print('***** Total number of patches to evaluate = %d batch_size= %d keep_value=%.2f'
              % (total_patches,batch_size,keep_value))


        last=0.0
        frame=0
        recon_image_list=[]
        while frame < imgshape[0]:
            count = 0
            print('***** Processing image %d/%d: (patches)' % (frame+1,imgshape[0]))

            while count < total_patches:
                idx_begin = count
                idx_end = idx_begin + batch_size
                if idx_end > total_patches:
                    idx_end = total_patches

                fraction=idx_begin/total_patches
                if fraction>=last:
                    print('***** \t\t\t [%6d,%6d]/%6d' % (idx_begin, idx_end,total_patches))
                    last+=0.1
                        
                current_indexes = patch_indexes[idx_begin:idx_end,:]
                current_patches = putil.getPatchesFromIndexes(testing_data.get_input_data()[frame],
                                                              current_indexes,
                                                              patch_shape,
                                                              padding='SAME',
                                                              dtype=np.float32)

                if threed:
                    current_patches=np.expand_dims(current_patches, 5)
                
                feed_dict={images_pointer: current_patches,
                           keep_pointer : keep_value }
                
                predictions = sess.run([model_output_image], feed_dict=feed_dict)
                result_patches[idx_begin:idx_end,:,:,:] = np.array(predictions[0])
                count += batch_size

            
            recon_image_list.append(putil.imagePatchRecon(testing_data.get_input_data()[0].shape, result_patches, patch_indexes,indent='  '))
            frame=frame+1
        # Now recon the result image from all the predicted patches into a single image

        print("***** Finished reconsructing %d images" % (len(recon_image_list)))
        return recon_image_list
    



