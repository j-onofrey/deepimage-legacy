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
    'debug' : False,
    'extra_debug' : False,
    'trainable' : True,
}

# ---------------------------------------------------------------------------
# Set variable creation to be optimizable or fixed
# ---------------------------------------------------------------------------

def create_variables_as_optimizable():
    INTERNAL['trainable']=True

def create_variables_as_fixed():
    INTERNAL['trainable']=False




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


def execute_command(cmd):
    print(".....\n..... E x e c u t i n g : \n.....\t "+cmd)
    if (os.system(cmd)!=0):
        sys.exit(0)
    print(".....")
    print("..... D o n e   E x e c u t i n g : \n.....\t "+cmd)


def handle_error_and_exit(e):
    print('!!!!!')
    print('!!!!! E r r o r :',e)
    print('!!!!! E x i t i n g')
    sys.exit(0)

# -----------------------------------------------------------------------------------
#
#  Core Utilities from John (originally)
#
# -----------------------------------------------------------------------------------
# explicitly setting device as cpu for summaries!!!
# -----------------------------------------------------------------------------------

def add_activation_summary(x,scope='Summaries/'):
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
    #with tf.device('/cpu:0'):
    with tf.variable_scope(scope):
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def add_scalar_summary(name,value,scope='Summaries/'):

    #with tf.device('/cpu:0'):
    with tf.variable_scope(scope):
        tf.summary.scalar(name,value)

def image_summary(img,name,image_size,num_frames=1,threed=False,max_outputs=3):


    #with tf.device('/cpu:0'):
    with tf.variable_scope('Summary/Img'):
        fimg=tf.cast(img,tf.float32)
    if threed:
        midslice=image_size[2]//2
        debug_print('+++++ \t\t extracting '+name+' midslice='+str(midslice)+' for summary')
        for i in range(0,num_frames):
            with tf.variable_scope('Summary/Img'):
                i_slices = tf.squeeze(tf.slice(fimg, (0, 0, 0, midslice, i), (-1, image_size[0], image_size[1], 1, 1)),4)
            tf.summary.image(name+'_slice_'+str(midslice)+'_fr_'+str(i+1), i_slices, max_outputs=max_outputs)
    else:
        for i in range(0,num_frames):
            with tf.variable_scope('Summary/Img'):
                i_slices = tf.slice(fimg, (0, 0, 0, i), (-1, image_size[0], image_size[1], 1))
            tf.summary.image(name+'_fr_'+str(i+1), i_slices, max_outputs=max_outputs)

# ---------------------------------------------------------------------------


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

    dtype =  tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=INTERNAL['trainable'])
    extra_debug_print('+++++ creating variable:'+name+', shape='+str(shape)+' optimizable='+str(INTERNAL['trainable']))
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
    dtype = tf.float32
    var = _variable_on_device(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')

    return var


# -----------------------------------------------------------------------------------
#
#  Set Parameter Ranges
#
# -----------------------------------------------------------------------------------
def getdimname(threed):
    if (threed):
        return "3d"
    return "2d"

def getdindex(threed):
    
    dindex=3
    if (threed):
        dindex=4
    return dindex


def getpoweroftwo(val,minv=16,maxv=128):

    """Returns the closest power of two for input

    Args:
    val: desired value
    minv: minimum value (should be power of two)
    maxv: maximum value (should be power of two)

    Returns:
    power of two that is closest to val
    """

    if (val==None):
        return None

    if val<minv:
        val=minv
    elif val>maxv:
        val=maxv

    a=math.floor(math.log(val,2.0)+0.5)
    val=int(math.pow(2,a))
    return val


def getdivisibleby(val,divider=10):

    if (val==None):
        return None

    if (val<divider):
        val=divider
    else:
        fraction=math.floor(val/divider+0.5)
        val=int(divider*fraction)

    return val


def getdivisibleby16(val,minv=16,maxv=256):

    val=getdivisibleby(val,divider=16)
    if val<minv:
        return minv

    if val>maxv:
        return maxv

    return val



def force_inrange(val,minv=0,maxv=1000):

    if (val==None):
        return None

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

    if initial=='':
        raise ValueError('No input checkpoint file/directory specified')

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

    if initial=='':
        raise ValueError('No output checkpoint file/directory specified')
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
        try:
            out[name]=reader.get_tensor('Parameters/'+name)
            if type(out[name]) is bytes:
                out[name]=out[name].decode("utf-8")
        except Exception as e:
            out[name]=None
    return out
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")




# -----------------------------------------------------------------------------------
#
#  Create DNN Elements
#
# -----------------------------------------------------------------------------------
def createConvolutionShape(filter_size=3,num1=1,num2=1,threed=False):
    if (threed):
        return [ filter_size,filter_size,filter_size,num1,num2 ]

    return [ filter_size,filter_size,num1,num2 ]


def createConvolutionShapeFromInput(input, filter_size=1, num_out_channels=1):
    in_shape = input.get_shape()
    # Automatically get the correct tensor shape for the next layer given the input layer
    out_shape = [filter_size,filter_size,in_shape[-1].value,num_out_channels]

    # Add another filter dim to the front for 3D data
    if len(in_shape) > 4:
        out_shape = [filter_size] + out_shape

    return out_shape



def createConvolutionLayer(input,name,shape,padvalue=-1,stdev=5e-2,wd=0.0,indent="",norelu=False,threed=False):
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
    cmode=getdimname(threed)
    if threed:
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

    debug_print('*****',indent,'Creating '+cmode+' convolution: num_layers = ',num_layers,' shape=',shape,
                    ' in_shape=',padding.get_shape(),
                    ' %s shape: %s' % (wname,conv.get_shape()))
    return [ conv,conv_b ]


def createActivitionFunctionLayer():

    return None


def createConvolutionLayerRELU(input,name,shape,padvalue=-1,stdev=5e-2,wd=0.0,indent="",norelu=False,threed=False):

    if not norelu:
        debug_print('*****',indent,' Creating convolution+relu ',name)
        indent=indent+'    '

    conv,conv_b=createConvolutionLayer(input=input,
                                       name=name,
                                       shape=shape,
                                       padvalue=padvalue,
                                       stdev=stdev,
                                       wd=wd,
                                       indent=indent,norelu=norelu,threed=threed)
    if not norelu:
        relu = tf.nn.relu( tf.nn.bias_add(conv, conv_b), name='relu_'+name)
        add_activation_summary(relu)
        return relu

    return conv



def createConvolutionLayerRELU_auto(input,name,filter_size=1,num_out_channels=1,padvalue=-1,stdev=5e-2,wd=0.0,indent="",norelu=False,threed=False):

    out_shape = createConvolutionShapeFromInput(input,filter_size=filter_size,num_out_channels=num_out_channels)
    if not norelu:
        debug_print('*****',indent,' Creating convolution+relu ',name)
        indent=indent+'    '

    conv,conv_b=createConvolutionLayer(input=input,
                                       name=name,
                                       shape=out_shape,
                                       padvalue=padvalue,
                                       stdev=stdev,
                                       wd=wd,
                                       indent=indent,norelu=norelu,threed=threed)
    if not norelu:
        relu = tf.nn.relu( tf.nn.bias_add(conv, conv_b), name='relu_'+name)
        add_activation_summary(relu)
        return relu

    return conv



def createPoolingLayer(input,size=2,stride=2,padding='VALID',name=None,mode='max',threed=False):
    # pool1 = tf.nn.max_pool3d(relu1_2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID', name='pool1')

    if threed:
        strides=[1,stride,stride,stride,1]
        ksize=[1,size,size,size,1 ]
    else:
        strides=[1,stride,stride,1]
        ksize=[1,size,size,1 ]

    cname=getdimname(threed)

    if mode=='max':
        if threed:
            pool = tf.nn.max_pool3d(input, ksize=ksize, strides=strides, padding='VALID', name='maxpool_'+name)
        else:
            pool = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='VALID', name='maxpool_'+name)
    else:
        mode='avg'
        if threed:
            pool = tf.nn.avg_pool3d(input, ksize=ksize, strides=strides, padding='VALID', name='avgpool_'+name)
        else:
            pool = tf.nn.avg_pool(input, ksize=ksize, strides=strides, padding='VALID', name='avgpool_'+name)

    debug_print('*****    Adding ',cname,mode,'pool shape: ',pool.get_shape(),' strides=',strides,' ksize=',ksize)
    return pool



def createConvolutionLayerRELUPool(input,name,shape,mode='max',
                                   padvalue=-1,in_ksize=2,in_strides=2,
                                   stdev=5e-2,wd=0.0,indent="",norelu=False,threed=False):

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
                                       threed=threed,
                                       wd=wd,
                                       indent=indent+'   ')
    else:
        relu,_=createConvolutionLayer(input=input,
                                      name=name,
                                      shape=shape,
                                      padvalue=padvalue,
                                      threed=threed,
                                      stdev=stdev,
                                      wd=wd,
                                      indent=indent+'    ',
                                      norelu=True)

    if threed:
        strides=[1,in_strides,in_strides,in_strides,1]
        ksize=[1,in_ksize,in_ksize,in_ksize,1 ]
    else:
        strides=[1,in_strides,in_strides,1]
        ksize=[1,in_ksize,in_ksize,1]

    cname=getdimname(threed)

    if mode=='max':
        if threed:
            pool = tf.nn.max_pool3d(relu, ksize=ksize, strides=strides, padding='VALID', name='maxpool_'+name)
        else:
            pool = tf.nn.max_pool(relu, ksize=ksize, strides=strides, padding='VALID', name='maxpool_'+name)
    else:
        mode='avg'
        if threed:
            pool = tf.nn.avg_pool3d(relu, ksize=ksize, strides=strides, padding='VALID', name='maxpool_'+name)
        else:
            pool = tf.nn.avg_pool(relu, ksize=ksize, strides=strides, padding='VALID', name='maxpool_'+name)

    debug_print('*****',indent,'    Adding ',cname,mode,'pool shape: ',pool.get_shape(),' strides=',strides,' ksize=',ksize)
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


def createFullyConnectedLayer(input,name,shape,stdev=5e-2,wd=0.0,indent="",threed=False):


    shapelength=len(shape)
    num_layers= shape[shapelength-1]
    cname=getdimname(threed)


    if indent=="":
        debug_print('*****',indent,' Creating fully connected '+cname+' layer ',name, ' shape=',shape,' input_data shape=',input.get_shape())
    else:
        extra_debug_print('*****',indent,' Creating fully connected '+cname+' layer ',name, ' shape=',shape,' input_data shape=',input.get_shape())

    fc1_w = _variable_with_weight_decay('w_'+name, shape=shape, stddev=stdev, wd=wd)
    fc1_b = _variable_on_device('b_'+name, shape=[num_layers], initializer=tf.constant_initializer(0.0))

    if threed==False:
        fc1_conv1 = tf.nn.conv2d(input, fc1_w, strides=[1,1,1,1], padding='SAME',name='conv2d_'+name)
    else:
        fc1_conv1 = tf.nn.conv3d(input, fc1_w, strides=[1,1,1,1,1], padding='SAME',name='conv3d_'+name)

    fc1_relu1 = tf.nn.relu(tf.nn.bias_add(fc1_conv1, fc1_b), name='relu1_'+name)
    add_activation_summary(fc1_relu1)
    if (indent==""):
        debug_print('*****      ',indent,name,':',fc1_relu1.get_shape())
    else:
        extra_debug_print('*****      '+indent+name+':',fc1_relu1.get_shape())
    return fc1_relu1


def createFullyConnectedLayerWithDropout(input,name,shape,keep_prob_tensor,stdev=5e-2,wd=0.0,indent="",threed=False):

    cname=getdimname(threed)
    debug_print('*****',indent,' Creating ',cname,' fully connected + dropout layer ',name)
    fc1_relu1=createFullyConnectedLayer(input=input,
                                        name=name,
                                        shape=shape,
                                        threed=threed,
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

def createDeconvolutionShape(size,in_channels,out_channels,threed=False):
    if (threed):
        return [ size,size,size,out_channels,in_channels ]

    return [ size,size,out_channels,in_channels ]


def createDeconvolutionShapeFromInput(input, filter_size=2, num_out_channels=1):
    in_shape = input.get_shape()
    # Automatically get the correct tensor shape for the next layer given the input layer
    out_shape = [filter_size,filter_size,num_out_channels,in_shape[-1]]

    # Add another filter dim to the front for 3D data
    if len(in_shape) > 4:
        out_shape = [filter_size] + out_shape

    return out_shape




def createDeconvolutionLayer(input,name,dynamic_output_shape,input_shape,
                             in_strides=2,stdev=5e-2,wd=0.0,indent="",threed=False):

    wname=name+'_w';
    bname=name+'_b';

    shapelength=len(input_shape)

    cname=getdimname(threed)
    if threed:
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
    debug_print('*****',indent,' Creating ',cname,' deconvolution ',name, ' shape=',deconv1.get_shape(),' ',
                ' strides=',strides)

    return deconv1


def createDeconvolutionLayer_auto(input,name,output_shape,filter_size=2,in_strides=2,num_out_channels=1,stdev=5e-2,wd=0.0,indent="",threed=False):


    input_shape = createDeconvolutionShapeFromInput(input,filter_size,num_out_channels)

    wname=name+'_w';
    bname=name+'_b';

    shapelength=len(input_shape)

    cname=getdimname(threed)
    if threed:
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
    debug_print('*****',indent,' Creating ',cname,' deconvolution ',name, ' shape=',deconv1.get_shape(),' ',
                ' strides=',strides)

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
                                 indent="",
                                 threed=False):

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
                                     threed=threed,
                                     indent=newindent)

    if nofuse:
        return deconv1

    fuse1 = tf.add(deconv1, fuse_input, name='fuse_'+name)
    add_activation_summary(fuse1)
    debug_print('*****',indent,' Creating deconvolution+fuse ',name, ' shape=',fuse1.get_shape())
    return fuse1




# -----------------------------------------------------------------------------------
def createSmoothnessLayer(outimg,
                          num_classes=1,
                          threed=False):

    print('===== \t\t adding back end smoothness computation, numclasses=',num_classes)
    if num_classes>1:
        outimg=tf.cast(outimg,tf.float32)

    with tf.variable_scope('Edge_Outputs'):

        if not threed:
            if (num_classes==1):
                grad_x = np.zeros([3, 1, 1, 1])
                grad_x[ 0, 0, : , :] = -1
                grad_x[ 1, 0, : , :] =  2
                grad_x[ 2, 0, : , :] = -1

                grad_y = np.zeros([1, 3, 1, 1])
                grad_y[ 0, 0, : , : ] = -1
                grad_y[ 0, 1, : , : ] =  2
                grad_y[ 0, 2, : , : ] = -1
            else:
                grad_x = np.zeros([2, 1, 1, 1])
                grad_x[ 0, 0, : , :] = -1
                grad_x[ 1, 0, : , :] =  1

                grad_y = np.zeros([1, 2, 1, 1])
                grad_y[ 0, 0, : , : ] = -1
                grad_y[ 0, 1, : , : ] =  1

            edge_conv_x = tf.nn.conv2d(outimg,grad_x,strides=[1,1,1,1],
                                       padding='SAME',name='smoothness_final_x');
            edge_conv_y = tf.nn.conv2d(outimg,grad_y,strides=[1,1,1,1],
                                       padding='SAME',name='smoothness_final_y');
            edgeloss=tf.nn.l2_loss(edge_conv_x)+tf.nn.l2_loss(edge_conv_y)
        else:
            if (num_classes==1):
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
            else:
                grad_x = np.zeros([2, 1, 1, 1, 1])
                grad_x[0, 0, 0, :, : ] = -1
                grad_x[1, 0, 0, :, : ] =  1

                grad_y = np.zeros([1, 2, 1, 1, 1])
                grad_y[0, 0, 0, :, : ] = -1
                grad_y[0, 1, 0, :, : ] =  1

                grad_z = np.zeros([1, 1, 2, 1, 1])
                grad_z[0, 0, 0, :, : ] = -1
                grad_z[0, 0, 1, :, : ] =  1

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
#  Train and Recon Functions
#
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
#
# Generic Training Code
#
# -----------------------------------------------------------------------------------
def run_training(training_data,
                 images_pointer,
                 targets_pointer,
                 keep_pointer,
                 step_pointer,
                 train_op,
                 loss_op,
                 output_model_path,
                 resume_model_path='',
                 keep_value=1.0,
                 batch_size=32,
                 max_steps=200,
                 model_name='fcn.ckpt',
                 step_fraction=50,
                 patch_size=[32,32,1],
                 patch_threed=False,
                 epsilon=0.1,
                 list_of_variables_to_read=None,
                 augmentation=False):


    #    max_steps=getdivisibleby(max_steps,step_fraction)

    #with tf.device('/cpu:0'):
    # Saver problem 1
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries
    summary_op = tf.summary.merge_all()

    # Build an init operation to run below
    # init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()



    # Start running operations on the Graph.
    if INTERNAL['extra_debug']:
        session=tf.Session(config=tf.ConfigProto(log_device_placement=True))
    else:
        session = tf.Session()
    session.run(init_op)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=session)

    try:
        output_model_path=get_write_checkpoint(output_model_path,model_name="")
    except Exception as e:
        handle_error_and_exit(e)

    print('+++++ Initializing output directory to ',output_model_path,' (from ',output_model_path,')')
    summary_writer = tf.summary.FileWriter(output_model_path, session.graph)

    last_loss=1e+30;

    if len(resume_model_path)>2:
        cname=get_read_checkpoint(resume_model_path)
        print("+++++ \t read initial tensor values from "+cname)
        if list_of_variables_to_read==None:
            list_of_variables_to_read=tf.global_variables()
        debug_print('+++++ Variables to read from'+cname+'= ',list_of_variables_to_read)
        loader = tf.train.Saver(list_of_variables_to_read)
        loader.restore(session, cname)
        begin_step = int(get_step_from_cname(cname))
    else:
        begin_step=0

    last_step=0
    checkpoint_outpath = get_write_checkpoint(output_model_path, model_name)

    
    if (max_steps>0):
        print("*****")
        print_dict(values={ 'steps' : max_steps,
                            'step_gap' : step_fraction,
                            'batch_size' : batch_size,
                            'keep_prob' : str(round(keep_value,4)),
                            'augment' :   augmentation,
                            'output' :    checkpoint_outpath},
                            extra="*****",
                   header="Beginning Optimization")
        print("*****")
        firstprint=True

        step4=step_fraction // 4
        step_fraction=step4*4

        for step in range(1,max_steps+1):

            truestep=step+begin_step
            start_time = time.time()
            batch = training_data.get_batch_of_patches(batch_size=batch_size,
                                                       augmentation=augmentation,
                                                       patch_size=patch_size,
                                                       patch_threed=patch_threed)
            #            print(batch[0][0].dtype,batch[0].shape,batch[0][1].dtype,batch[1].shape)

            feed_dict={
                images_pointer: batch[0],
                targets_pointer: batch[1],
                keep_pointer  : keep_value,
                step_pointer  : truestep
            }

            try:
                session.run(train_op, feed_dict=feed_dict)
            except Exception as e:
                handle_error_and_exit(e)

            duration = time.time() - start_time

            if ( (truestep % step_fraction == 0 or truestep % step4 ==0) or truestep == 1 or step == max_steps):

                if firstprint==True:
                    print("*****")
                    firstprint=False

                loss_value, summary_str = session.run([loss_op, summary_op], feed_dict=feed_dict)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                dloss=100.0
                if step>0:
                    dloss=(200.0*math.fabs(loss_value-last_loss))/(loss_value+last_loss)

                last_loss=loss_value
                num_exmaples_per_step = batch_size
                examples_per_sec = num_exmaples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('***** \t %s: step %4d, loss = %7.3f (%6.1f examples/s; %.3f s/batch)')

                s=datetime.now().strftime('%H:%M:%S')
                print (format_str % (s, truestep, loss_value, examples_per_sec, sec_per_batch),end='')

                if ( truestep % step_fraction == 0 or truestep == 1 or step == max_steps):
                    print(' (saving)')
                    summary_writer.add_summary(summary_str, truestep)
                    saver.save(session, checkpoint_outpath, global_step=truestep)
                else:
                    print('')

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
# quarter cut indices
# -----------------------------------------------------------------------------------
def quarter_cut_indices(patch_indexes,width,batch_size=8,threed=False):
                
    boxes= [ [ 0,0,0 ], [ width,0,0], [ 0, width,0 ],[ width,width,0 ],
             [ 0,0,width ], [ width,0,width], [ 0, width,width ],[ width,width,width ] ]

    ratio=4
    if threed:
        ratio=8
        
    oshp=patch_indexes.shape
    num_old=oshp[0]
    new_indices=np.zeros([num_old*ratio,oshp[1]],dtype=np.int32)

#   print('quartering ...', oshp,'-->',new_indices.shape,' boxes=',boxes)

    num_total=num_old*ratio

    first=0
    out_index=0
    in_index=0
    while first<num_old:
        this_batch_size=batch_size
        if first+batch_size>num_old:
            this_batch_size=int(num_old-first)
#       print('total=',oshp[0],', old=',first,' out_index=',out_index,'this_batch_size=',this_batch_size)
        for sector in range(0,ratio):
            for i in range(0,this_batch_size):
                in_index=first+i
#                print('i,sector=',[i,sector],' out_index=',out_index,'bx=',boxes[sector],' index=',patch_indexes[in_index])
                for j in range(0,3):
                    new_indices[out_index][j]=int(patch_indexes[in_index][j]+boxes[sector][j])
                out_index=out_index+1
        
        first=first+batch_size
    
            
    return new_indices

def offset_indices(patch_indexes,shift,threed=False):
                
    oshp=patch_indexes.shape
    new_indices=np.zeros(oshp,dtype=np.int32)
    maxj=2
    if threed:
        maxj=3
    
    for i in range(0,oshp[0]):
        for j in range(0,oshp[1]):
            if j<maxj:
                new_indices[i][j]=int(patch_indexes[i][j]+shift)
            else:
                new_indices[i][j]=int(patch_indexes[i][j])
    return new_indices



# Checks to see if two shape lists/tuples p and q have the same values
def is_shape_same(p,q,maxdim=None):

    mindim = min(len(p),len(q))
    if maxdim is None or maxdim < mindim:
        maxdim = mindim
    for i in range(0,maxdim):
        if p[i] != q[i]:
            return False

    return True



# -----------------------------------------------------------------------------------
#
# Generic Reconstuct Code
#
# -----------------------------------------------------------------------------------
# Returns list of reconstructed images
def reconstruct_images(checkpointname,
                       input_data,
                       images_pointer,
                       keep_pointer,
                       model_output_image,
                       patch_size,
                       threed,
                       one_hot_output=False,
                       num_classes=1,
                       stride_size=0.5,
                       sigma=0.0,
                       repeat=1,
                       keep_value=1.0,
                       actual_pad_size=None,
                       batch_size=32):


    print('***** Executing Reconstruction')
    
    # Create  a Saver

    # Saver problem 2
    #with tf.device('/cpu:0'):
    saver = tf.train.Saver(tf.global_variables())
        
    if INTERNAL['extra_debug']:
        sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
    else:
        sess = tf.Session()

    saver.restore(sess, checkpointname)
    global_step = get_step_from_cname(checkpointname)

    print('*****\t Restored model %s step=%s' % (checkpointname,global_step))

    imgshape=input_data.get_image_shape()
    # TODO: make this stride anisotropic
    strw=int(stride_size*patch_size[0]+0.5)
    print('***** \t Image Dimensions=',imgshape,' patch threed=',threed, ' patch_size=',str(patch_size),' stride=',str(stride_size),' sigma=',sigma)

    if (threed):
        stride = [int(stride_size*patch_size[0]+0.5), int(stride_size*patch_size[1]+0.5), int(stride_size*patch_size[2]+0.5)]
    else:
        stride = [int(stride_size*patch_size[0]+0.5), int(stride_size*patch_size[1]+0.5), int(stride_size*patch_size[2]+0.5)]

    print('***** \t Stride: '+str(stride))

    frame=0
    recon_image_list=[]
    while frame < imgshape[0]:
        print('*****')

        img = input_data.get_padded_input_data()[frame]
        min_dt = np.amin(img)
        max_dt = np.amax(img)

        patch_indexes = putil.getOrderedPatchIndexes(img,patch_size=patch_size, stride=stride,padding='SAME')
        total_patches = patch_indexes.shape[0]
        have_result_patches=False
        index_offset=0
        output_ratio=1

        print('***** Processing image %d/%d (range=%.2f,%.2f) shape: %s' % (frame+1,imgshape[0],min_dt,max_dt,str(img.shape)))
        print('***** \t Total number of patches to evaluate = %d batch_size= %d keep_value=%.2f repeat=%d'
              % (total_patches,batch_size,keep_value,repeat))

        start_time = time.time()

        repeat_count = 0
        while repeat_count < repeat:

            if (repeat>1):
                print('***** \t Repeat=%d/%d' % (repeat_count+1,repeat))
            count = 0
            last=0.0

            while count < total_patches:
                idx_begin = count
                idx_end = idx_begin + batch_size
                if idx_end > total_patches:
                    idx_end = total_patches

                fraction=idx_begin/total_patches
                if fraction>=last:
                    debug_print('***** \t\t\t [%6d,%6d]/%6d' % (idx_begin, idx_end,total_patches))
                    last+=0.1

                current_indexes = patch_indexes[idx_begin:idx_end,:]
                current_patches = putil.getPatchesFromIndexes(img,
                                                              current_indexes,
                                                              patch_size,
                                                              padding='SAME',
                                                              dtype=np.float32)

                if len(current_patches.shape) < 4:
                    current_patches = np.expand_dims(current_patches,3)
                if threed and len(current_patches.shape)<5:
                    current_patches=np.expand_dims(current_patches, 5)

                feed_dict={images_pointer: current_patches,
                           keep_pointer : keep_value }
                try:
                    predictions = np.array(sess.run([model_output_image], feed_dict=feed_dict)[0])
                except Exception as e:
                    handle_error_and_exit(e)


                if have_result_patches is False:
                    pshape=predictions.shape
                    output_ratio=int(pshape[0]/batch_size)
                    if output_ratio < 1:
                        output_ratio = 1
                    num_output_patches=total_patches*output_ratio

                    if output_ratio>1:
                        print('***** \t\t in cropped mode ... original num patches=',total_patches, 'ratio=',
                              output_ratio,'num_output_patches=',num_output_patches,' in_shape=',current_patches.shape,' -->',
                              pshape)
                    
                    # Evaluate the patch indexes in small batches
                    if (threed):
                        result_patches = np.zeros((num_output_patches, pshape[1],pshape[2], pshape[3], pshape[4]), dtype=np.float32)
                    else:
                        result_patches = np.zeros((num_output_patches, pshape[1], pshape[2], pshape[3]), dtype=np.float32)
                    have_result_patches = True

                    if output_ratio==1:
                        index_offset=int((patch_size[0]-pshape[1])/2)


                #                print('storing in ',[idx_begin*output_ratio,idx_end*output_ratio])
                result_patches[idx_begin*output_ratio:idx_end*output_ratio,:,:,:] = predictions
                count += batch_size

            # Now that you are done reconstructing combine
            #
            #  If size of out != size of img.shape then manipukate indexes
            #
            #  or ... interpolate patches ...
            #
            #
            # if output_ratio>1:
            #     recon_patch_indexes=quarter_cut_indices(patch_indexes,width=pshape[1],batch_size=batch_size,threed=threed)
            # else:
            #     recon_patch_indexes=offset_indices(patch_indexes,index_offset,threed=threed)
                
            # if index_offset>0:
            #     print('*****\t Recon center crop input shape = ',img.shape,' recon=',result_patches.shape, ' index_offset=',index_offset,' indices=',patch_indexes[0],recon_patch_indexes[0])
            # elif output_ratio>1:
            #     print('*****\t Recon regular input shape but cropped = ',img.shape,' recon=',result_patches.shape, ' index_offset=',index_offset,' indices=',patch_indexes[0],recon_patch_indexes[0])
            
            recon_image_shape = img.shape[:]
            predictions_shape = predictions.shape[1:]
            if total_patches == 1:
                recon_image_shape = predictions_shape
            if not is_shape_same(patch_size,predictions_shape):
                l=len(recon_image_shape)-1;
                rs=list(recon_image_shape);
                rs[l]=result_patches[0].shape[l];
                recon_image_shape=tuple(rs);
            # Adjust for one-hot encoding
            if num_classes > 2 or one_hot_output:
                recon_image_shape += (num_classes,)

            # Initialize the output container                
            if repeat_count < 1:
                out = np.zeros(recon_image_shape, dtype=result_patches.dtype)

            # Handle one-hot encoding output
            if num_classes > 2 or one_hot_output:
                print('Using one-hot encoding output with %d classes' % (num_classes))
                out += putil.imagePatchReconOneHot(recon_image_shape,
                                             result_patches, 
                                             patch_indexes,
                                             num_classes,
                                             indent='  ',
                                             sigma=sigma,
                                             threed=threed)
            else:
                out += putil.imagePatchRecon(recon_image_shape,
                                             result_patches, 
                                             patch_indexes,
                                             indent='  ',
                                             sigma=sigma,
                                             threed=threed)

            repeat_count += 1


        # if (min_dt>=0.0):
        #     out=(out>0)*out

        if repeat > 1:
            print('***** Averaging %d output recon images' % (repeat))
            out /= repeat


        # Collapse the images back if not one-hot encoding
        if num_classes > 2 and not one_hot_output:
            out = np.argmax(out, axis=len(out.shape)-1).astype(out.dtype)


        # Crop Image Back to size
        # 4D, 3D and 2D
        if actual_pad_size is not None:
            print('***** \t cropping all around by '+str(actual_pad_size))
            out = putil.cropImage(out,offset=actual_pad_size,threed=threed)
            
        recon_image_list.append(out)
        duration = time.time() - start_time
        print('***** \t\t reconstruction total time= %.2f seconds' % duration)
        print('*****')

        frame=frame+1
    # Now recon the result image from all the predicted patches into a single image

    print('*****')
    print("***** Finished reconsructing %d image(s)" % (len(recon_image_list)))

    return recon_image_list
