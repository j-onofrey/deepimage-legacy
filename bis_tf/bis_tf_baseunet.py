#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import numpy as np
import tensorflow as tf
import bis_tf_utils as bisutil
import bis_tf_base_segmentationmodel as bisbasesegmentationmodel
import bis_tf_loss_utils as bislossutil


# -------------------------------------------------------
# Unet Model Functions
# -------------------------------------------------------
def create_unet_model(input_data, 
                      num_conv_layers=2, 
                      num_convs_per_layer=2,
                      filter_size=3, 
                      num_frames=1, 
                      num_filters=64, 
                      keep_pointer=None, 
                      num_classes=2, 
                      name='U-Net', 
                      dodebug=False, 
                      threed=False, 
                      norelu=False,
                      poolmode='max'):


    print('===== Creating Concat UNET Model',name)
    
    with tf.variable_scope(name) as scope:

        # 1. Contracting convolution layers
        clayer_inputs = [ input_data ]
        clayer_outputs = [ ]
        num_out_channels = num_filters

        for clayer in range(1,num_conv_layers+1):
            cname = str(clayer)

            with tf.variable_scope("conv_"+cname) as scope:
                if dodebug:
                    print('====\n==== Convolution Layer '+cname)

                relu = bisutil.createConvolutionLayerRELU_auto(input=clayer_inputs[clayer-1],
                                                                name='convolution_'+cname+'_1',
                                                                filter_size=filter_size,
                                                                num_out_channels=num_out_channels,
                                                                padvalue=num_convs_per_layer,
                                                                norelu=norelu,
                                                                threed=threed)
                for i in range(1,num_convs_per_layer):
                	conv_name = str(i+1)
	                relu = bisutil.createConvolutionLayerRELU_auto(input=relu,
	                                                                name='convolution_'+cname+'_'+conv_name,
	                                                                filter_size=filter_size,
	                                                                num_out_channels=num_out_channels,
	                                                                padvalue=0,
	                                                                norelu=norelu,
	                                                                threed=threed)
                clayer_outputs.append(relu)

                pool = bisutil.createPoolingLayer(input=relu,
                                                  name='pooling'+cname,
                                                  mode=poolmode,
                                                  threed=threed)
                num_out_channels = 2*num_out_channels
                clayer_inputs.append(pool)


        # 2. Last convolution layer, no pool
        clayer = num_conv_layers+1
        cname = 'middle'
        with tf.variable_scope("conv_"+cname) as scope:
            if dodebug:
                print('====\n==== Middle Convolution Layer ')

            relu1 = bisutil.createConvolutionLayerRELU_auto(input=clayer_inputs[clayer-1],
                name='convolution_'+cname+'_1',
                filter_size=filter_size,
                num_out_channels=num_out_channels,
                padvalue=2,
                norelu=norelu,
                threed=threed)
            if keep_pointer is not None:
                relu1 = tf.nn.dropout(relu1, keep_prob=keep_pointer)

            relu_final = bisutil.createConvolutionLayerRELU_auto(input=relu1,
                name='convolution_'+cname+'_2',
                filter_size=filter_size,
                num_out_channels=num_out_channels,
                padvalue=0,
                norelu=norelu,
                threed=threed)
            if keep_pointer is not None:
                relu_final = tf.nn.dropout(relu_final, keep_prob=keep_pointer)

            clayer_inputs.append(relu_final)


        # 3. Expanding convolution (transpose) layers
        dindex=3
        if (threed):
            dindex=4
        dlayer_inputs = [ relu_final ]
        num_out_channels = int(num_out_channels/2)
        
        dlayer=num_conv_layers
        while (dlayer>0):
            dname=str(dlayer)
            with tf.variable_scope("deconv_"+dname):
                if dodebug:
                    print('=====\n===== Convolution Transpose Layer '+dname)

                clayer_in = clayer_inputs.pop()
                clayer_out = clayer_outputs.pop()
                
                input_shape = clayer_in.get_shape()
                output_shape = clayer_out.get_shape()
                
                upconv = bisutil.createDeconvolutionLayer_auto(input=dlayer_inputs[-1],
                    name='up-convolution_'+dname,
                    output_shape=tf.shape(clayer_out),
                    filter_size=2,
                    in_strides=2,
                    num_out_channels=output_shape[-1].value,
                    threed=threed)
                # Need to concat the two sets of features
                feature_concat = tf.concat([clayer_out, upconv], axis=dindex, name='concat_'+dname)
                
                relu = bisutil.createConvolutionLayerRELU_auto(input=feature_concat,
                    name='xconvolution_'+dname+'_1',
                    filter_size=filter_size,
                    num_out_channels=num_out_channels,
                    padvalue=num_convs_per_layer,
                    norelu=norelu,
                    threed=threed)

                for i in range(1,num_convs_per_layer):
	                conv_name = str(i+1)
	                relu = bisutil.createConvolutionLayerRELU_auto(input=relu,
	                    name='xconvolution_'+dname+'_'+conv_name,
	                    filter_size=filter_size,
	                    num_out_channels=num_out_channels,
	                    padvalue=0,
	                    norelu=norelu,
	                    threed=threed)

                num_out_channels = int(num_out_channels/2)
                dlayer_inputs.append(relu)
                dlayer=dlayer-1



        # 4. Final convolution layer
        with tf.variable_scope("deconv_final"):
            dname='final'
            if dodebug:
                print('=====\n===== Final Convolution Layer '+dname)

            conv_final = bisutil.createConvolutionLayerRELU_auto(input=dlayer_inputs[-1],
                name='xconvolution_'+dname+'_final',
                filter_size=1,
                num_out_channels=num_classes,
                padvalue=0,
                norelu=True,
                threed=threed)
            print("=====")


        return conv_final

# -----------------------------------------------------------------------------------------------------------

def create_fuse_unet_model(input_data, 
	                       num_conv_layers=2, 
	                       num_convs_per_layer=2,
                           filter_size=3, 
                           num_frames=1, 
                           num_filters=64, 
                           keep_pointer=None, 
                           num_classes=2, 
                           name='U-Net', 
                           dodebug=False, 
                           threed=False, 
                           norelu=False,
                           poolmode='max'):


    print('===== Creating FUSE UNET Model',name)
    
    with tf.variable_scope(name) as scope:

        # 1. Contracting convolution layers
        clayer_inputs = [ input_data ]
        clayer_outputs = [ ]
        num_out_channels = num_filters


        for clayer in range(1,num_conv_layers+1):
            cname = str(clayer)

            with tf.variable_scope("conv_"+cname) as scope:
                if dodebug:
                    print('====\n==== Convolution Layer '+cname)

                relu = bisutil.createConvolutionLayerRELU_auto(input=clayer_inputs[clayer-1],
                                                                name='convolution_'+cname+'_1',
                                                                filter_size=filter_size,
                                                                num_out_channels=num_out_channels,
                                                                padvalue=num_convs_per_layer,
                                                                norelu=norelu,
                                                                threed=threed)
                for i in range(1,num_convs_per_layer):
                    conv_name = str(i+1)
                    relu = bisutil.createConvolutionLayerRELU_auto(input=relu,
                                                                    name='convolution_'+cname+'_'+conv_name,
                                                                    filter_size=filter_size,
                                                                    num_out_channels=num_out_channels,
                                                                    padvalue=0,
                                                                    norelu=norelu,
                                                                    threed=threed)
                clayer_outputs.append(relu)

                pool = bisutil.createPoolingLayer(input=relu,
                                                  name='pooling'+cname,
                                                  mode=poolmode,
                                                  threed=threed)
                num_out_channels = 2*num_out_channels
                clayer_inputs.append(pool)


        # 2. Last convolution layer, no pool
        clayer = num_conv_layers+1
        cname = 'middle'
        with tf.variable_scope("conv_"+cname) as scope:
            if dodebug:
                print('====\n==== Middle Convolution Layer ')

            relu1 = bisutil.createConvolutionLayerRELU_auto(input=clayer_inputs[clayer-1],
                name='convolution_'+cname+'_1',
                filter_size=filter_size,
                num_out_channels=num_out_channels,
                padvalue=2,
                norelu=norelu,
                threed=threed)
            if keep_pointer is not None:
                relu1 = tf.nn.dropout(relu1, keep_prob=keep_pointer)

            relu_final = bisutil.createConvolutionLayerRELU_auto(input=relu1,
                name='convolution_'+cname+'_2',
                filter_size=filter_size,
                num_out_channels=num_out_channels,
                padvalue=0,
                norelu=norelu,
                threed=threed)
            if keep_pointer is not None:
                relu_final = tf.nn.dropout(relu_final, keep_prob=keep_pointer)

            clayer_inputs.append(relu_final)


        # 3. Expanding convolution (transpose) layers
        dindex=3
        if (threed):
            dindex=4
        dlayer_inputs = [ relu_final ]
        num_out_channels = int(num_out_channels/2)
        
        dlayer=num_conv_layers
        while (dlayer>0):
            dname=str(dlayer)
            with tf.variable_scope("deconv_"+dname):
                if dodebug:
                    print('=====\n===== Convolution Transpose Layer '+dname)

                clayer_in = clayer_inputs.pop()
                clayer_out = clayer_outputs.pop()
                
                input_shape = clayer_in.get_shape()
                output_shape = clayer_out.get_shape()
                
                upconv = bisutil.createDeconvolutionLayer_auto(input=dlayer_inputs[-1],
                    name='up-convolution_'+dname,
                    output_shape=tf.shape(clayer_out),
                    filter_size=2,
                    in_strides=2,
                    num_out_channels=output_shape[-1].value,
                    threed=threed)
                # Need to concat the two sets of features
                feature_add = tf.add(clayer_out,upconv, name='fuse_'+dname)

                relu = bisutil.createConvolutionLayerRELU_auto(input=feature_add,
                    name='xconvolution_'+dname+'_1',
                    filter_size=filter_size,
                    num_out_channels=num_out_channels,
                    padvalue=num_convs_per_layer,
                    norelu=norelu,
                    threed=threed)

                for i in range(1,num_convs_per_layer):
                    conv_name = str(i+1)
                    relu = bisutil.createConvolutionLayerRELU_auto(input=relu,
                        name='xconvolution_'+dname+'_'+conv_name,
                        filter_size=filter_size,
                        num_out_channels=num_out_channels,
                        padvalue=0,
                        norelu=norelu,
                        threed=threed)

                num_out_channels = int(num_out_channels/2)
                dlayer_inputs.append(relu)
                dlayer=dlayer-1



        # 4. Final convolution layer
        with tf.variable_scope("deconv_final"):
            if dodebug:
                print('=====\n===== Final Convolution Layer '+dname)

            conv_final = bisutil.createConvolutionLayerRELU_auto(input=dlayer_inputs[-1],
                name='xconvolution_'+dname+'_final',
                filter_size=1,
                num_out_channels=num_classes,
                padvalue=0,
                norelu=True,
                threed=threed)
            print("=====")


        return conv_final


# -----------------------------------------------------------------------------------------------------------    

def create_nobridge_unet_model(input_data, 
		                       num_conv_layers=2, 
		                       num_convs_per_layer=2,
                               filter_size=3, 
                               num_frames=1, 
                               num_filters=64, 
                               keep_pointer=None, 
                               num_classes=2, 
                               name='U-Net', 
                               dodebug=False, 
                               threed=False, 
                               norelu=False,
                               poolmode='max'):


    print('===== Creating NO BRIDGE UNET Model',name)
    
    with tf.variable_scope(name) as scope:

        # 1. Contracting convolution layers
        clayer_inputs = [ input_data ]
        clayer_outputs = [ ]
        num_out_channels = num_filters


        for clayer in range(1,num_conv_layers+1):
            cname = str(clayer)

            with tf.variable_scope("conv_"+cname) as scope:
                if dodebug:
                    print('====\n==== Convolution Layer '+cname)

                relu = bisutil.createConvolutionLayerRELU_auto(input=clayer_inputs[clayer-1],
                                                                name='convolution_'+cname+'_1',
                                                                filter_size=filter_size,
                                                                num_out_channels=num_out_channels,
                                                                padvalue=num_convs_per_layer,
                                                                norelu=norelu,
                                                                threed=threed)
                for i in range(1,num_convs_per_layer):
                    conv_name = str(i+1)
                    relu = bisutil.createConvolutionLayerRELU_auto(input=relu,
                                                                    name='convolution_'+cname+'_'+conv_name,
                                                                    filter_size=filter_size,
                                                                    num_out_channels=num_out_channels,
                                                                    padvalue=0,
                                                                    norelu=norelu,
                                                                    threed=threed)
                clayer_outputs.append(relu)


                pool = bisutil.createPoolingLayer(input=relu,
                                                  name='pooling'+cname,
                                                  mode=poolmode,
                                                  threed=threed)
                num_out_channels = 2*num_out_channels
                clayer_inputs.append(pool)


        # 2. Last convolution layer, no pool
        clayer = num_conv_layers+1
        cname = 'middle'
        with tf.variable_scope("conv_"+cname) as scope:
            if dodebug:
                print('====\n==== Middle Convolution Layer ')

            relu1 = bisutil.createConvolutionLayerRELU_auto(input=clayer_inputs[clayer-1],
                name='convolution_'+cname+'_1',
                filter_size=filter_size,
                num_out_channels=num_out_channels,
                padvalue=2,
                norelu=norelu,
                threed=threed)
            if keep_pointer is not None:
                relu1 = tf.nn.dropout(relu1, keep_prob=keep_pointer)

            relu_final = bisutil.createConvolutionLayerRELU_auto(input=relu1,
                name='convolution_'+cname+'_2',
                filter_size=filter_size,
                num_out_channels=num_out_channels,
                padvalue=0,
                norelu=norelu,
                threed=threed)
            if keep_pointer is not None:
                relu_final = tf.nn.dropout(relu_final, keep_prob=keep_pointer)

            clayer_inputs.append(relu_final)


        # 3. Expanding convolution (transpose) layers
        dindex=3
        if (threed):
            dindex=4
        dlayer_inputs = [ relu_final ]
        num_out_channels = int(num_out_channels/2)
        
        dlayer=num_conv_layers
        while (dlayer>0):
            dname=str(dlayer)
            with tf.variable_scope("deconv_"+dname):
                if dodebug:
                    print('=====\n===== Convolution Transpose Layer '+dname)

                clayer_in = clayer_inputs.pop()
                clayer_out = clayer_outputs.pop()
                
                input_shape = clayer_in.get_shape()
                output_shape = clayer_out.get_shape()
                
                upconv = bisutil.createDeconvolutionLayer_auto(input=dlayer_inputs[-1],
                    name='up-convolution_'+dname,
                    output_shape=tf.shape(clayer_out),
                    filter_size=2,
                    in_strides=2,
                    num_out_channels=output_shape[-1].value,
                    threed=threed)

                # No concat here
                relu = bisutil.createConvolutionLayerRELU_auto(input=upconv,
                    name='xconvolution_'+dname+'_1',
                    filter_size=filter_size,
                    num_out_channels=num_out_channels,
                    padvalue=num_convs_per_layer,
                    norelu=norelu,
                    threed=threed)

                for i in range(1,num_convs_per_layer):
                    conv_name = str(i+1)
                    relu = bisutil.createConvolutionLayerRELU_auto(input=relu,
                        name='xconvolution_'+dname+'_'+conv_name,
                        filter_size=filter_size,
                        num_out_channels=num_out_channels,
                        padvalue=0,
                        norelu=norelu,
                        threed=threed)

                num_out_channels = int(num_out_channels/2)
                dlayer_inputs.append(relu)
                dlayer=dlayer-1



        # 4. Final convolution layer
        with tf.variable_scope("deconv_final"):
            if dodebug:
                print('=====\n===== Final Convolution Layer '+dname)

            conv_final = bisutil.createConvolutionLayerRELU_auto(input=dlayer_inputs[-1],
                name='xconvolution_'+dname+'_final',
                filter_size=1,
                num_out_channels=num_classes,
                padvalue=0,
                norelu=True,
                threed=threed)
            print("=====")


        return conv_final

# -----------------------------------------------------------------------------------------------------------
#
# Model cleanup
#
# Fix Number of Levels
#
# -----------------------------------------------------------------------------------------------------------
def get_pool_mode(avg_pool=False):

    poolmode="max"
    if avg_pool:
        poolmode="avg"
    return poolmode

def fix_number_of_conv_layers(num_conv_layers,patch_size):

    orig=num_conv_layers
    # Get the minimum patch size dimension, excluding the final channel dimension
    min_dim = min(patch_size[0:len(patch_size)-1])
    
    # Check if num_conv_layers is not too big
    while (int(math.pow(2,num_conv_layers+1))>min_dim):
        num_conv_layers=num_conv_layers-1

    if orig!=num_conv_layers:
        print('===== Reduced conv_layers from '+str(orig)+' to '+str(num_conv_layers)+' as patch size is too small.')

    return num_conv_layers

# -----------------------------------------------------------------------------------------------------------
#
# Get appropriate unet model function
#
# -----------------------------------------------------------------------------------------------------------
def get_unet_model_function(bridge_mode):

    # Create Model
    fun=create_unet_model
    if (bridge_mode=='fuse'):
        fun=create_fuse_unet_model
    elif (bridge_mode=='none'):
        fun=create_nobridge_unet_model

    return fun

# -----------------------------------------------------------------------------------------------------------
#
# Create output dictionary for loss functions
#
# -----------------------------------------------------------------------------------------------------------
def create_single_output(model_output,
                         num_classes=1,
                         threed=False):

    if num_classes>1:
        dim=len(model_output.get_shape())-1
        return tf.expand_dims(tf.argmax(model_output, dimension=dim),dim=dim)

    return model_output


def create_output_dictionary(model_output,
                             num_classes=1,
                             edge_smoothness=0.0,
                             threed=False,
                             max_outputs=3,
                             scope_name="Outputs",
                             output_name="Output"):

    with tf.variable_scope(scope_name):
        if num_classes>1:
            print("===== \t in classification mode (adding annotation)")
            dim=len(model_output.get_shape())-1
            annotation_pred = tf.argmax(model_output, axis=dim, name='prediction')
            output= { "image":  tf.expand_dims(annotation_pred, dim=dim,name=output_name),
                      "logits": tf.identity(model_output,name="Logits") ,
                      "regularizer": None}
        else:
            print("===== \t in regression mode")
            output = { "image":  tf.identity(model_output,name="Output"),
                       "logits": tf.identity(model_output,name="Logits"),
                       "regularizer" : None };

        if max_outputs>0:
            o_shape=output['image'].get_shape()
            final_shape = [o_shape[1].value, o_shape[2].value, 1]
            if threed:
                final_shape[2] = o_shape[3].value

            bisutil.image_summary(output['image'],'prediction',final_shape,1,threed,
                                  max_outputs=max_outputs)

        if (edge_smoothness>0.0):
            output['regularizer']=bisutil.createSmoothnessLayer(output['image'],
                                                                num_classes,threed)
        else:
            print('===== \t\t not adding back end smoothness computation')

    return output


# -------------------------------------------------------
# Some Static Functions for UNET derivatives
# -------------------------------------------------------
def create_unet_parameters(obj):

    obj.params['no_relu']=False
    obj.params['avg_pool']=False
    obj.params['filter_size']=3
    obj.params['num_filters']=64
    obj.params['num_conv_layers']=3
    obj.commandlineparams['edge_smoothness']=0.0
    obj.donotreadlist.append('edge_smoothness')

            

def add_unet_commandline_parameters(parser,training):

    if training:
        parser.add_argument('--no_relu', help='If set no RELU units will be used',
                            default=None,action='store_true')
        parser.add_argument('--avg_pool', help='If set will use avg_pool instead of max_pool',
                            default=None, action='store_true')
        parser.add_argument('-l','--smoothness', help='Smoothness factor (Baysian training with edgemap)',
                                default=None,type=float)
        parser.add_argument('--num_conv_layers', help='Number of Convolution Layers',
                            default=None,type=int)
        parser.add_argument('--num_filters', help='Number of Convolution Filters',
                            default=None,type=int)
        parser.add_argument('--filter_size',   help='Filter Size  in Convolutional Layers',
                            default=None,type=int)


def extract_unet_commandline_parameters(obj,args,training=False):

    if training:
        bislossutil.extract_parser_loss_params(obj,args)
        obj.set_param_from_arg(name='no_relu',value=args.no_relu)
        obj.set_param_from_arg(name='avg_pool',value=args.avg_pool)
        obj.set_param_from_arg(name='num_conv_layers',
                                value=bisutil.force_inrange(args.num_conv_layers,minv=2,maxv=8))
        obj.set_param_from_arg(name='filter_size',
                                value=bisutil.force_inrange(args.filter_size,minv=3,maxv=11))
        obj.set_param_from_arg(name='num_filters',
                                value=bisutil.getpoweroftwo(args.num_filters,4,128))
        obj.set_commandlineparam_from_arg(name='edge_smoothness',
                                value=bisutil.force_inrange(args.smoothness,minv=0.0,maxv=10000.0))


        
# -------------------------------------------------------

class BaseUNET(bisbasesegmentationmodel.BaseSegmentationModel):

    def __init__(self):
        self.params['name']='UNet'
        create_unet_parameters(self)
        bislossutil.create_loss_params(self)
        super().__init__()

    def get_description(self):
        return " U-Net Model for 2D/3D images."

    def can_train(self):
        return False

    def add_custom_commandline_parameters(self,parser,training):
        add_unet_commandline_parameters(parser,training)
        bislossutil.add_parser_loss_params(parser)

    def extract_custom_commandline_parameters(self,args,training=False):

        if training:
            extract_unet_commandline_parameters(self,args,training)
            bislossutil.extract_parser_loss_params(self,args)


    def create_loss(self,output_dict):
        return bislossutil.create_loss_function(self,
                                                output_dict=output_dict,
                                                smoothness=self.commandlineparams['edge_smoothness'],
                                                batch_size=self.commandlineparams['batch_size']);




if __name__ == '__main__':

    UNetModel().execute()
