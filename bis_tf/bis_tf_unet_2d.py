#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import numpy as np
import tensorflow as tf
import bis_tf_utils as bisutil
import bis_tf_loss_utils as bislossutil
import bis_tf_base_segmentationmodel as bisbasesegmentationmodel



# -------------------------------------------------------
# Unet Model Functions
# -------------------------------------------------------
def create_model(input_data, 
                  num_conv_layers=3, 
                  num_convs_per_layer=2,
                  filter_size=3, 
                  num_frames=1, 
                  num_filters=64, 
                  keep_pointer=None, 
                  num_classes=2, 
                  bridge_mode='concat',
                  name='U-Net', 
                  dodebug=False):


    print('===== Creating U-Net Model',name)
    
    with tf.variable_scope(name) as scope:

        in_shape = input_data.get_shape()
        print('Input data shape: '+str(in_shape))

        # Global padding values
        pad_values=[[0,0],[num_convs_per_layer,num_convs_per_layer],[num_convs_per_layer,num_convs_per_layer],[0,0]]


        # 1. Contracting convolution layers
        clayer_inputs = [ input_data ]
        clayer_outputs = [ ]
        num_out_channels = num_filters

        for clayer in range(1,num_conv_layers+1):
            cname = str(clayer)

            with tf.variable_scope("conv_"+cname) as scope:
                if dodebug:
                    print('====\n==== Convolution Layer '+cname)

                padding = tf.pad(clayer_inputs[clayer-1], paddings=pad_values, mode='SYMMETRIC', name='padding_'+name)
                print('***** Adding padding: '+str(pad_values))

                relu = tf.layers.conv2d(inputs=padding, 
                                filters=num_out_channels, 
                                kernel_size=filter_size, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                activation=tf.nn.relu,
                                name='convolution_'+cname+'_1')
                print('feature shape: '+str(relu.get_shape()))

                for i in range(1,num_convs_per_layer):
                    conv_name = str(i+1)
                    relu = tf.layers.conv2d(inputs=relu, 
                                    filters=num_out_channels, 
                                    kernel_size=filter_size, 
                                    strides=1, 
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                    activation=tf.nn.relu,
                                    name='convolution_'+cname+'_'+conv_name)
                    print('feature shape: '+str(relu.get_shape()))

                clayer_outputs.append(relu)

                pool = tf.layers.max_pooling2d(inputs=relu,
                                  pool_size=2,
                                  strides=2)
                num_out_channels = 2*num_out_channels
                clayer_inputs.append(pool)


        # 2. Last convolution layer, no pool
        clayer = num_conv_layers+1
        cname = 'middle'
        with tf.variable_scope("conv_"+cname) as scope:
            if dodebug:
                print('====\n==== Middle Convolution Layer ')

            padding = tf.pad(clayer_inputs[clayer-1], paddings=pad_values, mode='SYMMETRIC', name='padding_'+name)
            print('***** Adding padding: '+str(pad_values))

            relu = tf.layers.conv2d(inputs=padding, 
                            filters=num_out_channels, 
                            kernel_size=filter_size, 
                            strides=1, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                            activation=tf.nn.relu,
                            name='convolution_'+cname+'_1')
            print('feature shape: '+str(relu.get_shape()))
            if keep_pointer is not None:
                relu = tf.nn.dropout(relu, keep_prob=keep_pointer, name='dropout_'+cname+'_1')

            for i in range(1,num_convs_per_layer):
                conv_name = str(i+1)
                relu = tf.layers.conv2d(inputs=relu, 
                                filters=num_out_channels, 
                                kernel_size=filter_size, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                activation=tf.nn.relu,
                                name='convolution_'+cname+'_'+conv_name)
                print('feature shape: '+str(relu.get_shape()))

            if keep_pointer is not None:
                relu = tf.nn.dropout(relu, keep_prob=keep_pointer, name='dropout_'+cname+'_'+conv_name)

            clayer_inputs.append(relu)



        # 3. Expanding convolution (transpose) layers
        dindex=3
        # if (threed):
        #     dindex=4
        dlayer_inputs = [ relu ]
        num_out_channels = int(num_out_channels/2)
        
        dlayer=num_conv_layers
        while (dlayer>0):
            dname=str(dlayer)
            with tf.variable_scope("conv_transpose_"+dname):
                if dodebug:
                    print('=====\n===== Convolution Transpose Layer '+dname)

                clayer_in = clayer_inputs.pop()
                clayer_out = clayer_outputs.pop()
                
                input_shape = clayer_in.get_shape()
                output_shape = clayer_out.get_shape()
                
                upconv = tf.layers.conv2d_transpose(inputs=dlayer_inputs[-1], 
                                  filters=output_shape[-1].value, 
                                  kernel_size=2, 
                                  strides=2, 
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                  activation=None, 
                                  name='up-convolution_'+dname)
                print('upconv shape: '+str(upconv.get_shape()))


                # Need to concat the two sets of features
                if bridge_mode == 'concat':
                    feature_concat = tf.concat([clayer_out, upconv], axis=dindex, name='concat_'+dname)
                else:
                    feature_concat = tf.identity(upconv, name='identity_'+dname)

                padding = tf.pad(feature_concat, paddings=pad_values, mode='SYMMETRIC', name='padding_'+name)
                print('***** Adding padding: '+str(pad_values))

                relu = tf.layers.conv2d(inputs=padding, 
                                filters=num_out_channels, 
                                kernel_size=filter_size, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                activation=tf.nn.relu,
                                name='convolution_'+dname+'_1')
                print('feature shape: '+str(relu.get_shape()))

                for i in range(1,num_convs_per_layer):
                    conv_name = str(i+1)
                    relu = tf.layers.conv2d(inputs=relu, 
                                    filters=num_out_channels, 
                                    kernel_size=filter_size, 
                                    strides=1, 
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                    activation=tf.nn.relu,
                                    name='convolution_'+dname+'_'+conv_name)
                    print('feature shape: '+str(relu.get_shape()))


                num_out_channels = int(num_out_channels/2)
                dlayer_inputs.append(relu)
                dlayer=dlayer-1


        # 4. Final convolution layer
        with tf.variable_scope("deconv_final"):
            if dodebug:
                print('=====\n===== Final Convolution Layer '+dname)

            conv_final = tf.layers.conv2d(inputs=dlayer_inputs[-1], 
                                filters=num_classes, 
                                kernel_size=1, 
                                strides=1, 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                use_bias=False,
                                name='convolution_'+dname+'_final')
            print('feature shape: '+str(conv_final.get_shape()))
            print("=====")


        return conv_final



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

    return output


# -------------------------------------------------------
# Some Static Functions for UNET derivatives
# -------------------------------------------------------
def create_unet_parameters(obj):

    obj.params['filter_size']=3
    obj.params['num_conv_layers']=3
    obj.params['num_filters']=64
            

def add_unet_commandline_parameters(parser,training):

    if training:
        parser.add_argument('--filter_size',   help='Filter Size  in Convolutional Layers',
                            default=None,type=int)
        parser.add_argument('--num_conv_layers', help='Number of Convolution Layers',
                            default=None,type=int)
        parser.add_argument('--num_convs_per_layer', help='Number of Convolutions Per Convolution Layer', 
                            default=None,type=int)
        parser.add_argument('--num_filters', help='Number of Convolution Filters',
                            default=None,type=int)
        parser.add_argument('--bridge_mode', help='Bridge Mode (one of concat, fuse, none)', 
                            default=None)


def extract_unet_commandline_parameters(obj,args,training=False):

    if training:
        bislossutil.extract_parser_loss_params(obj,args)
        obj.set_param_from_arg(name='filter_size',
                                value=bisutil.force_inrange(args.filter_size,minv=3,maxv=11))
        obj.set_param_from_arg(name='num_conv_layers',
                                value=bisutil.force_inrange(args.num_conv_layers,minv=2,maxv=8))
        obj.set_param_from_arg(name='num_filters',
                                value=bisutil.getpoweroftwo(args.num_filters,4,4096))
        obj.set_param_from_arg(name='num_convs_per_layer', 
                                value=bisutil.force_inrange(args.num_convs_per_layer,minv=1,maxv=8))
        obj.set_param_from_arg(name='bridge_mode',
                                value=args.bridge_mode,allowedvalues=obj.allowed_bridge_modes)

        
# -------------------------------------------------------

class UNet2DModel(bisbasesegmentationmodel.BaseSegmentationModel):

    allowed_bridge_modes = [ 'concat','fuse','none' ]

    def __init__(self):
        self.params['name']='U-Net'
        self.params['model_dim']=2
        self.params['num_convs_per_layer']=2 
        self.params['bridge_mode']='concat'  # one of concat, fuse, none
        self.donotreadlist.append('bridge_mode')
        create_unet_parameters(self)
        bislossutil.create_loss_params(self)
        super().__init__()

    def get_description(self):
        return "U-Net Model for 2D images."

    def can_train(self):
        return True

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
                                                batch_size=self.commandlineparams['batch_size']);

    def create_inference(self,training=True):

        if training:
            self.add_parameters_as_variables()

        print('===== Creating U-Net (2D) Inference Model ')
        print('=====')

        input_data=self.pointers['input_pointer']
        bisutil.print_dict({ 'Num Conv/Deconv Layers' : self.params['num_conv_layers'],
                             'Num Convs Per Layer': self.params['num_convs_per_layer'],
                             'Bridge Mode': self.params['bridge_mode'],
                             'Filter Size': self.params['filter_size'],
                             'Num Filters': self.params['num_filters'],
                             'Num Classes':  self.calcparams['num_classes'],
                             'Patch Size':   self.calcparams['patch_size'],
                             'Num Frames':   self.calcparams['num_frames']},
                           extra="=====",header="Model Parameters:")

        # Create Model
        model_output = create_model(input_data, 
                                      num_conv_layers=self.params['num_conv_layers'],
                                      num_convs_per_layer=self.params['num_convs_per_layer'],
                                      filter_size=self.params['filter_size'],
                                      num_frames=self.calcparams['num_frames'],
                                      num_filters=self.params['num_filters'],
                                      keep_pointer=self.pointers['keep_pointer'],
                                      num_classes=self.calcparams['num_classes'],
                                      bridge_mode=self.params['bridge_mode'],
                                      name='U-Net',
                                      dodebug=self.commandlineparams['debug'])        

        return create_output_dictionary(model_output,
                                        num_classes=self.calcparams['num_classes'],
                                        threed=self.calcparams['threed'],
                                        max_outputs=self.commandlineparams['max_outputs'])


if __name__ == '__main__':

    UNet2DModel().execute()
