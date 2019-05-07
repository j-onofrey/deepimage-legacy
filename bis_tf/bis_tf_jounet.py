#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import numpy as np
import tensorflow as tf
import bis_tf_utils as bisutil
import bis_tf_baseunet as bisbaseunet


class JOUnetModel(bisbaseunet.BaseUNET):

    allowed_bridge_modes = [ 'concat','fuse','none' ]
    
    def __init__(self):
        super().__init__()
        self.params['num_convs_per_layer']=2 
        self.params['bridge_mode']='concat'  # one of concat, fuse, none
        self.donotreadlist.append('bridge_mode')
        self.params['name']='JOUNet'

    def get_description(self):
        return "U-Net (Auto) Model for 2D/3D images."

    def can_train(self):
        return True

    def add_custom_commandline_parameters(self,parser,training):

        if training:
            super().add_custom_commandline_parameters(parser,training)
            parser.add_argument('--num_convs_per_layer', help='Number of Convolutions Per Convolution Layer', default=None,type=int)
            parser.add_argument('--bridge_mode', help='Bridge Mode (one of concat, fuse, none)', default=None)


    def extract_custom_commandline_parameters(self,args,training=False):

        if training:
            super().extract_custom_commandline_parameters(args,training)
            self.set_param_from_arg(name='num_convs_per_layer', value=bisutil.force_inrange(args.num_convs_per_layer,minv=1,maxv=8))
            self.set_param_from_arg(name='bridge_mode',value=args.bridge_mode,allowedvalues=self.allowed_bridge_modes)


    # --------------------------------------------------------------------------------------------
    # Variable length Inference
    # --------------------------------------------------------------------------------------------

    def create_inference(self,training=True):


        if training:
            self.add_parameters_as_variables()

        cname=bisutil.getdimname(self.calcparams['threed'])
        print('===== Creating '+cname+' U-Net Inference Model ')
        print('=====')

        
        input_data=self.pointers['input_pointer']
        poolmode=bisbaseunet.get_pool_mode(self.params['avg_pool'])
        self.params['num_conv_layers']=bisbaseunet.fix_number_of_conv_layers(self.params['num_conv_layers'],
                                                                 self.calcparams['patch_size'])
        bisutil.print_dict({ 'Num Conv/Deconv Layers' : self.params['num_conv_layers'],
                             'Num Convs Per Layer': self.params['num_convs_per_layer'],
                             'Bridge Mode': self.params['bridge_mode'],
                             'Filter Size': self.params['filter_size'],
                             'Num Filters': self.params['num_filters'],
                             'Num Classes':  self.calcparams['num_classes'],
                             'Patch Size':   self.calcparams['patch_size'],
                             'Num Frames':   self.calcparams['num_frames'],
                             'NoRelu':self.params['no_relu'],
                             'PoolMode':poolmode},
                           extra="=====",header="Model Parameters:")


        # Create Model
        model_function = bisbaseunet.get_unet_model_function(self.params['bridge_mode'])
        model_output = model_function(input_data, 
                                      num_conv_layers=self.params['num_conv_layers'],
                                      num_convs_per_layer=self.params['num_convs_per_layer'],
                                      filter_size=self.params['filter_size'],
                                      num_frames=self.calcparams['num_frames'],
                                      num_filters=self.params['num_filters'],
                                      keep_pointer=self.pointers['keep_pointer'],
                                      num_classes=self.calcparams['num_classes'],
                                      name='U-Net',
                                      dodebug=self.commandlineparams['debug'],
                                      threed=self.calcparams['threed'],
                                      norelu=self.params['no_relu'],
                                      poolmode=poolmode)
        

        return bisbaseunet.create_output_dictionary(model_output,
                                                num_classes=self.calcparams['num_classes'],
                                                edge_smoothness=self.commandlineparams['edge_smoothness'],
                                                threed=self.calcparams['threed'],
                                                max_outputs=self.commandlineparams['max_outputs'])

        


if __name__ == '__main__':

    JOUnetModel().execute()
