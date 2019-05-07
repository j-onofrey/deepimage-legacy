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
import bis_tf_image_utils as bisimageutil
import bis_tf_baseunet as bisbaseunet
# -----------------------------------------------------------------

class CropUnet(bisbaseunet.BaseUNET):

    def __init__(self):
        super().__init__()
        self.params['name']='CropUNet'

    def get_description(self):
        return " Crop U-Net (Auto) Model for 2D/3D images."

    def can_train(self):
        return True

    # --------------------------------------------------------------------------------------------
    # Main Create Inference
    # --------------------------------------------------------------------------------------------

    def create_inference(self,training=True):

        dodebug=self.commandlineparams['debug']
        poolmode=bisbaseunet.get_pool_mode(self.params['avg_pool'])
        input_data=self.pointers['input_pointer']
        threed=self.calcparams['threed']
        final_width=(self.commandlineparams['patch_size']/2)

        if training:
            self.add_parameters_as_variables()
            
        cname=bisutil.getdimname(threed)
        print('===== Creating '+cname+' Crop U-Net Inference Model')
        print('=====')


        self.params['num_conv_layers']=bisbaseunet.fix_number_of_conv_layers(self.params['num_conv_layers'],
                                                                             self.commandlineparams['patch_size'])
        
        bisutil.print_dict({ 'Num Conv/Deconv Layers' : self.params['num_conv_layers'],
                             'Filter Size': self.params['filter_size'],
                             'Num Filters': self.params['num_filters'],
                             'Num Classes':  self.calcparams['num_classes'],
                             'Orig Patch Size':   self.commandlineparams['patch_size'],
                             'Final Patch Size':final_width,
                             'Patch Size':   self.commandlineparams['patch_size'],
                             'Num Frames':   self.calcparams['num_frames'],
                             'NoRelu':self.params['no_relu'],
                             'PoolMode':poolmode},
                           extra="=====",header="Model Parameters:")
        
        # Get Model Function
        model_function=bisbaseunet.create_unet_model


        with tf.variable_scope("LowRes") as scope:
            new_input=bisimageutil.quarterImage(input_data,threed=threed,comment=" quarter input image ")
            new_target=bisimageutil.quarterImage(self.pointers['target_pointer'],
                                             threed=threed,comment=" quarter input image ")
            o_shape=new_input.get_shape()
            bisutil.image_summary(new_input,'cropped_input',o_shape[3].value,1,
                                  threed, max_outputs=self.commandlineparams['max_outputs'])
            bisutil.image_summary(new_target,'cropped_target',o_shape[3].value,1,
                                  threed, max_outputs=self.commandlineparams['max_outputs'])

            
        model_output=model_function(new_input,
                                    num_conv_layers=self.params['num_conv_layers'],
                                    filter_size=self.params['filter_size'],
                                    num_frames=self.calcparams['num_frames'],
                                    num_filters=self.params['num_filters'],
                                    keep_pointer=self.pointers['keep_pointer'],
                                    num_classes=self.calcparams['num_classes'],
                                    name="Crop-Unet",
                                    dodebug=self.commandlineparams['debug'],
                                    threed=threed,
                                    norelu=self.params['no_relu'],
                                    poolmode=poolmode)
        
        output=bisbaseunet.create_output_dictionary(model_output,
                                                    num_classes=self.calcparams['num_classes'],
                                                    edge_smoothness=self.commandlineparams['edge_smoothness'],
                                                    threed=self.calcparams['threed'],
                                                    max_outputs=self.commandlineparams['max_outputs'])
        
        output['cropped_target']= new_target
        return output

if __name__ == '__main__':

    CropUnet().execute()
