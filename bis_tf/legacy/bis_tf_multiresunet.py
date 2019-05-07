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
import bis_tf_optimizer_utils as bisoptutil
import bis_tf_baseunet as bisbaseunet
# -----------------------------------------------------------------

class MultiResUnetModel(bisbaseunet.BaseUNET):

    allowed_bridge_modes = [ 'concat','fuse','none' ]
    allowed_opt_modes = [ 'low','high','both' ]
    low_res_variables = []

    
    def __init__(self):
        super().__init__()
        self.params['name']='MultiResUNet'
        self.params['bridge_mode']='concat'  # one of concat, fuse, none
        self.params['opt_mode']='both'
        self.params['sigma']=2.0
        self.params['reduce']=2
        self.params['sigma2']=0.0
        self.params['reduce2']=1                
        self.params['crop_factor']=2
        self.params['center_crop']=False
        self.params['mask_high']=False

        self.donotreadlist.append('center_crop')
        self.donotreadlist.append('mask_high')
        self.donotreadlist.append('sigma2')
        self.donotreadlist.append('reduce2')
        self.donotreadlist.append('bridge_mode')
        self.donotreadlist.append('opt_mode')

    def get_description(self):
        return " Multires U-Net (Auto) Model for 2D/3D images."

    def can_train(self):
        return True

    def add_custom_commandline_parameters(self,parser,training):

        if training:
            super().add_custom_commandline_parameters(parser,training)
            parser.add_argument('--bridge_mode', help='Bridge Mode (one of concat, fuse, none)',
                                default=None)
            parser.add_argument('--opt_mode', help='Optimization Mode (one of low,high or both)',
                                default=None)
            parser.add_argument('--sigma', help='Gaussian sigma for smoothing (low resolution) prior to resolution reduction (default=2.0)',
                                default=None,type=float)
            parser.add_argument('--reduce', help='Resolution reduction factor (low resolution) (integer) for higher resolution (default=2)',
                                default=None,type=int)
            parser.add_argument('--sigma2', help='Gaussian sigma for smoothing (high resolution) prior to resolution reduction (default=0.0)',
                                default=None,type=float)
            parser.add_argument('--reduce2', help='Resolution reduction factor (high resolution) (integer) for higher resolution (default=1)',
                                default=None,type=int)
            parser.add_argument('--center_crop', help='If set will use center of patch for high res instead of whole patch',
                                default=None, action='store_true')
            parser.add_argument('--mask_high', help='If set will use mask output of high-res level close to boundary only',
                                default=None, action='store_true')
            parser.add_argument('--crop_factor', help='fraction 1/crop_factor of Image to keep for high res (default=2 --> 1/2 of image, values=1,2,4,8)',
                                default=None,type=int)
        


    def extract_custom_commandline_parameters(self,args,training=False):

        if training:
            super().extract_custom_commandline_parameters(args,training)
            self.set_param_from_arg(name='bridge_mode',value=args.bridge_mode,allowedvalues=self.allowed_bridge_modes)
            self.set_param_from_arg(name='opt_mode',value=args.opt_mode,allowedvalues=self.allowed_opt_modes)
            self.set_param_from_arg(name='sigma',
                                    value=bisutil.force_inrange(args.sigma,minv=0.1,maxv=100.0))
            self.set_param_from_arg(name='reduce',
                                    value=bisutil.getpoweroftwo(args.reduce,1,8))
            self.set_param_from_arg(name='reduce2',
                                    value=bisutil.getpoweroftwo(args.reduce2,1,8))
            self.set_param_from_arg(name='center_crop',value=args.center_crop)
            self.set_param_from_arg(name='mask_high',value=args.mask_high)
            self.set_param_from_arg(name='crop_factor',
                                    value=bisutil.getpoweroftwo(args.crop_factor,1,8))
            self.set_param_from_arg(name='sigma2',
                                    value=bisutil.force_inrange(args.sigma2,minv=0.0,maxv=100.0))

            
    def create_loss(self,output_dict):
        
        return bislossutil.create_loss_function(self,
                                                output_dict=output_dict,
                                                smoothness=self.commandlineparams['edge_smoothness'],
                                                batch_size=self.commandlineparams['batch_size']);
        


    def set_parameters_to_read_from_resume_model(self,path):

        self.list_of_variables_to_read=None

        
        # None means all global variables

        cname=bisutil.get_read_checkpoint(path)
        old=bisutil.get_tensor_list_from_checkpoint_file(cname, ['opt_mode']);
        oldmode=old['opt_mode']
        newmode=self.params['opt_mode']

        bisutil.debug_print('+++++ Scanning ' +cname+', Old mode=',oldmode,' New mode=',newmode)

        if oldmode=='low':
            # Only read low res variables
            self.list_of_variables_to_read=self.low_res_variables
            print("===== reading in only low_res_variables")


    # --------------------------------------------------------------------------------------------
    # Main Create Inference
    # --------------------------------------------------------------------------------------------

    def create_inference(self,training=True):

        dodebug=self.commandlineparams['debug']
        poolmode=bisbaseunet.get_pool_mode(self.params['avg_pool'])
        input_data=self.pointers['input_pointer']
        threed=self.calcparams['threed']

        if training:
            self.add_parameters_as_variables()
            
        cname=bisutil.getdimname(threed)
        print('===== Creating '+cname+' MultiRes U-Net Inference Model')
        print('=====')


        self.params['num_conv_layers']=bisbaseunet.fix_number_of_conv_layers(self.params['num_conv_layers'],
                                                                 self.commandlineparams['patch_size'])
        
        bisutil.print_dict({ 'Num Conv/Deconv Layers' : self.params['num_conv_layers'],
                             'Opt Mode:' : self.params['opt_mode'],
                             'Sigma:' : self.params['sigma'],
                             'Bridge Mode': self.params['bridge_mode'],
                             'Filter Size': self.params['filter_size'],
                             'Num Filters': self.params['num_filters'],
                             'Num Classes':  self.calcparams['num_classes'],
                             'Patch Size':   self.commandlineparams['patch_size'],
                             'Num Frames':   self.calcparams['num_frames'],
                             'NoRelu':self.params['no_relu'],
                             'Sigma=':self.params['sigma'],
                             'Reduce=':self.params['reduce'],
                             'Sigma2=':self.params['sigma2'],
                             'Reduce2=':self.params['reduce2'],
                             'Center Crop=':self.params['center_crop'],
                             'Mask High=':self.params['mask_high'],
                             'PoolMode':poolmode},
                           extra="=====",header="Model Parameters:")
        
        # Get Model Function
        model_function=bisbaseunet.get_unet_model_function(self.params['bridge_mode'])

        dindex=bisutil.getdindex(self.calcparams['threed'])


        
        with tf.variable_scope("LowResInp") as scope:
            
            print('=====\n===== Preparing Low Res Images sigma='+str(self.params['sigma'])+' reduce='+str(self.params['reduce']))
            lowres_input=bisimageutil.createReducedSizeImage(input_data,
                                                             threed=threed,
                                                             scope_name="Reduce_Input",
                                                             name="input",
                                                             sigma=self.params['sigma'],
                                                             reduce=self.params['reduce'],
                                                             num_classes=1,
                                                             max_outputs=self.commandlineparams['max_outputs'],
                                                             add_summary_images=True)
            
            
        print('=====\n===== Creating Low Res level input_image=',lowres_input.get_shape())
        cframes=lowres_input.get_shape()[dindex].value
        lowres_model_output=model_function(lowres_input,
                                           num_conv_layers=self.params['num_conv_layers'],
                                           filter_size=self.params['filter_size'],
                                           num_frames=cframes,
                                           num_filters=self.params['num_filters'],
                                           keep_pointer=self.pointers['keep_pointer'],
                                           num_classes=self.calcparams['num_classes'],
                                           name="LowRes",
                                           dodebug=self.commandlineparams['debug'],
                                           threed=threed,
                                           norelu=self.params['no_relu'],
                                           poolmode=poolmode)
        
        with tf.variable_scope("LowResOut") as scope:
            if (self.params['opt_mode']=='low'):
                low_res_dictionary=bisbaseunet.create_output_dictionary(lowres_model_output,
                                                                        num_classes=self.calcparams['num_classes'],
                                                                        edge_smoothness=self.commandlineparams['edge_smoothness'],
                                                                        threed=self.calcparams['threed'],
                                                                        max_outputs=self.commandlineparams['max_outputs'],
                                                                        scope_name='Output_Low')
                low_res_dictionary['cropped_target']=bisimageutil.createReducedSizeImage(self.pointers['target_pointer'],
                                                                                         threed=threed,
                                                                                         scope_name="Reduce_Target",
                                                                                         name="target",
                                                                                         num_classes=self.calcparams['num_classes'],
                                                                                         sigma=self.params['sigma'],
                                                                                         reduce=self.params['reduce'],
                                                                                         max_outputs=self.commandlineparams['max_outputs'],
                                                                                         add_summary_images=True)
                
            else:
                
                low_res_dictionary = {
                    'image' : bisbaseunet.create_single_output(lowres_model_output,
                                                               num_classes=self.calcparams['num_classes'],
                                                               threed=self.calcparams['threed'])
                }
                
            upsampled_output=bisimageutil.resampleTo(tf.cast(low_res_dictionary['image'],dtype=tf.float32),
                                                     target_image=input_data,
                                                     threed=threed,
                                                     interp=1)
            o_shape=upsampled_output.get_shape()
            bisutil.image_summary(upsampled_output,'lowres_pred',o_shape[3].value,1,
                                  threed, max_outputs=self.commandlineparams['max_outputs'])
            
            self.low_res_variables=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope="LowRes")

            
        if self.params['opt_mode']=='low':
            with tf.variable_scope('Outputs'):
                output_image=tf.identity(upsampled_output,name="Output")
            return low_res_dictionary
            
        # Essentially this is now full resolution, will crop later
        upsampled_lowres_model_output=bisimageutil.resampleTo(lowres_model_output,
                                                              target_image=input_data, # specifies dimensions
                                                              threed=threed,
                                                              interp=1)
        print('===== Created Upsampled low level output_image=',upsampled_lowres_model_output.get_shape())

        print('====== = = = = = = = = = = = = = = = = = = = = = = = = =')

        # Crop Input data ...
        # ...................
        reduce_size=self.commandlineparams['patch_size']/self.params['crop_factor']
        num_crop_pixels=int((self.commandlineparams['patch_size']-reduce_size)/2)
        
        # --------------- Adding High Res Layer ------------------------

        print('=====')
        print('===== On to High Res sigma2=',self.params['sigma2'])
        
        with tf.variable_scope('HighResInp'):

            crop_input=input_data
            if self.params['center_crop']:
                crop_input = bisimageutil.cropImageBoundary(input_data,threed=threed,comment='input_image',numpixels=num_crop_pixels)

            highres_input=crop_input                
            if self.params['sigma2']>0.0 or self.params['reduce2']>1:
                print('=====\t smoothing highres input sigma=',self.params['sigma2'],' reduce=',self.params['reduce2'],'\n')
                highres_input=bisimageutil.smoothAndReduce(crop_input,threed=threed,comment='smoothing high res input',name='smoothed_highres',
                                                           reduce=self.params['reduce2'],
                                                           sigma=self.params['sigma2'])
                
        print('===== Created High Res level input_image=',highres_input.get_shape(),' crop=',self.params['center_crop'],' (num_crop_pixels=',
              num_crop_pixels,') smoothing='+str(self.params['sigma2'])+' reduce='+str(self.params['reduce2'])+')')


        o_shape=upsampled_output.get_shape()
        with tf.variable_scope('MHResInput'):
            bisutil.image_summary(highres_input,'highres_input',o_shape[3].value,1,
                                  threed, max_outputs=self.commandlineparams['max_outputs'])

        
        cframes=highres_input.get_shape()[dindex].value                                             
        highres_model_output=model_function(highres_input,
                                            num_conv_layers=self.params['num_conv_layers'],
                                            filter_size=self.params['filter_size'],
                                            num_frames=cframes,
                                            num_filters=self.params['num_filters'],
                                            keep_pointer=self.pointers['keep_pointer'],
                                            num_classes=self.calcparams['num_classes'],
                                            name='HighRes',
                                            dodebug=self.commandlineparams['debug'],
                                            threed=threed,
                                            norelu=self.params['no_relu'],
                                            poolmode=poolmode)



        with tf.variable_scope("N_Combined") as scope:

            crop_upsampled_lowres_model_output=upsampled_lowres_model_output
            if (self.params['center_crop']):
                # crop upsampled low res model to bring to cropped space
                crop_upsampled_lowres_model_output= bisimageutil.cropImageBoundary(upsampled_lowres_model_output,threed=threed,comment='upsampled_lowresoutput',numpixels=num_crop_pixels)
            
                if self.calcparams['num_classes']==1:
                    tmp=crop_upsampled_lowres_model_output
                else:
                    dim=len(crop_upsampled_lowres_model_output.get_shape())-1
                    tmp=tf.expand_dims(tf.argmax(crop_upsampled_lowres_model_output, dimension=dim, name='prediction'),dim=dim)
                    o_shape=tmp.get_shape()
                    bisutil.image_summary(tmp,'crop_upsampled_lowres_model_output_pred',o_shape[3].value,1,
                                          threed, max_outputs=self.commandlineparams['max_outputs'])

                print('===== Creating Crop Low Res Output=',crop_upsampled_lowres_model_output.get_shape(),' (num_crop_pixels=',num_crop_pixels,')')


            upsampled_highres_model_output=highres_model_output                
            if self.params['reduce2']>1:
                upsampled_highres_model_output=bisimageutil.resampleTo(highres_model_output,
                                                                       target_image=crop_input, # specifies dimensions
                                                                       threed=threed,
                                                                       interp=1)


            if self.params['mask_high']:
                o_shape=input_data.get_shape()
                k=self.params['reduce']*4
                if k<3:
                    k=3
                name="average"+str(k)


                crop_upsampled_output=upsampled_output
                if (self.params['center_crop']):
                    # crop upsampled low res model to bring to cropped space
                    crop_upsampled_output= bisimageutil.cropImageBoundary(upsampled_output,threed=threed,comment='upsampled_lowresoutput',numpixels=num_crop_pixels)

                
                if threed:
                    mask1 = tf.nn.avg_pool3d(crop_upsampled_output, ksize=[1,k,k,k,1], strides=[1,1,1,1,1], padding='SAME', name=name)
                else:
                    mask1 = tf.nn.avg_pool(crop_upsampled_output, ksize=[1,k,k,1],strides=[1,1,1,1], padding='SAME', name=name)        
                    
                mask2=1.0-4.0*tf.square(0.5-mask1)

                if threed:
                    highres_mask = tf.nn.max_pool3d(mask2, ksize=[1,k,k,k,1], strides=[1,1,1,1,1], padding='SAME', name=name)
                else:
                    highres_mask = tf.nn.max_pool(mask2, ksize=[1,k,k,1],strides=[1,1,1,1], padding='SAME', name=name)        

                bisutil.image_summary(highres_mask,'highres_mask',o_shape[3].value,1,threed, max_outputs=self.commandlineparams['max_outputs'])


                masked_upsampled_highres_model_output=tf.multiply(highres_mask,upsampled_highres_model_output)
                print('===== Creating Masked Upsampled High Res Output=',upsampled_highres_model_output.get_shape(),' k-width='+str(k))
            else:
                print('===== Creating Upsampled High Res Output=',upsampled_highres_model_output.get_shape())
                masked_upsampled_highres_model_output=upsampled_highres_model_output
                
            
            combined_output = tf.add(crop_upsampled_lowres_model_output,masked_upsampled_highres_model_output, name='combined_outputs')
            

        print('====== = = = = = = = = = = = = = = = = = = = = = = = = =')
            
        output_dictionary=bisbaseunet.create_output_dictionary(combined_output,
                                                               num_classes=self.calcparams['num_classes'],
                                                               edge_smoothness=self.commandlineparams['edge_smoothness'],
                                                               threed=self.calcparams['threed'],
                                                               max_outputs=self.commandlineparams['max_outputs'])



        with tf.variable_scope("N_Combined") as scope:

            print("=====")

            if (self.params['center_crop']):
                print("===== \t creating center crop high res target")
                tmp_target=bisimageutil.cropImageBoundary(self.pointers['target_pointer'],
                                                          threed=threed,comment='upsampled_lowresoutput',numpixels=num_crop_pixels)
                output_dictionary['cropped_target']=tmp_target

            
        # -------- Set Exclude List and Optimizer Stuff -------------------
        
        if self.params['opt_mode']=='high':
            print('===== Not Optimizing low res unet, keeping this fixed')
            self.exclude_list=self.low_res_variables
            toopt=list(set(tf.trainable_variables()) - set(self.exclude_list))
            bisoptutil.set_list_of_variables_to_optimize(self,toopt)
        
        return output_dictionary

if __name__ == '__main__':

    MultiResUnetModel().execute()
