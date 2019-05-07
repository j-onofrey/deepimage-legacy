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
import bis_tf_baseunet as bisbaseunet
import bis_tf_optimizer_utils as bisoptutil

class DualSegm(bisbaseunet.BaseUNET):

    allowed_modes = [ 'encode','classify','both', 'separate' ]
    exclude_list = [];

    autoencoder_variables = []
    classifier_variables = []


    def __init__(self):
        super().__init__()
        self.params['mode']='classify'
        self.donotreadlist.append('mode')
        self.commandlineparams['recon_weight']=0.1
        self.params['name']='Dual-UNet'
        self.commandlineparams['metric']='ce'
        self.commandlineparams['pos_weight']=1.0

    def get_description(self):
        return " Dual Dual U-Net Model for 2D/3D images."

    def can_train(self):
        return True

    def add_custom_commandline_parameters(self,parser,training):

        if training:
            super().add_custom_commandline_parameters(parser,training)
            parser.add_argument('--mode', help='Bridge Mode (one of encode,classify,both, default=classify)',
                                default=None)
            parser.add_argument('--recon_weight', help='Weight of reconstruction similarity (when mode=encode or both)',
                                default=None,type=float)
    def extract_custom_commandline_parameters(self,args,training=False):

        if training:
            super().extract_custom_commandline_parameters(args,training)
            self.set_param_from_arg(name='mode',value=args.mode,allowedvalues=self.allowed_modes)
            self.set_commandlineparam_from_arg(name='recon_weight',value=bisutil.force_inrange(args.recon_weight,0.0,100.0))

    def create_loss(self,output_dict):

        batch_size=self.commandlineparams['batch_size'];
        class_loss=bislossutil.create_loss_function(self,
                                                    output_dict=output_dict,
                                                    smoothness=self.commandlineparams['edge_smoothness'],
                                                    batch_size=batch_size,name="Loss/Class_Loss")
        
        total_loss=class_loss
        
        if self.params['mode']!='classify' and self.commandlineparams['recon_weight']>0.0:

            print("===== Creating combined encode/classify metric. Encode weight (l2) =",self.commandlineparams['recon_weight'])
            input_data=self.pointers['input_pointer']
            recon_image=output_dict['encoding']

            with tf.variable_scope('Loss/Pred_Loss'):
                print('+++++ Creating Similary Function for encoding comparing:',recon_image.get_shape(),input_data.get_shape())
                normalizer_scalar=bislossutil.compute_normalizer(input_data.get_shape().as_list(),batch_size)*0.01
                loss_op=bislossutil.l2_loss(pred_image=tf.cast(recon_image,dtype=tf.float32),
                                            targets=input_data,
                                            normalizer=normalizer_scalar)

            with tf.variable_scope('Loss/Combined'):
                total_loss=tf.add(loss_op*self.commandlineparams['recon_weight'],class_loss,name='Combined')
                bisutil.add_scalar_summary('GrandTotal', total_loss,scope='Metrics/')
                

        if (self.params['mode']=="encode"):
            print('===== Optimizing Encoding Only')
            self.exclude_list=self.classifier_variables
        elif (self.params['mode']=="classify"):
            print('===== Optimizing Classification Only')
            self.exclude_list=self.autoencoder_variables
            
        toopt=list(set(tf.trainable_variables()) - set(self.exclude_list))
        bisoptutil.set_list_of_variables_to_optimize(self,toopt)

        return total_loss


    def set_parameters_to_read_from_resume_model(self,path):
        # None means all global variables

        cname=bisutil.get_read_checkpoint(path)
        old=bisutil.get_tensor_list_from_checkpoint_file(cname, ['mode']);
        oldmode=old['mode']
        newmode=self.params['mode']

        bisutil.debug_print('+++++ Scanning ' +cname+', Old mode=',oldmode,' New mode=',newmode)

        if oldmode=='both' or oldmode=='separate':
            # Read them all
            self.list_of_variables_to_read=None
        elif  oldmode=='classify':
            if newmode!='classify':
                self.list_of_variables_to_read=self.classifier_variables
        else:
            if newmode!='classify':
                self.list_of_variables_to_read=None
            else:
                raise ValueError('Cannot resume classify from an autoencoder only session ('+cname+')')

    
    # --------------------------------------------------------------------------------------------
    # Variable length Inference
    # --------------------------------------------------------------------------------------------

    def create_inference(self,training=True):

        if (self.params['mode']=='separate' or self.params['mode']=='both') and self.commandlineparams['recon_weight']<0.0001:
            self.commandlineparams['recon_weight']=1.0
        
        dodebug=self.commandlineparams['debug']
        poolmode=bisbaseunet.get_pool_mode(self.params['avg_pool'])

        norelu=False
        input_data=self.pointers['input_pointer']
        threed=self.calcparams['threed']
        imgshape=input_data.get_shape()
        self.params['num_conv_layers']=bisbaseunet.fix_number_of_conv_layers(self.params['num_conv_layers'],
                                                                             self.commandlineparams['patch_size'])
        num_conv_layers=int(self.params['num_conv_layers'])
        p=self.commandlineparams['patch_size']

        if training:
            self.add_parameters_as_variables()


        cname=bisutil.getdimname(threed)
        print('===== Creating '+cname+' BONE Segmentation Model (mode='+str(self.params['mode'])+'). Inputs shape= %s ' % (imgshape))
        print('=====')

        bisutil.print_dict({'Num Conv/Deconv Layers' : self.params['num_conv_layers'],
                            'Filter Size': self.params['filter_size'],
                            'Num Filters': self.params['num_filters'],
                            'Num Classes':  self.calcparams['num_classes'],
                            'Patch Size':   self.commandlineparams['patch_size'],
                            'Num Frames':   self.calcparams['num_frames'],
                            'Smoothness':   self.commandlineparams['edge_smoothness']},
                           extra="=====",header="Model Parameters:")


 
        output={
            'regularizer' : None,
            'image' : None,
            'logits' : None,
            'encoding': None,
            'classification' : None,
        }
        self.exclude_list= []

        output = {
            'image' : None,
            'logits' : None,
            'regularizer' : None
        }

        second_input=input_data
        
        # Create Model
        print('=====\n===== C r e a t i n g  M o d e l s  '+self.params['mode']+'\n=====')
        if (self.params['mode']!="classify"):
            autoencoder_output = bisbaseunet.create_nobridge_unet_model(input_data, 
                                                                  num_conv_layers=num_conv_layers,
                                                                  filter_size=self.params['filter_size'],
                                                                  num_frames=self.calcparams['num_frames'],
                                                                  num_filters=self.params['num_filters'],
                                                                  keep_pointer=self.pointers['keep_pointer'],
                                                                  num_classes=1,
                                                                  name='Autoencoder',
                                                                  dodebug=dodebug,
                                                                  threed=threed,
                                                                  norelu=norelu,
                                                                  poolmode=poolmode)
            self.autoencoder_variables=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope="Autoencoder")

            if self.params['mode']!='separate':
                second_input=autoencoder_output
        
            with tf.variable_scope('Outputs/Autoencoder'):
                output['encoding']= tf.identity(autoencoder_output,name="Encoding")
                o_shape=output['encoding'].get_shape()
                bisutil.image_summary(output['encoding'],'encoding',o_shape[3].value,1,self.calcparams['threed'],max_outputs=self.commandlineparams['max_outputs'])
            
        classifier_output = bisbaseunet.create_unet_model(second_input,
                                                          num_conv_layers=num_conv_layers,
                                                          filter_size=self.params['filter_size'],
                                                          num_frames=1,
                                                          num_filters=self.params['num_filters']/2,
                                                          keep_pointer=self.pointers['keep_pointer'],
                                                          num_classes=self.calcparams['num_classes'],
                                                          name="Classifier",
                                                          dodebug=dodebug,
                                                          threed=threed,
                                                          norelu=norelu,
                                                          poolmode=poolmode)
        
        self.classifier_variables=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope="Classifier")

        with tf.variable_scope('Outputs/Classifier'):

            
            output["logits"] = tf.identity(classifier_output,name="Logits")
            dim=len(classifier_output.get_shape())-1
            class_pred=tf.argmax(classifier_output, dimension=dim, name='classification')
            output["classification"]=tf.expand_dims(class_pred, dim=dim,name="Classification_output")
            
            o_shape=output['classification'].get_shape()
            bisutil.image_summary(output['classification'],'classification',o_shape[3].value,1,self.calcparams['threed'],max_outputs=self.commandlineparams['max_outputs'])
            
            if (self.commandlineparams['edge_smoothness']>0.0 and training):
                output['regularizer']=bisutil.createSmoothnessLayer(output['classification'],
                                                                    self.calcparams['num_classes'],
                                                                    self.calcparams['threed']);
            elif dodebug:
                output['regularizer']=None
                print('===== Not adding back end smoothness computation')
                
        with tf.variable_scope('Outputs'):
            if self.params['mode']!="classify":
                output['image']=tf.identity(output['encoding'],name="Output")            
            else:
                output['image']=tf.identity(output['classification'],name="Output")

        return output



if __name__ == '__main__':

    DualSegm().execute()
