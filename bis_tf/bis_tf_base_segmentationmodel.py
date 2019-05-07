#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import platform
import re
import time
import math
import numpy as np
import json
from datetime import datetime
import argparse
from six.moves import xrange

import tensorflow as tf
from tensorflow.python.framework.errors_impl import PermissionDeniedError
from tensorflow.python.framework import test_util

import bis_tf_utils as bisutil
import bis_tf_optimizer_utils as bisoptutil
import bis_tf_imageset as pimage
import bis_tf_basemodel as basemodel
from tensorflow.python.client import device_lib

class BaseSegmentationModel(basemodel.BaseModel):

    # -----------------------------------------------
    # Computed Model Parameters -- used in inference
    # these derive from both params and input data
    # -----------------------------------------------
    calcparams = {
        'num_classes': 2,
        'threed' : False,
        'num_frames' : 1,
        'patch_size' : None,
        'transpose' : None,
    }

    # --------------------------------------
    # External Functions
    # --------------------------------------
    external = {
        'training' : bisutil.run_training,
        'recon'    : bisutil.reconstruct_images
    }


    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.params['name']='BaseSegmentationModel';
        self.params['keep_prob']=0.85;

        self.commandlineparams["batch_size"]= 16;
        self.commandlineparams["pad_size"]=None;
        self.commandlineparams["pad_type"]='zero';
        self.commandlineparams["no_augment"]=False;
        self.commandlineparams["one_hot_output"]=False;
        self.commandlineparams["patch_size"]=None;
        self.commandlineparams["transpose"]=None;
        bisoptutil.create_opt_params(self)
        return None

    def get_description(self):
        return "Base Segmentation Model. Can be used to reconstruct."


    # ------------------------------------------------------------------
    # Command line parsing
    # ------------------------------------------------------------------
    def add_common_commandline_parameters(self,parser,training=False):

        super().add_common_commandline_parameters(parser,training);
        
        if training==True:

            parser.add_argument('-p','--patch_size', nargs='+', 
                                help='patch size (should be even). If no value is given, the patch size is the same as the \
                                      input data. If more than one value is given the patch will be set according to inputs. \
                                      Default value is None.',
                                default=None,type=int)
            parser.add_argument('--no_augment',
                                help='If true do not augment patches by flipping and rotation (default=augment)',
                                default=None,action='store_true')
            parser.add_argument('--transpose', nargs='+', 
                                help='transpose the data for model training (and saves the value for recon). \
                                      Input is in the form of a list of data axis integers, eg 1 0 will flip the x and y dimensions. \
                                      If no value is given, the data remains in the same order.  \
                                      Default value is None.',
                                default=None,type=int)
        else:
            parser.add_argument('--one_hot_output',
                                help='If true, output the segmentation result as a one-hot version of the image.',
                                default=None,action='store_true')
            parser.add_argument('--stride_size', help='Stride for reconstruction -- default =0.5 = 1/2 patch width',
                                default=0.5,type=float)
            parser.add_argument('--sigma', help='smoothing for patches, default=0.25 no smoothing',
                                default=0.25,type=float)
            parser.add_argument('--repeat', help='repeat inference for probabilitic reasoning, default=1 run once',
                                default=1,type=int)
            self.params['keep_prob']=1.0
            self.commandlineparams['batch_size']=32

        parser.add_argument('--keep_prob',   help='Keep Probability for Dropoff',
                            default=None,type=float)
        parser.add_argument('-b','--batch_size', help='Num of training samples per mini-batch -- mapped to closest power of two',
                            default=None,type=int)
        parser.add_argument('--pad_size', nargs='+', help='Amount of padding to add. Functionality is similar to that of \
                                                           of patch_size, where a speficied value will override the default value (zero) \
                                                           for that dimension. Default value is None',
                            default=None,type=int)
        parser.add_argument('--pad_type', help='type of padding to use, default is "zero"', choices=['edge','reflect','zero'],
                            default='zero')



    def extract_common_commandline_parameters(self,args,training=False):

        super().extract_common_commandline_parameters(args,training);
        
        if training:
            self.set_commandlineparam_from_arg(name='no_augment',value=args.no_augment,defaultvalue=False)
            self.set_commandlineparam_from_arg(name='transpose',value=args.transpose)
            self.set_commandlineparam_from_arg(name='patch_size',value=args.patch_size)
            keepv=self.params['keep_prob']
            maxbatch=512
        else:
            maxbatch=4096
            keepv=1.0
            self.commandlineparams['one_hot_output']=False
            self.commandlineparams['stride_size']=0.5
            self.commandlineparams['sigma']=0.25
            self.commandlineparams['repeat']=1
            self.set_commandlineparam_from_arg(name='one_hot_output',value=args.one_hot_output,defaultvalue=False)
            self.set_commandlineparam_from_arg('stride_size', bisutil.force_inrange(args.stride_size,0.0,0.95))
            self.set_commandlineparam_from_arg('sigma', bisutil.force_inrange(args.sigma,0.0,1.0))
            self.set_commandlineparam_from_arg('repeat', bisutil.force_inrange(args.repeat,1,100))

        self.set_commandlineparam_from_arg(name='batch_size',
                                           value=bisutil.getpoweroftwo(args.batch_size,4,maxbatch))
        self.set_commandlineparam_from_arg(name='pad_size',value=args.pad_size)
        self.set_commandlineparam_from_arg(name='pad_type',
                                           value=args.pad_type)
        self.set_param_from_arg(name='keep_prob',
                                value=bisutil.force_inrange(args.keep_prob,0.0,1.0),
                                defaultvalue=keepv)


    # ------------------------------------------------------------------------
    # Initialize network, create input placeholders and calculate parameters
    # ------------------------------------------------------------------------

    def initialize_dimension(self,input_data,patch_shape):

        final_input_patch_shape = input_data.compute_input_patch_shape(patch_shape);
        
        self.calcparams['patch_size'] = final_input_patch_shape;
        self.calcparams['num_frames'] = input_data.get_num_frames();

        self.calcparams['threed']=input_data.get_threed();
        if (self.calcparams['threed'] and final_input_patch_shape[2]==1):
            self.calcparams['threed']=False;

        print('+++++ threed='+str(self.calcparams['threed'])+' num_frames='+str(self.calcparams['num_frames']));
    
    def initialize(self,input_data,patch_shape,max_outputs=3):
        
        self.initialize_dimension(input_data,patch_shape);
        
        # Check if the targets are different size

        final_input_patch_shape=self.calcparams['patch_size'];
        final_target_patch_shape = input_data.compute_target_patch_shape(final_input_patch_shape);
        

        print('+++++ Initializing input data ',input_data.get_image_shape(),'\n+++++\t ... and allocating placeholders size='+str(self.calcparams['patch_size']),' threed='+str(self.calcparams['threed'])+
              ', num_frames='+str(self.calcparams['num_frames']))


        if self.calcparams['threed']:
            imgshape=[ final_input_patch_shape[0],final_input_patch_shape[1],final_input_patch_shape[2],self.calcparams['num_frames']];
            targshape=[ final_target_patch_shape[0],final_target_patch_shape[1],final_target_patch_shape[2],final_target_patch_shape[3]];
        else:
            imgshape=[ final_input_patch_shape[0],final_input_patch_shape[1],self.calcparams['num_frames']];
            targshape=[ final_target_patch_shape[0],final_target_patch_shape[1],final_target_patch_shape[2]];


        with tf.variable_scope('Constants') as scope:
            keep_prob = tf.placeholder(tf.float32, name='Keep_Probability')
        self.pointers['keep_pointer']=keep_prob
        self.initialize_placeholders(imgshape,targshape,self.calcparams['threed'],max_outputs);


    # --------------------------------------------------------------------------------------------
    # Model-specific variables initialization
    # --------------------------------------------------------------------------------------------
    def initialize_model_variables(self):
        if self.commandlineparams['debug']:
            print("+++++ Initializing base model variables")


    # -----------------------------------------------------------------------------
    # "Virtual" Functions to be specified in derived classes
    # -----------------------------------------------------------------------------

    # Create inference
    # return a dictionary of results ('image','logits','regularizer') etc.
    def create_inference(self,training=True):
        raise ValueError('Create_Inference not implemented')

    # Create AND returns loss function (output_dict = output of create_inference)
    def create_loss(self,output_dict):
        raise ValueError('Create Loss not implemented')

    # Create AND returns optimizer (loss_op = output of create_loss)
    def create_optimizer(self,loss_op):
        return bisoptutil.create_opt_function(self,loss_op)

    def set_parameters_to_read_from_resume_model(self,path):
        # None means all global variables
        self.list_of_variables_to_read=None

    # ------------------------------------------------------------------------
    # Most of the work is done in bisutil.reconstruct_image
    # ------------------------------------------------------------------------
    # Image Reconstruction method
    def reconstruct_images(self,input_data,model_path,batch_size=16):

        # Keep prob for recon can be different than what model specified
        self.donotreadlist.append('keep_prob')
        self.donotreadlist.append('edge_smoothness')


        try:
            cname=self.read_parameters_from_checkpoint(model_path)
        except RuntimeError as e:
            bisutil.handle_error_and_exit(e)

        undo_transpose = None
        transpose = self.calcparams['transpose']
        if transpose is not None:
            print('+++++ Transposing data: '+str(transpose))
            undo_transpose = input_data.transpose(transpose)
            print('+++++ Returned undo transpose ordering: '+str(undo_transpose))


        pad_type = self.commandlineparams['pad_type']
        pad_size = self.commandlineparams['pad_size']

        print('+++++\n+++++ Padding images by '+str(pad_size)+' with pad type: '+pad_type)
        actual_pad_size = input_data.pad(pad_size=pad_size,pad_type=pad_type)

        with tf.Graph().as_default() as graph:

            if (self.commandlineparams['fixed_random']==True):
                np.random.seed(0)
                tf.set_random_seed(0)
                print("+++++ SETTING FIXED RANDOM SEED. Next number should be fourty-four =  ", np.random.random_integers(0,100))

            
            if self.commandlineparams['not_read_graph']:
                # Get the images and their labels for the images (3D!!!)
                self.initialize(input_data=input_data,
                                max_outputs=0,
                                patch_shape=self.calcparams['patch_size'])
                self.initialize_model_variables()

                #Build a Graph that computes the logits predictions from the inference model
                self.print_line()
                self.create_inference(training=False);

            else:
                # Read the graph!!!!
                metaname=cname+".meta"
                print('+++++ \t importing graph from '+metaname)
                model_reader = tf.train.import_meta_graph(metaname,clear_devices=True)
                print('+++++')

            result_image_list = self.external['recon'](checkpointname=cname,
                                          input_data=input_data,
                                          images_pointer=graph.get_tensor_by_name('Inputs/Input:0'),
                                          model_output_image=graph.get_tensor_by_name('Outputs/Output:0'),
                                          keep_pointer = graph.get_tensor_by_name('Constants/Keep_Probability:0'),
                                          keep_value = self.params['keep_prob'],
                                          patch_size=self.calcparams['patch_size'],
                                          actual_pad_size=actual_pad_size,
                                          threed=self.calcparams['threed'],
                                          one_hot_output=self.commandlineparams['one_hot_output'],
                                          num_classes=self.calcparams['num_classes'],
                                          stride_size=self.commandlineparams['stride_size'],
                                          sigma=self.commandlineparams['sigma'],
                                          repeat=self.commandlineparams['repeat'],
                                          batch_size=batch_size);

            if undo_transpose is not None:
                print('+++++ Transposing data: '+str(undo_transpose))
                # input_data.transpose(undo_transpose)
                # Now change the result images over too
                for i in range(0,len(result_image_list)):
                    orig_shape = result_image_list[i].shape
                    result_image_list[i] = np.transpose(result_image_list[i],undo_transpose)
                    print('-+-+- Transpose result image data shape[',i,']: ',str(orig_shape),' to ',str(result_image_list[i].shape))

            return result_image_list

    # ------------------------------------------------------------------------
    # Network training method(s)
    # Most of the work is done in bisutil.run_training
    # ------------------------------------------------------------------------
    def initialize_network_tensor_values(self):
        if self.commandlineparams['debug']:
            print("+++++ Using automatically initialized tensor values")



    def train(self,
              training_data,
              output_model_path,
              resume_model_path,
              patch_size,
              max_outputs=3,
              max_steps=1000,
              step_gap=50,
              batch_size=10,
              augmentation=True,
              epsilon=0.1):


        if (len(output_model_path)<2):
            bisutil.handle_error_and_exit("No output path specified")

        self.print_line()
        print('+++++ B e g i n n i n g  t r a i n i n g  with  device=',self.commandlineparams['device'],self.devicespec)
        print('+++++')

        if (len(resume_model_path)>2):
            print("+++++ Resuming training: ignoring commandline model structure parameters.\n+++++")
            try:
                cname=self.read_parameters_from_checkpoint(resume_model_path)
            except RuntimeError as e:
                bisutil.handle_error_and_exit(e)
            print("+++++")


        transpose = self.commandlineparams['transpose']
        if transpose is not None:
            print('+++++ Transposing data: '+str(transpose))
            training_data.transpose(transpose)
            self.calcparams['transpose'] = transpose


        pad_type = self.commandlineparams['pad_type']
        pad_size = self.commandlineparams['pad_size']
        
        print('+++++\t Padding images by '+str(pad_size)+' with pad type: '+pad_type)
        training_data.pad(pad_size=pad_size,pad_type=pad_type)



        with tf.Graph().as_default():
            #with tf.device(self.commandlineparams['device']):
            if (self.commandlineparams['fixed_random']==True):
                np.random.seed(0)
                tf.set_random_seed(0)
                print("+++++ SETTING FIXED RANDOM SEED. Next number should be fourty-four =  ", np.random.random_integers(0,100))

            # All of these calls communicate by storing results in self.pointers
            # Initialize pointer placeholders

            self.initialize(input_data=training_data,
                            max_outputs=max_outputs,
                            patch_shape=patch_size)
            self.initialize_model_variables()


            # Check training data
            d=training_data.get_target_data()
            m1=math.floor(d[0].min())
            m2=math.floor(d[0].max())
            numcl=math.floor(m2-m1+1)
            if (numcl<20 and m1==0):
                print('+++++ Target range',[m1,m2],' setting num_classes='+str(numcl))
                self.calcparams['num_classes']=numcl
            else:
                self.calcparams['num_classes'] = self.calcparams['num_frames']
                

            # Build a Graph that computes the logits predictions from the inference model

            print('+++++')
            output_dict=self.create_inference(training=True);
            print('+++++')

            # Initialize values (from pre-trained network etc.)
            self.initialize_network_tensor_values()

            # Calculate the loss.
            loss_op=self.create_loss(output_dict)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op=self.create_optimizer(loss_op)

            self.calcparams['total_num_parameters'] = self.get_total_number_of_parameters()
            print('+++++ \t total number of trainable parameters='+str(self.get_total_number_of_parameters()))
            print('+++++')

            if (self.commandlineparams['dry_run']==True):
                self.print_line()
                print("+++++ D r y  R u n.  N O T  T R A I N I N G!")
                max_steps=-1


            # ---------------------------------------------------------
            # --- Last Hook
            # ---------------------------------------------------------
            if len(resume_model_path)>2:
                self.set_parameters_to_read_from_resume_model(resume_model_path)

            # From here use results in self.pointers to run training

            start_time = time.time()
            start_date = datetime.now()

            oname=self.getoutputname(self.calcparams['threed'])

            
            with tf.variable_scope('Save') as scope:
                try:
                    jsonfile=self.external['training'](
                        training_data=training_data,
                        images_pointer = self.pointers['input_pointer'],
                        targets_pointer = self.pointers['target_pointer'],
                        keep_pointer =  self.pointers['keep_pointer'],
                        step_pointer = self.pointers['global_step_pointer'],
                        keep_value   = self.params['keep_prob'],
                        loss_op = loss_op,
                        train_op= train_op,
                        output_model_path=output_model_path,
                        resume_model_path=resume_model_path,
                        batch_size=batch_size,
                        max_steps=max_steps,
                        step_fraction=step_gap,
                        model_name=oname,
                        patch_size=self.calcparams['patch_size'],
                        patch_threed=self.calcparams['threed'],
                        epsilon=epsilon,
                        list_of_variables_to_read=self.list_of_variables_to_read,
                        augmentation=augmentation)
                except PermissionDeniedError as e:
                    a=str(e).split('\n')
                    bisutil.handle_error_and_exit('Failed to write output in '+str(a[0]))

            end_time = time.time()
            end_date = datetime.now()
            duration = end_time - start_time

            systeminfo=self.get_system_info();
            systeminfo['duration']=duration;
            s=systeminfo['date'];

            print('+++++\n+++++ Optimization done. Started at:',start_date.strftime('%H:%M:%S'),
                  ' ended at:',end_date.strftime('%H:%M:%S'),
                  ' duration=%.2fs' %(duration))
            print('+++++')

            self.save_parameters_to_file(jsonfile,systeminfo)

    # -----------------------------------------------------------------------------
    # Main training method -- takes command line args loads images and calls train
    # -----------------------------------------------------------------------------
    def train_main(self,argv=None):

        print('+++++ L o a d i n g  d a t a . . .')
        print('+++++')
        training_data=pimage.ImageSet()
        try:
            training_data.load(image_filename=self.commandlineparams['input_data'],
                               target_filename=self.commandlineparams['target_data'],
                               secondinput_filename=self.commandlineparams['second_input'],
                               secondtarget_filename=self.commandlineparams['second_target'])
        except ValueError as e:
            bisutil.handle_error_and_exit(e)

        print('+++++ Training model output=',os.path.abspath(self.commandlineparams['output_model_path']))
        if (len(self.commandlineparams['resume_model_path'])>2):
            print('+++++ Resuming from '+os.path.abspath(self.commandlineparams['resume_model_path']))
        print('+++++')

        augmentation=True
        if self.commandlineparams['no_augment']:
            augmentation=False

        self.train(training_data=training_data,
                   output_model_path=self.commandlineparams['output_model_path'],
                   resume_model_path=self.commandlineparams['resume_model_path'],
                   patch_size=self.commandlineparams['patch_size'],
                   max_outputs=self.commandlineparams['max_outputs'],
                   max_steps=self.commandlineparams['max_steps'],
                   step_gap=self.commandlineparams['step_gap'],
                   augmentation=augmentation,
                   batch_size=self.commandlineparams['batch_size'])


    # ------------------------------------------------------------------------
    # Main reconstruction method -- loads image, calls reconstructs and then saves
    # ------------------------------------------------------------------------
    def recon_main(self,argv=None):


        print('+++++ L o a d i n g  d a t a . . .')
        print('+++++')
        input_data=pimage.ImageSet()
        try:
            input_data.load(image_filename=self.commandlineparams['input_data'],
                              secondinput_filename=self.commandlineparams['second_input'])
        except ValueError as e:
            bisutil.handle_error_and_exit(e)

        self.print_line()

        #with tf.device(self.commandlineparams['device']):
        print('+++++ B e g i n n i n g  r e c o n s t r u c t i o n  with device=',self.commandlineparams['device'],self.devicespec)
        print('+++++')
        result_images=self.reconstruct_images(input_data=input_data,
                                              model_path=self.commandlineparams['model_path'],
                                              batch_size=self.commandlineparams['batch_size'])
        self.print_line()

        try:
            input_data.save_reconstructed_image_data(data=result_images,path=self.commandlineparams['output_path'])
        except ValueError as e:
            bisutil.handle_error_and_exit(e)




    # ------------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------------
    def execute(self):


        if (sys.version_info[0]<3):
            print('\n .... this tool is incompatible with Python v2. You are using '+str(platform.python_version())+'. Use Python v3.\n')
            sys.exit(0)
        elif (sys.version_info[1]<4):
            print('\n .... this tool needs at least Python version 3.4. You are using '+str(platform.python_version())+'\n')
            sys.exit(0)


        print('+++++\n+++++\n+++++ B e g i n n i n g  e x e c u t i o n  model='+str(self.params['name'])+' (python v'+str(platform.python_version())+' tf='+str(tf.__version__)+' cuda='+str(test_util.IsGoogleCudaEnabled())+')')



        self.parse_commandline()
        self.initialize_tf_util()
        self.print_line()

        if not self.commandlineparams['forcetrain']:
            self.recon_main()
        else:
            self.train_main()
        self.print_line()
        print('+++++ E x e c u t i o n  c o m p l e t e d\n+++++')


if __name__ == '__main__':

    # This works for recon
    BaseSegmentationModel().execute()
