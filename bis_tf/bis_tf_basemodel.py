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
from tensorflow.python.client import device_lib

class BaseModel:

    # ------------------------------------------
    # Core Model Parameters -- used in inference
    # ------------------------------------------
    params = {
        'name': 'BaseModel',
    }


    # ---------------------------------
    # Pointers to objects and functions
    # ---------------------------------
    pointers = {
        'input_pointer' : None,
        'target_pointer' : None,
    }

    # --------------------------------------
    # Command line parameters
    # Used to specify things like filenames
    #  and number of iterations
    # (parameters as to how to run the model
    #  as opposed to how to create it )
    #
    # Force train is special and is used to
    #   force training is --train is used
    # --------------------------------------
    commandlineparams = {
        "debug": False,
        "dry_run": False,
        "extra_debug": False,
        "max_steps": 128,
        "step_gap": 64,
        "input_data" : None,
        "fixed_random" : False,
        "device" : '/gpu:0',
        'max_outputs' : 3,
    }

    # --------------------------------------
    # Parameters not to read from .cpkt file
    # if parsing
    # --------------------------------------
    donotreadlist = [ 'name','device', 'dry_run' , 'fixed_random', 'max_outputs']
    devicespec ="NONE"

    # Optimizer and Session Restorer Lists
    list_of_variables_toopt=None
    list_of_variables_to_read=None


    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------

    def __init__(self):
        return None

    def can_train(self):
        return False

    def get_description(self):
        return "Base Model. Useless"

    def print_line(self):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+++++')

    # --------------------------------------------------------------------
    # Set Parameters from Dictionary elements params and commandlineparams
    # --------------------------------------------------------------------
    def set_commandline_path_from_arg(self,name,value):
        if (value!='' and value!=None):
            self.commandlineparams[name]=os.path.abspath(value)
        elif (not name in self.commandlineparams):
            self.commandlineparams[name]=""

    def set_param_from_arg(self,name,value,defaultvalue=None,commandline=False,allowedvalues=None):

        if (allowedvalues!=None and value!=None):
            value=value.lower()
            if not value in allowedvalues:
                if (defaultvalue==None):
                    value=allowedvalues[0]
                else:
                    value=None

        if (value!=None):
            if commandline:
                self.commandlineparams[name]=value
            else:
                self.params[name]=value
        elif (defaultvalue!=None):
            if (commandline==True) and (name not in self.commandlineparams):
                self.commandlineparams[name]=defaultvalue

            if (commandline==False) and (name not in self.params):
                self.params[name]=defaultvalue

    def set_commandlineparam_from_arg(self,name,value,allowedvalues=None,defaultvalue=None):
        self.set_param_from_arg(name=name,value=value,commandline=True,defaultvalue=defaultvalue,allowedvalues=None)


    def set_parameters_from_dictionary(self,dict):

        try:
            new_params=dict['parameters']
            new_commandlineparams=dict['commandlineparams']

            name=new_params['name']
            if (name!=self.params['name']):
                raise ValueError('Bad parameter file. Name '+str(name)+' does not match expected name '+self.params['name'])
        except Exception as e:
            print('!!!!!\n!!!!! Bad parameter dictionary'+str(dict))
            bisutil.handle_error_and_exit(e)

        for key in self.params:
            if ((not key in self.donotreadlist) and (key in new_params)):
                if (new_params[key]!=''):
                    #                  print('+++++ setting params['+key+'] to '+str(new_params[key]))
                    self.params[key]=new_params[key]


        for key in new_commandlineparams:
            if (new_commandlineparams[key]!=''):
                #               print('+++++ setting commandlineparams['+key+'] to '+str(new_commandlineparams[key]))
                self.commandlineparams[key]=new_commandlineparams[key]



                
    def save_parameters_to_file(self,filename,systeminfo=None):




        
        odict = {
            'info' : {
                'calculated' : self.calcparams,
            },
            'commandlineparams' : self.commandlineparams,
            'parameters' : self.params
        }

        if systeminfo!=None:
            odict['info']['system']=systeminfo

            

        with open(filename, 'w') as fp:
            json.dump(odict, fp, sort_keys=True,indent=4)

            print('+++++ S t o r e d  p a r a m e t e r s  in ',filename)



    # ------------------------------------------------------------------
    # Command line parsing
    # ------------------------------------------------------------------
    def add_common_commandline_parameters(self,parser,training=False):

        if training==True:

            self.commandlineparams['target_data']=None
            self.commandlineparams['output_model_path']=None

            parser.add_argument('-i','--input',
                                 help='txt files containing list of image files')
            parser.add_argument('-t','--target',
                                help='txt files containing list of target files')
            parser.add_argument('--second_input',
                                help='txt files containing list of second input files (previous results or eigenvalues or ...)')
            parser.add_argument('--second_target',
                                help='txt files containing list of second target files (e.g. segmentation map or ...)')
            parser.add_argument('-o','--output',
                                help='Path for saving the learned model')
            parser.add_argument('--resume_model', help='Path (directory or .meta file) to read existing model to resume from',
                                default='')
            parser.add_argument('--param_file', help='JSON File to read parameters from, same format as output .json',
                                default='')
            parser.add_argument('-s','--steps', help='Number of steps (iterations)',
                                default=None,type=int)
            parser.add_argument('--step_gap', help='Save every step_gap steps',
                                default=None,type=int)
            parser.add_argument('--dry_run', help='If true, model is created but not trained.',
                                default=None,action='store_true')
            bisoptutil.add_parser_opt_params(parser)
        else:

            self.commandlineparams['model_path']=None
            self.commandlineparams['output_path']=None

            parser.add_argument('-m','--model_path',
                                help='path where to find the learned model (directory or .meta file)')
            parser.add_argument('-i','--input',
                                help='Input image to reconstuct (NIfTI image) (or text file listing images)')
            parser.add_argument('-o','--output',
                                help='Output classification result (NIfTI image or directory if input is text file)')
            parser.add_argument('--second_input',
                                help='txt files containing list of second input files (previous results etc.)')
            parser.add_argument('--second_target',
                                help='txt files containing list of second target files (previous results etc.)')
            parser.add_argument('--not_read_graph', help='If true, model is NOT read from .meta file and IS recreated in code', action='store_true')


        parser.add_argument('--fixed_random', help='If true, the random seed is set explitily (for regression testing)',
                            default=None,action='store_true')
        parser.add_argument('-d','--debug', help='Extra Debug Output',
                            default=None,action='store_true')
        parser.add_argument('--extra_debug', help='Extra Debug Output',
                            default=None,action='store_true')
        parser.add_argument('--device', help='Train using device',
                            default='/gpu:0')

    def add_custom_commandline_parameters(self,parser,training):
        return

    def extract_parameters_from_param_file(self,args):

        if (args.param_file==''):
            # No parameter file specified
            return

        try:
            json_data=open(args.param_file).read()
            data = json.loads(json_data)
        except Exception as e:
            print('!!!!!\n!!!!! Failed to read or parse',args.param_file,end='')
            bisutil.handle_error_and_exit(e)

        self.set_parameters_from_dictionary(data)


    def extract_common_commandline_parameters(self,args,training=False):

        if training:

            self.set_commandline_path_from_arg('input_data',args.input)
            self.set_commandline_path_from_arg('target_data',args.target)
            self.set_commandline_path_from_arg('second_input',args.second_input)
            self.set_commandline_path_from_arg('second_target',args.second_target)
            self.set_commandline_path_from_arg('output_model_path',args.output)
            self.set_commandline_path_from_arg('resume_model_path',args.resume_model)
            self.set_commandlineparam_from_arg('max_steps',args.steps)
            self.set_commandlineparam_from_arg('step_gap',args.step_gap)
            self.set_commandlineparam_from_arg(name='dry_run',value=args.dry_run,defaultvalue=False)
            bisoptutil.extract_parser_opt_params(self,args) 
        else:
            self.set_commandline_path_from_arg('model_path',args.model_path)
            self.set_commandline_path_from_arg('input_data',args.input)
            self.set_commandline_path_from_arg('output_path',args.output)
            self.set_commandline_path_from_arg('second_input',args.second_input)
            self.set_commandline_path_from_arg('second_target',args.second_target)
            self.set_commandlineparam_from_arg(name='not_read_graph',value=args.not_read_graph)


        self.set_commandlineparam_from_arg(name='fixed_random',value=args.fixed_random,defaultvalue=False)
        self.set_commandlineparam_from_arg(name='debug',value=args.debug)
        self.set_commandlineparam_from_arg(name='extra_debug',value=args.extra_debug)
        self.set_commandlineparam_from_arg(name='device',value=args.device)

    def extract_custom_commandline_parameters(self,args,training=False):
        return


    def validate_commandline(self,training=False):

        if (self.commandlineparams["input_data"]==None):
            bisutil.handle_error_and_exit('No input data specified (-i)')

        if (training):
            if self.commandlineparams['target_data']==None:
                bisutil.handle_error_and_exit('No target data specified (-i second argument)')
            if self.commandlineparams['output_model_path']==None:
                bisutil.handle_error_and_exit('No output_model_path specified (-o)')

        else:
            if self.commandlineparams['model_path']==None:
                bisutil.handle_error_and_exit('No (input) model path specified (-m)')

            if self.commandlineparams['output_path']==None:
                bisutil.handle_error_and_exit('No output path to save reconstructed images specified. (-o)')



    # --------------------------------------------------------------------------------------------------------------
    # Parse Command Line
    # --------------------------------------------------------------------------------------------------------------
    def parse_commandline(self,training=False):

        self.commandlineparams['forcetrain']=False
        if (self.can_train()):
            extra=" Use the --train flag to switch to training mode. (Use --train -h to see training help)"
            if (training!=True):
                training=False

            if '--train' in sys.argv:
                self.commandlineparams['forcetrain']=True
                training=True;
        else:
            training=False
            extra=""

        if training:
            parser = argparse.ArgumentParser(description='Train: '+self.get_description())
        else:
            parser = argparse.ArgumentParser(description='Recon: '+self.get_description()+extra)


        if self.commandlineparams['forcetrain']:
            parser.add_argument('--train', help='force training mode', action='store_true')

        self.add_common_commandline_parameters(parser,training)
        self.add_custom_commandline_parameters(parser,training)

        args = parser.parse_args()

        if training:
            self.extract_parameters_from_param_file(args)
        self.extract_common_commandline_parameters(args,training)
        self.extract_custom_commandline_parameters(args,training)
        self.validate_commandline(training)




    # ------------------------------------------------------------------
    # Read Parameters from checkpoint file
    # ------------------------------------------------------------------
    def clean_params_prior_to_serialization(self,prm):

        
        for key in prm:
            v=prm[key]
            if type(v)==np.int32:
                if v.size==1:
                    prm[key]=int(prm[key])

            if type(v)==np.float32:
                if v.size==1:
                    prm[key]=float(prm[key])
                    
            if type(v)==np.bool_:
                if v.size==1:
                    prm[key]=bool(prm[key])
        

    # Read Parameters
    def read_parameters_from_checkpoint(self,model_path):

        cname=bisutil.get_read_checkpoint(model_path)

        list_to_get=[];

        for key in self.params:
            if (not key in self.donotreadlist):
                list_to_get.append(key)

        
        values=bisutil.get_tensor_list_from_checkpoint_file(cname,list_to_get)

        for key in values:
            if (values[key]!=None):
                self.params[key]=values[key]
                try:
                    a=self.params[key]
                    self.params[key]=a
                except AttributeError:
                    a=0

        list_to_get=[]
        for key in self.calcparams:
            if (not key in self.donotreadlist):
                list_to_get.append(key)

        values2=bisutil.get_tensor_list_from_checkpoint_file(cname,list_to_get)
        for key in values2:
            self.calcparams[key]=values2[key]
            try:
                a=self.calcparams[key]
                self.calcparams[key]=a
            except AttributeError:
                a=0


        self.clean_params_prior_to_serialization(self.calcparams);
        self.clean_params_prior_to_serialization(self.params);
                
        print("+++++ Read parameters from checkpoint file "+cname)
        if self.commandlineparams['debug']:
            values.update(values2)
            bisutil.print_dict(values,extra='+++++')
        return cname

    def add_parameters_as_variables(self):

        #with tf.device('/cpu:0'):
        with tf.variable_scope('Parameters') as scope:
            for key in self.params:
                tf.Variable(self.params[key],name=key,trainable=False)
            for key in self.calcparams:
                if self.calcparams[key] is not None:
                    tf.Variable(self.calcparams[key],name=key,trainable=False)

    def get_total_number_of_parameters(self):

        lst=bisoptutil.get_list_of_variables_to_optimize(self)

        param_count = 0
        for var in lst:
            shape = var.get_shape()

            var_param_count = 1
            for dim in shape:
                var_param_count *= dim.value

            param_count += var_param_count

        return param_count

    # ------------------------------------------------------------------------
    # Initialize flags for bis_tf_util
    # ------------------------------------------------------------------------
    def initialize_tf_util(self):

        if self.commandlineparams['extra_debug']:
            self.commandlineparams['debug']=True

        if self.commandlineparams['extra_debug']:
            os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
        else:
            os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        if (self.commandlineparams['device']=="/gpu:0"):
            os.environ['CUDA_VISIBLE_DEVICES']="0"
        elif (self.commandlineparams['device']=="/gpu:1"):
            os.environ['CUDA_VISIBLE_DEVICES']="1"
            self.commandlineparams['device']="/gpu:0"
        elif self.commandlineparams['device']=='/cpu:0':
            os.environ['CUDA_VISIBLE_DEVICES']=''

        if (self.commandlineparams['debug']):
            print('+++++\n+++++ Looking for CUDA devices')
            print('+++++ \t',self.commandlineparams['device'],' CUDA_VISIBLE_DEVICES=',os.environ['CUDA_VISIBLE_DEVICES'])


        lst=device_lib.list_local_devices()
        n=len(lst)
        found=False
        i=0
        while i<n and found==False:
            name=lst[i].name
            if (self.commandlineparams['debug']):
                print('+++++ \t',i,'name=',name)
            if (self.commandlineparams['device']==name):
                found=True
                self.devicespec=lst[i].physical_device_desc
            i=i+1;

        if (found==False):
            print('+++++\n+++++ Forcing device name to /cpu:0 (was',self.commandlineparams['device'],')')
            self.commandlineparams['device']="/cpu:0"
            self.devicespec=lst[0].physical_device_desc
            os.environ['CUDA_VISIBLE_DEVICES']=''


        # Set Internal variables
        bisutil.INTERNAL['extra_debug']=self.commandlineparams['extra_debug']
        bisutil.INTERNAL['debug']=self.commandlineparams['debug']
        if (self.commandlineparams['debug']):
            bisutil.debug_print('+++++\n+++++ bis_tf_utils ',bisutil.INTERNAL)


    # ------------------------------------------------------------------------
    # Initialize network, create input placeholders and calculate parameters
    # ------------------------------------------------------------------------
    def initialize_placeholders(self,input_shape,target_shape,isthreed=False,max_outputs=3):
        

        print('+++++ \t ... and allocating placeholders size='+str(input_shape),' target='+str(target_shape)+', threed='+str(isthreed));

        iname="Input"

        with tf.variable_scope('Inputs') as scope:

            dtype=tf.float32

            # Allocate placeholders
            if isthreed:
                maxd=3;
                images = tf.placeholder(dtype, shape=(None,input_shape[0],input_shape[1],input_shape[2],input_shape[3]), name=iname)
                targets = tf.placeholder(dtype, shape=(None,target_shape[0],target_shape[1],target_shape[2],target_shape[3]), name='Target')
            else:
                maxd=2;
                images = tf.placeholder(dtype, shape=(None,input_shape[0],input_shape[1],input_shape[2]), name=iname)
                targets = tf.placeholder(dtype, shape=(None,target_shape[0],target_shape[1],target_shape[2]), name='Target')

            if max_outputs > 0:
                bisutil.image_summary(images,'input',input_shape,input_shape[maxd],isthreed,max_outputs=max_outputs)
                bisutil.image_summary(targets,'target',target_shape,target_shape[maxd],isthreed,max_outputs=max_outputs)

        with tf.variable_scope('Constants') as scope:
            global_step = tf.Variable(0, trainable=False,name="Global_Step")
            bisutil.add_scalar_summary('global_step', global_step,'Metrics/')

        self.pointers['input_pointer']=images;
        self.pointers['target_pointer']=targets;
        self.pointers['global_step_pointer']=global_step
        
        print('+++++ input image=',images.get_shape());
        print('+++++ target image=',targets.get_shape());


        if self.commandlineparams['debug']:
            print('+++++ All Initialized')
            bisutil.print_dict(self.params,extra='+++++',header='params:')
            bisutil.print_dict(self.calcparams,extra='+++++',header='calcparams:')
            bisutil.print_dict(self.commandlineparams,extra='+++++',header='commandlineparams:')
            print('+++++')


    # --------------------------------------------------------------------------------------------
    # Model-specific variables initialization
    # --------------------------------------------------------------------------------------------
    def initialize_model_variables(self):
        if self.commandlineparams['debug']:
            print("+++++ Initializing base model variables")




    # ------------------------------------------------------------------------
    # Get Checkpoint output name
    # ------------------------------------------------------------------------
    def getoutputname(self,isthreed):
        outputname=self.params['name'];
        if isthreed:
            outputname+="_3d.ckpt"
        else:
            outputname+="_2d.ckpt"
        return outputname


    # -----------------------------------------------------------------------------
    # "Virtual" Functions to be specified in derived classes
    # -----------------------------------------------------------------------------

    # Create AND returns optimizer (loss_op = output of create_loss)
    def create_optimizer(self,loss_op):
        return bisoptutil.create_opt_function(self,loss_op)

    def set_parameters_to_read_from_resume_model(self,path):
        # None means all global variables
        self.list_of_variables_to_read=None


    # --------------------------------
    # Info
    # --------------------------------
    def get_system_info(self):

        tform='%Y-%m-%d %H:%M:%S'
        s=datetime.now().strftime(tform)
        
        return {
            'os'   : platform.system()+' '+platform.release(),
            'date' : s,
            'node' : platform.node(),
            'machine' : platform.machine(),
            'python'  : platform.python_version(),
            'tensorflow' :  str(tf.__version__),
            'numpy' : np.version.version,
            'pwd'   : os.getcwd(),
            'devicespec' : self.devicespec,
            'rawcommandline' :  ' '.join(sys.argv),
        };

