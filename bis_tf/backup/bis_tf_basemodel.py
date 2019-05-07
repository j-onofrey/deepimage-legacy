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

import bis_tf_utils as bisutil
import bis_image_patchutil as putil
import bis_tf_imageset as pimage


class BaseModel:

    # ------------------------------------------
    # Core Model Parameters -- used in inference
    # ------------------------------------------
    params = {
        'name': 'BaseModel',
        'keep_prob': 0.85
    }

    # -----------------------------------------------
    # Computed Model Parameters -- used in inference
    # these derive from both params and input data
    # -----------------------------------------------
    calcparams = {
        'patch_size' : 64,
        'num_classes': 2,
        'threed' : False,
        'num_frames' : 1,
        'max_outputs' : 3,
    }

    # ---------------------------------
    # Pointers to objects and functions
    # ---------------------------------
    pointers = {
        'input_image': None,
        'label_image': None,
        'input_pointer' : None,
        'target_pointer' : None,
        'loss_op' : 'None',
        'output_dict' : None,
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
        'forcetrain' : False,
        'resume_model_path' : ''
    }

    # --------------------------------------
    # Parameters not to read from .cpkt file
    # if parsing
    # --------------------------------------
    donotreadlist = [ 'name',',use_gpu' ]

    
    # ------------------------------------------------------------------
    # Basic Stuff, constructor and descriptor
    # ------------------------------------------------------------------
  
    def __init__(self):
        return

    def can_train(self):
        return False
    
    def get_description(self):
        return "Base CNN Model. Can be used to test."

    def print_line(self):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+++++')
        
    
    # ------------------------------------------------------------------
    # Command line parsing
    # ------------------------------------------------------------------
    def add_common_commandline_parameters(self,parser,training=False):

        if training==True:
            parser.add_argument('-i','--input', nargs=2, help='txt files containing list of image and label data files')
            parser.add_argument('-o','--output', help='Directory for saving the learned model')
            parser.add_argument('--resume_model', help='Directory to read existingmodel to resume from',default='')    
            parser.add_argument('-s','--steps', help='Number of steps (iterations) -- mapped to closest multiple of 10', default=300,type=int)
            parser.add_argument('-p','--patch_size', help='patch size -- mapped to closest power of two', default=64,type=int)
            parser.add_argument('--keep_prob',   help='Keep Probability for Dropoff',default=0.85,type=float)
            parser.add_argument('--no_augment',
                                help='If true do not augment patches by flipping and rotation (default=augment)',
                                action='store_true')
            parser.add_argument('--learning_rate',   help='Learning rate for optimizer',default=1e-4,type=float)
            parser.add_argument('--dry_run', help='If true, model is created but not trained.', action='store_true')
        else:
            parser.add_argument('-m','--model_path', help='path where to find the learned model (directory or .meta file)')    
            parser.add_argument('-i','--input', help='Input image to reconstuct (NIfTI image) (or text file listing images)')
            parser.add_argument('-o','--output', help='Output classification result (NIfTI image or directory if input is text file)')
            if (self.can_train()):
                parser.add_argument('--no_read_graph', help='If true, model is read from .meta file and not recreated in code', action='store_true')
            parser.add_argument('--keep_prob',   help='Keep Probability for Dropoff',default=1.0,type=float)
        
        parser.add_argument('-b','--batch_size', help='Number of training samples per mini-batch -- mapped to closest power of two', default=64,type=int)
        parser.add_argument('-d','--debug', help='Extra Debug Output', action='store_true')
        parser.add_argument('--extra_debug', help='Extra Debug Output', action='store_true')
        parser.add_argument('-g','--use_gpu', help='Train using GPU', action='store_true')
        parser.add_argument('--use_fp16', help='Use 16-bit floatsd GPU', action='store_true')


    def add_custom_commandline_parameters(self,parser,training):
        return 
        
    def extract_common_commandline_parameters(self,args,training=False):

        if training:
            self.commandlineparams['output_model_path']=args.output
            self.commandlineparams['target_data']=args.input[1]
            self.commandlineparams['input_data']=args.input[0]
            self.commandlineparams['max_steps']=args.steps
            self.commandlineparams['dry_run']=args.dry_run
            self.commandlineparams['patch_size']=bisutil.getpoweroftwo(args.patch_size,8,128)
            self.commandlineparams['learning_rate']=args.learning_rate
            self.commandlineparams['resume_model_path']=args.resume_model
            self.commandlineparams['no_augment']=args.no_augment
            maxbatch=512
        else:
            self.commandlineparams['model_path']=args.model_path
            self.commandlineparams['input_data']=args.input
            self.commandlineparams['output_path']=args.output
            if (self.can_train()):
                self.commandlineparams['no_read_graph']=args.no_read_graph
            else:
                self.commandlineparams['no_read_graph']=False
            maxbatch=4096
            
        self.commandlineparams['batch_size']=bisutil.getpoweroftwo(args.batch_size,4,maxbatch)
        self.params['keep_prob']=bisutil.force_inrange(args.keep_prob,0.0,1.0)
        self.commandlineparams['use_gpu']=args.use_gpu
        self.commandlineparams['debug']=args.debug
        self.commandlineparams['extra_debug']=args.extra_debug
        self.commandlineparams['use_fp16']=args.use_fp16


    def extract_custom_commandline_parameters(self,args,training=False):
        return
        
    def parse_commandline(self,training=False):


        if (self.can_train()):

            extra=" Use the --train flag to switch to training mode. (Use --train -h to see training help)"
            if (training!=True):
                training=False
                
            self.commandlineparams['forcetrain']=False
            if '--train' in sys.argv:
                self.commandlineparams['forcetrain']=True
                training=True;
        else:
            training=False
            extra=""
                
        if training:
            parser = argparse.ArgumentParser(description='Train: '+self.get_description()) 
        else:
            parser = argparse.ArgumentParser(description='Test/Apply: '+self.get_description()+extra)


        if self.commandlineparams['forcetrain']:
            parser.add_argument('--train', help='force training mode', action='store_true')

        self.add_common_commandline_parameters(parser,training)
        self.add_custom_commandline_parameters(parser,training)

        args = parser.parse_args()
        
        self.extract_common_commandline_parameters(args,training)
        self.extract_custom_commandline_parameters(args,training)


    # ------------------------------------------------------------------
    # Read Parameters from checkpoint file
    # ------------------------------------------------------------------

    # Read Parameters
    def read_parameters_from_checkpoint(self,model_path):

        cname=bisutil.get_read_checkpoint(model_path)
        
        list_to_get=[];

        for key in self.params:
            if (not key in self.donotreadlist):
                list_to_get.append(key)
                
                
        values=bisutil.get_tensor_list_from_checkpoint_file(cname,list_to_get)
        for key in values:
            self.params[key]=values[key]
            try:
                a=self.params[key].item()
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
                a=self.calcparams[key].item()
                self.calcparams[key]=a
            except AttributeError:
                a=0


        print("+++++ Read parameters from checkpoint file "+cname+" (from "+model_path+")")
        if self.commandlineparams['debug']:
            values.update(values2)
            bisutil.print_dict(values,extra='+++++')
        return cname

    def add_parameters_as_variables(self):
        with tf.variable_scope('Parameters') as scope:
            for key in self.params:
                tf.Variable(self.params[key],name=key,trainable=False)
            for key in self.calcparams:
                tf.Variable(self.calcparams[key],name=key,trainable=False)

    # ------------------------------------------------------------------------
    # Initialize flags for bis_tf_util
    # ------------------------------------------------------------------------
    def initialize_tf_util(self):

        if self.commandlineparams['extra_debug']:
            self.commandlineparams['debug']=True
            
        if self.commandlineparams['extra_debug']:
            os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
        elif self.commandlineparams['debug']:
            os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
        else:
            os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"


        bisutil.INTERNAL['extra_debug']=self.commandlineparams['extra_debug']
        bisutil.INTERNAL['debug']=self.commandlineparams['debug']
        bisutil.INTERNAL['use_fp16']=self.commandlineparams['use_fp16']
        bisutil.INTERNAL['use_gpu']=self.commandlineparams['use_gpu']
        bisutil.set_device(self.commandlineparams['use_gpu'])
        bisutil.debug_print('+++++ bis_tf_utils ',bisutil.INTERNAL)
        
    # ------------------------------------------------------------------------
    # Initialize network, create input placeholders and calculate parameters
    # ------------------------------------------------------------------------
    def initialize(self,imgshape,patch_size=64,max_outputs=3):

        
        shapelength=len(imgshape)
            
        if (shapelength<4):
            raise RuntimeError('Bad Input image')

        offset=1
            
        if (shapelength==4):
            self.calcparams['num_frames']=1;
        else:
            self.calcparams['num_frames']=imgshape[4];

        if (imgshape[3]>1):
            self.calcparams['threed']=True;
        else:
            self.calcparams['threed']=False;
                
        self.calcparams['patch_size']= bisutil.getpoweroftwo(patch_size,8,128);

        image_size=self.calcparams['patch_size']
        num_frames=self.calcparams['num_frames']
        
        print('+++++ Initializing input image ',imgshape,' and allocating placeholders size='+str(image_size))


        iname="Input"
        if self.commandlineparams['use_fp16']:
            iname="input_precast"
            
        with tf.variable_scope('Inputs') as scope:
        
        # Let python feed the data to the graph
            if self.calcparams['threed']:
                images = tf.placeholder(tf.float32, shape=(None,image_size,image_size,image_size,
                                                           num_frames), name=iname)
                labels = tf.placeholder(tf.float32, shape=(None,image_size,image_size,image_size,1), name='Target')
                midslice=image_size//2
                print('+++++ Extracting midslice='+str(midslice)+' for summary images')
                i_slices = tf.slice(images, (0, 0, 0, midslice, 0), (-1, image_size, image_size, 1, 1))
                tf.summary.image('input_midslice', tf.squeeze(i_slices, 4), max_outputs=max_outputs)

                l_slices = tf.slice(labels, (0, 0, 0, midslice, 0), (-1, image_size, image_size, 1, 1))
                tf.summary.image('target_midslice', tf.squeeze(l_slices, 4), max_outputs=max_outputs)
            else:
                images = tf.placeholder(tf.float32, shape=(None,image_size,image_size,num_frames), name=iname)
                labels = tf.placeholder(tf.float32, shape=(None,image_size,image_size,1), name='Target')
                tf.summary.image('input', images, max_outputs=max_outputs)
                tf.summary.image('target', tf.cast(labels,tf.float32), max_outputs=max_outputs)
            
            if self.commandlineparams['use_fp16']:
                images = tf.cast(images, tf.float16,name="Input")
            keep_prob = tf.placeholder(tf.float32, name='Keep_Probability')

                
        self.pointers['input_pointer']=images;
        self.pointers['target_pointer']=labels;
        self.pointers['keep_pointer']=keep_prob

        if self.commandlineparams['debug']:
            print('+++++ All Initialized')
            bisutil.print_dict(self.params,extra='+++++',header='params:')
            bisutil.print_dict(self.calcparams,extra='+++++',header='calcparams:')
            bisutil.print_dict(self.commandlineparams,extra='+++++',header='commandlineparams:')
            print('+++++')

                        

        
    # ------------------------------------------------------------------------
    # Get Checkpoint output name
    # ------------------------------------------------------------------------
    def getoutputname(self):
        outputname=self.params['name'];
        if self.calcparams['threed']:
            outputname+="_3d.ckpt"
        else:
            outputname+="_2d.ckpt"
        return outputname


    # -----------------------------------------------------------------------------
    # "Virtual" Functions to be specified in derived classes
    # -----------------------------------------------------------------------------

    # Create inference
    # stores result in self.pointers['output_dict'] and also retuns this as output
    def create_inference(self,training=True):
        raise ValueError('Create_Inference not implemented')

    # Create loss
    # stores in self.pointers['loss_op'] and also returns this as output
    def create_loss(self):
        raise ValueError('Create Loss not implemented')

    # Create optimizer -- the default is here
    # stores in self.pointers['train_op'] and also returns this as output
    def create_optimizer(self):

        learning_rate=self.commandlineparams['learning_rate']
        
        loss_op=self.pointers['loss_op']
        if (loss_op==None):
            raise ValueError('No loss_op has been specified')

        print('+++++ Creating AdamOptimizer, learning_rate='+str(learning_rate))
        
        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads = optimizer.compute_gradients(loss_op, tf.trainable_variables())
            self.pointers['train_op']=optimizer.apply_gradients(grads,name="Optimizer")
            return self.pointers['train_op']


    # ------------------------------------------------------------------------
    # Reconstruct is inner testing method
    # Most of the work is done in bisutil.reconstruct_image
    # ------------------------------------------------------------------------

    
    # Image Reconstruction method
    def reconstruct_images(self,testing_data,model_path,batch_size=16,no_read_graph=True):

        # Keep prob for recon can be different than what model specified
        self.donotreadlist.append('keep_prob')
        self.donotreadlist.append('edge_smoothness')

        try:
            cname=self.read_parameters_from_checkpoint(model_path)
        except RuntimeError as e:
            bisutil.handle_error_and_exit(e)

            
        with tf.Graph().as_default() as graph:

            if no_read_graph==True:
                # Get the images and their labels for the images (3D!!!)
                self.initialize(imgshape=testing_data.get_image_shape(),
                                patch_size=self.calcparams['patch_size'])

            
                #Build a Graph that computes the logits predictions from the inference model
                self.print_line()
                self.create_inference(training=False);
            else:
                # Read the graph!!!!
                metaname=cname+".meta"
                print('+++++ Importing graph from '+metaname)
                model_reader = tf.train.import_meta_graph(metaname)
                
            self.print_line()
            
            return bisutil.reconstruct_images(checkpointname=cname,
                                              testing_data=testing_data,
                                              images_pointer=graph.get_tensor_by_name('Inputs/Input:0'),
                                              model_output_image=graph.get_tensor_by_name('Outputs/Output:0'),
                                              keep_pointer = graph.get_tensor_by_name('Inputs/Keep_Probability:0'),
                                              keep_value = self.params['keep_prob'],
                                              patch_width=self.calcparams['patch_size'],
                                              threed=self.calcparams['threed'],
                                              batch_size=batch_size);
        
    # ------------------------------------------------------------------------
    # Network training method(s)
    # Most of the work is done in bisutil.run_training
    # ------------------------------------------------------------------------
    def set_network_tensor_values(self):
        if self.commandlineparams['debug']:
            print("+++++ Using automatically initialized tensor values")
        
    
    def train(self,
              training_data,
              output_model_path,
              resume_model_path,
              patch_size,
              max_outputs=3,
              max_steps=1000,
              batch_size=10,
              augmentation=True,
              epsilon=0.1):



        if (len(resume_model_path)>2):
            print("+++++ Resuming training: ignoring commandline model parameters")
            try:
                cname=self.read_parameters_from_checkpoint(resume_model_path)
            except RuntimeError as e:
                bisutil.handle_error_and_exit(e)
            print("+++++")
        
        with tf.Graph().as_default():


            # All of these calls communicate by storing results in self.pointers
            # Get the images and their labels for the images (Fix 3D)
            self.initialize(imgshape=training_data.get_image_shape(),
                            max_outputs=max_outputs,
                            patch_size=patch_size)
            
            # Build a Graph that computes the logits predictions from the inference model
            self.print_line()
            self.create_inference(training=True);
            self.print_line()
            
            # Initialize values (from pre-trained network etc.)
            self.set_network_tensor_values()
            
            # Calculate the loss.
            self.create_loss()
            
            # Build a Graph that trains the model with one batch of examples and 
            # updates the model parameters.
            self.create_optimizer()

            
            if (self.commandlineparams['dry_run']==True):
                self.print_line()
                print("+++++ D r y  R u n.  N O T  T R A I N I N G!")
                max_steps=-1
            
            # From here use results in self.pointers to run training

            start_time = time.time()
            start_date = datetime.now()

            oname=self.getoutputname()
            self.print_line()
            with tf.variable_scope('Save') as scope:
                try:                
                    jsonfile=bisutil.run_training(
                        training_data=training_data,
                        images_pointer = self.pointers['input_pointer'],
                        labels_pointer = self.pointers['target_pointer'],
                        keep_pointer =  self.pointers['keep_pointer'],
                        keep_value   = self.params['keep_prob'],
                        loss_op = self.pointers['loss_op'],
                        train_op= self.pointers['train_op'],
                        output_model_path=output_model_path,
                        resume_model_path=resume_model_path,
                        batch_size=batch_size,
                        max_steps=max_steps,
                        model_name=oname,
                        patch_size=self.calcparams['patch_size'],
                        epsilon=epsilon,
                        augmentation=augmentation)
                except PermissionDeniedError as e:
                    a=str(e).split('\n')
                    bisutil.handle_error_and_exit('Failed to write output in '+str(a[0]))

            end_time = time.time()
            end_date = datetime.now()
            duration = end_time - start_time
            tform='%Y-%m-%d %H:%M:%S'
            s=datetime.now().strftime(tform)
            

            print('+++++\n+++++ Training done. Started at:',start_date.strftime('%H:%M:%S'),
                  ' ended at:',end_date.strftime('%H:%M:%S'),
                  ' duration=%.2fs' %(duration))
            print('+++++')
                  
            
            odict = {
                'parameters' : self.params,
                'calculated' : self.calcparams,
                'commandlineparams' : self.commandlineparams,
                'system' : {
                    'os'   : platform.system()+' '+platform.release(),
                    'date' : s,
                    'duration' : duration,
                    'node' : platform.node(),
                    'machine' : platform.machine(),
                    'python'  : platform.python_version(),
                    'tensorflow' :  str(tf.__version__),
                    'numpy' : np.version.version,
                    'pwd'   : os.getcwd(),
                    'rawcommandline' :  ' '.join(sys.argv),
                }
            }
            
            with open(jsonfile, 'w') as fp:
                json.dump(odict, fp, sort_keys=True,indent=4)
                
            print('+++++ Stored parameters in ',jsonfile)

    # -----------------------------------------------------------------------------
    # Main training method -- takes command line args loads images and calls train
    # -----------------------------------------------------------------------------
    def train_main(self,argv=None):

        training_data=pimage.ImageSet()
        try:
            training_data.load(self.commandlineparams['input_data'],
                               self.commandlineparams['target_data'])
        except ValueError as e:
            bisutil.handle_error_and_exit(e)

        print('+++++')
        print('+++++ Training model output=',os.path.abspath(self.commandlineparams['output_model_path']))
        if (len(self.commandlineparams['resume_model_path'])>2):
            print('+++++ Resuming from '+os.path.abspath(self.commandlineparams['resume_model_path']))
        self.print_line()

        augmentation=True
        if self.commandlineparams['no_augment']:
            augmentation=False
        
        self.train(training_data=training_data,
                   output_model_path=self.commandlineparams['output_model_path'],
                   resume_model_path=self.commandlineparams['resume_model_path'],
                   patch_size=self.commandlineparams['patch_size'],
                   max_outputs=self.calcparams['max_outputs'],
                   max_steps=self.commandlineparams['max_steps'],
                   augmentation=augmentation,
                   batch_size=self.commandlineparams['batch_size'])
            
                
    # ------------------------------------------------------------------------
    # Main testing method -- loads image, calls reconstructs and then saves
    # ------------------------------------------------------------------------
    def test_main(self,argv=None):


        print('+++++ Loading data...')
        testing_data=pimage.ImageSet()
        try:
            testing_data.load(self.commandlineparams['input_data'])
        except ValueError as e:
            bisutil.handle_error_and_exit(e)
            
        print('+++++ Loaded input image with shape: %s' % (testing_data.get_image_shape() ))
        self.print_line()

        result_images=self.reconstruct_images(testing_data=testing_data,
                                              model_path=self.commandlineparams['model_path'],
                                              batch_size=self.commandlineparams['batch_size'],
                                              no_read_graph=self.commandlineparams['no_read_graph'])
        self.print_line()
        try:
            testing_data.save_reconstructed_image_data(data=result_images,path=self.commandlineparams['output_path'])
        except ValueError as e:
            bisutil.handle_error_and_exit(e)




    # ------------------------------------------------------------------------
    # Main testing method -- loads image, calls reconstructs and then saves
    # ------------------------------------------------------------------------
    def execute(self):

        print('\n+++++ Beginning to execute model=',self.params['name'],' using python',platform.python_version())
        self.parse_commandline()
        self.initialize_tf_util()
        self.print_line()

        if not self.commandlineparams['forcetrain']:
            self.test_main()
        else:
            self.train_main()
        self.print_line()
        print('+++++ Execute completed\n')


if __name__ == '__main__':

    # This works for testing
    BaseModel().execute()

        
        
