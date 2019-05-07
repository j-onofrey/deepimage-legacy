from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import numpy as np
import tensorflow as tf
import bis_tf_utils as bisutil
import bis_tf_basemodel as bisbasemodel
import bis_tf_loss_utils as bislossutil

def create_fcn_model(input_data, 
                     num_conv_layers=1, 
                     filter_size=3, 
                     num_frames=1, 
                     num_filters=64, 
                     keep_pointer=None, 
                     num_classes=2, 
                     name='FCN', 
                     dodebug=False, 
                     threed=False, 
                     norelu=False,
                     nofuse=False,
                     num_connected=512,
                     num_fully_connected_layers=2,
                     poolmode='max'):

    with tf.variable_scope(name) as scope:

        clayer_inputs = [ input_data ]
        clayer_outputs = [ ]
        c1=filter_size
        c2=num_frames
        c3=num_filters

        for clayer in range(0,num_conv_layers):

            cname=str(clayer+1)
            with tf.variable_scope("c"+cname) as scope:
                if dodebug:
                    print('=====\n===== Convolution Layer '+cname);
                    
                shape11=bisutil.createConvolutionShape(c1,#self.params['filter_size'],
                                                       c2,#self.calcparams['num_frames'],
                                                       c3,#self.params['num_filters'],
                                                       threed)
                relu1=bisutil.createConvolutionLayerRELU(input=clayer_inputs[clayer],#input_data,
                                                         name='convolution_'+cname+'_1',
                                                         shape=shape11,
                                                         padvalue=-1,norelu=norelu,
                                                         threed=threed);
            
                shape12=bisutil.createConvolutionShape(c1,#self.params['filter_size'],
                                                       c3,#self.params['num_filters'],
                                                       c3,#self.params['num_filters']
                                                       threed)
                
                pool3=bisutil.createConvolutionLayerRELUPool(input=relu1,
                                                             mode=poolmode,
                                                             name='convolution_'+cname+'_2',
                                                             shape=shape12,
                                                             padvalue=0,
                                                             norelu=norelu,threed=threed);
                
                c2=c3
                c3=c3*2
                clayer_inputs.append(pool3)
                
                
        # Fully connected layers

        full_inputs = [ pool3 ]

        for flayer in range(0,num_fully_connected_layers):
            fname=str(flayer+1)
                            
            with tf.variable_scope("f"+fname):

                if dodebug:
                    print('=====\n===== Fully Connected Layer '+fname);

                if (flayer==0):
                    shapef1=bisutil.createFullyConnectedShape(pool3.get_shape().as_list(),
                                                              c2,
                                                              num_connected,
                                                              threed);
                else:
                    shapef1=bisutil.createSquareShape(1,num_connected,num_connected,threed)
                    
                if (flayer<num_fully_connected_layers-1):
                    fc1=bisutil.createFullyConnectedLayerWithDropout(input=full_inputs[flayer],
                                                                     shape=shapef1,
                                                                     name="fullyconnected_"+fname,
                                                                     keep_prob_tensor=keep_pointer,
                                                                     threed=threed)
                else:
                    fc1=bisutil.createFullyConnectedLayer(input=full_inputs[flayer],
                                                          shape=shapef1,
                                                          name='fullyconnected_'+fname,
                                                          threed=threed)
                full_inputs.append(fc1)
                            
                    
        # Needed from here on below
        dindex=3
        if (threed):
            dindex=4

        dlayer_inputs = [ fc1 ]
        d1=4
        d2=num_connected

        for dlayer in range(0,num_conv_layers-1):
            dname=str(dlayer+1)

            with tf.variable_scope("d"+dname):
                if dodebug:
                    print('=====\n===== Deconv Layer '+dname);

                pool=clayer_inputs[num_conv_layers-(dlayer+1)]
                deconv_shape1 = pool.get_shape()
                shaped1=bisutil.createSquareShape(d1,deconv_shape1[dindex].value,
                                                  d2,
                                                  threed)
                out_shape1=bisutil.createDeconvolutionDynamicOutputShape(pool,input_data,nofuse,threed)
                d2=deconv_shape1[dindex]

                fuse1=bisutil.createDeconvolutionLayerFuse(input=dlayer_inputs[dlayer],
                                                           fuse_input=pool,
                                                           dynamic_output_shape=out_shape1,
                                                           shape=shaped1,
                                                           nofuse=nofuse,
                                                           threed=threed,
                                                           name="deconv_"+dname)
                dlayer_inputs.append(fuse1)

        dlayer=num_conv_layers-1
        dname=str(dlayer+1)
        with tf.variable_scope("d"+dname):
            if dodebug:
                print('=====\n===== Deconv Layer '+dname+' (final)');

            shaped_final=bisutil.createSquareShape(4,
                                                   num_classes,
                                                   d2,
                                                   threed)
            out_shapefinal=bisutil.createDeconvolutionDynamicOutputShape(input=input_data,
                                                                         orig_input_data=input_data,
                                                                         nofuse=True,
                                                                         threed=threed,
                                                                         num_classes=num_classes)


                
            deconv_final=bisutil.createDeconvolutionLayer(input=dlayer_inputs[dlayer],
                                                          name="deconv_final",
                                                          input_shape=shaped_final,
                                                          threed=threed,
                                                          dynamic_output_shape=out_shapefinal)

            return deconv_final

class FCNModel(bisbasemodel.BaseModel):

    def __init__(self):
        self.params['name']='FCN'
        self.params['no_relu']=False
        self.params['no_fuse']=False
        self.params['avg_pool']=False
        self.params['filter_size']=3
        self.params['num_connected']=512
        self.params['num_filters']=32
        self.params['num_conv_layers']=2
        self.params['num_fully_connected_layers']=2
        self.commandlineparams['edge_smoothness']=0.0
        self.donotreadlist.append('edge_smoothness')
        bislossutil.create_loss_params(self)
        super().__init__()

    def get_description(self):
        return " Fully convolutional NN for 2D/3D images."

    def can_train(self):
        return True

    def add_custom_commandline_parameters(self,parser,training):

        if training:
            parser.add_argument('--no_relu', help='If set no RELU units will be used', 
                                default=None,action='store_true')
            parser.add_argument('--no_fuse', help='If set no FUSE units will be used in deconvolution', 
                                default=None,action='store_true')
            parser.add_argument('--avg_pool', help='If set will use avg_pool instead of max_pool',
                                default=None, action='store_true')
            parser.add_argument('-l','--smoothness', help='Smoothness factor (Baysian training with edgemap)',
                                default=None,type=float)
            parser.add_argument('--num_conv_layers', help='Number of Convolution Layers',
                                default=None,type=int)
            parser.add_argument('--num_connected', help='Number of Elements in Connected Layers',
                                default=None,type=int)
            parser.add_argument('--num_fully_connected_layers', help='Number of Fully Connected Layers',
                                default=None,type=int)
            parser.add_argument('--num_filters', help='Number of Convolution Filters',
                                default=None,type=int)
            parser.add_argument('--filter_size',   help='Filter Size  in Convolutional Layers',
                                default=None,type=int)
            bislossutil.add_parser_loss_params(parser)
            
    def extract_custom_commandline_parameters(self,args,training=False):

        if training:
            self.set_param_from_arg(name='no_relu',value=args.no_relu)
            self.set_param_from_arg(name='no_fuse',value=args.no_fuse)
            self.set_param_from_arg(name='avg_pool',value=args.avg_pool)

            self.set_param_from_arg(name='num_conv_layers',
                                    value=bisutil.force_inrange(args.num_conv_layers,minv=1,maxv=8))
            
            self.set_param_from_arg(name='num_fully_connected_layers',
                                    value=bisutil.force_inrange(args.num_fully_connected_layers,minv=1,maxv=8))
            
            self.set_param_from_arg(name='num_fully_connected_layers',
                                    value=bisutil.force_inrange(args.num_fully_connected_layers,minv=1,maxv=8))
            
            self.set_commandlineparam_from_arg(name='edge_smoothness',
                                    value=bisutil.force_inrange(args.smoothness,minv=0.0,maxv=10000.0))
            
            self.set_param_from_arg(name='filter_size',
                                    value=bisutil.force_inrange(args.filter_size,minv=3,maxv=11))
            self.set_param_from_arg(name='num_connected',
                                    value=bisutil.getpoweroftwo(args.num_connected,8,2048))
            self.set_param_from_arg(name='num_filters',
                                    value=bisutil.getpoweroftwo(args.num_filters,4,128))
            bislossutil.extract_parser_loss_params(self,args)


    def create_loss(self,output_dict):
        return bislossutil.create_loss_function(self,
                                                output_dict=output_dict,
                                                smoothness=self.commandlineparams['edge_smoothness'],
                                                batch_size=self.commandlineparams['batch_size']);


    # --------------------------------------------------------------------------------------------
    # Variable length Inference
    # --------------------------------------------------------------------------------------------
    
    def create_inference(self,training=True):

        dodebug=self.commandlineparams['debug']
        poolmode="max"
        if self.params['avg_pool']:
            poolmode="avg"
            
        norelu=self.params['no_relu']
        nofuse=self.params['no_fuse']
        input_data=self.pointers['input_pointer']
        threed=self.calcparams['threed']
        imgshape=input_data.get_shape()
        num_conv_layers=int(self.params['num_conv_layers'])
        num_fully_connected_layers=int(self.params['num_fully_connected_layers'])
        p=self.commandlineparams['patch_size']

        cname=bisutil.getdimname(threed)
        print('===== Creating '+cname+' FCN Inference Model (training='+str(training)+'). Inputs shape= %s ' % (imgshape))
        print('=====')
        
        # Check if num_conv_layers is not too big
        while (int(math.pow(2,num_conv_layers))>p):
            num_conv_layers=num_conv_layers-1
            
        if self.params['num_conv_layers']!=num_conv_layers:
            print('===== Reduced conv_layers from '+str(self.params['num_conv_layers'])+' to '+str(num_conv_layers)+' as patch size is too small.')
            self.params['num_conv_layers']=num_conv_layers
            

        bisutil.print_dict({ 'Num Conv/Deconv Layers' : self.params['num_conv_layers'],
                             'Num Fully Connected  Layers'  : self.params['num_fully_connected_layers'],
                             'Filter Size': self.params['filter_size'],
                             'Num Filters': self.params['num_filters'],
                             'Num Connected': self.params['num_connected'],
                             'Num Classes':  self.calcparams['num_classes'],
                             'Patch Size':   self.commandlineparams['patch_size'],
                             'Num Frames':   self.calcparams['num_frames'],
                             'NoRelu':norelu,
                             'PoolMode':poolmode,
                             'NoFuse':nofuse },
                           extra="=====",header="Model Parameters:")

        if training:
            self.add_parameters_as_variables()

            
        deconv_final=create_fcn_model(input_data,
                                      num_conv_layers=num_conv_layers,
                                      filter_size=self.params['filter_size'],
                                      num_frames=self.calcparams['num_frames'],
                                      keep_pointer=self.pointers['keep_pointer'],
                                      num_classes=self.calcparams['num_classes'],
                                      name='FCN',
                                      dodebug=dodebug,
                                      threed=threed,
                                      norelu=norelu,
                                      nofuse=nofuse,
                                      num_connected=self.params['num_connected'],
                                      num_fully_connected_layers=self.params['num_fully_connected_layers'],
                                      poolmode=poolmode);

        
        print("=====")

        with tf.variable_scope('Outputs'):
                
            if self.calcparams['num_classes']>1:
                print("===== In classification mode (adding annotation)")
                dim=len(deconv_final.get_shape())-1
                annotation_pred = tf.argmax(deconv_final, dimension=dim, name='prediction')
                output= { "image":  tf.expand_dims(annotation_pred, dim=dim,name="Output"),
                          "logits": tf.identity(deconv_final,name="Logits") ,
                          "regularizer": None}
            else:
                print("===== In regression mode")
                output = { "image":  tf.identity(deconv_final,name="Output"),
                           "logits": tf.identity(deconv_final,name="Logits"),
                           "regularizer" : None };
                
            o_shape=output['image'].get_shape()
            bisutil.image_summary(output['image'],'prediction',o_shape[3].value,1,self.calcparams['threed'],max_outputs=self.commandlineparams['max_outputs'])

                        
            if (self.commandlineparams['edge_smoothness']>0.0 and training):
                output['regularizer']=bisutil.createSmoothnessLayer(output['image'],
                                                                    self.calcparams['num_classes'],
                                                                    self.calcparams['threed']);
            elif dodebug:
                print('===== Not adding back end smoothness computation')

        return output



if __name__ == '__main__':

    FCNModel().execute()

        
