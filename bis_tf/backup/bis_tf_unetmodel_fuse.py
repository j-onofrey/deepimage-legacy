from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import numpy as np
import tensorflow as tf
import bis_tf_utils as bisutil
import bis_image_patchutil as putil
import bis_tf_basemodel as bisbasemodel
import bis_tf_loss_utils as bislossutil



def create_unet_model(input_data, 
    num_conv_layers=1, 
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

                relu1 = bisutil.createConvolutionLayerRELU_auto(input=clayer_inputs[clayer-1],
                                                                name='convolution_'+cname+'_1',
                                                                filter_size=filter_size,
                                                                num_out_channels=num_out_channels,
                                                                padvalue=2,
                                                                norelu=norelu,
                                                                threed=threed)
                relu2 = bisutil.createConvolutionLayerRELU_auto(input=relu1,
                                                                name='convolution_'+cname+'_2',
                                                                filter_size=filter_size,
                                                                num_out_channels=num_out_channels,
                                                                padvalue=0,
                                                                norelu=norelu,
                                                                threed=threed)
                clayer_outputs.append(relu2)

                pool = bisutil.createPoolingLayer(input=relu2,
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

                relu1 = bisutil.createConvolutionLayerRELU_auto(input=feature_add,
                    name='xconvolution_'+dname+'_1',
                    filter_size=filter_size,
                    num_out_channels=num_out_channels,
                    padvalue=2,
                    norelu=norelu,
                    threed=threed)
                relu2 = bisutil.createConvolutionLayerRELU_auto(input=relu1,
                    name='xconvolution_'+dname+'_2',
                    filter_size=filter_size,
                    num_out_channels=num_out_channels,
                    padvalue=0,
                    norelu=norelu,
                    threed=threed)

                num_out_channels = int(num_out_channels/2)
                dlayer_inputs.append(relu2)
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






class UNetModel(bisbasemodel.BaseModel):

    def __init__(self):
        self.params['name']='UNet'
        self.params['edge_smoothness']=0.0
        self.params['no_relu']=False
        self.params['no_fuse']=False
        self.params['avg_pool']=False
        self.params['filter_size']=3
        self.params['num_connected']=512
        self.params['num_filters']=64
        self.params['num_conv_layers']=3
        self.params['num_fully_connected_layers']=2
        self.donotreadlist.append('edge_smoothness')
        bislossutil.create_loss_params(self)
        



    def get_description(self):
        return " U-Net Model for 2D/3D images."

    def can_train(self):
        return True

    def add_custom_commandline_parameters(self,parser,training):

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
            bislossutil.add_parser_loss_params(parser)


    def extract_custom_commandline_parameters(self,args,training=False):

        if training:
            bislossutil.extract_parser_loss_params(self,args)
            self.set_param_from_arg(name='no_relu',value=args.no_relu)
            self.set_param_from_arg(name='avg_pool',value=args.avg_pool)
            self.set_param_from_arg(name='num_conv_layers',
                                    value=bisutil.force_inrange(args.num_conv_layers,minv=2,maxv=8))
            self.set_param_from_arg(name='edge_smoothness',
                                    value=bisutil.force_inrange(args.smoothness,minv=0.0,maxv=10000.0))
            self.set_param_from_arg(name='filter_size',
                                    value=bisutil.force_inrange(args.filter_size,minv=3,maxv=11))
            self.set_param_from_arg(name='num_filters',
                                    value=bisutil.getpoweroftwo(args.num_filters,4,128))





    def create_loss(self,batch_size=-1):
        return bislossutil.create_loss_function(self,batch_size)


    # --------------------------------------------------------------------------------------------
    # Variable length Inference
    # --------------------------------------------------------------------------------------------

    def create_inference(self,training=True):

        dodebug=self.commandlineparams['debug']
        poolmode="max"
        if self.params['avg_pool']:
            poolmode="avg"

        norelu=self.params['no_relu']
        input_data=self.pointers['input_pointer']
        threed=self.calcparams['threed']
        imgshape=input_data.get_shape()
        num_conv_layers=int(self.params['num_conv_layers'])
        p=self.calcparams['patch_size']

        if training:
            self.add_parameters_as_variables()


        cname=bisutil.getdimname(threed)
        print('===== Creating '+cname+' U-Net Inference Model (training='+str(training)+'). Inputs shape= %s ' % (imgshape))
        print('=====')

        # Check if num_conv_layers is not too big
        while (int(math.pow(2,num_conv_layers+1))>p):
            num_conv_layers=num_conv_layers-1

        if self.params['num_conv_layers']!=num_conv_layers:
            print('===== Reduced conv_layers from '+str(self.params['num_conv_layers'])+' to '+str(num_conv_layers)+' as patch size is too small.')
            self.params['num_conv_layers']=num_conv_layers


        bisutil.print_dict({ 'Num Conv/Deconv Layers' : self.params['num_conv_layers'],
                             'Filter Size': self.params['filter_size'],
                             'Num Filters': self.params['num_filters'],
                             'Num Classes':  self.calcparams['num_classes'],
                             'Patch Size':   self.calcparams['patch_size'],
                             'Num Frames':   self.calcparams['num_frames'],
                             'NoRelu':norelu,
                             'PoolMode':poolmode},
                           extra="=====",header="Model Parameters:")




        # Create Model
        model_output = create_unet_model(input_data, 
            num_conv_layers=num_conv_layers,
            filter_size=self.params['filter_size'],
            num_frames=self.calcparams['num_frames'],
            num_filters=self.params['num_filters'],
            keep_pointer=self.pointers['keep_pointer'],
            num_classes=self.calcparams['num_classes'],
            name='U-Net',
            dodebug=dodebug,
            threed=threed,
            norelu=norelu,
            poolmode=poolmode)


        with tf.variable_scope('Outputs'):

            if self.calcparams['num_classes']>1:
                print("===== In classification mode (adding annotation)")
                dim=len(model_output.get_shape())-1
                annotation_pred = tf.argmax(model_output, dimension=dim, name='prediction')
                output= { "image":  tf.expand_dims(annotation_pred, dim=dim,name="Output"),
                          "logits": tf.identity(model_output,name="Logits") ,
                          "regularizer": None}
            else:
                print("===== In regression mode")
                output = { "image":  tf.identity(model_output,name="Output"),
                           "logits": tf.identity(model_output,name="Logits"),
                           "regularizer" : None };
            o_shape=output['image'].get_shape()
            bisutil.image_summary(output['image'],'prediction',o_shape[3].value,1,self.calcparams['threed'],max_outputs=self.calcparams['max_outputs'])


            if (self.params['edge_smoothness']>0.0 and training):
                output['regularizer']=bisutil.createSmoothnessLayer(output['image'],
                                                                    self.calcparams['num_classes'],
                                                                    self.calcparams['threed']);
            elif dodebug:
                print('===== Not adding back end smoothness compilation')


        self.pointers['output_dict']=output;
        return output






if __name__ == '__main__':

    UNetModel().execute()
