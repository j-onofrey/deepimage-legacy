#!/usr/bin/env python3

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
import bis_tf_unetmodel as bisunet
import bis_tf_jounet as bisjounet
import bis_tf_loss_utils as bislossutil


def crop_image(inp,radius,patch_size,threed=False,name="valid"):

    radius=int(radius)
    r=2*radius+1
    sz=patch_size-2*radius
    if not threed:
        valid=tf.slice(inp,begin=[0,radius,radius,0],size=[-1,sz,sz,-1],name=name)
    else:
        valid=tf.slice(inp,begin=[0,radius,radius,radius,0],size=[-1,sz,sz,sz,-1],name=name)

    return valid

# -----------------------------------------------------------------------------------
def createWeightedSmoothnessLayer(outimg,
                                  maskimg,
                                  sigma=2.0,
                                  radius=2,
                                  alpha=0.5,
                                  obj=None,
                                  threed=False):


    radius=int(radius)
    r=2*radius+1
    print('===== \t adding back end weighted smoothness, window size=',r,' alpha=',alpha,' sigma=',sigma)

    with tf.variable_scope('Mask_Smooth_Outputs'):
        mask=tf.cast(maskimg,tf.float32)
        outimg=tf.cast(outimg,tf.float32)
        if threed:
            pad=tf.pad(mask,paddings=[ [0,0],[radius,radius],[radius,radius],[radius,radius],[0,0]], mode='CONSTANT',name='pad')
            sm_mask = tf.nn.avg_pool3d(pad, ksize=[1,r,r,r,1], strides=[1,1,1,1,1], padding='VALID', name='Sm_Mask3D')
        else:
            pad=tf.pad(mask,paddings=[ [0,0],[radius,radius],[radius,radius],[0,0]], mode='CONSTANT',name='pad')
            sm_mask = tf.nn.avg_pool(pad, ksize=[1,r,r,1], strides=[1,1,1,1], padding='VALID', name='Sm_Mas23D')


        if (alpha<0.05):
            alpha=0.05
        a=(0.5-sm_mask)/alpha
        # Quadratic distance from a
        final_weight=tf.identity(tf.multiply(a,a),name='weight_map')
        o_shape=final_weight.get_shape()

        bisutil.image_summary(final_weight,'sm_weight',o_shape[3].value,1,threed,max_outputs=obj.calcparams['max_outputs'])
#        bisutil.image_summary(sm_mask,'sm_mask',o_shape[3].value,1,threed,max_outputs=obj.calcparams['max_outputs'])


        if not threed:
            grad_x = np.zeros([3, 1, 1, 1])
            grad_x[ 0, 0, : , :] = -1
            grad_x[ 1, 0, : , :] =  2
            grad_x[ 2, 0, : , :] = -1

            grad_y = np.zeros([1, 3, 1, 1])
            grad_y[ 0, 0, : , : ] = -1
            grad_y[ 0, 1, : , : ] =  2
            grad_y[ 0, 2, : , : ] = -1

            edge_conv_x = tf.nn.conv2d(outimg,grad_x,strides=[1,1,1,1],
                                       padding='SAME',name='smoothness_final_x');
            edge_conv_y = tf.nn.conv2d(outimg,grad_y,strides=[1,1,1,1],
                                       padding='SAME',name='smoothness_final_y');

            sum_image=tf.identity(tf.multiply(edge_conv_x,edge_conv_x)+tf.multiply(edge_conv_y,edge_conv_y))

        else:
            grad_x = np.zeros([3, 1, 1, 1, 1])
            grad_x[2, 0, 0, :, : ] = -1
            grad_x[1, 0, 0, :, : ] =  2
            grad_x[0, 0, 0, :, : ] = -1

            grad_y = np.zeros([1, 3, 1, 1, 1])
            grad_y[0, 0, 0, :, : ] = -1
            grad_y[0, 1, 0, :, : ] =  2
            grad_y[0, 2, 0, :, : ] = -1

            grad_z = np.zeros([1, 1, 3, 1, 1])
            grad_z[0, 0, 0, :, : ] = -1
            grad_z[0, 0, 1, :, : ] =  2
            grad_z[0, 0, 2, :, : ] = -1

            conv_x = tf.nn.conv3d(outimg,grad_x,strides=[1,1,1,1,1],
                                  padding='SAME',name='smoothness_final_x');
            conv_y = tf.nn.conv3d(outimg,grad_y,strides=[1,1,1,1,1],
                                  padding='SAME',name='smoothness_final_y');
            conv_z = tf.nn.conv3d(outimg,grad_z,strides=[1,1,1,1,1],
                                  padding='SAME',name='smoothness_final_z');
            sum_image=tf.identity(tf.multiply(edge_conv_x,edge_conv_x)+
                                  tf.multiply(edge_conv_y,edge_conv_y)+
                                  tf.multiply(edge_conv_z,edge_conv_z))

#        bisutil.image_summary(sum_image,'sm_image',o_shape[3].value,1,threed,max_outputs=obj.calcparams['max_outputs'])

        valid=crop_image(tf.multiply(sum_image,final_weight),radius=radius,patch_size=o_shape[1].value,threed=threed,name="valid_product")
        bisutil.image_summary(valid,'roughness',o_shape[3].value,1,threed,max_outputs=obj.calcparams['max_outputs'])

        edgeloss =  tf.reduce_mean(valid,name='MRegularizer')
        return edgeloss

def createUnWeightedSmoothnessLayer(outimg,
                                    maskimg,
                                    obj=None,
                                    threed=False):


    o_shape=maskimg.get_shape()

    radius=1
    r=2*radius+1
    print('===== \t adding back end unweighted smoothness, window size=',r)

    with tf.variable_scope('Mask_Smooth_Outputs'):
        outimg=tf.cast(outimg,tf.float32)

        if not threed:
            grad_x = np.zeros([3, 1, 1, 1])
            grad_x[ 0, 0, : , :] = -1
            grad_x[ 1, 0, : , :] =  2
            grad_x[ 2, 0, : , :] = -1

            grad_y = np.zeros([1, 3, 1, 1])
            grad_y[ 0, 0, : , : ] = -1
            grad_y[ 0, 1, : , : ] =  2
            grad_y[ 0, 2, : , : ] = -1

            edge_conv_x = tf.nn.conv2d(outimg,grad_x,strides=[1,1,1,1],
                                       padding='SAME',name='smoothness_final_x');
            edge_conv_y = tf.nn.conv2d(outimg,grad_y,strides=[1,1,1,1],
                                       padding='SAME',name='smoothness_final_y');

            sum_image=tf.identity(tf.multiply(edge_conv_x,edge_conv_x)+tf.multiply(edge_conv_y,edge_conv_y))

        else:
            grad_x = np.zeros([3, 1, 1, 1, 1])
            grad_x[2, 0, 0, :, : ] = -1
            grad_x[1, 0, 0, :, : ] =  2
            grad_x[0, 0, 0, :, : ] = -1

            grad_y = np.zeros([1, 3, 1, 1, 1])
            grad_y[0, 0, 0, :, : ] = -1
            grad_y[0, 1, 0, :, : ] =  2
            grad_y[0, 2, 0, :, : ] = -1

            grad_z = np.zeros([1, 1, 3, 1, 1])
            grad_z[0, 0, 0, :, : ] = -1
            grad_z[0, 0, 1, :, : ] =  2
            grad_z[0, 0, 2, :, : ] = -1

            conv_x = tf.nn.conv3d(outimg,grad_x,strides=[1,1,1,1,1],
                                  padding='SAME',name='smoothness_final_x');
            conv_y = tf.nn.conv3d(outimg,grad_y,strides=[1,1,1,1,1],
                                  padding='SAME',name='smoothness_final_y');
            conv_z = tf.nn.conv3d(outimg,grad_z,strides=[1,1,1,1,1],
                                  padding='SAME',name='smoothness_final_z');
            sum_image=tf.identity(tf.multiply(edge_conv_x,edge_conv_x)+
                                  tf.multiply(edge_conv_y,edge_conv_y)+
                                  tf.multiply(edge_conv_z,edge_conv_z))


        valid=crop_image(sum_image,radius=radius,patch_size=obj.calcparams['patch_size'],threed=threed,name="valid_product")
        bisutil.image_summary(valid,'edgemap',o_shape[3].value,1,threed,max_outputs=obj.calcparams['max_outputs'])
        edgeloss =  tf.reduce_mean(valid,name='MRegularizer')
        return edgeloss


class MaskedAutoEncoder(bisbasemodel.BaseModel):

    def __init__(self):
        self.params['name']='AutoEncoder'
        self.commandlineparams['sigma']=2.0
        self.commandlineparams['alpha']=0.5
        self.commandlineparams['weighted_smoothing']=False
        self.commandlineparams['radius']=2
        self.donotreadlist.append('sigma')
        self.donotreadlist.append('radius')
        self.donotreadlist.append('alpha')
        self.donotreadlist.append('weighted_smoothing')
        bisunet.create_unet_parameters(self)
        bislossutil.create_loss_params(self)
        self.commandlineparams['metric']='l2'
        super().__init__()

    def get_description(self):
        return "U-Net Masked Autoencoder Model for 2D/3D images."

    def can_train(self):
        return True

    def add_custom_commandline_parameters(self,parser,training):

        if training:
            bisunet.add_unet_commandline_parameters(parser,training);
            bislossutil.add_parser_loss_params(parser)
            parser.add_argument('--sigma',   help='Gaussian Filter sigma (voxels) for mask smoothing',
                                default=None,type=float)
            parser.add_argument('--radius',   help='radius of filter to avg mask over for computing smoothness weight  (voxels)',
                                default=None,type=float)
            parser.add_argument('--weighted_smoothing', help='If true, perform weighted smoothing else regular smoothing  (if smoothness>0.0)',
                                default=None,action='store_true')




    def extract_custom_commandline_parameters(self,args,training=False):

        if training:
            # Restrict this to regression stuff
            bislossutil.force_regression()
            bislossutil.extract_parser_loss_params(self,args)
            bisunet.extract_unet_commandline_parameters(self,args,training)
            self.set_commandlineparam_from_arg(name='sigma',
                                    value=bisutil.force_inrange(args.sigma,0.5,10.0))
            self.set_commandlineparam_from_arg(name='radius',
                                    value=bisutil.force_inrange(args.radius,1,10))
            self.set_commandlineparam_from_arg(name='weighted_smoothing',value=args.weighted_smoothing,defaultvalue=False)


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
        input_data=self.pointers['input_pointer']
        threed=self.calcparams['threed']
        imgshape=input_data.get_shape()
        num_conv_layers=int(self.params['num_conv_layers'])
        p=self.calcparams['patch_size']

        if training:
            self.add_parameters_as_variables()


        cname=bisutil.getdimname(threed)
        print('===== Creating '+cname+' MAE U-Net Model (training='+str(training)+'). Inputs shape= %s ' % (imgshape))
        print('=====')

        # Check if num_conv_layers is not too big
        while (int(math.pow(2,num_conv_layers+1))>p):
            num_conv_layers=num_conv_layers-1

        if self.params['num_conv_layers']!=num_conv_layers:
            print('===== Reduced conv_layers from '+str(self.params['num_conv_layers'])+' to '+str(num_conv_layers)+' as patch size is too small.')
            self.params['num_conv_layers']=num_conv_layers

        if (not self.commandlineparams['weighted_smoothing']):
            self.commandlineparams['radius']=1

        bisutil.print_dict({ 'Num Conv/Deconv Layers' : self.params['num_conv_layers'],
                             'Filter Size': self.params['filter_size'],
                             'Num Filters': self.params['num_filters'],
                             'Num Classes':  self.calcparams['num_classes'],
                             'Patch Size':   self.calcparams['patch_size'],
                             'Num Frames':   self.calcparams['num_frames'],
                             'NoRelu':norelu,
                             'Radius' : self.commandlineparams['radius'],
                             'PoolMode':poolmode},
                           extra="=====",header="Model Parameters:")




        # Create Model
        model_output = bisjounet.create_nobridge_unet_model(input_data,
                                                            num_conv_layers=num_conv_layers,
                                                            filter_size=self.params['filter_size'],
                                                            num_frames=self.calcparams['num_frames'],
                                                            num_filters=self.params['num_filters'],
                                                            keep_pointer=self.pointers['keep_pointer'],
                                                            num_classes=1,
                                                            name='UN-MAE',
                                                            dodebug=dodebug,
                                                            threed=threed,
                                                            norelu=norelu,
                                                            poolmode=poolmode)


        with tf.variable_scope('Outputs'):

            output = { "image":  tf.identity(model_output,name="Output"),
                       "logits": tf.identity(model_output,name="Logits"),
                       # This is critical!
                       "cropped_target" : input_data,
                       "regularizer" : None };
            o_shape=output['image'].get_shape()
            bisutil.image_summary(output['image'],'prediction',o_shape[3].value,1,self.calcparams['threed'],max_outputs=self.calcparams['max_outputs'])

            if (self.commandlineparams['edge_smoothness']>0.0 and training):
                if (self.commandlineparams['weighted_smoothing']):
                    output['regularizer']=createWeightedSmoothnessLayer(outimg=output['image'],
                                                                        maskimg=self.pointers['target_pointer'],
                                                                        sigma=self.commandlineparams['sigma'],
                                                                        radius=self.commandlineparams['radius'],
                                                                        alpha=0.5,
                                                                        obj=self,
                                                                        threed=self.calcparams['threed']);
                else:
                    output['regularizer']=createUnWeightedSmoothnessLayer(outimg=output['image'],
                                                                          maskimg=self.pointers['target_pointer'],
                                                                          obj=self,
                                                                          threed=self.calcparams['threed']);


                ci=crop_image(output['image'],patch_size=self.calcparams['patch_size'],radius=self.commandlineparams['radius'],threed=threed,name="valid_prediction")
                output['cropped_image']=ci
                output['cropped_logits']=ci
                #bisutil.image_summary(output['cropped_image'],'crop_pred',o_shape[3].value,1,self.calcparams['threed'],max_outputs=self.calcparams['max_outputs'])
                output['cropped_target']=crop_image(input_data,
                        patch_size=self.calcparams['patch_size'],radius=self.commandlineparams['radius'],name="valid_target",threed=threed)
                #bisutil.image_summary(output['cropped_target'],'crop_targ',o_shape[3].value,1,self.calcparams['threed'],max_outputs=self.calcparams['max_outputs'])

            elif dodebug:
                print('===== Not adding back end mask smoothness added')

        return output



if __name__ == '__main__':

    MaskedAutoEncoder().execute()
