#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import numpy as np
import tensorflow as tf
import bis_tf_utils as bisutil


# --------------------------------------------------------------
#
# Create Gaussian Filter
#
# --------------------------------------------------------------

def createGaussian2D(sigma=1.0,axis=0,radius=-1):

    if sigma<0.001:
        sigma=0.0
        filt=np.zeros([1,1,1,1],dtype=np.float32)
        filt[0,0,0,0]=1.0
    else:
        if radius<1:
            radius=int(2.0*sigma+0.5)
        width=2*radius+1
        half=radius
    
        filt=np.zeros([width,1,1,1],dtype=np.float32)
        
        sum=0.0
        
        for k in range(0,width):
            r=(k-radius)
            r2=-(r*r)/(2.0*sigma*sigma)
            G=math.exp(r2)
            sum=sum+G
            filt[k,0,0,0]=G
            
        norm=1.0/sum
        filt=filt*norm


        # Transpose 
        if axis==1:
            filt=np.transpose(filt,(1,0,2,3))


    return filt
    

def createGaussian(sigma=1.0,axis=0,threed=False,radius=-1):

    if not threed:
        return createGaussian2D(sigma,axis,radius)

    if sigma<0.001:
        sigma=0.0
        filt=np.zeros([1,1,1,1],dtype=np.float32)
        for i in range(0,1):
            filt[0,0,0,i]=1.0
    else:
        if radius<1:
            radius=int(2.0*sigma+0.5)
        width=2*radius+1
        half=radius
    
        filt=np.zeros([width,1,1,1,1],dtype=np.float32)
        
        sum=0.0
        
        for k in range(0,width):
            r=(k-radius)
            r2=-(r*r)/(2.0*sigma*sigma)
            G=math.exp(r2)
            sum=sum+G
            filt[k,0,0,0,0]=G
            
        norm=1.0/sum
        filt=filt*norm


        # Transpose 
        if axis==1:
            filt=np.transpose(filt,(1,0,2,3,4))
        elif axis==2:
            filt=np.transpose(filt,(2,1,0,3,4))

            
    return filt

# --------------------------------------------------------------
#
#  Bilinear interpolation of image to match original
#
# --------------------------------------------------------------
def upSample(input,newshape,threed=False,interp=1):

    if interp==0 or interp==False:
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    else:
        method=tf.image.ResizeMethod.BILINEAR
    
    with tf.variable_scope('UpSample'):
        if threed:
            size=[newshape[0],newshape[1],newshape[2]]
        else:
            size=[ newshape[0],newshape[1]]

        return tf.image.resize_images(input, method=method,size=size)

def resampleTo(input,target_image,threed=False,interp=1):


    inpshape=target_image.get_shape()
    if threed:
        ishape=[ inpshape[1].value,inpshape[2].value,inpshape[3].value]
    else:
        ishape=[ inpshape[1].value,inpshape[2].value ]
    return upSample(input,ishape,threed,interp);


# --------------------------------------------------------------
#
# cropImage numpixels from boundary
#
# --------------------------------------------------------------

def cropImageBoundary(input,numpixels=2,threed=False,comment='image'):

    
    width=input.get_shape()[1].value
    origin=numpixels
    sz=width-2*numpixels

    bisutil.debug_print('+_+_+ \t\t cropping '+comment+' original width=',width,' origin=',origin,' newwidth=',sz)

    with tf.variable_scope('Crop_'+str(origin)+"_"+str(sz)):
        if not threed:
            return tf.slice(input,begin=[0,origin,origin,0],size=[-1,sz,sz,-1])

        return tf.slice(input,begin=[0,origin,origin,origin,0],size=[-1,sz,sz,sz,-1])


    
def cropImageCenter(input,threed=False,comment='image'):

    width=input.get_shape()[1].value
    numpixels=int(width/4)
    return cropImageBoundary(input,numpixels=numpixels,threed=threed,comment=comment)

def cropImageCenterWidth(input,output_width,threed=False,comment='image'):

    width=input.get_shape()[1].value
    numpixels=int((width-output_width)/2)
    return cropImageBoundary(input,numpixels=numpixels,threed=threed,comment=comment)


def smoothAndReduce(input,threed=False,comment='input',name='Reduced',reduce=2,sigma=1.0):


    bisutil.debug_print('+_+_+ \t  beginning smoothing ('+str(sigma)+') & shrinking x '+str(reduce)+' '+comment+', input=',input.get_shape())
    
    if (sigma<0.001 and reduce==1):
        bisutil.debug_print('+_+_+ \t\t smooth+shrink -> identity as (sigma='+str(sigma)+') & shrinking x='+str(reduce)+' '+comment)
        return tf.identity(input,name=name)

    if reduce!=2 and reduce!=4 and reduce!=8:
        reduce=1

    with tf.variable_scope('SmAndR'):

        num_frames=1

        radius=int(sigma)*2
        if radius<1:
            radius=1
        
        smooth_x = createGaussian(sigma=sigma,axis=0,threed=threed,radius=radius)
        smooth_y = createGaussian(sigma=sigma,axis=1,threed=threed,radius=radius)

        if (threed):
            smooth_z = createGaussian(sigma=sigma,axis=2,threed=threed,radius=radius)
            
        if not threed:
            srx = tf.nn.conv2d(input,smooth_x,strides=[1,reduce,1,1],
                                padding='SAME',name='reduce_x');
            output = tf.nn.conv2d(srx,smooth_y,strides=[1,1,reduce,1],
                                  padding='SAME',name='reduce_y');

        else:
            srx = tf.nn.conv3d(input,smooth_x,strides=[1,reduce,1,1],
                                  padding='SAME',name='reduce_x');
            sry = tf.nn.conv3d(srx,smooth_y,strides=[1,1,reduce,1,1],
                                  padding='SAME',name='reduce_y');

            output = tf.nn.conv3d(sry,smooth_z,strides=[1,1,1,reduce,1],
                               padding='SAME',name='reduce_z');

    if reduce==2:
        bisutil.debug_print('+_+_+ \t\t smoothing ('+str(sigma)+') & shrinking x '+str(reduce)+' '+comment+', input=',input.get_shape(),' output=',output.get_shape())
    else:
        bisutil.debug_print('+_+_+ \t\t smoothing (sigma='+str(sigma)+') '+comment+' input=',input.get_shape(),' output=',output.get_shape())

    return tf.identity(output,name=name)

def averageAndReduce(input,threed=False,comment='input',name='Reduced',reduce=2,sigma=1.0):


    bisutil.debug_print('+_+_+ \t  beginning averaging & shrinking x '+str(reduce)+' '+comment+', input=',input.get_shape())
    
    if (reduce==1):
        bisutil.debug_print('+_+_+ \t\t average+shrink -> identity as shrinking x='+str(reduce)+' '+comment)
        return tf.identity(input,name=name)

    if reduce!=2 and reduce!=4 and reduce!=8:
        reduce=1

    if threed:
        strides=[1,reduce,reduce,reduce,1]
        ksize=[1,reduce,reduce,reduce,1 ]
    else:
        strides=[1,reduce,reduce,1]
        ksize=[1,reduce,reduce,1 ]
        
    with tf.variable_scope('AvgAndR'):

        if threed:
            output = tf.nn.avg_pool3d(input, ksize=ksize, strides=strides, padding='SAME', name=name)
        else:
            output = tf.nn.avg_pool(input, ksize=ksize, strides=strides, padding='SAME', name=name)
            
    bisutil.debug_print('+_+_+ \t\t averaging & shrinking x '+str(reduce)+' '+comment+', input=',input.get_shape(),' output=',output.get_shape())


    return output


# --------------------------------------------------------------

def quarterImage(input,threed=False,comment='image'):

    
    width=input.get_shape()[1].value
    sz=int(width/2)

    bisutil.debug_print('+_+_+ \t\t quartering '+comment+' original width=',width,' half=',sz)

    boxes= [ [ 0,0,0 ], [ sz,0,0], [ 0, sz,0 ],[ sz,sz,0 ],
             [ 0,0,sz ], [ sz,0,sz], [ 0, sz,sz ],[ sz,sz,sz ] ]

    imglist=[]
    
    if threed:
        for i in range(0,8):
            imglist.append(tf.slice(input,begin=[0,boxes[i][0],boxes[i][1],boxes[i][2],0],size=[-1,sz,sz,sz,-1]))
    else:
        for i in range(0,4):
            imglist.append(tf.slice(input,begin=[0,boxes[i][0],boxes[i][1],0],size=[-1,sz,sz,-1]))
            
    out=tf.concat(imglist, axis=0)
    bisutil.debug_print('+_+_+ \t\t\t quartering '+comment+' done out=',out.get_shape)
    return out


def upSampleAndQuarter(input,threed=False,interp=1,comment='image'):

    inpshape=input.get_shape()
    if threed:
        outshape=[ 2*inpshape[1].value,2*inpshape[2].value,2*inpshape[3].value]
    else:
        outshape=[ 2*inpshape[1].value,2*inpshape[2].value ]
        
    bisutil.debug_print('+_+_+ \t\t upsample and quartered, inp=', inpshape)
        
    upsampled=upSample(input,newshape=outshape,threed=threed,interp=interp)

    bisutil.debug_print('+_+_+ \t\t\t upsampled, ', upsampled.get_shape())
    
    quartered=quarterImage(upsampled,threed,comment)

    bisutil.debug_print('+_+_+ \t\t\t quartered, ', quartered.get_shape())

    return quartered

# --------------------------------------------------------------
#
# create Multi Resolution Pyramid
#
# --------------------------------------------------------------
def fixInputData(input_data,num_classes=1):
    if num_classes > 1:
        sq=3
        if len(input_data.get_shape())==5:
            sq=4
            
        print("***** \t\t  converting to one-hot before smoothing or cropping")
        output_data=tf.squeeze(tf.one_hot(indices=tf.cast(input_data,tf.int32),
                                          depth=num_classes,
                                          on_value=1.0,
                                          off_value=0.0,
                                          axis=sq,
                                          dtype=tf.float32),squeeze_dims=[sq+1])
    else:
        output_data=tf.cast(input_data,tf.float32)

    return output_data
    
# ----------------------------------------------------------------------------------------------------
def createReducedSizeImage(input_data,
                           threed=False,
                           scope_name="Half_Size",
                           num_classes=1,
                           sigma=1.0,
                           max_outputs=3,
                           name="input",
                           reduce=2,
                           add_summary_images=True):

    print('***** \t creating reduced size '+name+' image: num_classes='+str(num_classes)+' sigma='+str(sigma)+' reduce factor='+str(reduce))
    
    with tf.variable_scope(scope_name) as scope:
    
        input_data=fixInputData(input_data,num_classes)
        smooth_fun=smoothAndReduce
        if num_classes>1:
            smooth_fun=averageAndReduce
            
        output=smooth_fun(input_data,threed,
                          comment=name,
                          sigma=sigma,
                          reduce=reduce)
        
        if num_classes > 1:
            dim=len(output.get_shape())-1
            output=tf.expand_dims(tf.argmax(output, dimension=dim, name='onehot_fix'),dim=dim)
                
        if add_summary_images:
            c_shape=output.get_shape()
            bisutil.image_summary(output,name+'_half',
                                  c_shape[3].value,1,threed,max_outputs=max_outputs)
                    
    return output
# ----------------------------------------------------------------------------------------------------


def createPureSmoothingPyramid(input_data,num_levels,num_classes,sigma,name,threed,add_summary_images=False,max_outputs=3):

    crop_images= [ input_data ]

    fun=smoothAndReduce
    if num_classes>1:
        fun=averageAndReduce
    
    for i in range(0,num_levels):
        if (i>0):
            crop_images.append(fun(crop_images[i-1],threed,
                                   comment=name+" level "+str(i),
                                   sigma=sigma,reduce=1))

        if add_summary_images and num_classes<2:
            c_shape=crop_images[i].get_shape()
            bisutil.image_summary(crop_images[i],'inp_crop_'+str(num_levels-i),
                                  c_shape[3].value,1,threed=threed,max_outputs=max_outputs)
            
    return crop_images

def createImagePyramid(input_data,
                       num_levels=3,
                       final_width=32,
                       threed=False,
                       just_smooth=False,
                       scope_name="Data_Prep",
                       num_classes=1,
                       sigma=1.0,
                       max_outputs=3,
                       name="input",
                       add_summary_images=True):



    print('*****\n***** Creating Image Pyramid, levels='+str(num_levels)+' just_smooth='+str(just_smooth)+' num_classes='+str(num_classes)+' width='+str(final_width))
    
    with tf.variable_scope("Data_Prep") as scope:
    
        input_data=fixInputData(input_data,num_classes)
                
        if just_smooth:
            crop_images= createPureSmoothingPyramid(input_data,
                                                    num_levels=num_levels,
                                                    name=name,
                                                    num_classes=num_classes,
                                                    sigma=sigma,
                                                    threed=threed,
                                                    add_summary_images=add_summary_images,
                                                    max_outputs=max_outputs)

        else:
            last_input= input_data
            crop_images = [ ]
            fun=smoothAndReduce
            if num_classes>1:
                fun=averageAndReduce
    


            for i in range(0,num_levels):

                bisutil.debug_print('*****\n***** Working on pyramid level '+str(i))
                
                if (i>0):
                    last_input=fun(last_input,threed,
                                   comment=name+" level "+str(i),
                                   sigma=sigma)
                    
                if (i<num_levels-1):
                    crop_images.append(cropImageCenterWidth(last_input,output_width=final_width,
                                                            threed=threed,comment=" reduced input image "+str(i)))
                else:
                    crop_images.append(last_input)
                    
                bisutil.debug_print('===== \t '+name+' crop['+str(i)+']=',crop_images[i].get_shape())

                if add_summary_images and num_classes<2:
                    c_shape=crop_images[i].get_shape()
                    bisutil.image_summary(crop_images[i],'inp_crop_'+str(num_levels-i),
                                          c_shape[3].value,1,threed,max_outputs=max_outputs)
                    
            last_input=None


        if num_classes > 1:
            print("***** \t  Closing pyramid by arg_max on one-hot input data")
            old_crop_images=crop_images
            crop_images= []
            dim=len(old_crop_images[0].get_shape())-1
            for i in range(0,len(old_crop_images)):
                crop_images.append(tf.expand_dims(tf.argmax(old_crop_images[i], dimension=dim, name='onehot'+str(i+1)),dim=dim))
                
                if add_summary_images:
                    c_shape=crop_images[i].get_shape()
                    bisutil.image_summary(crop_images[i],'inp_crop_'+str(num_levels-i),
                                          c_shape[3].value,1,threed,max_outputs=max_outputs)
                    
    # Images by level
    return crop_images


# --------------------------------------------------------------
def createImageQuarterPyramid(input_data,
                              num_levels=3,
                              final_width=32,
                              threed=False,
                              scope_name="Data_Prep",
                              num_classes=1,
                              sigma=1.0,
                              max_outputs=3,
                              name="input",
                              add_summary_images=True):



    print('*****\n***** Creating Image Pyramid, levels='+str(num_levels)+' num_classes='+str(num_classes)+' width='+str(final_width))
    
    with tf.variable_scope(scope_name) as scope:
    
        input_data=fixInputData(input_data,num_classes)
        last_input= input_data
        crop_images = [ ]
        smooth_fun=smoothAndReduce
        if num_classes>1:
            smooth_fun=averageAndReduce
            
        for i in range(0,num_levels):
            
            bisutil.debug_print('*****\n***** Working on pyramid level '+str(i)+' ',last_input)
            
            if (i>0):
                last_input=smooth_fun(last_input,threed,
                                      comment=name+" level "+str(i),
                                      sigma=sigma,
                                      reduce=2)
                
            if (i<num_levels-1):
                crop_images.append(quarterImage(last_input,threed=threed,comment=" quarter input image "+str(i)))
            else:
                crop_images.append(last_input)
                    
            bisutil.debug_print('===== \t '+name+' crop['+str(i)+']=',crop_images[i].get_shape())

            if add_summary_images and num_classes<2:
                c_shape=crop_images[i].get_shape()
                bisutil.image_summary(crop_images[i],'inp_crop_'+str(num_levels-i),
                                      c_shape[3].value,1,threed,max_outputs=max_outputs)
                
        last_input=None


        if num_classes > 1:
            print("***** \t  Closing pyramid by arg_max on one-hot input data")
            old_crop_images=crop_images
            crop_images= []
            dim=len(old_crop_images[0].get_shape())-1
            for i in range(0,len(old_crop_images)):
                crop_images.append(tf.expand_dims(tf.argmax(old_crop_images[i], dimension=dim, name='onehot'+str(i+1)),dim=dim))
                
                if add_summary_images:
                    c_shape=crop_images[i].get_shape()
                    bisutil.image_summary(crop_images[i],'inp_crop_'+str(num_levels-i),
                                          c_shape[3].value,1,threed,max_outputs=max_outputs)
                    
    # Images by level
    return crop_images


# --------------------------------------------------------------

if __name__ == '__main__':

    a=createGaussian(0.85,0,False,radius=1)
    print(a)
    

    
    
