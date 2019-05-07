from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import numpy as np
import nibabel as nib
from six.moves import xrange

import tensorflow as tf
import bis_tf_utils as bisutil
import bis_image_patchutil as putil
from PIL import Image as PILImage



class ImageSet():

    input_images = {
        'matrices': [], #list of nifti header of data per image
        'filenames': [], # list of filenames read
	'data'  : None, #actual data,
        'datashape' : [], # shape of data,
        'setupname' : ""
    }

    target_images = {
        'matrices': [], #list of nifti header of data per image
        'filenames': [], # list of filenames read
	'data'  : None, #actual data,
        'datashape' : [], # shape of data,
        'setupname' : ""
    }


    
    def __init__(self):
        a=0
        
    def get_image_shape(self):
        return self.input_images['datashape']

    def get_input_data(self):
        return self.input_images['data']

    def get_extension(self,fname):
        ext=os.path.splitext(fname)[1]
        base=os.path.splitext(fname)[0]
        if (ext==".gz"):
            ext=os.path.splitext(base)[1]+ext
        return ext
    
    def load_images_fromlist(self,filenames,setupname):

        dict = {
            'matrices' : [],
            'filenames' : [],
            'data'  : [],
            'datashape' : None,

        }

        for fname in filenames:
            
            ext=self.get_extension(fname).lower()
            
                    
            mode=""
            reader = tf.WholeFileReader()
            
            extlist = [ ".nii.gz",".nii",".jpg",".jpeg",".png" ]
            if ext in extlist:
                if (ext==".nii" or ext==".nii.gz"):
                    tmp = nib.load(fname)
                    imagedata = tmp.get_data()
                    imagematrix= tmp.affine
                    mode="nii"
                else:
                    imagedata=np.asarray(PILImage.open(fname))
                    mode=ext[1:]
                    imagematrix=np.eye(4)
            else:
                raise ValueError('Unknown filetype for file: ' + fname+' (extension=\''+ext+'\')')
                
            if len(imagedata.shape) < 3:
                imagedata = np.expand_dims(imagedata, 2)


            dict['data'].append(imagedata)
            dict['filenames'].append(fname)
            dict['matrices'].append(imagematrix)
            bisutil.extra_debug_print('-+-+- \t Loaded %s image: %s ' % (mode,fname),imagedata.shape)

        dict['datashape']=[ len(dict['data']) ]
        for i in xrange(len(imagedata.shape)):
            dict['datashape'].append(imagedata.shape[i])
                
        print('-+-+- Loaded %d image samples from %s' % ( len(dict['data']),setupname), dict['datashape'])
        print('-+-+-')
        return dict



    def load_image_data_from_setup(self,setupname):


        newfilenames = []
        pathname = os.path.abspath(os.path.dirname(setupname))
        print("-+-+- Loading image data (pathname=",setupname,")")

        if (os.path.splitext(setupname)[1]!=".txt"):
            newfilenames.append(setupname)
        else:
            try: 
                image_input_file = open(setupname)
            except IOError as e:
                raise ValueError('Bad setup file '+setupname+'\n\t ('+str(e)+')')
            
            filenames = image_input_file.readlines()
            
            

            for f in filenames:
                fname = f.rstrip()

                if (len(fname)>0):
                    if not os.path.isabs(fname):
                        fname=os.path.abspath(os.path.join(pathname,fname))
                    
                    if not os.path.isfile(fname):
                        raise ValueError('Failed to find file: ' + fname)
                    
                    newfilenames.append(fname)

        dict=self.load_images_fromlist(newfilenames,setupname)
        dict['setupname']=os.path.abspath(setupname)
        return dict

    #
    #  Load Data
    #
    def load(self,image_filename,target_filename=""):

        input_images= {}
        target_images={}
        
        self.input_images=self.load_image_data_from_setup(image_filename)
        if (target_filename!=""):
            self.target_images=self.load_image_data_from_setup(target_filename)
            if (self.input_images['datashape']!=self.target_images['datashape']):
                raise ValueError('Not matching dimensions in %s, %s ' %(image_filename,target_filename))


        return True
        
    def get_batch(self,
                  batch_size=16,
                  augmentation=True,
                  patch_size=32):

        depth=self.input_images['datashape'][3]

        training_data=[ self.input_images['data'],self.target_images['data'] ]

        
        
        batch = putil.get_batch(training_data,
                                batch_size=batch_size,
                                augmentation=True,
                                image_size=patch_size)
        return batch


    def save_reconstructed_image_data(self,data,path=""):

        numimages=len(data)
        if numimages!= len(self.input_images['filenames']):
            raise ValueError(' Unqueal number of images in save_reconstructed from read')

        print('-+-+- saving ',numimages, ' image(s) in:',path)
        innames = self.input_images['filenames']
        outnames = []
        if os.path.isdir(path):
            for i in range(0,numimages):
                fname=os.path.basename(innames[i])
                f=os.path.join(path,"recon_"+fname)
                outnames.append(f)
        elif numimages==1:
            outnames.append(path)
        else:
            basename=path[0:ext-1]
            for i in range(0,numimages):
                fname=os.path.basename(innames[i])
                f=basename+"recon_"+fname
                outnames.append(f)

        for i in range(0,numimages):
            ext=self.get_extension(outnames[i]).lower()
            if (ext==".nii" or ext==".nii.gz"):
                out_image = nib.Nifti1Image(data[i], self.input_images['matrices'][i])
                print('-+-+- Saving NII output image in ',outnames[i],' dim=',out_image.get_data().shape)
                nib.save(out_image, outnames[i])
            else:
                im = PILImage.fromarray(np.squeeze(data[i],axis=2).astype(np.uint8))
                print('-+-+- Saving output image in ',outnames[i])
                try:
                    im.save(outnames[i])
                except:
                    raise ValueError('Failed to save file in '+outnames[i])
                


        
