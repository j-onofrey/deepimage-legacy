from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
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
        'dimensions' : [],
        'setupname' : "",
        'padded_data' : []
    }

    target_images = {
        'matrices': [], #list of nifti header of data per image
        'filenames': [], # list of filenames read
        'data'  : None, #actual data,
        'datashape' : [], # shape of data,
        'setupname' : "",
        'dimensions' : [],
        'padded_data' : []
    }

    first_patches=True
    num_frames=1
    threed=False
    textfileinput=False
    
    def __init__(self):
        return None
        
    def get_image_shape(self):
        return self.input_images['datashape']

    def get_target_shape(self):
        return self.target_images['datashape']

    def get_input_data(self):
        return self.input_images['data']

    def get_padded_input_data(self):
        return self.input_images['padded_data']

    def get_target_data(self):
        return self.target_images['data']

    
    def get_threed(self):
        return self.threed

    def get_num_frames(self):
        return self.num_frames



    def get_extension(self,fname):
        ext=os.path.splitext(fname)[1]
        base=os.path.splitext(fname)[0]
        if (ext==".gz"):
            ext=os.path.splitext(base)[1]+ext
        return ext
    
    def compute_input_patch_shape(self,patch_shape=None):

        data_shape = self.get_image_shape()
        output_patch_shape = data_shape[1:]
        output_patch_shape+=[1];
                
        if patch_shape is not None:
            for i in range(0,len(patch_shape)):
                if patch_shape[i] < 0:
                    output_patch_shape[i] = data_shape[i+1]
                else:
                    output_patch_shape[i] = bisutil.force_inrange(patch_shape[i],1,data_shape[i+1])

                    #        print('input patch_shape=',patch_shape,' --> data_shape=',data_shape,' output_patch_shape=',output_patch_shape);
                    
        return output_patch_shape;
    
    
    def compute_target_patch_shape(self,patch_shape=None):
        
        maxdim=3
        if self.threed==True:
            maxdim=4

        input_shape = self.input_images['datashape']
        target_shape = self.target_images['datashape']

        i_len = len(input_shape);
        t_len = len(target_shape);
        p_len = len(patch_shape);
        if (t_len<i_len):
            target_shape+=[1];
            t_len = len(target_shape);
        
        

        if i_len < maxdim or t_len <maxdim or p_len <maxdim or p_len < t_len:
            raise ValueError('Data dimensions are incompatible somehow ... inp=',input_shape,' targ=',target_shape,' patch=',patch_shape,' maxdim=',maxdim,i_len,t_len,p_len,' threed=',self.threed);

        o_patch_shape=patch_shape[:];
        for i in range(1,t_len):
            #            print('comparing i=',i,' patch_shape[i-1]=',o_patch_shape[i-1],' target=',target_shape[i]);
            o_patch_shape[i-1] = bisutil.force_inrange(o_patch_shape[i-1],1,target_shape[i])


        #print('shapes=',input_shape,target_shape,' patch=',patch_shape,', o_patch_shape=',o_patch_shape);
        #sys.exit(0);
        return o_patch_shape;


    def transpose(self,transpose=None):

        print('JAO: transpose='+str(transpose))
        print('JAO: data_shape='+str(self.input_images['data'][0].shape))
        undo_transpose = None
        if transpose is not None:
            # Raise error if tranpose value are not correct
            if len(transpose) is not len(self.input_images['data'][0].shape):
                raise ValueError('Tranpose input='+str(transpose)+' does not have the same length ('+str(len(transpose))+') as the input data ('+str(len(self.input_images['data'][0].shape))+')')

            self.transpose = transpose[:]
            undo_transpose = np.asarray(transpose[:])
            undo_transpose = list(undo_transpose[transpose])

            if self.input_images['data'] is not None:
                for i in range(0,len(self.input_images['data'])):
                    orig_shape = self.input_images['data'][i].shape
                    self.input_images['data'][i] = np.transpose(self.input_images['data'][i],transpose)
                    print('-+-+- Transpose input data shape[',i,']: ',str(orig_shape),' to ',str(self.input_images['data'][i].shape))

                data_shape = self.input_images['datashape'][:]
                for i in range(0,len(transpose)):
                    data_shape[i+1] = self.input_images['datashape'][transpose[i]+1]
                self.input_images['datashape'] = data_shape

            if self.target_images['data'] is not None:
                for i in range(0,len(self.target_images['data'])):
                    orig_shape = self.target_images['data'][i].shape
                    self.target_images['data'][i] = np.transpose(self.target_images['data'][i],transpose)
                    print('-+-+- Transpose target data shape[',i,']: ',str(orig_shape),' to ',str(self.target_images['data'][i].shape))

                data_shape = self.target_images['datashape'][:]
                for i in range(0,len(transpose)):
                    data_shape[i+1] = self.target_images['datashape'][transpose[i]+1]
                self.target_images['datashape'] = data_shape

        return undo_transpose

    
    def load_images_fromlist(self,filenames,setupname,setparams=False):

        dict = {
            'matrices' : [],
            'filenames' : [],
            'data'  : [],
            'datashape' : None,
            'dimensions' : [],

        }

        firsttime=True

        for fname in filenames:
            
            ext=self.get_extension(fname).lower()
            
                    
            mode=""
            reader = tf.WholeFileReader()
            
            extlist = [ ".nii.gz",".nii",".jpg",".jpeg",".png" ]
            if ext in extlist:
                if (ext==".nii" or ext==".nii.gz"):
                    tmp = nib.load(fname)
                    imagedata = tmp.get_data();
                    imagematrix= tmp.affine
                    mode="nii"
                    if firsttime==True and setparams==True:
                        self.threed=False
                        if (len(imagedata.shape)>2):
                            if (imagedata.shape[2]>1):
                                self.threed=True
                        firsttime=False
                else:
                    imagedata=np.asarray(PILImage.open(fname))
                    mode=ext[1:]
                    imagematrix=np.eye(4)
                    if firsttime==True and setparams==True:
                        self.Threed=False
                        self.num_frames=imagedata.shape[2];
                        firsttime=False
                        
            else:
                raise ValueError('Unknown filetype for file: ' + fname+' (extension=\''+ext+'\')')
                
            # if len(imagedata.shape) < 3:
            #     imagedata = np.expand_dims(imagedata, 2)

            shp=None
            if (len(imagedata.shape)>3):
                # Are we 3D+t
                if (imagedata.shape[2]>1):
                    dat=np.transpose(imagedata,(3,0,1,2))
                    print('-+-+- \t\t this is a 4D image (',imagedata.shape,') transposing --> (',dat.shape,')')
                    shp=dat.shape
                    nf=imagedata.shape[3]
                    for i in range(0,nf):
                        dict['data'].append(dat[i])   
                        iname="%04d" % ( (i+1))
                        fname2=fname[0:len(fname)-len(ext)]+'_'+iname+ext
                        dict['filenames'].append(fname2)
                        dict['matrices'].append(imagematrix)
                        dict['dimensions'].append(dat[i].shape)
                        bisutil.extra_debug_print('-+-+- \t\t loaded 4d %s -->%s image: ' % (fname,fname2),' ',imagedata.shape,' -->',dat.shape)
                else:
                    dat=np.transpose(imagedata,(0,1,3,2))
                    dat=np.squeeze(dat,axis=3)
                    dict['data'].append(dat)
                    dict['filenames'].append(fname)
                    dict['matrices'].append(imagematrix)
                    dict['dimensions'].append(dat.shape)
                    if len(dict['data'])<2:
                        self.num_frames=dat.shape[2]
                        print('+++++ \t\t Need to transpose as we are loading 2d+t .nii images',imagedata.shape,'--->',dat.shape);
                        self.threed=False;
                        shp=dat.shape;
            
            else:
                dict['data'].append(imagedata)
                dict['filenames'].append(fname)
                dict['matrices'].append(imagematrix)
                dict['dimensions'].append(imagedata.shape)

            bisutil.extra_debug_print('-+-+- \t\t loaded %s image: %s ' % (mode,fname),imagedata.shape)


        if (shp!=None):
            dict['datashape']=shp
        else:
            dict['datashape']=[ len(dict['data']) ]

            for i in xrange(len(dict['data'][0].shape)):
                dict['datashape'].append(dict['data'][0].shape[i])

        bisutil.debug_print('-+-+- \t\t dimension set=',dict['dimensions'])
                
        # Max dimensions here
        # I.e. make sure output
        for i in range(0,len(dict['dimensions'])):
            for j in range (0,len(dict['dimensions'][0])):
                if (dict['datashape'][j+1]<dict['dimensions'][i][j]):
                    dict['datashape'][j+1]=dict['dimensions'][i][j]
        

                
        print('-+-+- \t\t loaded %d image samples.' % ( len(dict['data'])),' Shape='+str(dict['datashape']), end='')
        if (setparams==True):
            print(', threed='+str(self.threed))
        else:
            print('')
        print('-+-+-')
        return dict



    def load_image_data_from_setup(self,setupname,comment='input',setparams=False):


        newfilenames = []
        pathname = os.path.abspath(os.path.dirname(setupname))
        print("-+-+- \t loading "+comment+" data (pathname=",setupname,")")

        
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
            self.textfileinput=True
            
        dict=self.load_images_fromlist(newfilenames,setupname,setparams)
        dict['setupname']=os.path.abspath(setupname)
        return dict

    # ---------------------------------------------------------------------
    # Utility Code
    #
    # Compare Sizes
    # --------------------------------
    def ensure_same_size(self,data_list,names,fnames,mode):

        master_dim=data_list[0]
        nf=len(master_dim)
        ndim=len(master_dim[0])
        bisutil.extra_debug_print('+++++ Ensuring same size for ', len(data_list),' image sets. Num frames of ',names[0],' = ',nf,' dimensions of frame 1=',master_dim[0],', ndim=',ndim)
        
        for inpindex in range(1,len(data_list)):

            target_dim=data_list[inpindex]
            nt=len(target_dim)
            bisutil.extra_debug_print('+++++ comparing ', names[0],' and ',names[inpindex], ' nf=',nf,' ',nt)


            if (mode[inpindex]==True and nf!=nt):
                raise ValueError('Bad number of images in %s (%s) (input has %d, target has %d)' %( names[inpindex],fnames[inpindex], nf, nt ))

            if (mode[inpindex]==False):
                if (nf!=nt and nt!=1):
                    raise ValueError('Bad number of images in %s (%s) (input has %d, target has %d which is neither %d or 1)' %( names[inpindex],fnames[inpindex], nf, nt, nf ))
            
                
            if (mode[inpindex]==True or nf==nt):
                for fr in range(0,nf):
                    sum=0
                    bisutil.extra_debug_print('+++++ \t comparing frame ', fr+1,' and ',fr+1,' dimensions=',master_dim[fr],' ',target_dim[fr])
                    for d in range(0,ndim):
                        sum+=math.fabs(master_dim[fr][d]-target_dim[fr][d])
                    if (sum>0):
                        raise ValueError('Not matching dimensions in %s, %s dimensions=' %(fnames[0],fnames[inpindex]),master_dim[fr],target_dim[0])
            else:
                for fr in range(1,nf):
                    sum=0
                    bisutil.extra_debug_print('+++++ \t comparing frame', fr+1,' and ',0,' dimensions=',master_dim[fr],' ',target_dim[0])
                    for d in range(0,ndim):
                        sum+=math.fabs(master_dim[fr][d]-target_dim[0][d])
                    if (sum>0):
                        raise ValueError('Not matching dimensions in %s, %s dimensions=' %(fnames[0],fnames[inpindex]),master_dim[fr],target_dim[0])

    def combine_images(self,first,second,name):

        if second==None:
            return
        
        # create 4d frames in input_data and modify shape

        axis=3;
        axisplus=4
        if (self.threed==False):
            axis=2
            axisplus=3

        nf=first['datashape'][0]
        nf2=second['datashape'][0]

        dat=[]
        print('+++++ Stacking ',name,' max =',first['datashape'],' and ',second['datashape'])
        for i in range(0,nf):
            bisutil.extra_debug_print('+++++ \t stacking frame ',i+1,' shapes=',first['data'][i].shape,' ',second['data'][i].shape)
                
            j=0
            if (nf2==nf):
                j=i
                

            tmp=np.concatenate( (np.expand_dims(first['data'][i],axisplus),
                                 np.expand_dims(second['data'][j],axisplus)),axis=axis)

            if (self.threed==False):
                tmp=np.squeeze(tmp,axis=3)

            
            dat.append(tmp)

        first['data']=dat 
        first['datashape'].append(2)
        bisutil.extra_debug_print('+++++')
                    
        
    # ---------------------------------------------------------------------
    #
    #  Load Data
    #
    def load(self,image_filename,target_filename="",secondinput_filename="",secondtarget_filename=""):


        lst=[]
        names=[]
        fnames=[]
        mode=[]
        self.textfileinput=False
        
        self.input_images=self.load_image_data_from_setup(image_filename,comment='input',setparams=True)
        lst.append(self.input_images['dimensions'])
        names.append('input')
        fnames.append(image_filename)
        mode.append(True)
        
        if (target_filename!=""):
            self.target_images=self.load_image_data_from_setup(target_filename,comment='target')
            lst.append(self.target_images['dimensions'])
            names.append('target')
            fnames.append(target_filename);
            mode.append(True)
        if (secondinput_filename!=""):
            secondinput_images=self.load_image_data_from_setup(secondinput_filename,comment='secondinput')
            lst.append(secondinput_images['dimensions'])
            names.append('secondinput')
            fnames.append(secondinput_filename);
            mode.append(False)
        else:
            secondinput_images=None

        if (secondtarget_filename!=""):
            secondtarget_images=self.load_image_data_from_setup(secondtarget_filename,comment='secondtarget')
            lst.append(secondtarget_images['dimensions'])
            names.append('secondtarget')
            fnames.append(secondtarget_filename);
            mode.append(False)
        else:
            secondtarget_images=None
            
        # self.ensure_same_size(lst,names,fnames,mode)

        if secondinput_images!=None:
            self.combine_images(self.input_images,secondinput_images,name="second input")
            self.num_frames=2
            print('+++++\t combined input shape (with second input as frame 2)=',self.get_image_shape())
            print('+++++')
            
        if secondtarget_images!=None:
            self.combine_images(self.target_images,secondtarget_images,name="second target")
            self.num_frames=2
            print('+++++\t combined target shape (with second target as frame 2)=',self.get_image_shape())
            print('+++++')

        print('+++++ Data Loaded, text=',    self.textfileinput,' numframes=',self.num_frames)

        return True
        

    def save_reconstructed_image_data(self,data,path=""):


            
        numimages=len(data)
        if numimages!= len(self.input_images['filenames']):
            raise ValueError(' Unqueal number of images in save_reconstructed from read')

        print('-+-+- S a v i n g ',numimages, '  i m a g e (s)  in:',path)
        print('-+-+-')
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
            for i in range(0,numimages):
                fname=os.path.basename(innames[i])
                f=path+"_recon_"+fname
                outnames.append(f)

        ext=self.get_extension(outnames[0]).lower()
        if (ext==".nii" or ext==".nii.gz" or self.threed==True or ext==""):
            isnifti=True
        else:
            isnifti=False
            
        if (numimages == 1 or isnifti==False or  self.textfileinput==True):

            for i in range(0,numimages):
                if isnifti==True or  self.threed==True:
                    ext=self.get_extension(outnames[i]).lower()
                    if (ext!=".nii.gz" and ext!=".nii"):
                        outnames[i]=outnames[i]+".nii.gz"

                    out_image = nib.Nifti1Image(data[i], self.input_images['matrices'][i])
                    print('-+-+- \t writing NII image in ',outnames[i],' dim=',out_image.get_data().shape)
                    nib.save(out_image, outnames[i])


                else:

                    shp=data[i].shape;
                    iscolor=False;
                    if (len(shp)==3):
                        if shp[2]==3:
                            iscolor=True;
                        
                    
                    min_dt=np.amin(data[i])
                    max_dt=np.amax(data[i])
                        
                    if (max_dt<2):
                        scalefactor=255
                    elif (max_dt<3):
                        scalefactor=100
                    elif (max_dt<6):
                        scalefactor=50
                    elif (max_dt<11):
                        scalefactor=25
                    else:
                        scalefactor=1

                    if not iscolor:
                        convimg=(scalefactor*np.squeeze(data[i],axis=2)+0.5).astype(np.uint8)
                        cmin_dt=np.amin(convimg)
                        cmax_dt=np.amax(convimg)
                        im = PILImage.fromarray(convimg)
                        
                    else:
                        convimg=(scalefactor*data[i]).astype(np.uint8)

                    print('-+-+- \t writing image (color='+str(iscolor)+') in '+outnames[i]+' sc='+str(scalefactor),' dim=',convimg.shape)
                    try:
                        im.save(outnames[i])
                    except:
                        raise ValueError('Failed to save file in '+outnames[i])

        else:
            f=path

            ext=self.get_extension(outnames[0]).lower()
            print('-+-+-+ extension=',ext)
            if (ext!=".gz" and ext!=".nii"):
                f=f+".nii.gz"


            if (self.threed==True):
                print('-+-+-+ \t creating 4d nifti image')
                s=data[0].shape
                newdata=np.zeros((s[0],s[1],s[2],numimages),dtype=np.float32)
                for i in range(0,numimages):
                    #print('old shape =', data[i].shape,' --->' , newdata.shape)
                    newdata[0:s[0],0:s[1],0:s[2],i:i+1]=data[i][0:s[0],0:s[1],0:s[2],0:1 ]
            else:
                print('-+-+-+ \t creating 2d+t nifti image')
                s=data[0].shape
                newdata=np.zeros((s[0],s[1],numimages,s[2]),dtype=np.float32)
                for i in range(0,numimages):
                    newdata[0:s[0],0:s[1],i:i+1,0]=data[i][0:s[0],0:s[1],0:1]
                newdata=np.transpose(newdata,(0,1,3,2))

            
                
            print('\n-+-+-+ \t Final output shape=',newdata.shape)
            out_image = nib.Nifti1Image(newdata, self.input_images['matrices'][0])
            print('-+-+- \t writing NII image in ',f,' dim=',out_image.get_data().shape)
            nib.save(out_image, f);
            
                

    # -----------------------------------------------------------------------------------
    #
    # Pad training data 
    #
    # -----------------------------------------------------------------------------------
    def internal_pad(self,images,pad_size,pad_type='zero'):

        data = images['data']
        
        num_images = len(data)
        images['padded_data']=[]

        half = pad_size[:]
        for i in range(0,len(pad_size)):
            half[i] = int(pad_size[i]/2)
        bisutil.debug_print('-+-+- Padding by ',pad_size)

        # Set the numpy pad type string
        np_pad_type = 'constant'
        if pad_type == 'reflect':
            np_pad_type = 'reflect'
        elif pad_type == 'edge':
            np_pad_type = 'edge'

        for i in range(0,num_images):
            # imgshape = data[i].shape
            # if imgshape[2] > 1:
            #     if len(imgshape)>3:
            #         newdata = np.pad(data[i],((half[0],half[0]),(half[1],half[1]),(half[2],half[2]),(0,0)),np_pad_type)
            #     else:
            #         newdata = np.pad(data[i],((half[0],half[0]),(half[1],half[1]),(half[2],half[2])),np_pad_type)
            # else:
            #     newdata = np.pad(data[i],((half[0],half[0]),(half[1],half[1]),(0,0)),np_pad_type)

            dims = len(data[i].shape)
            padding = ()
            for j in range(0,dims):
                padding += ((half[j],half[j]),)
            newdata = np.pad(data[i], padding, np_pad_type)

            bisutil.debug_print('-+-+- padding ',i,' from ',data[i].shape,' to ',newdata.shape)
            images['padded_data'].append(newdata)

        return half


    def pad(self,pad_size=None,pad_type='zero'):

        dims = len(self.get_image_shape())-1
        actual_pad_size = []
        for i in range(0,dims):
            actual_pad_size += [0]
        if pad_size is not None:
            for i in range(0,len(pad_size)):
                actual_pad_size[i] = bisutil.force_inrange(pad_size[i],0,512)

        h = self.internal_pad(self.input_images,pad_size=actual_pad_size,pad_type=pad_type)
        if self.target_images['data'] != None:
            self.internal_pad(self.target_images,pad_size=actual_pad_size,pad_type=pad_type)
        return h
    
    # -----------------------------------------------------------------------------------
    #
    # Batch Code 3dt,3D and 2Dt, 2D
    #
    # -----------------------------------------------------------------------------------
    def get_batch_of_patches_3d_plust(self, image_training_data, target_training_data, batch_size, augmentation=False,patch_size=None,dtype=np.float32):

        # Get the random sample indexing
        num_samples = len(image_training_data)
        sample_idx = (((num_samples-1)*np.random.rand(batch_size))+0.5).astype(int)
        # print('Sampling index: %s' % (sample_idx,))

        num_frames=image_training_data[0].shape[3]
        images = np.zeros((batch_size,patch_size[0], patch_size[1], patch_size[2], num_frames), dtype=dtype)
        targets = np.zeros((batch_size,patch_size[0], patch_size[1], patch_size[2], 1), dtype=dtype)

        if (self.first_patches==True):
            print('+++++ \t\t In 3D+t Patches, numframes=',num_frames,' images_shape',images.shape,' target_shape=',targets.shape)
            self.first_patches=False
            
        
        flip_count = 0
        for i in range(0,batch_size):
            # Only get a single patch at a time
            # [p1, idx] = image_training_data[sample_idx[i]].getPatches(1)
            [p1, idx] = putil.getRandomPatches(image_training_data[sample_idx[i]], patch_size, num_patches=1, dtype=dtype)
            # [p2, idx] = target_training_data[sample_idx[i]].getPatchesFromIndexes(idx)
            p2 = putil.getPatchesFromIndexes(target_training_data[sample_idx[i]], idx, patch_size, dtype=dtype)
        
            # print('Get patch index: %s' % (idx,))
            images[i,:,:,:,:] = p1
            targets[i,:,:,:,0] = p2
            if augmentation:
                # 50% chance of flipping in a random dimension
                if np.random.random() > 0.5:
                    # flip_dim = np.random.randint(0,3)
                    # Fix this for Left-Right flipping (assumes rigid registration to an atlas/template)
                    flip_dim = 0
                    images[i,:,:,:,:] = np.flip(images[i,:,:,:,:],flip_dim)
                    targets[i,:,:,:,0] = np.flip(targets[i,:,:,:,0],flip_dim)
                    flip_count += 1
                
        return [images, targets]

    def get_batch_of_patches_3d(self, image_training_data, target_training_data, batch_size, augmentation=False,patch_size=None,dtype=np.float32):

        # Get the random sample indexing
        num_samples = len(image_training_data)
        sample_idx = (((num_samples-1)*np.random.rand(batch_size))+0.5).astype(int)
        # print('Sampling index: %s' % (sample_idx,))

        images = np.zeros((batch_size,patch_size[0], patch_size[1], patch_size[2], 1), dtype=dtype)
        targets = np.zeros((batch_size,patch_size[0], patch_size[1], patch_size[2], 1), dtype=dtype)

        if (self.first_patches==True):
            print('+++++ \t\t In 3D Patches images_shape',images.shape,' target_shape=',targets.shape)
            self.first_patches=False

        
        flip_count = 0
        for i in range(0,batch_size):
            # Only get a single patch at a time
            # [p1, idx] = image_training_data[sample_idx[i]].getPatches(1)
            [p1, idx] = putil.getRandomPatches(image_training_data[sample_idx[i]], patch_size, num_patches=1, dtype=dtype)
            # [p2, idx] = target_training_data[sample_idx[i]].getPatchesFromIndexes(idx)
            p2 = putil.getPatchesFromIndexes(target_training_data[sample_idx[i]], idx, patch_size, dtype=dtype)
        
            # print('Get patch index: %s' % (idx,))
            images[i,:,:,:,0] = p1
            targets[i,:,:,:,0] = p2
            if augmentation:
                # 50% chance of flipping in a random dimension
                if np.random.random() > 0.5:
                    # flip_dim = np.random.randint(0,3)
                    # Fix this for Left-Right flipping (assumes rigid registration to an atlas/template)
                    flip_dim = 0
                    images[i,:,:,:,0] = np.flip(images[i,:,:,:,0],flip_dim)
                    targets[i,:,:,:,0] = np.flip(targets[i,:,:,:,0],flip_dim)
                    flip_count += 1
                
        return [images, targets]


    def get_batch_of_patches_2d_plust(self, image_training_data, target_training_data, batch_size, augmentation=False,patch_size=None,dtype=np.float32):

        # Get the random sample indexing
        num_samples = len(image_training_data)
        sample_idx = (((num_samples-1)*np.random.rand(batch_size))+0.5).astype(int)
        # print('Sampling index: %s' % (sample_idx,))

        # num_frames=image_training_data[0].shape[2]
        # num_frames = patch_size[2]
        targ_size = patch_size[:]

        # Assumption: if we have patches that are the same size as the data, just use the data sizes
        if bisutil.is_shape_same(image_training_data[0].shape,patch_size):
            targ_size = target_training_data[0].shape

        images = np.zeros((batch_size, patch_size[0], patch_size[1], patch_size[2]), dtype=dtype)
        if len(targ_size)>=3:
            targets = np.zeros((batch_size, targ_size[0], targ_size[1], targ_size[2]), dtype=dtype)
        else:
            # Single channel target
            targets = np.zeros((batch_size, targ_size[0], targ_size[1],1), dtype=dtype)


        if (self.first_patches==True):
            print('+++++ \t\t In 2D(+t) Patches images_shape',images.shape,' target_shape=',targets.shape)
            self.first_patches=False

        flip_count = 0
        for i in range(0,batch_size):
            # Only get a single patch at a time
            # [p1, idx] = image_training_data[sample_idx[i]].getPatches(1, dtype=dtype)
            [p1, idx] = putil.getRandomPatches(image_training_data[sample_idx[i]], patch_size, num_patches=1, dtype=dtype)
            # [p2, idx]< = target_training_data[sample_idx[i]].getPatchesFromIndexes(idx, dtype=dtype)
            p2 = putil.getPatchesFromIndexes(target_training_data[sample_idx[i]], idx, targ_size, dtype=dtype)
            # print('Get patch index: %s' % (idx,))

            if len(p1.shape) < 4:
                p1 = np.expand_dims(p1, axis=3)
            if len(p2.shape) < 4:
                p2 = np.expand_dims(p2, axis=3)

            images[i,...] = p1[0,...] #np.squeeze(p1, 3)
            targets[i,...] = p2[0,...] #np.squeeze(p2, 3)
            if augmentation:
                # 50% chance of flipping in a random dimension
                if np.random.random() > 0.5:
                    # flip_dim = np.random.randint(0,2)
                    flip_dim = 0
                    # TODO: this can probably be made to have ellipses syntax
                    images[i,:,:,:] = np.flip(images[i,:,:,:],flip_dim)
                    targets[i,:,:,0] = np.flip(targets[i,:,:,0],flip_dim)
                    flip_count += 1


        return [images, targets]
    

    def get_batch_of_patches(self, batch_size, augmentation=False,patch_size=None,dtype=np.float32,patch_threed=False):

        # print('JAO: patch_size: '+str(patch_size))

        using_padded=False
        image_training_data = self.input_images['data']
        target_training_data = self.target_images['data']
            
        if (len(self.input_images['padded_data'])>0):
            using_padded=True
            image_training_data=self.input_images['padded_data']
            target_training_data=self.target_images['padded_data']


        if len(image_training_data) != len(target_training_data):
            raise ValueError('Image and target data counts do not match')

        if self.first_patches:
            print('+++++ \t Getting Patches (padded='+str(using_padded)+'): input_shape=',image_training_data[0].shape,target_training_data[0].shape,' dtype=',dtype)

        if (self.num_frames>1 and patch_threed==True):
            return self.get_batch_of_patches_3d_plust(image_training_data, target_training_data, batch_size, augmentation, patch_size, dtype=dtype)
            
        if (patch_threed==True):
            return self.get_batch_of_patches_3d(image_training_data, target_training_data, batch_size, augmentation, patch_size, dtype=dtype)
            
        return self.get_batch_of_patches_2d_plust(image_training_data, target_training_data, batch_size, augmentation, patch_size, dtype=dtype)


