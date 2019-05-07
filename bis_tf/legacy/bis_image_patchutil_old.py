from __future__ import print_function

import argparse
import os.path
import nibabel as nib
import numpy as np
import math

from six.moves import xrange

"""
  getRandomPatchIndexs -- indices for random patches
  getOrderedPatchIndexes  -- indices for a regularly sampled grid
  getPatchesFromIndexes -- getPatchesFromIndexes
  getRandomPatches -- (RandomPatchIndexes -> PatchesFromIndexes)
  getOrderedPatches -- (OrderedPatchIndexes -> PatchesFromIndexes)
  imagePatchRecon -- Recon Image given patches and indices

  getbatch3d -- get batch of 3d patches
  getbatch   -- get batch of images -- if 3D call getbatch3d

"""



# -------------------------------------------------------------------------------
#
# Core Patch Code
#
# -------------------------------------------------------------------------------
# Get Indices
# -------------------------------------------------------------------------------

def getRandomPatchIndexes(image, patch_size, num_patches=1, padding='VALID'):
   """Get image patch samples from a regularly sampled grid

   Create the
   
   Returns:
      indexes: the index of the image denoting the top-left corner of the patch
   """

   indexes = np.zeros((num_patches,3), dtype=np.int32)

   image_limits = (image.shape[0], 
                   image.shape[1],
                   image.shape[2])

   if padding is 'VALID':
      image_limits = (image.shape[0] - patch_size[0], 
                      image.shape[1] - patch_size[1],
                      image.shape[2] - patch_size[2])

   for i in range(0,num_patches):

      p_idx = (np.random.random_integers(0, image_limits[0]),
               np.random.random_integers(0, image_limits[1]), 
               np.random.random_integers(0, image_limits[2]))
      indexes[i,:] = p_idx

   return indexes




def getOrderedPatchIndexes(image, patch_size, stride=[1,1,1], padding='VALID'):
   """Get image patch samples from a regularly sampled grid

   Create the
   
   Returns:
      indexes: the index of the image denoting the top-left corner of the patch
   """


   # Set the index sampling ranges
   i_idx = np.arange(1, image.shape[0], stride[0])
   j_idx = np.arange(1, image.shape[1], stride[1])
   k_idx = np.arange(1, image.shape[2], stride[2])

   if padding is 'VALID':
      image_limits = (image.shape[0] - patch_size[0], 
                      image.shape[1] - patch_size[1],
                      image.shape[2] - patch_size[2])
      i_idx = i_idx[i_idx <= image_limits[0]]
      j_idx = j_idx[j_idx <= image_limits[1]]
      k_idx = k_idx[k_idx <= image_limits[2]]

   if i_idx.size < 1:
      i_idx = np.array([1])
   if j_idx.size < 1:
      j_idx = np.array([1])
   if k_idx.size < 1:
      k_idx = np.array([1])

   total_patches = i_idx.size*j_idx.size*k_idx.size
   indexes = np.zeros((total_patches,3), dtype=np.int32)
   n = 0
   for k in range(0, k_idx.size):
      for j in range(0, j_idx.size):
         for i in range(0, i_idx.size):
            indexes[n,:] = [i_idx[i], j_idx[j], k_idx[k]]
            n += 1 

   # Make sure to use 0 indexing 
   indexes -= 1

   return indexes


# -------------------------------------------------------------------------------
# Get Patches
# -------------------------------------------------------------------------------


def getPatchesFromIndexes(image, indexes, patch_size, padding='VALID', dtype=None):
   """Get image patches from specific positions in the image.

   Returns:
      patches: the image patches as a 4D numpy array
      indexes: the indexes of the image denoting the top-left corner of the patch in the image
               (just pass through really)
   """

   if not(dtype):
      dtype = image.dtype

   assert indexes.shape[1] == 3

   num_patches = indexes.shape[0]
   if (len(image.shape)>3 and patch_size[2]>1):
      numframes=image.shape[3]
      patches = np.zeros((num_patches, patch_size[0], patch_size[1], patch_size[2],numframes), dtype=image.dtype)

      # Need to pad the image if padding is 'SAME'
      if padding is 'SAME':
         # This will be an issue in 3D
         print('\n\n\n\n\n\n PADDING ......')
         image = np.lib.pad(image,((0,patch_size[0]-1), (0,patch_size[1]-1), (0,patch_size[2]-1),(0,numframes-2)), 'edge') 
         
      image_limits = (image.shape[0] - patch_size[0], 
                      image.shape[1] - patch_size[1],
                      image.shape[2] - patch_size[2])

      for i in range(0,num_patches):
         p_idx = indexes[i,:]
         if p_idx[0] <= image_limits[0] and p_idx[1] <= image_limits[1] and p_idx[2] <= image_limits[2]:
            patches[i,:,:,:,:] = image[p_idx[0]:p_idx[0] + patch_size[0], 
                                       p_idx[1]:p_idx[1] + patch_size[1], 
                                       p_idx[2]:p_idx[2] + patch_size[2],:]
            
   elif (patch_size[2]==1 and image.shape[2]>1):

      numframes=image.shape[2]      
      #      print('2d+t ... ',image.shape,'numf=',numframes)

      patches = np.zeros((num_patches, patch_size[0], patch_size[1], numframes), dtype=image.dtype)

      # Need to pad the image if padding is 'SAME'
      if padding is 'SAME':
         print('\n\n\n\n\n\n PADDING ......')
         image = np.lib.pad(image,((0,patch_size[0]-1), (0,patch_size[1]-1), (0,numframes-2)), 'edge')

      image_limits = (image.shape[0] - patch_size[0], 
                      image.shape[1] - patch_size[1])

      for i in range(0,num_patches):
         p_idx = indexes[i,:]
         if p_idx[0] <= image_limits[0] and p_idx[1] <= image_limits[1]:
            patches[i,:,:,:] = image[p_idx[0]:p_idx[0] + patch_size[0], 
                                     p_idx[1]:p_idx[1] + patch_size[1], :]
                                     
            

   else:
      patches = np.zeros((num_patches, patch_size[0], patch_size[1], patch_size[2]), dtype=image.dtype)

      # Need to pad the image if padding is 'SAME'
      if padding is 'SAME':
         print('\n\n\n\n\n\n PADDING ......')
         image = np.lib.pad(image,((0,patch_size[0]-1), (0,patch_size[1]-1), (0,patch_size[2]-1)), 'edge')

      image_limits = (image.shape[0] - patch_size[0], 
                      image.shape[1] - patch_size[1],
                      image.shape[2] - patch_size[2])

      for i in range(0,num_patches):
         p_idx = indexes[i,:]
         if p_idx[0] <= image_limits[0] and p_idx[1] <= image_limits[1] and p_idx[2] <= image_limits[2]:
            patches[i,:,:,:] = image[p_idx[0]:p_idx[0] + patch_size[0], 
                                     p_idx[1]:p_idx[1] + patch_size[1], 
                                     p_idx[2]:p_idx[2] + patch_size[2]]
            
   return patches.astype(dtype)




def getRandomPatches(image, patch_size, num_patches=1, padding='VALID', dtype=None):
   """Get image patch samples from a regularly sampled grid

   Create the
   
   Returns:
      patches: the image patches as a 4D numpy array
      indexes: the index of the image denoting the top-left corner of the patch
   """

   indexes = getRandomPatchIndexes(image, patch_size, num_patches=num_patches, padding=padding)
   patches = getPatchesFromIndexes(image, indexes, patch_size=patch_size, padding=padding, dtype=dtype)
   return [patches, indexes]




def getOrderedPatches(image, patch_size, stride=[1,1,1], num_patches=0, padding='VALID', dtype=None):
   """Get image patch samples from a regularly sampled grid

   Create the
   
   Returns:
      patches: the image patches as a 4D numpy array
      indexes: the index of the image denoting the top-left corner of the patch
   """

   indexes = getOrderedPatchIndexes(image, patch_size, stride=stride, padding=padding)

   total_patches = indexes.shape[0]
   if num_patches > total_patches:
      num_patches = total_patches

   if num_patches > 0:
      indexes = indexes[0:num_patches,:]

   patches = getPatchesFromIndexes(image, indexes, patch_size=patch_size, padding=padding, dtype=dtype)

   return [patches, indexes]


# -------------------------------------------------------------------------------
#
# Image Patch Reconstruction
#
# -------------------------------------------------------------------------------
def imagePatchSmoothRecon(output_size, patches, indexes, dtype=None,indent='',threed=False,sigma=0.0,orig_patch_shape=None):


   if not(dtype):
      dtype = patches.dtype

   num_patches = patches.shape[0]
   patch_size = (patches.shape[1], patches.shape[2], patches.shape[3])
   if (orig_patch_shape==None):
      orig_patch_size = (patches.shape[1], patches.shape[2], patches.shape[3])
   else:
      orig_patch_size = (orig_patch_shape[1], orig_patch_shape[2], orig_patch_shape[3])
            
   if threed:
      pad_size = (output_size[0]+orig_patch_size[0]-1, output_size[1]+orig_patch_size[1]-1, output_size[2]+orig_patch_size[2]-1,1)
      weight_mat=np.zeros((patch_size[0],patch_size[1],patch_size[2],1),dtype=np.float32)
   else:
      pad_size = (output_size[0]+orig_patch_size[0]-1, output_size[1]+orig_patch_size[1]-1, 1)
      weight_mat=np.zeros((patch_size[0],patch_size[1],1),dtype=np.float32)
   

   shp=weight_mat.shape
   sigma=shp[0]*sigma
   midpoint=np.zeros(len(shp),dtype=np.float32);
   for i in range(0,len(shp)):
      midpoint[i]=(shp[i]-1)/2.0
   sigma2=2.0*sigma*sigma

   if (threed):
      for i in range(0,shp[0]):
         for j in range(0,shp[1]):
            for k in range(0,shp[2]):
               r=pow(i-midpoint[0],2.0)+pow(j-midpoint[1],2.0)+pow(k-midpoint[2],2.0)
               weight_mat[i,j,k,0]=math.exp(-r/sigma2)
   else:
      for i in range(0,shp[0]):
         for j in range(0,shp[1]):
            r=pow(i-midpoint[0],2.0)+pow(j-midpoint[1],2.0)
            weight_mat[i,j,0]=math.exp(-r/sigma2)

   padded_image = np.zeros(pad_size, dtype=patches.dtype)
   sum_image = np.zeros(padded_image.shape, dtype=np.float32)

   print("_____",indent,"imagePatchSmoothRecon: patch_shape=",patches.shape," out=",output_size," padded size=",padded_image.shape,' sigma_vox=',sigma, ' shp=',shp,' orig_patch_size=',orig_patch_size)

   
   for i in xrange(0, num_patches):
      idx = indexes[i,:]
      if threed:
         padded_image[idx[0]+2:idx[0]+patch_size[0]-2,
                      idx[1]+2:idx[1]+patch_size[1]-2,
                      idx[2]+2:idx[2]+patch_size[2]-2,:] += np.multiply(patches[i,2:patch_size[0]-2,2:patch_size[1]-2,2:patch_size[2]-2],
                                                                        weight_mat[2:patch_size[0]-2,2:patch_size[1]-2,2:patch_size[2]-2])

         sum_image[idx[0]+2:idx[0]+patch_size[0]-2,
                   idx[1]+2:idx[1]+patch_size[1]-2,
                   idx[2]+2:idx[2]+patch_size[2]-2] += weight_mat[2:patch_size[0]-2,2:patch_size[1]-2,2:patch_size[2]-2]

      else:
         
         padded_image[idx[0]:idx[0]+patch_size[0],
                      idx[1]:idx[1]+patch_size[1],
                      idx[2]:idx[2]+patch_size[2]] += np.multiply(patches[i,0:patch_size[0],0:patch_size[1],0:patch_size[2]],weight_mat)
         sum_image[idx[0]:idx[0]+patch_size[0],
                   idx[1]:idx[1]+patch_size[1],
                   idx[2]:idx[2]+patch_size[2]] += weight_mat
         
   sum_image[sum_image<0.01] = 1

   image = np.true_divide(padded_image, sum_image);
   output = image[0:output_size[0], 0:output_size[1], 0:output_size[2]]
   #   print('_____'+indent+'    output-shape=',output.shape)
   return output.astype(dtype)


def imagePatchRecon(output_size, patches, indexes, dtype=None,indent='',threed=False,sigma=0.0,orig_patch_shape=None):

   
   if not(dtype):
      dtype = patches.dtype

   if sigma>0:
      return imagePatchSmoothRecon(output_size,patches,indexes,dtype,indent,threed,sigma,orig_patch_shape)
   

   if not(dtype):
      dtype = patches.dtype

   num_patches = patches.shape[0]
   patch_size = (patches.shape[1], patches.shape[2], patches.shape[3])

   if (orig_patch_shape==None):
      orig_patch_size = (patches.shape[1], patches.shape[2], patches.shape[3])
   else:
      orig_patch_size = (orig_patch_shape[1], orig_patch_shape[2], orig_patch_shape[3])

   
   if threed:
      pad_size = (output_size[0]+orig_patch_size[0]-1, output_size[1]+orig_patch_size[1]-1, output_size[2]+orig_patch_size[2]-1,1)
      weight_mat=np.zeros((patch_size[0],patch_size[1],patch_size[2],1),dtype=np.float32)
   else:
      pad_size = (output_size[0]+orig_patch_size[0]-1, output_size[1]+orig_patch_size[1]-1, 1)
      weight_mat=np.zeros((patch_size[0],patch_size[1],1),dtype=np.float32)
   

   shp=weight_mat.shape
   sigma=0.5*shp[0]
   midpoint=np.zeros(len(shp),dtype=np.float32);
   for i in range(0,len(shp)):
      midpoint[i]=(shp[i]-1)/2.0
   sigma2=2.0*sigma*sigma
   factor=1.0/(math.sqrt(math.pi*sigma2))


   if (threed):
      for i in range(0,shp[0]):
         for j in range(0,shp[1]):
            for k in range(0,shp[2]):
               r=pow(i-midpoint[0],2.0)+pow(j-midpoint[1],2.0)+pow(k-midpoint[2],2.0)
               weight_mat[i,j,k,0]=math.exp(-r/sigma2)*factor
   else:
      for i in range(0,shp[0]):
         for j in range(0,shp[1]):
            r=pow(i-midpoint[0],2.0)+pow(j-midpoint[1],2.0)
            weight_mat[i,j,0]=math.exp(-r/sigma2)*factor

   padded_image = np.zeros(pad_size, dtype=patches.dtype)
   sum_image = np.zeros(padded_image.shape, dtype=np.float32)

   print("_____",indent,"imagePatchRecon: patch_shape=",patches.shape," out=",output_size," padded size=",padded_image.shape,' orig_patch_size=',orig_patch_size)

   
   for i in xrange(0, num_patches):
      idx = indexes[i,:]
      if threed:
         padded_image[idx[0]+2:idx[0]+patch_size[0]-2,
                      idx[1]+2:idx[1]+patch_size[1]-2,
                      idx[2]+2:idx[2]+patch_size[2]-2,:] += patches[i,2:patch_size[0]-2,2:patch_size[1]-2,2:patch_size[2]-2]

         sum_image[idx[0]+2:idx[0]+patch_size[0]-2,
                   idx[1]+2:idx[1]+patch_size[1]-2,
                   idx[2]+2:idx[2]+patch_size[2]-2] += 1 

      else:
         padded_image[idx[0]:idx[0]+patch_size[0],
                      idx[1]:idx[1]+patch_size[1],:] += patches[i,0:patch_size[0],0:patch_size[1]:]
         sum_image[idx[0]:idx[0]+patch_size[0],
                   idx[1]:idx[1]+patch_size[1],
                   :] += 1
            
   sum_image[sum_image<1] = 1

   image = np.true_divide(padded_image, sum_image);
   output = image[0:output_size[0], 0:output_size[1], 0:output_size[2]]
   #   print('_____'+indent+'    output-shape=',output.shape)
   return output.astype(dtype)


# -------------------------------------------------------------------------------
# Crop Image
# -------------------------------------------------------------------------------


def cropImage(image, offset=0,threed=False):
   
   dtype = image.dtype
   imgshape=image.shape
   bd=2*offset
   
   if threed==True:
      if len(imgshape)>3:
         newdata=np.zeros([imgshape[0]-bd,imgshape[1]-bd,imgshape[2]-bd,imgshape[3]],dtype=np.float32)
         newshape=newdata.shape
         newdata[:,:,:,:]=image[offset:offset+newshape[0],offset:offset+newshape[1],offset:offset+newshape[2],:]
      else:
         newdata=np.zeros([imgshape[0]-bd,imgshape[1]-bd,imgshape[2]-bd],dtype=np.float32)
         newshape=newdata.shape
         newdata[:,:,:]=image[offset:offset+newshape[0],offset:offset+newshape[1],offset:offset+newshape[2]]

   else:
         newdata=np.zeros([imgshape[0]-bd,imgshape[1]-bd,imgshape[2]],dtype=np.float32)
         newshape=newdata.shape
         newdata[:,:,:]=image[offset:offset+newshape[0],offset:offset+newshape[1],:]

   return newdata

# -----------------------------------------------------------------------------------
#
# Main Function
#
# -----------------------------------------------------------------------------------

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Load an image for patch sampling.')
  parser.add_argument('input', nargs=1, help='NIfTI image input file.')
  parser.add_argument('output', nargs=1, help='NIfTI image patch output file.')
  parser.add_argument('-n','--num_samples', type=int, help='number of image patch samples to extract', default=0)
  parser.add_argument('-r','--random', help='Perform random patch sampling from the image', action='store_true')
  parser.add_argument('-p','--patch_size', type=int, nargs='+', help='Set the patch size in voxels', default=[16, 16])
  parser.add_argument('-s','--stride', type=int, help='Set the patch stride in voxels (isotropically)', default=1)
  parser.add_argument('--recon', help='File name for to create a reconstructed image from the sampled patches')
  args = parser.parse_args()


  if not os.path.isfile(args.input[0]):
    raise ValueError('Failed to find the file: ' + f)
  print('Loading file: %s' % args.input[0])
  nifti_image = nib.load(args.input[0])
  image = nifti_image.get_data()
  print('Loaded image with data of size: %s' % (image.shape,))


  # psize = (args.patch_size, args.patch_size, args.patch_size)
  psize = [1, 1, 1]

  if len(image.shape) < 3:
    image = np.expand_dims(image, 2)

  dims = min(len(image.shape),len(args.patch_size))
  for i in range(0,dims):
    psize[i] = min(image.shape[i],args.patch_size[i])


  print('Patch size = %s' %(psize,))
  print('Random sampling = %r' % args.random)

  if args.random:
    [patches, indexes] = getRandomPatches(image, psize, num_patches=args.num_samples, padding='VALID')
  else:
    [patches, indexes] = getOrderedPatches(image, psize, stride=[args.stride, args.stride, args.stride], padding='VALID', num_patches=args.num_samples)

  print('Patch sampling complete.')
  print('Got %d patches from the image...' % patches.shape[0])


  out_patches = np.zeros(psize + [patches.shape[0]], dtype=image.dtype)
  for i in range(0,patches.shape[0]):
    out_patches[:,:,:,i] = patches[i,:,:,:]

  print('Saving the patch image out: %s' % args.output[0])
  output = nib.Nifti1Image(out_patches, nifti_image.affine)
  nib.save(output, args.output[0])


  if args.recon:
    print('Saving reconstructed image out: %s' % args.recon)


    r_image = imagePatchRecon(image.shape, patches, indexes)
    output = nib.Nifti1Image(r_image, nifti_image.affine, header=nifti_image.header)
    nib.save(output, args.recon)


