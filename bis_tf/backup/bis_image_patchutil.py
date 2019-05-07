from __future__ import print_function

import argparse
import os.path
import nibabel as nib
import numpy as np


from six.moves import xrange


# -------------------------------------------------------------------------------
#
# Core Patch Code
#
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
   patches = np.zeros((num_patches, patch_size[0], patch_size[1], patch_size[2]), dtype=image.dtype)

   # Need to pad the image if padding is 'SAME'
   if padding is 'SAME':
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


def imagePatchRecon(output_size, patches, indexes, dtype=None,indent=''):


   if not(dtype):
      dtype = patches.dtype


   if (len(patches.shape)>4):
      threed=True
      depth=1
      if (len(output_size)>3):
         depth=output_size[3]
   else:
      threed=False

   num_patches = patches.shape[0]
   patch_size = (patches.shape[1], patches.shape[2], patches.shape[3])
   if threed:
      pad_size = (output_size[0]+patch_size[0]-1, output_size[1]+patch_size[1]-1, output_size[2]+patch_size[2]-1,depth)
   else:
      pad_size = (output_size[0]+patch_size[0]-1, output_size[1]+patch_size[1]-1, output_size[2]+patch_size[2]-1)

   
   padded_image = np.zeros(pad_size, dtype=patches.dtype)
   sum_image = np.zeros(padded_image.shape, dtype=np.float32)

   print("_____",indent,"imagePatchRecon: patch_shape=",patches.shape," out=",output_size," padded size=",padded_image.shape)
   
   for i in xrange(0, num_patches):
      idx = indexes[i,:]
      if threed:
         padded_image[idx[0]+2:idx[0]+patch_size[0]-2,
                      idx[1]+2:idx[1]+patch_size[1]-2,
                      idx[2]+2:idx[2]+patch_size[2]-2,:] += patches[i,2:patch_size[0]-2,2:patch_size[1]-2:,2:patch_size[2]-2]
         sum_image[idx[0]+2:idx[0]+patch_size[0]-2,
                   idx[1]+2:idx[1]+patch_size[1]-2,
                   idx[2]+2:idx[2]+patch_size[2]-2] += 1.0

      else:
         padded_image[idx[0]+2:idx[0]+patch_size[0]-2,
                      idx[1]+2:idx[1]+patch_size[1]-2,
                      idx[2]:idx[2]+patch_size[2]] += patches[i,2:patch_size[0]-2,2:patch_size[1]-2:,:]
         sum_image[idx[0]+2:idx[0]+patch_size[0]-2,
                   idx[1]+2:idx[1]+patch_size[1]-2,
                   idx[2]:idx[2]+patch_size[2]] += 1.0
         
   sum_image[sum_image<1] = 1

   image = np.true_divide(padded_image, sum_image);
   output = image[0:output_size[0], 0:output_size[1], 0:output_size[2]]
   print('_____'+indent+'    output-shape=',output.shape)
   return output.astype(dtype)





# -----------------------------------------------------------------------------------
#
# Batch Code 
#
# -----------------------------------------------------------------------------------
def get_batch3d(TRAINING_DATA, batch_size, augmentation=False,image_size=64):
    image_training_data = TRAINING_DATA[0]
    label_training_data = TRAINING_DATA[1]
    
    # print('Image training data shape: %s' % (image_training_data.shape,))
    # print('Label training data shape: %s' % (label_training_data.shape,))  
    
    if len(image_training_data) != len(label_training_data):
        raise ValueError('Image and label data counts do not match')

    #    IMAGE_SIZE = fcn_model.IMAGE_SIZE
    
    # Get the random sample indexing
    num_samples = len(image_training_data)
    sample_idx = (((num_samples-1)*np.random.rand(batch_size))+0.5).astype(int)
    # print('Sampling index: %s' % (sample_idx,))
    
    patch_size = (image_size, image_size, image_size)
    
    images = np.zeros((batch_size,image_size, image_size, image_size, 1), dtype=np.float32)
    labels = np.zeros((batch_size,image_size, image_size, image_size, 1), dtype=np.float32)
    flip_count = 0
    for i in range(0,batch_size):
        # Only get a single patch at a time
        # [p1, idx] = image_training_data[sample_idx[i]].getPatches(1)
        [p1, idx] = getRandomPatches(image_training_data[sample_idx[i]], patch_size, num_patches=1, dtype=np.float32)
        # [p2, idx] = label_training_data[sample_idx[i]].getPatchesFromIndexes(idx)
        p2 = getPatchesFromIndexes(label_training_data[sample_idx[i]], idx, patch_size, dtype=np.float32)
        
        # print('Get patch index: %s' % (idx,))
        images[i,:,:,:,0] = p1
        labels[i,:,:,:,0] = p2
        if augmentation:
            # 50% chance of flipping in a random dimension
            if np.random.random() > 0.5:
                # flip_dim = np.random.randint(0,3)
                # Fix this for Left-Right flipping (assumes rigid registration to an atlas/template)
                flip_dim = 0
                images[i,:,:,:,0] = np.flip(images[i,:,:,:,0],flip_dim)
                labels[i,:,:,:,0] = np.flip(labels[i,:,:,:,0],flip_dim)
                flip_count += 1
                
    return [images, labels]

def get_batch(TRAINING_DATA, batch_size, augmentation=False,image_size=64):
    image_training_data = TRAINING_DATA[0]
    label_training_data = TRAINING_DATA[1]

    depth=image_training_data[0].shape[2]
    if (depth>1):
        return get_batch3d(TRAINING_DATA,batch_size,augmentation,image_size)
    

    if len(image_training_data) != len(label_training_data):
        raise ValueError('Image and label data counts do not match')

    # Get the random sample indexing
    num_samples = len(image_training_data)
    sample_idx = (((num_samples-1)*np.random.rand(batch_size))+0.5).astype(int)
    # print('Sampling index: %s' % (sample_idx,))

    patch_size = (image_size, image_size, 1)

    images = np.zeros((batch_size,image_size, image_size, 1), dtype=np.float32)
    labels = np.zeros((batch_size,image_size, image_size, 1), dtype=np.float32)
    flip_count = 0
    for i in range(0,batch_size):
        # Only get a single patch at a time
        # [p1, idx] = image_training_data[sample_idx[i]].getPatches(1, dtype=np.float32)
        [p1, idx] = getRandomPatches(image_training_data[sample_idx[i]], patch_size, num_patches=1, dtype=np.float32)
        # [p2, idx]< = label_training_data[sample_idx[i]].getPatchesFromIndexes(idx, dtype=np.float32)
        p2 = getPatchesFromIndexes(label_training_data[sample_idx[i]], idx, patch_size, dtype=np.float32)
        # print('Get patch index: %s' % (idx,))
        images[i,:,:,0] = np.squeeze(p1, 3)
        labels[i,:,:,0] = np.squeeze(p2, 3)
        if augmentation:
            # 50% chance of flipping in a random dimension
            if np.random.random() > 0.5:
                    # flip_dim = np.random.randint(0,2)
                flip_dim = 0
                images[i,:,:,0] = np.flip(images[i,:,:,0],flip_dim)
                labels[i,:,:,0] = np.flip(labels[i,:,:,0],flip_dim)
                flip_count += 1

    # print('%d of %d flipped' % (flip_count, batch_size))

    return [images, labels]

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
   parser.add_argument('-p','--patch_size', type=int, help='Set the patch size in voxels (isotropically)', default=16)
   parser.add_argument('-s','--stride', type=int, help='Set the patch stride in voxels (isotropically)', default=1)
   parser.add_argument('--recon', help='File name for to create a reconstructed image from the sampled patches')
   args = parser.parse_args()


   if not os.path.isfile(args.input[0]):
      raise ValueError('Failed to find the file: ' + f)
   print('Loading file: %s' % args.input[0])
   nifti_image = nib.load(args.input[0])
   image = nifti_image.get_data()
   print('Loaded image with data of size: %s' % (image.shape,))



   psize = (args.patch_size, args.patch_size, args.patch_size)

   if len(image.shape) < 3:
      image = np.expand_dims(image, 2)
      psize = (args.patch_size, args.patch_size, 1)


   print('Patch size = %s' %(psize,))
   print('Random sampling = %r' % args.random)

   if args.random:
      [patches, indexes] = getRandomPatches(image, psize, num_patches=args.num_samples, padding='SAME')
   else:
      [patches, indexes] = getOrderedPatches(image, psize, stride=[args.stride, args.stride, args.stride], padding='SAME', num_patches=args.num_samples)

   print('Patch sampling complete.')
   print('Got %d patches from the image...' % patches.shape[0])


   out_patches = np.zeros(psize + (patches.shape[0],), dtype=image.dtype)
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


