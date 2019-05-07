
## Infrastructure

* Add documentation to code
* Add regression tests


## Functionality

* _DONE_ in bis_tf_util.load_training_data and load_single_image -- make output be dictionary that contains
	
		{
			'shape' : shape of data,
			'header': nifti header of data (first image is OK)
			'data'  : actual data,
		}
	also added general jpg/png reading as well
		
* _DONE_: Figure out how to load graph from cpkt.meta file 
  [See this page](https://www.tensorflow.org/programmers_guide/meta_graph)
  
* Create variable length fcn_model (num convolution/deconvolution layers)

* _DONE_ Fix shape stuff in deconvolutions to break link

* _Mostly Done_: Understand shape, tf.shape etc, get_shape() etc.

* Figure out how to use _super_ in both Python 2 and Python 3

* Take a look at git clone https://github.com/galeone/dynamic-training-bench dtb
