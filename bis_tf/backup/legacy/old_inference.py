    def create_inference_old(self,training=True):

        dodebug=self.commandlineparams['debug']
        poolmode="max"
        if self.params['avg_pool']:
            poolmode="avg"
            
        norelu=self.params['no_relu']
        nofuse=self.params['no_fuse']
        input_data=self.pointers['input_pointer']
        threed=self.calcparams['threed']
        imgshape=input_data.get_shape()
        
        print('===== Creating FCN Inference Model (training='+str(training)+'). Inputs shape= %s ' % (imgshape))
        bisutil.print_dict({ 'Filter Size': self.params['filter_size'],
                             'Num Filters': self.params['num_filters'],
                             'Num Connected': self.params['num_connected'],
                             'Num Classes':  self.calcparams['num_classes'],
                             'Patch Size':   self.calcparams['patch_size'],
                             'Num Frames':   self.calcparams['num_frames'],
                             'NoRelu':norelu,
                             'NoFuse':nofuse },
                           extra="=====",header="Model Parameters:")

        if training:
            self.add_parameters_as_variables()
            
        # Create Model
        with tf.variable_scope('Inference') as scope:
                
            with tf.variable_scope('c1'):
                if dodebug:
                    print('=====\n===== Convolution Layer 1');
                    
                shape11=bisutil.createConvolutionShape(self.params['filter_size'],
                                                       self.calcparams['num_frames'],
                                                       self.params['num_filters'],threed)
                relu1=bisutil.createConvolutionLayerRELU(input=input_data,
                                                         name='convolution_1_1',
                                                         shape=shape11,
                                                         padvalue=-1,norelu=norelu);

                shape12=bisutil.createConvolutionShape(self.params['filter_size'],
                                                       self.params['num_filters'],
                                                       self.params['num_filters'],threed)
                
                pool1=bisutil.createConvolutionLayerRELUPool(input=relu1,
                                                             mode=poolmode,
                                                             name='convolution_1_2',
                                                             shape=shape12,
                                                             padvalue=0,norelu=norelu);
                
            with tf.variable_scope('c2'):
                if dodebug:
                    print('=====\n===== Convolution Layer 2');

                shape21=bisutil.createConvolutionShape(self.params['filter_size'],
                                                       self.params['num_filters'],
                                                       self.params['num_filters']*2,threed)
                    
                relu2=bisutil.createConvolutionLayerRELU(input=pool1,
                                                         name='convolution_2_1',
                                                         shape=shape21,
                                                         padvalue=-1,norelu=norelu);
                shape22=bisutil.createConvolutionShape(self.params['filter_size'],
                                                       self.params['num_filters']*2,
                                                       self.params['num_filters']*2,threed)

                pool2=bisutil.createConvolutionLayerRELUPool(input=relu2,
                                                             mode=poolmode,
                                                             name='convolution_2_2',
                                                             shape=shape22,
                                                             padvalue=0,norelu=norelu);
                
            with tf.variable_scope('c3'):
                if dodebug:
                    print('=====\n===== Convolution Layer 3');
                    
                shape31=bisutil.createConvolutionShape(self.params['filter_size'],
                                                       self.params['num_filters']*2,
                                                       self.params['num_filters']*4,threed)

                relu3=bisutil.createConvolutionLayerRELU(input=pool2,
                                                         name='convolution_3_1',
                                                         shape=shape31,
                                                         padvalue=-1,norelu=norelu);

                shape32=bisutil.createConvolutionShape(self.params['filter_size'],
                                                       self.params['num_filters']*4,
                                                       self.params['num_filters']*4,threed)
                pool3=bisutil.createConvolutionLayerRELUPool(input=relu3,
                                                             mode=poolmode,
                                                             name='convolution_3_2',
                                                             shape=shape32,
                                                             padvalue=0,norelu=norelu);
                
            # Fully connected layers
            with tf.variable_scope('f1'):
                if dodebug:
                    print('=====\n===== Fully Connected Layer 1');
                shapef1=bisutil.createFullyConnectedShape(pool3.get_shape().as_list(),
                                                          self.params['num_filters']*4,
                                                          self.params['num_connected'],
                                                          threed);
                fc1=bisutil.createFullyConnectedLayerWithDropout(input=pool3,
                                                                 shape=shapef1,
                                                                 name='fullyconnected_1',
                                                                 keep_prob_tensor=self.pointers['keep_pointer'])
                
            with tf.variable_scope('f2'):
                if dodebug:
                    print('=====\n===== Fully Connected Layer 2');

                shapef2=bisutil.createSquareShape(1,self.params['num_connected'],
                                                  self.params['num_connected'],threed)
                    
                fc2=bisutil.createFullyConnectedLayerWithDropout(input=fc1,
                                                                 name='fullyconnected_2',
                                                                 shape=shapef2,
                                                                 keep_prob_tensor=self.pointers['keep_pointer'])
                
            with tf.variable_scope('f3'):
                if dodebug:
                    print('=====\n===== Fully Connected Layer 3');
                    
                shapef3=bisutil.createSquareShape(1,self.params['num_connected'],
                                                  self.params['num_connected'],threed)
                fc3=bisutil.createFullyConnectedLayer(input=fc2,
                                                      name='fullyconnected_3',
                                                      shape=shapef3);

            # Needed from here on below
            dindex=3
            if (threed):
                dindex=4


            # Deconvolve (upscale the image)
            with tf.variable_scope('d1'):
                if dodebug:
                    print('=====\n===== Deconv Layer 1');

                deconv_shape1 = pool2.get_shape()
                shaped1=bisutil.createSquareShape(4,deconv_shape1[dindex].value,
                                                  self.params['num_connected'] ,
                                                  threed)
                out_shape1=bisutil.createDeconvolutionDynamicOutputShape(pool2,input_data,nofuse,threed)
                
                fuse1=bisutil.createDeconvolutionLayerFuse(input=fc3,
                                                           fuse_input=pool2,
                                                           dynamic_output_shape=out_shape1,
                                                           shape=shaped1,
                                                           nofuse=nofuse,
                                                           name="deconv_1")
                
                
            with tf.variable_scope('d2'):
                deconv_shape2 = pool1.get_shape()
                if dodebug:
                    print('=====\n===== Deconv Layer 2');

                shaped2=bisutil.createSquareShape(4,deconv_shape2[dindex].value,
                                                  deconv_shape1[dindex].value,
                                                  threed)

                out_shape2=bisutil.createDeconvolutionDynamicOutputShape(pool1,input_data,nofuse,threed)
                
                fuse2=bisutil.createDeconvolutionLayerFuse(input=fuse1,
                                                           fuse_input=pool1,
                                                           dynamic_output_shape=out_shape2,
                                                           shape=shaped2,
                                                           nofuse=nofuse,
                                                           name="deconv_2")
                
            with tf.variable_scope('d3'):
                if dodebug:
                    print("=====\n===== Deconv layer 3")
                shaped3=bisutil.createSquareShape(4,
                                                  self.calcparams['num_classes'],
                                                  deconv_shape2[dindex].value,
                                                  threed)

                shape = input_data.get_shape()
                if (threed):
                    out_shape = tf.stack([tf.shape(input_data)[0],
                                          shape[1].value, shape[2].value, shape[3].value,
                                          self.calcparams['num_classes']])
                else:
                    out_shape = tf.stack([tf.shape(input_data)[0],
                                          shape[1].value, shape[2].value,
                                          self.calcparams['num_classes']])
                    
                deconv3=bisutil.createDeconvolutionLayer(input=fuse2,
                                                         name="deconv_3",
                                                         input_shape=shaped3,
                                                         dynamic_output_shape=out_shape);
            if dodebug:
                print("=====")

        with tf.variable_scope('Outputs'):
                
            if self.calcparams['num_classes']>1:
                print("===== In classification mode (adding annotation)")
                dim=len(deconv3.get_shape())-1
                annotation_pred = tf.argmax(deconv3, dimension=dim, name='prediction')
                output= { "image":  tf.expand_dims(annotation_pred, dim=dim,name="Output"),
                          "logits": tf.identity(deconv3,name="Logits") ,
                          "regularizer": None}
            else:
                print("===== In regression mode")
                output = { "image":  tf.identity(deconv3,name="Output"),
                           "logits": tf.identity(deconv3,name="Logits"),
                           "regularizer" : None };
                
            if not self.calcparams['threed']:
                tf.summary.image('prediction',
                                 tf.cast(output['image'], tf.float32), max_outputs=self.calcparams['max_outputs'])
            else:
                o_shape=output['image'].get_shape()
                midslice=o_shape[3].value//2
                p_slices = tf.slice(output['image'], (0, 0, 0, midslice, 0),
                                    (-1, o_shape[1].value, o_shape[2].value, 1, 1))
                tf.summary.image('prediction_midslice',
                                 tf.squeeze(p_slices, 4),  max_outputs=self.calcparams['max_outputs'])
                        
            if (self.params['edge_smoothness']>0.0 and training):
                output['regularizer']=bisutil.createSmoothnessLayer(output['image'],
                                                                    self.calcparams['num_classes'],
                                                                    self.calcparams['threed']);
            elif dodebug:
                print('===== Not adding back end smoothness compilation')

        self.pointers['output_dict']=output;
        return output
    
