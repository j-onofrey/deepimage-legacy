from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bis_tf_utils as bisutil
import tensorflow as tf

#XP: If adding metric -- add it here

allowedmetrics = [ 'cc', 'ce', 'wce', 'l2', 'nl2'  ]
regressionmetrics = [ 'cc', 'l2', 'nl2' ]

# -----------------------------------------------------------------------------------
# Only allow regression metrics
# -----------------------------------------------------------------------------------
def force_regression():
    allowedmetrics=regressionmetrics

# -------------------------------------------------------
# Global Helpers (common fcnet,unet)
# -------------------------------------------------------

def is_metric_regression(name):

    if name in regressionmetrics:
        return True

    return False


def create_loss_params(obj):
    obj.commandlineparams['pos_weight']=1.0
    obj.commandlineparams['metric']="ce"
    obj.donotreadlist.append('pos_weight')
    obj.donotreadlist.append('metric')

def add_parser_loss_params(parser):
    parser.add_argument('--metric', help='Metric (one of ce, wce, cc or l2)',
                        default=None)
    parser.add_argument('--pos_weight',   help='Positive weight for classification accuracy (or intensity weighted in nl2)',
                        default=None,type=float)


def extract_parser_loss_params(obj,args):

    obj.set_commandlineparam_from_arg(name='pos_weight',
                           value=bisutil.force_inrange(args.pos_weight,0.01,1000.0))

    #XP: If adding metric -- add name(s) here so it is valid

    obj.set_commandlineparam_from_arg(name='metric',value=args.metric,allowedvalues=allowedmetrics)



# -----------------------------------------------------------------------------------
#
#  Normalizer for functions
#
# -----------------------------------------------------------------------------------
def compute_normalizer(image_shape,batch_size=1):

    depth=1;
    if (len(image_shape)==5):
        depth=image_shape[3]

    return 1.0/float(depth*image_shape[2]*image_shape[1]*batch_size)


# -----------------------------------------------------------------------------------
#
#  Standard Loss Functions
#
# -----------------------------------------------------------------------------------
def cross_entropy_loss(logits, targets):
    """Add Classification Loss

    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    targets = tf.cast(targets, tf.int32)

    sq=3
    if len(targets.get_shape())==5:
        sq=4

    num_classes=logits.get_shape()[sq]
    if (num_classes>2):
        return weighted_cross_entropy_loss(logits=logits,targets=targets,pos_weight=1.0)

    
    print('*****\t setting up entropy_loss function: dim=',logits.shape,'sq=',sq)
    entropyloss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(targets, squeeze_dims=[sq]),
                                                       name='entropy'))
    bisutil.add_scalar_summary('entropy', entropyloss,scope='Metrics/')
    return entropyloss


def normalized_l2_loss(pred_image, targets,normalizer=1.0,pos_weight=1.0):
    """Add L2Loss

    Returns:
    Loss tensor of type float.
"""
    print('***** Setting up normalized l2_loss function: dim=',pred_image.shape,' pos_weight=',pos_weight,' normalizer='+str(normalizer))


    mean=1.0+tf.abs(tf.reduce_mean(targets))
    mean=500.0
    imean=1.0/(mean)
    m2=mean*mean

    factor=m2
    if (pos_weight<0.0):
        pos_weight=0.0

    if (pos_weight>0):
        iposweight=1.0/pos_weight
    else:
        iposweight=1.0


    l=tf.subtract(pred_image,targets)
    l2=tf.square(l)
    robust=tf.divide(l2,l2+factor)
    weighted=iposweight*tf.multiply( 1.0+pos_weight*targets,robust*imean)

    l2 =  tf.identity( tf.reduce_mean(l2),name='L2_loss')
    nl2=  tf.identity( tf.reduce_mean(robust),name='Rob_L2_loss')
    wnl2 = tf.identity( tf.reduce_mean(weighted),name='WgtRob_loss')

    bisutil.add_scalar_summary('L2', l2,scope='Metrics/')
    bisutil.add_scalar_summary('NL2', nl2,scope='Metrics/')
    bisutil.add_scalar_summary('WNL2', wnl2,scope='Metrics/')
    
    return wnl2

def l2_loss(pred_image, targets,normalizer=1.0):
    """Add L2Loss

    Returns:
    Loss tensor of type float.
"""
    print('***** Setting up l2_loss function: norm='+str(normalizer)+'. Pred=',pred_image,'target=',targets)

    a=tf.subtract(pred_image,targets)
    b=tf.reduce_mean(tf.square(a))

    l2_loss = tf.identity(b*normalizer,name='L2_loss')
    bisutil.add_scalar_summary('Mean_L2', l2_loss,scope='Metrics/')

    return l2_loss


def cross_correlation_loss(pred_image,targets):

    m1=tf.reduce_mean(pred_image)
    m2=tf.reduce_mean(targets)

    v1=tf.sqrt(2.0*tf.nn.l2_loss(pred_image-m1))
    v2=tf.sqrt(2.0*tf.nn.l2_loss(targets-m2))

    c=tf.reduce_sum(tf.multiply((pred_image-m1)/(v1+0.001),
                                (targets-m2)/(v2+0.001)))
    bisutil.add_scalar_summary('CC', c,scope='Metrics/')
    a=(1.0-c)
    bisutil.add_scalar_summary('CC_loss', a,scope='Metrics/')
    return a


def weighted_cross_entropy_loss(logits,targets,pos_weight=1.0):

    sq=3
    if len(targets.get_shape())==5:
        sq=4

    num_classes=logits.get_shape()[sq]
    print("***** \t  weighted cross_entropy pos_weight="+str(pos_weight)+" num_classes="+str(num_classes))


    onehot_targets=tf.squeeze(tf.one_hot(indices=tf.cast(targets,tf.int32),
                                         depth=num_classes,
                                         on_value=1.0,
                                         off_value=0.0,
                                         axis=sq,
                                         dtype=tf.float32),squeeze_dims=[sq+1])



    if (num_classes==2):
        classes_weights = tf.constant([1.0, pos_weight])
    elif num_classes==3:
        classes_weights = tf.constant([1.0, pos_weight, pos_weight])
    elif num_classes==4:
        classes_weights = tf.constant([1.0, pos_weight, pos_weight, pos_weight])
    elif num_classes==5:
        classes_weights = tf.constant([1.0, pos_weight, pos_weight, pos_weight, pos_weight])
    elif num_classes==6:
        classes_weights = tf.constant([1.0, pos_weight, pos_weight, pos_weight, pos_weight, pos_weight])
    elif num_classes==7:
        classes_weights = tf.constant([1.0, pos_weight, pos_weight, pos_weight, pos_weight, pos_weight, pos_weight])
    elif num_classes==8:
        classes_weights = tf.constant([1.0, pos_weight, pos_weight, pos_weight, pos_weight, pos_weight, pos_weight, pos_weight])
    else:
        raise ValueError('Too many classes for weighted cross entropy, max=8')

    cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=onehot_targets, pos_weight=classes_weights),name="weighted_entropy")

    bisutil.add_scalar_summary('Weighted_Entropy', cross_entropy,scope='Metrics/')
    return cross_entropy





def compute_loss(pred_image,logits,targets,regularizer,metric="cross_entropy",normalizer=1.0,pos_weight=1.0,smoothness=0.0):


    #XP: If adding metric -- add clause here to call it appropriately

    if  metric == "cc":
        print("*****   Creating cross_correlation_loss function")
        dataloss=cross_correlation_loss(pred_image=pred_image,
                                        targets=targets)
    elif metric == "ce":
        print("*****   Creating cross_entropy_loss function")
        dataloss=cross_entropy_loss(logits=logits,
                                    targets=targets)
    elif metric == "wce":
        print("*****   Creating weighted_entropy (weighted cross entropy) loss function")
        dataloss=weighted_cross_entropy_loss(logits=logits,
                                             targets=targets,
                                             pos_weight=pos_weight)
    elif metric=="nl2":
        dataloss=normalized_l2_loss(pred_image=pred_image,
                                    targets=targets,
                                    normalizer=normalizer,
                                    pos_weight=pos_weight)
    else:
        dataloss=l2_loss(pred_image=pred_image,
                         targets=targets,
                         normalizer=normalizer)


    if smoothness>0.0:
        print('*****\t and adding regularizer smoothness='+str(smoothness))
        reg=regularizer*normalizer;
        bisutil.add_scalar_summary('Regularizer', reg,scope='Metrics/')
        total=tf.identity(dataloss+reg*smoothness,name="Total")
        bisutil.add_scalar_summary('Total', total,scope='Metrics/')
    else:
        bisutil.debug_print('*****\t and not adding regularizer smoothness='+str(smoothness))
        total=tf.identity(dataloss,name='Total')

    return total



# --------------------------------------------------------------------------------
#
# MAIN CREATE LOSS FUNCTION
#
# --------------------------------------------------------------------------------
def create_loss_function(obj,output_dict,smoothness=0.0,batch_size=1,name="Loss"):

    if batch_size<1:
        batch_size=obj.commandlineparams['batch_size'];

    if (output_dict==None):
        raise RuntimeError('No output dictionary has been defined.')


    print('*****   Preparing loss function metric=',obj.commandlineparams['metric'])
    
    if not 'cropped_target' in output_dict:
        output_dict['cropped_target']=obj.pointers['target_pointer']
    else:
        print('***** \t using cropped target ',output_dict['cropped_target'].get_shape())
    

    if not 'cropped_logits' in output_dict:
        output_dict['cropped_logits']=output_dict['logits']
    else:
        print('***** \t using cropped output_logits ',output_dict['cropped_logits'].get_shape())
    

    if not 'cropped_image' in output_dict:
        output_dict['cropped_image']=output_dict['image']
    else:
        print('***** \t using cropped output_image ',output_dict['cropped_image'].get_shape())
    

    input_data=obj.pointers['input_pointer']
    normalizer_scalar=compute_normalizer(output_dict['cropped_target'].get_shape().as_list(),batch_size)
    with tf.variable_scope(name):
        loss_op = compute_loss(pred_image=output_dict["cropped_image"],
                               logits=output_dict["cropped_logits"],
                               targets=output_dict["cropped_target"],
                               regularizer=output_dict["regularizer"],
                               smoothness=smoothness,
                               metric=obj.commandlineparams['metric'],
                               pos_weight=obj.commandlineparams['pos_weight'],
                               normalizer=normalizer_scalar)
    return loss_op;
