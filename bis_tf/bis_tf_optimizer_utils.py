from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bis_tf_utils as bisutil
import tensorflow as tf

# -------------------------------------------------------
# Global Helpers (common fcnet,unet)
# -------------------------------------------------------

# ---- List of variables to Optimizer ---



def set_list_of_variables_to_optimize(obj,lst):
    obj.list_of_variables_toopt=lst

def get_list_of_variables_to_optimize(obj):
    if obj.list_of_variables_toopt==None:
        return tf.trainable_variables()
    return obj.list_of_variables_toopt

# ----------------------------------------------

def create_opt_params(obj):

    obj.commandlineparams['learning_rate']= 0.0001
    obj.commandlineparams['learning_decay_rate']=0.0
    obj.commandlineparams['opt_numexamples_epoch']=1000
    obj.commandlineparams['opt_numepochs_decay']=4
    obj.commandlineparams['opt_moving_average_decay']=0.9999

# ---- Parser ---

def add_parser_opt_params(parser):

    parser.add_argument('--learning_rate',   help='Learning rate for optimizer',
                        default=None,type=float)
    parser.add_argument('--learning_decay_rate',   help='Learning decay rate for optimizer',
                        default=None,type=float)
    parser.add_argument('--opt_numexamples_epoch', help='number of examples per epoch (1000)',
                        default=None,type=int)
    parser.add_argument('--opt_numepochs_decay', help='number of epochs per decay (4)',
                        default=None,type=int)
    parser.add_argument('--opt_moving_average_decay', help='moving average for decay (0.9999)',
                        default=None,type=float)


def extract_parser_opt_params(obj,args):

    obj.set_commandlineparam_from_arg(name='learning_rate',value=args.learning_rate)
    obj.set_commandlineparam_from_arg(name='learning_decay_rate',value=args.learning_decay_rate)
    obj.set_commandlineparam_from_arg(name='opt_numexamples_epoch', value=args.opt_numexamples_epoch)
    obj.set_commandlineparam_from_arg(name='opt_numepochs_decay', value=args.opt_numepochs_decay)
    obj.set_commandlineparam_from_arg(name='opt_moving_average_decay', value=args.opt_moving_average_decay)

def create_opt_function(obj,loss_op):

    learning_rate=obj.commandlineparams['learning_rate']
    learning_decay_rate=obj.commandlineparams['learning_decay_rate']
    num_examples_per_epoch_for_train = obj.commandlineparams['opt_numexamples_epoch']
    num_epochs_per_decay = obj.commandlineparams['opt_numepochs_decay']
    moving_average_decay = obj.commandlineparams['opt_moving_average_decay']

    if (loss_op==None):
        raise ValueError('No loss_op has been specified')


    print('ooooo')
    if learning_decay_rate > 1e-8:

        init_learning_rate=obj.commandlineparams['learning_rate']

        # Variables that affect learning rate.
        num_batches_per_epoch = num_examples_per_epoch_for_train / obj.commandlineparams['batch_size']
        decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

        print('ooooo Creating AdamOptimizer, learning_rate='+str(learning_rate)+' decay_rate='+str(learning_decay_rate))
        print('ooooo \t num_batches_per_epoch = %f' % num_batches_per_epoch)
        print('ooooo \t decay_steps = %f' % decay_steps)
        print('ooooo \t moving_average = %f' % moving_average_decay)


        with tf.variable_scope('Decay_Optimizer'):

            opt_global_step = tf.Variable(0, trainable=False,name="Optimization_Global_Step")
            bisutil.add_scalar_summary('optimization_global_step', opt_global_step,'Metrics/')

            learning_rate = tf.train.exponential_decay(init_learning_rate,
                                                       opt_global_step,
                                                       decay_steps,
                                                       learning_decay_rate,
                                                       staircase=True)
            bisutil.add_scalar_summary('learning_rate',learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads = optimizer.compute_gradients(loss_op, get_list_of_variables_to_optimize(obj))
            apply_gradient_op = optimizer.apply_gradients(grads,global_step=opt_global_step,name="Optimizer")

            variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
            variable_averages_op = variable_averages.apply(get_list_of_variables_to_optimize(obj))

            with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
                train_op = tf.no_op(name='Train')

    else:
        print('ooooo Creating AdamOptimizer, learning_rate='+str(learning_rate))

        with tf.variable_scope('Optimizer'):
            bisutil.add_scalar_summary('learning_rate',learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads = optimizer.compute_gradients(loss_op, get_list_of_variables_to_optimize(obj))
            train_op=optimizer.apply_gradients(grads,name="Optimizer")

    bisutil.debug_print('ooooo Variables to optimize',get_list_of_variables_to_optimize(obj))

    return train_op
