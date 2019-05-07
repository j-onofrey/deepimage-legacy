#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import device_lib
from tensorflow.python.framework import test_util

import sys
import os
import platform
import numpy as np
import tensorflow as tf
import bis_tf_utils as bisutil


print('\n---------------------------------------------------------------------------\n')

systeminfo= {
    'os'   : platform.system()+' '+platform.release(),
    'node' : platform.node(),
    'machine' : platform.machine(),
    'python'  : platform.python_version(),
    'tensorflow' :  str(tf.__version__),
    'numpy' : np.version.version,
    'cuda_enabled': test_util.IsGoogleCudaEnabled()
}

bisutil.print_dict(systeminfo,header="System Information")
print('\n')
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"



lst=device_lib.list_local_devices()
n=len(lst)
i=0
while i<n:
    name=lst[i].name
    print('============ Device',i,'==========================================')
    print('name:  ',lst[i].name)
    print('type:  ',lst[i].device_type)
    print('memory:',lst[i].memory_limit/(1024*1024*1024), 'GB')
    if (lst[i].physical_device_desc!=False):
        print('desc:  ',lst[i].physical_device_desc)
    print('')
    i=i+1

print('---------------------------------------------------------------------------\n')

