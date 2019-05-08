#!/bin/bash
bis_tf_jounet.py  --train -i ../tdata/mouse/ct_train.txt -t ../tdata/mouse/rois_train.txt --metric ce --device /gpu:1 -p 32 -b 32 -s 2048 --num_conv_layers 2 --num_filters 32 -o last_unet_ce_p32_sm1 --smoothness 1.0 --resume_model last_unet_ce_p32


