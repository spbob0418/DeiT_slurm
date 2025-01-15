#!/bin/bash

export OMP_NUM_THREADS=1

DIR=output
VERSION=finegrained_wgfp_qkl_vertex_register_token_raw_patch_init

mkdir -p ${DIR}/${VERSION}

#fourbits_deit_small_patch16_224 --> for quantized version 
#deit_small_patch16_224 --> for fullprecision
#main_original_constant_lr
#main_original

nohup taskset -c 32-63 python3 ./main_original.py \
--model fourbits_deit_small_patch16_224 \
--epochs 90 \
--weight-decay 0.05 \
--batch-size 256 \
--data-path /data/ILSVRC2012 \
--lr 5e-4 \
--output_dir ${DIR}/${VERSION} \
--distributed > ${DIR}/${VERSION}/output.log 2>&1 &

