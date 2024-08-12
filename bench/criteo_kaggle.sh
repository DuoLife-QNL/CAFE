#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#WARNING: must have compiled PyTorch

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option
CUDA_VISIBLE_DEVICES=0 \
python dlrm_s_pytorch.py \
--use-gpu \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=128 \
--print-freq=1024 \
--nepochs=100 \
--print-time \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--cat-path="datasets/criteo_kaggle/cafe/data_cat.npy" \
--dense-path="datasets/criteo_kaggle/cafe/data_int.npy" \
--label-path="datasets/criteo_kaggle/cafe/data_label.npy" \
--count-path="datasets/criteo_kaggle/cafe/data_count.npy" \
$dlrm_extra_option 2>&1 | tee run_kaggle_pt.log

echo "done"

# --cat-path="../criteo_kaggle/kaggle_processed_sparse_sep.bin" \
# --dense-path="../criteo_kaggle/kaggle_processed_dense.bin" \
# --label-path="../criteo_kaggle/kaggle_processed_label.bin" \
# --count-path="../criteo_kaggle/kaggle_processed_count.bin" \
