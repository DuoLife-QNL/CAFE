#!/bin/bash

# 创建时间戳
time_stamp=$(date +"%Y%m%d_%H%M%S")

# 创建日志目录
log_dir="logs/multi_run/$time_stamp"
mkdir -p "$log_dir"

# 定义压缩率数组
compress_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)


# 执行ada方法
for rate in "${compress_rates[@]}"; do
    log_file="$log_dir/uniinit_ada_${rate}.log"
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
        --print-time \
        --test-mini-batch-size=16384 \
        --test-num-workers=16 \
        --cat-path="datasets/criteo_kaggle/cafe/data_cat.npy" \
        --dense-path="datasets/criteo_kaggle/cafe/data_int.npy" \
        --label-path="datasets/criteo_kaggle/cafe/data_label.npy" \
        --count-path="datasets/criteo_kaggle/cafe/data_count.npy" \
        --ada-flag \
        --compress-rate=$rate \
        --nepochs=5 \
        --test-freq=100000 \
        | tee "$log_file"
done

echo "所有实验已完成。日志文件保存在 $log_dir 目录下。"
