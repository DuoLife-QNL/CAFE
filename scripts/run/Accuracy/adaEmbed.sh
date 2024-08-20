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
    --compress-rate=0.2 \
    --nepochs=5 \
    --test-freq=100000 \
    | tee logs/ada.log