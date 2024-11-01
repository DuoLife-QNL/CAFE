python dlrm_s_pytorch.py \
    --use-gpu \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="0-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=moments \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.001 \
    --mini-batch-size=32 \
    --print-freq=1024 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --nepochs=20 \
    --test-freq=1000 \
    | tee logs/moments/full.log

