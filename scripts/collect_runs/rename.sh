#!/bin/bash

# 设置目标目录
DIR="logs/multi_run/20240910_155347"

# 遍历目录中的所有文件
for file in "$DIR"/freq_prune_*.log; do
    # 提取文件名
    filename=$(basename "$file")
    
    # 构建新文件名
    newname=$(echo "$filename" | sed 's/freq_prune_/freq_prune_uniinit_/')
    
    # 重命名文件
    mv "$file" "$DIR/$newname"
    
    echo "Renamed $filename to $newname"
done