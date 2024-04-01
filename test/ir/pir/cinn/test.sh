#!/bin/bash

# 定义要测试的模块和形状
graph_tested=("TestCinnLayerNorm" "TestCinnSoftmax" "TestLlamaRMSNorm")
shape_tested=("[128,128,768]" "[128,12,128,128]" "[1,1,4096]" "[1,17,4096]")

# 用于存储输出的关联数组（如果需要区分不同MODULE和SHAPE的输出）
declare -A outputs

# 循环遍历每个模块和形状
for MODULE in "${graph_tested[@]}"; do
  for SHAPE in "${shape_tested[@]}"; do
    # 构建命令数组
    CMD=("nsys" "profile" "--stats" "true" "-w" "true" "-t" "cuda,nvtx,osrt,cudnn,cublas" "python" "test_cinn_sub_graph.py" "$MODULE" "$SHAPE")

    # 执行命令并捕获输出
    # output=$("${CMD[@]}" 2>&1) # 2>&1 也会捕获错误输出
    # 如果你只想捕获标准输出，可以这样写：
    output=$("${CMD[@]}" | grep -A 10 "cuda_gpu_kern_sum")

    # 将输出存储在关联数组中
    outputs["$MODULE,$SHAPE"]="$output"

    # 检查命令是否成功执行
    if [ $? -eq 0 ]; then
      echo "Command succeeded for $MODULE,$SHAPE"
    else
      echo "Command failed for $MODULE,$SHAPE"
    fi
  done
done

# 输出结果
for MODULE in "${graph_tested[@]}"; do
  for SHAPE in "${shape_tested[@]}"; do
    key="$MODULE,$SHAPE"
    echo "$key:"
    echo "${outputs[$key]}"
    echo "-----"
  done
done