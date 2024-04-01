#!/bin/bash

# 初始化一个空数组来存储数字
declare -a numbers=()

# 读取文件，查找行，并提取数字
while IFS= read -r line; do
  # 提取第四列
  number=$(echo "$line" | awk '{print $4}')
  # 将数字添加到数组中
  numbers+=("$number")
done < <(grep 'fn_elementwise' outlog_dse.log)

i=0
j=0
# 打印提取的数字
for number in "${numbers[@]}"; do
  if ((i == 9))
  then
    echo "warp " 4*$j
    ((j++))
    i=0
  fi
  ((i++))
  echo "$number"
done