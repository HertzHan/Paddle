#!/bin/bash

# graph_tested=("TestCinnLayerNorm" "TestCinnSoftmax")
graph_tested=("TestCinnSoftmax")
# graph_tested=("TestLlamaRMSNorm")
# 使用正确的bash数组语法，并且每个形状都被引号包围
shape_tested=("[128,128,768]" "[128,12,128,128]" "[1,1,4096]" "[1,17,4096]")
# shape_tested=("[1,1,4096]")
declare -a output 
i=0
for MODULE in ${graph_tested[@]}
do
    for SHAPE in ${shape_tested[@]}
    do
        echo "********************************"
        echo "debug:"$MODULE $SHAPE $i
        echo '/n'
        CMD="CUDA_VISIBLE_DEVICES=1 FLAGS_support_reduce_stride_read=1 GLOG_vmodule=compiler=3 FLAGS_pir_apply_shape_optimization_pass=1 FLAGS_enable_pir_api=1 
FLAGS_cinn_new_group_scheduler=1 FLAGS_group_schedule_tiling_first=1 FLAGS_cinn_bucket_compile=True FLAGS_cinn_compile_with_nvrtc=True FLAGS_nvrtc_compile_to_cubin=True 
nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas python test_cinn_sub_graph.py $MODULE $SHAPE"
        # echo "Running command: $CMD"

        # eval $CMD
          eval $CMD | grep -A 10 "cuda_gpu_kern_sum"
        # output[$i]=$(eval $CMD | grep -A 10 "cuda_gpu_kern_sum")
        # echo "$output[$i]"
        ((i++))
    done
done



# 调用Python程序进行性能分析


# 你可以在这里添加额外的步骤来处理nsys生成的性能数据
