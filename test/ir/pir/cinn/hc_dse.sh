#!/bin/bash

MODULE="TestLlamaRMSNorm"
# shape_tested=("[128,128,768]" "[128,12,128,128]" "[1,1,4096]" "[1,17,4096]")
SHAPE="[1,1,4096]"
warp_num=("1" "4" "8" "12" "16" "20" "24" "28" "32")
tree_reduce_num=("1" "2" "4" "8" "16" "32" "64" "128"  "256")
declare -a output 
i=0
for WARP in ${warp_num[@]}
do
    for TREE in ${tree_reduce_num[@]}
    do
        echo "********************************"
        echo "debug:"$MODULE $SHAPE $i
        echo "warp_num:"$WARP "tree_reduce_num:"$TREE >>outlog_dse.log
        echo '/n'
        CMD="CUDA_VISIBLE_DEVICES=1 FLAGS_warp_num=$WARP FLAGS_tree_reduce_num=$TREE GLOG_v=0 FLAGS_support_reduce_stride_read=1 FLAGS_pir_apply_shape_optimization_pass=1 FLAGS_enable_pir_api=1 
          FLAGS_prim_enable_dynamic=true FLAGS_prim_all=true FLAGS_cinn_new_group_scheduler=1 FLAGS_group_schedule_tiling_first=1 
          FLAGS_cinn_bucket_compile=True FLAGS_cinn_compile_with_nvrtc=True FLAGS_nvrtc_compile_to_cubin=True 
          nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas python test_cinn_sub_graph.py $MODULE $SHAPE"
        # echo "Running command: $CMD"

        # eval $CMD
          eval $CMD | grep -C 1 "fn_elementwise_mul_2_reduce_sum_0_scale_1_scale_2_rsqrt_0" >>outlog_dse.log 
        # output[$i]=$(eval $CMD | grep -A 10 "cuda_gpu_kern_sum")
        # echo "$output[$i]"
        ((i++))
    done
done

# i=0
# for MODULE in ${graph_tested[@]}
# do
#     for SHAPE in ${shape_tested[@]}
#     do
#       echo $i"-"$MODULE","$SHAPE":"
#       echo "$output[$i]"
#       ((i++))
#   done
# done
