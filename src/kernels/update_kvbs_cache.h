#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"


template<typename T>
void launchUpdateKVBSCache(TensorWrapper<T> * kv_src, 
                           TensorWrapper<T> * kv_dist,
                                        int * parent_beam,
                         TensorWrapper<int> * his_len,
                         TensorWrapper<int> * layer_id);