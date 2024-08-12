#pragma once
#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <stdio.h>
#include <fstream>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include"src/kernels/update_kvbs_cache.h"

// beamsearch layer --- update kv cache

template<typename T>
__global__ void update_kvbs_cache(const T *k_src,
                                        T *k_dist,
                                  const int  beam_width,
                                  const int  layer_offset,
                                  const int  kv_head_num,
                                  const int  max_seq_len,
                                  const int  head_size,
                                  const int  history_length){
    //将[batch, beam_width] 所有数据进行搬运
    int batch_id = blockIdx.x / beam_width;
    int beam_id = blockIdx.x % beam_width;
    int head_id = blockIdx.y;
    int tx = threadIdx.x;
    // 将k_src [batch_size, beam_width, kv_head_num * head_size] 放到第history_length行
    T *k_cache_dist = k_dist + layer_offset;
    int src_offset = batch_id * beam_width * kv_head_num * head_size +
                     beam_id * kv_head_num * head_size + 
                     head_id * head_size +
                     tx;
    // k_dist{num_layers, batch_size * beam_width, head_num, max_seq_len, head_size}
    int dist_offset = batch_id * beam_width * kv_head_num * max_seq_len * head_size + 
                      beam_id * kv_head_num * max_seq_len * head_size + 
                      head_id * max_seq_len * head_size + 
                      history_length * head_size +
                      tx;
    k_cache_dist[dist_offset] = k_src[src_offset];
    
}
template<typename T>
__global__ void bs_memcpy(      T *k_dist,
                          const int beam_id,
                          const int parent_id,
                          const int layer_offset,
                          const int kv_head_num,
                          const int max_seq_len,
                          const int head_size){
    //k_dist num_layers * batch_size * beam_width * kv_head_num * max_seq_len * head_size;
    int row = blockIdx.x;
    int head_id = blockIdx.y;
    int tx = threadIdx.x;
    int src_offset = layer_offset + parent_id * kv_head_num *  max_seq_len * head_size
                      + head_id * max_seq_len * head_size;
    int dist_offset = layer_offset + beam_id   * kv_head_num *  max_seq_len * head_size
                       + head_id * max_seq_len * head_size;
    
    T *k_cache_src = k_dist + src_offset;
    T *k_cache_dist = k_dist + dist_offset;
    // k_dist{num_layers, batch_size * beam_width, head_num, max_seq_len, head_size}
    

    k_cache_dist[row * head_size + tx] = k_cache_src[row * head_size + tx]; 

}

// self-decoder的输入ksrc: shape[batch_size, beam_width, kv_head_num * head_size]   
// 将ksrc 存入对应parent[0], parent[1],..., parent[beam_width-1] 的kv-cache k_dist中 
// k_dist:shape [num_layers, batch_size * beam_width, head_num, max_seq_len, head_size]
template<typename T>
void launchUpdateKVBSCache(TensorWrapper<T> * kv_src, 
                           TensorWrapper<T> * kv_dist,
                                        int * parent,
                         TensorWrapper<int> * his_len,
                         TensorWrapper<int> * layer_id){
    
    const int batch_size = kv_src->shape[0];
    const int beam_width = kv_src->shape[1];
    const int kv_head_num = kv_src->shape[2];
    const int head_size = kv_src->shape[3];

    const int num_layers = kv_dist->shape[0];
    const int max_seq_len = kv_dist->shape[4];

    const int layer = layer_id->getVal();
    const int history_len = his_len->getVal();

    const int blockSize = head_size;
    T * kv_src_val = kv_src->data;
    T * kv_dist_val = kv_dist->data;
    
    // Attention: ksrc对应的parent (认为beam_id = 0 == parent[0], beam_id=1 == parent[1])
    // 如果不相等就需要将 parent[beam_id]的内存放到beam_id处
    // parent[0] = 0; parent[1] = 0; parent[1] 复制到 beam_id=1
    #pragma unroll
    for(int i = 0; i<batch_size; ++i){
        for(int j=0; j<beam_width; ++j){
            // 将整个batch parent[beam_id] != beam_id那么进行数据拷贝
            if(parent[i * batch_size + j] != j){
                dim3 grid_(history_len, kv_head_num);
                dim3 block_(blockSize);
                const int beam_id = j;
                const int parent_id = parent[i * beam_width + j];
                int layer_offset = layer * batch_size * kv_head_num * beam_width * max_seq_len * head_size;
                bs_memcpy<T><<<grid_, block_>>>(kv_dist_val, 
                                                beam_id,      parent_id, // parent_id指向内存复制一份给beam_id内存
                                                layer_offset, kv_head_num, max_seq_len, head_size);
                parent[i * beam_width + j] = beam_id;
            }
        }
    }
    dim3 grid(batch_size * beam_width, kv_head_num);
    dim3 block(blockSize);
    int layer_offset = layer * batch_size * beam_width * kv_head_num * max_seq_len * head_size;
    int history_length = 2;
    //将k_src [batch_size, beam_width, kv_head_num * head_size] 放到对应beam_id的第history_length行
    update_kvbs_cache<T><<<grid, block>>>(kv_src_val,     kv_dist_val,
                                          beam_width,  layer_offset, 
                                          kv_head_num, max_seq_len,
                                          head_size,   history_length);
}

template void launchUpdateKVBSCache(TensorWrapper<float> * kv_src, 
                                    TensorWrapper<float> * kv_dist,
                                                     int * parent_beam,
                                      TensorWrapper<int> * his_len,
                                      TensorWrapper<int> * layer_id);

template void launchUpdateKVBSCache(TensorWrapper<int> * kv_src, 
                                    TensorWrapper<int> * kv_dist,
                                                   int * parent_beam,
                                    TensorWrapper<int> * his_len,
                                    TensorWrapper<int> * layer_id);
                           