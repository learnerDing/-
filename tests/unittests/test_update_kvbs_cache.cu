#include <algorithm> // std::fill_n
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <stdlib.h>  // rand
#include <string>    // std::string
#include <vector>    // std::vector

#include <cuda.h>
#include "src/kernels/update_kvbs_cache.h"
int main()
{

    const int batch_size = 1;
    const int beam_width = 2;
    const int kv_head_num = 4;
    const int head_size = 8;
    const int num_layers = 1;
    // const int layer_id = 0;
    const int max_seq_len = 4;

    
    int *history_len_h_ = new int[batch_size];
    history_len_h_[0] = 2;
    int history_length_h = 2;    

    int size_src = batch_size * beam_width * kv_head_num * head_size;//1 * 2* 4 * 8 = 64
    int size_dist = num_layers * batch_size * beam_width * kv_head_num * max_seq_len * head_size;
    // 1 * 1 * 2 * 4 * 4 * 8 = 256
    int *k_src_h = new int[size_src];
    int *k_dist_h = new int [size_dist];
    for(int i=0; i<size_src; ++i){
        k_src_h[i] = i;
    }
    int pre_temp = num_layers * batch_size * beam_width * kv_head_num;
    for(int i=0; i < pre_temp; ++i){
        for(int j=0; j<history_length_h; ++j){
            for(int k=0; k<head_size; ++k){
                k_dist_h[i * max_seq_len * head_size + // note: max_seq_len
                         j * head_size +
                         k] = k + i;
            }
        }
    }
    int* k_src_d;
    int * k_dist_d;
    int *layer_id_h = new int[batch_size];
    layer_id_h[0] = 0;
    // int *history_len_d;
    CHECK(cudaMalloc((void**)&k_src_d, sizeof(int) * size_src));
    CHECK(cudaMalloc((void**)&k_dist_d, sizeof(int) * size_dist));

    // CHECK(cudaMalloc((void**)&history_len_d, sizeof(int) * batch_size));
    CHECK(cudaMemcpy(k_src_d, k_src_h, sizeof(int) * size_src, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(k_dist_d, k_dist_h, sizeof(int) * size_dist, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(history_len_d, history_len_h_ , sizeof(int) * batch_size, cudaMemcpyHostToDevice));

    int *parent = new int [batch_size * 2];
    parent[0] = 0;
    parent[1] = 0;

    DataType type_int = getTensorType<int>();
    // 将k_src [batch_size, beam_width, kv_head_num * head_size]
    TensorWrapper<int> *in_ksrc = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, beam_width, kv_head_num, head_size}, k_src_d);
    // k_dist:shape [num_layers, batch_size * beam_width, head_num, max_seq_len, head_size]
    TensorWrapper<int> *in_kdist = new TensorWrapper<int>(Device::GPU, type_int, {num_layers, batch_size, beam_width, kv_head_num, max_seq_len, head_size}, k_dist_d);
    // TensorWrapper<float> *in_vsrc = new TensorWrapper<float>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size}, d_v_src);
    TensorWrapper<int> *history_len = new TensorWrapper<int>(Device::CPU, type_int, {batch_size}, history_len_h_);
    TensorWrapper<int> *layer_id = new TensorWrapper<int>(Device::CPU, type_int, {batch_size}, layer_id_h);


    launchUpdateKVBSCache(in_ksrc, 
                          in_kdist,
                          parent,
                          history_len,
                          layer_id);
                    

    CHECK(cudaMemcpy(k_dist_h, k_dist_d, sizeof(int) * size_dist, cudaMemcpyDeviceToHost));
    // k_dist = num_layers * batch_size * beam_width * kv_head_num * max_seq_len * head_size;
    for(int i=0; i<num_layers; ++i){
        for(int j=0; j<batch_size; ++j){
            std::cout<<"batch: "<<j<<std::endl;
            for(int h=0; h < beam_width;++h){
                std::cout<<"beam: "<<h<<std::endl;
                for(int k=0; k<kv_head_num; ++k){
                    std::cout<<"head_id: "<<k % kv_head_num<<std::endl;
                    for(int z=0; z<max_seq_len; ++z){
                        std::cout<<"lane_id: "<<z<<"--";
                        for(int w=0; w< head_size; ++w){
                            std::cout.width(3);
                            std::cout<<k_dist_h[i * batch_size * beam_width * kv_head_num * max_seq_len * head_size + 
                                            j * beam_width * kv_head_num * max_seq_len * head_size +
                                            h * kv_head_num * max_seq_len * head_size +
                                            k * max_seq_len * head_size +
                                            z * head_size
                                            +w] <<" ";
                        }
                        std::cout<<std::endl;
                    }
                    std::cout<<std::endl;
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
    }
        
    delete k_src_h;
    delete k_dist_h;
    delete layer_id_h;
    delete history_len_h_;
    delete parent;
    cudaFree(k_src_d);
    cudaFree(k_dist_d);
    // cudaFree(layer_id_d);
    // cudaFree(history_len_d);


}