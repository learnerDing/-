#include "src/kernels/build_casual_mask.h"
// XuLin-1017: 此算子仅使用在context decoder阶段，用于遮盖掉seq当前位置之后的信息，防止模型使用未来的信息
// 而self decoder是一个自回归模型，本就没有未来的信息
// mask shape =  [bs, max_q_len, max_k_len] 因为Q矩阵大小为n*hidden_size，Kt矩阵shape:hidden_size*m，所以Q*Kt矩阵为n*m维度
template<typename T>
__global__ void BuildCausalMasksConsideringContextPastKV(T* mask,//mask张量开头的地址
                                                const int* q_lens,  //input lens,q数组， shape=[batch size]   batchsize个句子
                                                const int* k_lens,  //context lens, shape=[batch size]
                                                int max_q_len,  // max(q_lens)
                                                int max_k_len){ // max(k_lens)
    int tid = threadIdx.x;
    // XuLin-1017: 核函数共分配了bs个block，可以方便得通过block id来访问q_lens和k_lens数组中的值
    // 一个block负责处理一个bs大小中的数
    int qlen = q_lens[blockIdx.x];
    int klen = k_lens[blockIdx.x];
    // 偏移一个bs大小的空间
    // 即blockIdx.x==0时，指向mask数组的开头；blockIdx.x==1时，指向mask数组偏移了max_q_len * max_k_len大小后的位置
    mask += blockIdx.x * max_q_len * max_k_len;
    // offset用于表示每个bs内部的偏移
    int offset = threadIdx.x;
    // note: this judgement confirms we dont exceed data boundry
    while (offset < max_q_len * max_k_len){//循环生成矩阵shape为max_q_len行，max_k_len列
        // 分别求出行号q和列号k
        int q = offset / max_k_len;//该线程（数据）在矩阵的第q行
        int k = offset % max_k_len;//该线程（数据）在矩阵的第k列
        // 此处与视频中的代码不同，k考虑了多轮对话的上下文序列，但设置mask时 k >= klen - qlen 将旧序列一并遮去了
        // 下图为支持多轮对话的mask，第二轮对话中有些token需要掩盖，但是第一轮对话中的所有token都是已知的，所有左边矩阵全为一
        // 1 1 1 | 1 -inf -inf
        // 1 1 1 | 1   1  -inf
        // 1 1 1 | 1   1    1
        // 下图为不支持多轮对话的mask，上一轮矩阵数据全部被屏蔽
        // -inf -inf -inf | 1 -inf -inf
        // -inf -inf -inf | 1   1  -inf
        // -inf -inf -inf | 1   1    1
        // "|"符号前表示旧的对话序列，符号后表示当前轮的对话序列
        bool is_one = q < qlen && k < klen && k <= q + (klen - qlen) && k >= klen - qlen;//判断该矩阵哪些位置应该填1
        mask[offset] = static_cast<T>(is_one);

        // 保证遍历完一个bs中所有的空间
        offset += blockDim.x;
    }
}

template<typename T>
void launchBuildCausalMasks(TensorWrapper<T>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens)
{
    int batch_size = mask->shape[0];
    int max_q_len = mask->shape[1];
    int max_k_len = mask->shape[2];
    // XuLin-1017: 此处的max_q_len和max_k_len是经过统计后得出的外部输入
    BuildCausalMasksConsideringContextPastKV<T><<<batch_size, 256>>>(mask->data, q_lens->data, k_lens->data, max_q_len, max_k_len);
}

template void launchBuildCausalMasks(TensorWrapper<float>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens);

template void launchBuildCausalMasks(TensorWrapper<half>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens);
