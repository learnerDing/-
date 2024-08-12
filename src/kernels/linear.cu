#include <iostream>
#include <fstream>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/linear.h"
// TODO: when abstracted weight class, replace T with class
// all matmul cases:模型中所有的矩阵乘法计算
//qkv线性变换
// context qkv linear: [num_tokens, qhiddenunits] * [qhiddenunits, hiddenunits] = {num_tokens, qkv_head_num,  head_size}
// context attn output linear: {num_tokens, head_num, head_size} * {q hidden units, q hidden units} = {num_tokens, q hidden units}
// selfattention qkv linear: [bs, q hidden units] * [qhiddenunits, hiddenunits] = {bs, qkv_head_num,  head_size}}
// selfattention attn output linear: {batch_size, q hidden_units} * [qhiddenunits, qhiddenunits] = [bs, q hiddenunits]
// lmhead linear: [bs, q hidden units] * [vocab size, q hiden units], need transpose B
// gate:[bs/token nums, q hidden units] * [q hidden units, inter size] = [bs/token nums, inter size]
// up:[bs/token nums, q hidden units] * [q hidden units, inter size] = [bs/token nums, inter size]
// fusedGateUpGemm: [bs/token nums, q hidden units] * [q hidden units, 2 * inter size] = [bs/token nums, 2 * inter size]
// down:[bs/token nums, inter size] * [q hidden units, inter size] = [bs/token nums, q hidden units]
template <typename T>
void launchLinearGemm(TensorWrapper<T> *input,
                      BaseWeight<T> &weight,
                      TensorWrapper<T> *output,
                      cublasWrapper *cublas_wrapper,//关于cublas的一切操作都封装到cublas类里面
                      bool trans_a,//a矩阵是否转置
                      bool trans_b)//b矩阵是否转置
{
    int Am = weight.shape[1];//A矩阵为weight
    int Ak = weight.shape[0];
    int Bk = input->shape[1];//B矩阵为输入
    int Bn = input->shape[0];
    int Cm = output->shape[1];
    int Cn = output->shape[0];
    // for ctx attn and self attn qkv linear, assume [bs/token nums, qkv h ead num, head size]
    // for gate & up linear, assume weight.shape=[hidden,2*intersize], output.shape=[bs, 2, inter size]
    Cm = output->shape.size() == 3 ? output->shape[1] * output->shape[2] : output->shape[1];
    // for ctx attn output linear
    Bk = input->shape.size() == 3 ? input->shape[1] * input->shape[2] : input->shape[1];
    int lda = Am;
    int ldb = Bk;
    int ldc = Cm;

    // for lmhead linear and ffn all lieanrs
    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (!trans_a && !trans_b)//如果AB都不转置
    {
        LLM_CHECK_WITH_INFO(Ak == Bk, "2nd dim of input MUST = 1st dim of weight");
    }
    cublas_wrapper->Gemm(transA,
                         transB,
                         trans_b ? Ak : Am, // m
                         Cn,                // n, when load real weight, lmhead weight is same as pre embedding, which shape = [vocab, hidden], so here should transpose b
                         Bk,
                         weight.data,  // A, cur_input_len is for context decoder lmhead
                         lda,          // lda
                         input->data,  // B
                         ldb,          // ldb 
                         output->data, // C
                         ldc,          // ldc
                         1.0f,
                         0.0f);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#else
#endif
}

template <typename T>
void launchLinearStridedBatchGemm(TensorWrapper<T> *input1,//shape:[batchsize,head nums,seqlen,headsize]
                                  TensorWrapper<T> *input2,//shape:[batchsize,head nums,seqlen,headsize]
                                  TensorWrapper<T> *output,
                                  cublasWrapper *cublas_wrapper,
                                  bool trans_a,
                                  bool trans_b)
{
    // B.T A.T = C.T
    // TODO:currently only consider trans_b
    int Bm = input1->shape[2]; // 输入tensor的第三个维度:seqlen,也就是矩阵的行数     // len q
    int Bk = input1->shape[3]; // 输入tensor的第四个维度:headsize,也就是矩阵的列数
    int Ak = input2->shape[2]; // len k       // len k
    int An = input2->shape[3]; // head size   // head size 多头注意力的头的数量
    int Cm = output->shape[2]; // len q       // len q
    int Cn = output->shape[3]; // len k       // head size
    int lda = An;
    int ldb = Bk; // ld should be val before transpose
    int ldc = Cn;
    int64_t strideA = Ak * An; //batchsize*head num // stride should be val after transpose
    int64_t strideB = Bm * Bk;
    int64_t strideC = Cm * Cn;
    // TODO:check 4nd dim of input = 3rd dim of weight
    // TODO:check batchCount of two matrix is equal
    int batchCount = input1->shape[0] * input1->shape[1];

    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublas_wrapper->stridedBatchedGemm(transA,
                                       transB,
                                       Cn,           // m
                                       Cm,           // n
                                       Bk,           // k
                                       input2->data, // A,[Bk, Bn]=[bs, head num,  head size,max k len]
                                       lda,
                                       strideA,
                                       input1->data, // B [Ak, An]=[bs, head num,  head size,max q len]
                                       ldb,
                                       strideB,
                                       output->data, // C [[bs, head num,  max k len, max q len]
                                       ldc,
                                       strideC,
                                       batchCount,
                                       1.0f,
                                       0.0f);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#else
#endif
}

template void launchLinearGemm(TensorWrapper<float> *input,
                               BaseWeight<float> &weight,
                               TensorWrapper<float> *output,
                               cublasWrapper *cublas_wrapper,
                               bool trans_a,
                               bool trans_b);

template void launchLinearGemm(TensorWrapper<half> *input,
                               BaseWeight<half> &weight,
                               TensorWrapper<half> *output,
                               cublasWrapper *cublas_wrapper,
                               bool trans_a,
                               bool trans_b);

template void launchLinearStridedBatchGemm(TensorWrapper<float> *input1,
                                           TensorWrapper<float> *input2,
                                           TensorWrapper<float> *output,
                                           cublasWrapper *cublas_wrapper,
                                           bool trans_a,
                                           bool trans_b);

template void launchLinearStridedBatchGemm(TensorWrapper<half> *input1,
                                           TensorWrapper<half> *input2,
                                           TensorWrapper<half> *output,
                                           cublasWrapper *cublas_wrapper,
                                           bool trans_a,
                                           bool trans_b);
