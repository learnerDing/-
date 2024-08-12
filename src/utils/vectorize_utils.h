#pragma once //避免重定义
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>//hf数据类型才能用
//(RussWong)note: below 5 overloaded function can convert different scalar type data to specified vector type data.
//以下五个函数功能是可以将不同类型的标量数据转换为指定的向量类型的data
template<typename T_OUT, typename T_IN>
inline __device__ T_OUT scalar_cast_vec(T_IN val)
{
    return val;
}

template<>
inline __device__ half2 scalar_cast_vec<half2, float>(float val)
{
    return __float2half2_rn(val);
}

template<>
inline __device__ float4 scalar_cast_vec<float4, float>(float val)
{
    return make_float4(val, val, val, val);
}

template<>
inline __device__ float2 scalar_cast_vec<float2, float>(float val)
{
    return make_float2(val, val);
}

template<>
inline __device__ half2 scalar_cast_vec<half2, half>(half val)
{
    //(RussWong)note: __half2half2 cant be parsed by my nvcc compiler, so I give it up
    //return __half2half2(val);
    half2 res;
    res.x = val;
    res.y = val;
    return res;
}
//定义基本数据结构：向量 用来在cuda程序里面向量化读取数据
template<typename T>
struct Vec {
    using Type = T;//申明Type是T的别名
    static constexpr int size = 0;
};
template<>
struct Vec<half> {
    using Type = half2; 
    static constexpr int size = 2;//向量化长度
};
template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};
//(RussWong)note: temply dont know which LLM use two continuous elements do RoPE
struct TwoFloat2{
    float2 x;
    float2 y;
};
