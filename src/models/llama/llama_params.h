#pragma once
struct LLaMAAttentionStaticParams {
    int   rotary_embedding_dim;//旋转编码维度
    float rotary_embedding_base;
    int   max_position_embeddings;//最大位置编码
    bool  use_dynamic_ntk; // for dyn scaling rope 是否使用dynamic_ntk
};

// (RussWong)note: llama类模型里面动态改变的变量, 注意非全部必需
struct LLaMAAttentionDynParams {
    int batch_size;
    int num_tokens;
    int max_q_len;
    int max_k_len;
    int num_layers;
    bool is_ctx = false;
};
