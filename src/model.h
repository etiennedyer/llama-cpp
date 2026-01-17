#pragma once
#include <vector>
#include <string>
#include "layers.h" 

struct Llama {
    // global config
    LlamaConfig conf;

    // global Weights (Outside the blocks)
    Tensor token_embedding_table; // [vocab_size, dim]
    Tensor rms_final_weight;      // [dim]
    Tensor w_cls; // [dim, vocab_size], Unembedding matrix

    // 32 transformer blocks go here
    std::vector<TransformerBlock> layers;

    // compute the cos/sin tables once here and share them with all layers.
    RoPE rope;

    // constructor
    Llama(const LlamaConfig& config);
    void load_safetensors(const std::string& path);

    // forward diffusion 
    Tensor forward(int token, int pos, KVCache& cache);
    Tensor forward_prefill(const Tensor& X_in, KVCache& cache);
    //outputs logits
};