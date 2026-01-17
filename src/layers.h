#pragma once
#include "tensor.h"
#include <vector>

// 1. Hyperparameters
struct LlamaConfig {

    int dim;        // Transformer dimension, defined ahead of time, shows up in the size of E, attention calculation, RoPE, head dimension, other stuff 
    int hidden_dim;  // Size of the FFN inner layer (Step 7)
    int n_layers;     // Number of layers / transformer blocks, how many times we repeat Steps 3-7 (Step 8)
    int n_heads;      // Query heads
    int n_kv_heads;    // Key/Value heads (GQA)
    int vocab_size; // vocab size
    int seq_len;    // Context window size
    const int head_dim;

    // construct and initialize
    LlamaConfig(
        int dim_in = 4096,
        int hidden_dim_in = 14336,
        int n_layers_in = 32,
        int n_heads_in = 32,
        int n_kv_heads_in = 8,
        int vocab_size_in = 128256,
        int seq_len_in = 8192
    ) :
        dim(dim_in),
        hidden_dim(hidden_dim_in),
        n_layers(n_layers_in),
        n_heads(n_heads_in),
        n_kv_heads(n_kv_heads_in),
        vocab_size(vocab_size_in),
        seq_len(seq_len_in),
        head_dim(dim_in / n_heads_in) {}
};


// RoPE
// We declare it here, but implementation goes in .cpp
struct RoPE {
    Tensor freqs_cos;
    Tensor freqs_sin;
    
    RoPE(const LlamaConfig& conf); // declare constructor, uses seq_len and head_dim from conf
    // could have implemented the constructor here, but best practice is to have implementations in the cpp file
    void forward(Tensor& q, Tensor& k, int pos) const;
};

// KV Cache
struct KVCache {
    // We store Keys and Values for all layers in two giant tensors.
    // Shape: [n_layers, seq_len, n_kv_heads, head_dim]
    Tensor k_cache; 
    Tensor v_cache;

    // Which token are we currently generating? (0, 1, 2...)
    // This tells us where to write the new data in the cache.
    int current_pos; 

    KVCache(const LlamaConfig& conf);
};

// Transformer block
struct TransformerBlock {
    // Attention Weights (Step 4a: WQ, WK, WV)
    Tensor wq; 
    Tensor wk; 
    Tensor wv; 
    Tensor wo; // Output projection (Step 5)
    
    // FeedForward Weights (Step 7: SwiGLU)
    Tensor w1; //wgate
    Tensor w2; //wup
    Tensor w3; //wdown
    
    // Normalization Weights (Step 3 & 7)
    Tensor rms_att_weight; 
    Tensor rms_ffn_weight; 

    // We keep a reference to config to know 'd' and 'heads'
    int head_dim;
    int n_heads;
    int n_kv_heads;

    TransformerBlock(const LlamaConfig& conf);
    
    // The loop body: takes x, returns processed x
    Tensor forward(Tensor x, int pos, const LlamaConfig& conf, const RoPE& rope, KVCache& cache, int layer_idx); 

    Tensor forward_prefill(Tensor X, const LlamaConfig& conf, const RoPE& rope, KVCache& cache, int layer_idx);
};

