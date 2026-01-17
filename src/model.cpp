#include "model.h"
#include "safetensors.h"
#include <cmath>
#include <cstring>

// Constructor: Build the 32 layers
Llama::Llama(const LlamaConfig& config) : 
    conf(config), 
    rope(config),
    token_embedding_table({conf.vocab_size, conf.dim}),
    rms_final_weight({conf.dim}),
    w_cls({conf.dim, conf.vocab_size}) {
    // Create the 32 blocks
    layers.reserve(conf.n_layers);
    for(int i = 0; i < conf.n_layers; i++) {
        layers.emplace_back(conf);
    }
}

Tensor Llama::forward(int token, int pos, KVCache& cache) {
    
    // embedding for token x_i is the (token_ID_x_i)th row of E.
    Tensor x({conf.dim});
    float* embedding_row = token_embedding_table.data.data() + (token * conf.dim);
    std::memcpy(x.data.data(), embedding_row, conf.dim * sizeof(float));

    // loop over layers
    for(int i = 0; i < conf.n_layers; i++) {
        x = layers[i].forward(x, pos, conf, rope, cache, i);
    }

    // final normalization
    rms_norm_inplace(x, rms_final_weight);

    // unembed
    Tensor logits = matmul(x, w_cls); 

    return logits;
}

Tensor Llama::forward_prefill(const Tensor& X_in, KVCache& cache) {
    
    Tensor X = X_in; // copy so we can modify

    // loop over layers
    for(int i = 0; i < conf.n_layers; i++) {
        X = layers[i].forward_prefill(X, conf, rope, cache, i);
    }

    // final normalization
    rms_norm_inplace(X, rms_final_weight);

    // unembed
    Tensor logits = matmul(X, w_cls); 

    return logits;
}

void Llama::load_safetensors(const std::string& path) {
    SafeTensorsLoader loader(path);

    loader.load_into("model.embed_tokens.weight", token_embedding_table, false);

    for (int i = 0; i < conf.n_layers; ++i) {
        const std::string prefix = "model.layers." + std::to_string(i) + ".";
        loader.load_into(prefix + "self_attn.q_proj.weight", layers[i].wq, true);
        loader.load_into(prefix + "self_attn.k_proj.weight", layers[i].wk, true);
        loader.load_into(prefix + "self_attn.v_proj.weight", layers[i].wv, true);
        loader.load_into(prefix + "self_attn.o_proj.weight", layers[i].wo, true);
        loader.load_into(prefix + "input_layernorm.weight", layers[i].rms_att_weight, false);
        loader.load_into(prefix + "post_attention_layernorm.weight", layers[i].rms_ffn_weight, false);
        loader.load_into(prefix + "mlp.gate_proj.weight", layers[i].w1, true);
        loader.load_into(prefix + "mlp.up_proj.weight", layers[i].w3, true);
        loader.load_into(prefix + "mlp.down_proj.weight", layers[i].w2, true);
    }

    loader.load_into("model.norm.weight", rms_final_weight, false);
    loader.load_into("lm_head.weight", w_cls, true);
}
