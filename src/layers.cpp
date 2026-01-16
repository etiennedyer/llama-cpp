#include "layers.h"
#include <algorithm>
#include <cmath>
#include <cstring>

RoPE::RoPE(const LlamaConfig& conf) : freqs_cos({conf.seq_len, conf.head_dim / 2}), freqs_sin({conf.seq_len, conf.head_dim / 2}) {
    // RoPE:: refers to the RoPE struct in layers.h,
    // and RoPE::RoPE means we're using the constructor function RoPE() from the RoPE struct
    // note we don't declare a type for constructors

    // The Llama 3 hyperparameter for frequency
    float theta_base = 500000.0f; 
    
    int head_dim = conf.head_dim; // i.e., 128

    // 2. Pre-compute the "Thetas" (frequencies) for each dimension pair
    // Formula: theta_i = base^(-2i / dim)
    std::vector<float> thetas(head_dim / 2);
    for (int i = 0; i < head_dim / 2; ++i) {
        float exponent = -2.0f * i / head_dim;
        thetas[i] = powf(theta_base, exponent);
    }

    // 3. Build the full table for every possible position (0 to 8192)
    for (int pos = 0; pos < conf.seq_len; ++pos) {
        for (int i = 0; i < head_dim / 2; ++i) {
            float angle = pos * thetas[i];
            
            // Calculate index in the 1D data array
            int idx = pos * (head_dim / 2) + i;
            
            freqs_cos.data[idx] = cosf(angle);
            freqs_sin.data[idx] = sinf(angle);
        }
    }
}

void RoPE::forward(Tensor& q, Tensor& k, int pos) const {
    // We access the pre-computed tables using the current token position 'pos'
    // This gives us a pointer to the start of the row for this specific position.
    const float* cos_table = &freqs_cos.data[pos * (freqs_cos.shape[1])];
    const float* sin_table = &freqs_sin.data[pos * (freqs_sin.shape[1])];

    // apply to Query
    // We iterate in steps of 2 because we process pairs
    for (int i = 0; i < q.data.size(); i += 2) {
        
        // Find which "pair index" we are at within the rotation table.
        // We use modulo (%) because the rotation pattern repeats for every head.
        // e.g., if head_dim is 128 (64 pairs), pair 0 and pair 64 use the same angle.
        int head_dim_idx = (i / 2) % freqs_cos.shape[1];
        
        float x = q.data[i];
        float y = q.data[i+1];
        
        float c = cos_table[head_dim_idx];
        float s = sin_table[head_dim_idx];

        // rotation formula
        q.data[i]   = x * c - y * s;
        q.data[i+1] = x * s + y * c;
    }

    // apply to Key, same logic
    for (int i = 0; i < k.data.size(); i += 2) {
        int head_dim_idx = (i / 2) % freqs_cos.shape[1];
        
        float x = k.data[i];
        float y = k.data[i+1];
        float c = cos_table[head_dim_idx];
        float s = sin_table[head_dim_idx];

        k.data[i]   = x * c - y * s;
        k.data[i+1] = x * s + y * c;
    }
};

// construct the cache
KVCache::KVCache(const LlamaConfig& conf) : 
    current_pos(0), 
    k_cache({conf.n_layers, conf.seq_len, conf.n_kv_heads, conf.head_dim}), 
    v_cache({conf.n_layers, conf.seq_len, conf.n_kv_heads, conf.head_dim}) { };

// attention with raw pointers
// manually iterates over memory because we can't use Tensor objects here.
Tensor attention_impl(Tensor& Q, float* k_cache_layer, float* v_cache_layer, int pos, const LlamaConfig& conf) {
    
    int dim = conf.head_dim;
    int n_heads = conf.n_heads;
    int n_kv_heads = conf.n_kv_heads;
    int head_stride = n_kv_heads * dim; // How far to jump to get to the next token's data in the cache
    
    // output container -- same shape as Q, [n_heads * head_dim]
    Tensor output({n_heads * dim}); 
    std::fill(output.data.begin(), output.data.end(), 0.0f); // Initialize with 0

    // iterate over every Query head
    for (int h = 0; h < n_heads; h++) {
        
        // GQA: which KV head does this query head use?
        // e.g. If 32 Q-heads and 8 KV-heads, heads 0-3 all use KV-head 0.
        int kv_head = h / (n_heads / n_kv_heads);

        // Get pointer to the current Q vector
        float* q_vec = Q.data.data() + (h * dim);

        // calculate Attention Scores (Q @ K.T)
        std::vector<float> scores(pos + 1);
        
        for (int t = 0; t <= pos; t++) {
            // find K vector for token 't', head 'kv_head'
            // Formula: Start + (Time Step Offset) + (Head Offset)
            float* k_vec = k_cache_layer + (t * head_stride) + (kv_head * dim);
            
            // Dot Product
            float score = 0.0f;
            for (int i = 0; i < dim; i++) {
                score += q_vec[i] * k_vec[i];
            }
            scores[t] = score / sqrtf((float)dim);
        }

        // --- Step B: Softmax ---
        // 1. Find max for numerical stability
        float max_val = scores[0];
        for(int t=1; t<=pos; t++) if(scores[t] > max_val) max_val = scores[t];

        // 2. Exp and Sum
        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; t++) {
            scores[t] = expf(scores[t] - max_val);
            sum_exp += scores[t];
        }

        // 3. Normalize
        for (int t = 0; t <= pos; t++) {
            scores[t] /= sum_exp;
        }

        // --- Step C: Weighted Sum (Score @ V) ---
        // Get pointer to where we write the result
        float* out_vec = output.data.data() + (h * dim);

        for (int t = 0; t <= pos; t++) {
            // find V vector for token 't'
            float* v_vec = v_cache_layer + (t * head_stride) + (kv_head * dim);
            float weight = scores[t];

            // Accumulate
            for (int i = 0; i < dim; i++) {
                out_vec[i] += weight * v_vec[i];
            }
        }
    }

    return output;
}

// initialize TransformerBlock

TransformerBlock::TransformerBlock(const LlamaConfig& conf) : 
    n_heads(conf.n_heads), 
    head_dim(conf.head_dim), 
    n_kv_heads(conf.n_kv_heads),

    // 1. Attention Weights
    // Query: [dim, dim] 
    wq({conf.dim, conf.n_heads * conf.head_dim}), 
    
    // Key/Value: [dim, n_kv_heads * head_dim] (Smaller because of GQA)
    wk({conf.dim, conf.n_kv_heads * conf.head_dim}),
    wv({conf.dim, conf.n_kv_heads * conf.head_dim}),
    
    // Output: [dim, dim]
    wo({conf.n_heads * conf.head_dim, conf.dim}),

    // 2. FeedForward Weights
    // Gate/Up: [dim, hidden_dim] (Expands the vector)
    w1({conf.dim, conf.hidden_dim}),
    w3({conf.dim, conf.hidden_dim}),
    
    // Down: [hidden_dim, dim] (Shrinks it back)
    w2({conf.hidden_dim, conf.dim}),

    // 3. Normalization Weights
    // Just a 1D vector scaling each dimension
    rms_att_weight({conf.dim}),
    rms_ffn_weight({conf.dim}) 
    { }

Tensor TransformerBlock::forward(Tensor x, int pos, const LlamaConfig& conf, const RoPE& rope, KVCache& cache, int layer_idx) {

    Tensor x_norm = x; // copy so we don't lose the original x
    // rms norm
    rms_norm_inplace(x_norm, rms_att_weight);

    // get K/Q/V
    
    Tensor K = matmul(x_norm, wk);
    Tensor V = matmul(x_norm, wv);
    Tensor Q = matmul(x_norm, wq);

    // apply RoPE
    rope.forward(Q, K, pos);

    // GQA 
    // calculate offset (which transformer block we're on + which head)
    size_t layer_offset = layer_idx * conf.seq_len * conf.n_kv_heads * conf.head_dim;
    size_t pos_offset   = pos * conf.n_kv_heads * conf.head_dim;
    size_t base_offset  = layer_offset + pos_offset;

    std::memcpy(cache.k_cache.data.data() + base_offset, K.data.data(), K.data.size() * sizeof(float));
    // memcpy copies bytes from one location to another
    // memcpy(void *dest, const void *src, size_t count)
    // so here we're sending the data from the K matrix to the offset k_cache
    // *dest := address of the first byte where bytes will be written
    // cache.k_cache.data.data() + base_offset effectively says: 
    // "Start at the beginning of the array, and skip over base_offset number of floats."

    std::memcpy(cache.v_cache.data.data() + base_offset, V.data.data(), V.data.size() * sizeof(float));

    // Calculate Pointers to the start of this layer's history
    float* k_ptr = cache.k_cache.data.data() + layer_offset;
    float* v_ptr = cache.v_cache.data.data() + layer_offset;

    // 4.c)-f) attention
    Tensor attn_out = attention_impl(Q, k_ptr, v_ptr, pos, conf);

    // 5. projection
    Tensor output = matmul(attn_out, wo); 

    // 6. residuals
    // Add the original input 'x' back to the result
    add_inplace(output, x);

    // 7. FFNN
    
    // Create a residual copy for the SECOND part
    Tensor x_ffn = output; // Start with the result of attention

    // Norm 2
    rms_norm_inplace(x_ffn, rms_ffn_weight);

        // SwiGLU : w2( SiLU(w1(x)) * w3(x) )
    // We need 3 matrix multiplications.
    
    // A. Calculate the two "Gate" paths
    Tensor gate = matmul(x_ffn, w1); // shape: [dim, hidden_dim]
    Tensor up   = matmul(x_ffn, w3); // shape: [dim, hidden_dim]

    // B. Apply SiLU Activation to the 'gate' (w1)
    // SiLU(x) = x * sigmoid(x)
    for(int i=0; i<gate.data.size(); i++) {
        float val = gate.data[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        gate.data[i] = val * sigmoid;
    }
    // C. Element-wise Multiply (Gate * Up)
    // We can reuse 'gate' to store the result to save memory
    for(int i=0; i<gate.data.size(); i++) {
        gate.data[i] = gate.data[i] * up.data[i];
    }

    // D. Final Projection (Down)
    Tensor ffn_out = matmul(gate, w2);

    // 4. Second Residual Connection
    add_inplace(output, ffn_out);

    return output;

}
