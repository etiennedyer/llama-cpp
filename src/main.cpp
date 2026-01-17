#include "model.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <random>

// for debug / tiny mode, fills a tensor with noise
static void fill_tensor(Tensor& t, std::mt19937& rng, float scale = 0.02f) {
    std::normal_distribution<float> dist(0.0f, scale);
    for (float& v : t.data) v = dist(rng);
}

// fills all tensors with random weights for tiny mode
static void init_random_weights(Llama& model) {
    std::mt19937 rng(123);
    fill_tensor(model.token_embedding_table, rng);
    fill_tensor(model.rms_final_weight, rng);
    fill_tensor(model.w_cls, rng);
    for (auto& layer : model.layers) {
        fill_tensor(layer.wq, rng);
        fill_tensor(layer.wk, rng);
        fill_tensor(layer.wv, rng);
        fill_tensor(layer.wo, rng);
        fill_tensor(layer.w1, rng);
        fill_tensor(layer.w2, rng);
        fill_tensor(layer.w3, rng);
        fill_tensor(layer.rms_att_weight, rng);
        fill_tensor(layer.rms_ffn_weight, rng);
    }
}

int main(int argc, char** argv) {

    bool use_tiny = false;
    bool use_prefill = false;

    // default to 1 iteration
    int iters = 1;
    int prefill_len = 0;

    // parse arg for flags
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tiny") {
            use_tiny = true;
            if (i + 1 < argc && std::isdigit(argv[i + 1][0])) {
            iters = std::stoi(argv[++i]);
            }
        } else if (arg == "--prefill" && i + 1 < argc) {
            use_prefill = true;
            prefill_len = std::stoi(argv[++i]);
        }
    }

    try { 
        if (use_tiny) { 
            LlamaConfig conf( // configure custom small dimensions
                512,  // dim 
                2048, // hidden_dim 
                6,   // n_layers 
                8,   // n_heads
                4,   // n_kv_heads
                4096, // vocab_size
                128); // seq_len 

            Llama model(conf);
            KVCache cache(conf);
            init_random_weights(model);

            // intialize logits
            Tensor logits({conf.vocab_size});

            // if ran with prefill flag, create a random vector of length T
            if (use_prefill) {
                int T = prefill_len; // from --prefill T
                Tensor X({T, conf.dim});
                std::mt19937 rng(123);
                fill_tensor(X, rng);

                logits = model.forward_prefill(X, cache);

            } else { // run in decode mode

                // the token we will predict on
                int token = 1;
                int pos = 0;



                // loop for {iters} iterations
                for (int i = 0; i < iters; ++i) {
                    logits = model.forward(token, pos, cache);
                }
            }
            
            std::cout << "use_prefill: " << use_prefill << std::endl;
            std::cout << "prefill_len: " << prefill_len << std::endl;
            std::cout << "Iters: " << iters << std::endl;
            std::cout << "Logits size: " << logits.data.size() << std::endl;
            std::cout << "First logits: ";
            int to_print = 8;
            if (static_cast<size_t>(to_print) > logits.data.size()) {
                to_print = static_cast<int>(logits.data.size());
            }

            for (int i = 0; i < to_print; ++i) {
                std::cout << std::fixed << std::setprecision(6) << logits.data[i];
                if (i + 1 < to_print) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        } else {

        std::string weights_path = "./llama/weights/model.safetensors.index.json";

        LlamaConfig conf;
        Llama model(conf);
        KVCache cache(conf);

        std::cout << "Loading weights from: " << weights_path << std::endl;
        model.load_safetensors(weights_path);

        int token = 1;
        int pos = 0;
        Tensor logits = model.forward(token, pos, cache);

        std::cout << "Logits size: " << logits.data.size() << std::endl;
        std::cout << "First logits: ";
        int to_print = 8;
        if (static_cast<size_t>(to_print) > logits.data.size()) {
            to_print = static_cast<int>(logits.data.size());
        }
        for (int i = 0; i < to_print; ++i) {
            std::cout << std::fixed << std::setprecision(6) << logits.data[i];
            if (i + 1 < to_print) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
    } 

    return 0;
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }
}
