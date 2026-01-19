This is my implementation of Llama-3 in C++. It's part 2 of what I intend to be a 3 part series, covering a naive C++ implementation, optimization for CPU (blocked GEMM/GEMV, vectorization, multithreading...), and optimization for GPU (GEMM/vectorized matmul kernel, FlashAttention kernel...). You can find part 1 [here].

You can get the real weights [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (about 15gb, BF16 safetensors, model-00001 to model-00004 + model.safetensors.index.json, behind an auth wall), and download them to ./weights/, or just run a small version with the --tiny flag.

You can read about how I'm optimizing it [here](https://github.com/etiennedyer/llama-cpp/optimization.md). The writeup serves as a good intro to tiled GEMV if you aren't familiar with the concept.

## Repo structure

├── README.md
├── optimization.md     // optimization writeup
├── src
│   ├── layers.cpp      // defines RoPE, attention, transformer forward pass, etc.
│   ├── layers.h
│   ├── main.cpp        // execution loop, define parameters here
│   ├── model.cpp       // defines the full forward loop
│   ├── model.h
│   ├── safetensors.cpp // loads weights BF16 weights
│   ├── safetensors.h
│   ├── tensor.cpp      // tensor class, matmul (GEMM/GEMV)
│   └── tensor.h
└── weights             // empty, fill with Llama-3 8B weights


## Running

To run:
Compile with

```console
g++ -std=c++17 -O2 -o llama_main src/*.cpp 
```

run with
```console
./llama_main 
```

runtime flags: 
```
--tiny
```

runs the loop with placeholder matrices, with parameters specified in main.cpp.

```
--tiny N 
```
to run the loop for N iterations (defaults to 1 if N ommited)

```
--prefill T
```
run in prefill mode with random prefill of size T. Only available in tiny mode

