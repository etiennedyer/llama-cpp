This is my implementation of Llama-3 in C++. It's part 1 of what I intend to be a 3 part series, covering a naive C++ implementation, optimization for CPU (blocked GEMM, vectorization, multithreading...), and optimization for GPU (GEMM/vectorized matmul kernel, flashattention kernel...).

You can get the real weights [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (about 15gb, BF16 safetensors, model-00001 to model-00004 + model.safetensors.index.json, behind an auth wall), and download them to ./weights/, or just run a small version with the --tiny flag.

To run:
Compile with
```console
g++ -std=c++17 -O2 -o llama_main src/*.cpp 
```

runtime flags: 
--tiny
runs the loop with placeholder matrices, with parameters specified in main.cpp
optionally --tiny N to run the loop for N iterations (defaults to 1 if N ommited)


*Thoughts on implementation* 

Overall, this was more straightforward than I expected. The main implementation challenge I ran into was managing the KV cache.  I initially planned on just using matmul for the multiplication attention step (i.e., Q @ K.T), but realized that this is inefficient because of the way we actually construct the cache. My mental model was that we'd start with a small K/V cache and append the new K/V to the existing cache tensor. But this is in fact very inneficient, because we'd need to copy all the old data to a new address in memory each time.

So what we do instead is initialize a tensor of 0s of size [number_of_layers, sequence_length, number_of_kv_heads, head_dimension], and write directly to memory to replace the 0s with the values we compute. Now, to do the matrix multiplication, we could extract the data to a matrix object (a member of our Tensor struct of size [current_position + 1, head_dim]) and use our matmul() algorithm, but this is also expensive, as we'd need to copy the necessary data to a new location in memory. Instead, we can compute the dot product for each entry of the output matrix by fetching the data at the correct address in memory.
