This is my implementation of Llama-3 in C++. It's part 2 of what I intend to be a 3 part series, covering a naive C++ implementation, optimization for CPU (blocked GEMM, vectorization, multithreading...), and optimization for GPU (GEMM/vectorized matmul kernel, flashattention kernel...). You can find part 1 [here](https://github.com/etiennedyer/llama-cpp/tree/part-1).

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

*Thoughts on implementation: Part 2, Profiling*

Profiling:
CPU: i5-7600K CPU @ 3.80GHz

first version:
Performance counter stats for './llama_main --tiny --iters 20':

6,369,559,149    cycles                                                                
5,680,240,739    instructions                     #    0.89  insn per cycle            
165,210,482      cache-misses                                                          

1.519746884 seconds time elapsed

1.482653000 seconds user
0.037016000 seconds sys

We're looking at instructions per cycle (IPC) and cache-misses per 1000 instructions (MPKI) to understand what's happening. IPC is fairly self-explanatory, it's the 

On an i5‑7600K (Kaby Lake/Skylake‑class), a rough ceiling for IPC is ~4 instructions retired per cycle. That usually means:

< 1.0 IPC: memory‑bound or branch‑heavy.
1–2 IPC: mixed workload, some stalls.
2–3.5 IPC: good compute‑bound kernels.
~4 IPC: very tight, well‑optimized, mostly in L1.

A CPU cache is a small, very fast memory that stores frequently used data or instructions so the CPU doesn't have to retrieve them in RAM. It's organized in levels (L1, L2, L3) where L1 is smallest and fastest (1 cache per CPU core), L2 is larger/slower (sometimes shared across cores), and L3 is largest/slowest (shared across all cores).

A cache miss happens when a CPU goes to look in a cache for a memory address, but doesn't find what it is looking for, so it needs to to fetch it from a slower level. (i.e., the CPU looks at L1 -> L2 -> L3 -> RAM)

Miss rate = cache-misses / cache-references. This tells you what fraction of cache accesses miss.
MPKI (misses per 1,000 instructions) = cache-misses / instructions * 1000.
Rules of thumb (for LLC misses):

MPKI < 1: very low (cache‑friendly).
MPKI 1–5: moderate.
MPKI > 10: high, likely memory‑bound.

IPC is what you're trying to optimize, and cache-missed is one indicator of what could be improved. If you have low IPC and high cache-misses, then that points to caching as one thing you should fix. If you have low IPC and low cache misses, then your issue is something else (e.g., poor vectorization).

In our case, we have the former. We will start by creating a tiled (or blocked) matrix multiplication. The idea behind tiled matmul is that there’s a lot of data that we reuse between computations, and so it’s more efficient to process it in tiles it in cache 
